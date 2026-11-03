"""FarmWise web app — FastAPI backend.

This is the Session 2 web layer. It does not reimplement any AI logic; it simply
wraps the Session 1 ``farmwise`` package:

* ``POST /api/predict``  -> ``farmwise.ml.predict.predict_crop``
* ``POST /api/ask``      -> ``farmwise.rag.assistant.ask`` (RAG) or
                            ``ask_without_rag`` (bare LLM), chosen per request
* ``POST /api/chat``     -> ``farmwise.rag.assistant.chat``

Reading is open to anyone who can reach the app. Changing the knowledge base goes
through :func:`require_admin`.
"""

from __future__ import annotations

import ipaddress
import os
import re
import secrets
import tempfile
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import urlparse

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from farmwise import config
from farmwise.llm.client import OllamaError, is_ollama_running
from farmwise.ml.predict import CropFeatures, ModelNotFoundError, predict_crop
from farmwise.rag.assistant import ask, ask_without_rag, chat
from farmwise.rag.indexer import index_document, index_url
from farmwise.rag.loader import SUPPORTED_SUFFIXES, load_file
from farmwise.rag.vector_store import ChromaVectorStore

STATIC_DIR = Path(__file__).parent / "static"

MAX_JSON_BODY_BYTES = 1024 * 1024
UPLOAD_PATH = "/api/ingest/file"
MULTIPART_OVERHEAD_BYTES = 1024 * 1024
_READ_BLOCK_BYTES = 1024 * 1024


def _max_upload_bytes() -> int:
    return config.MAX_UPLOAD_MB * 1024 * 1024


def _too_large_message() -> str:
    return f"That file is over the {config.MAX_UPLOAD_MB} MB upload limit."


class BodySizeLimitMiddleware:
    """Refuse oversized requests before the body is buffered to disk.

    FastAPI parses a multipart body before it runs dependencies, so this has to
    sit in front of the route rather than inside it.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        if scope["path"] == UPLOAD_PATH:
            limit = _max_upload_bytes() + MULTIPART_OVERHEAD_BYTES
            message = _too_large_message()
        else:
            limit = MAX_JSON_BODY_BYTES
            message = "Request body is too large."

        declared = dict(scope["headers"]).get(b"content-length", b"")
        if declared.isdigit() and int(declared) > limit:
            response = JSONResponse({"detail": message}, status_code=413)
            await response(scope, receive, send)
            return

        received = 0

        async def limited_receive():
            nonlocal received
            event = await receive()
            if event["type"] == "http.request":
                received += len(event.get("body", b""))
                if received > limit:
                    raise HTTPException(status_code=413, detail=message)
            return event

        await self.app(scope, limited_receive, send)


app = FastAPI(title="FarmWise", description="Smart farming assistant (Session 2 web app)")
app.add_middleware(BodySizeLimitMiddleware)


@app.exception_handler(RequestValidationError)
async def readable_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Return one readable message that the pages can show as it is.

    The default response echoes the rejected input, and a value such as infinity
    cannot be written as JSON, which would turn a bad request into a server error.
    """
    problems = []
    for error in exc.errors():
        field = ".".join(str(part) for part in error["loc"] if part != "body")
        problems.append(f"{field}: {error['msg']}" if field else error["msg"])
    return JSONResponse({"detail": "; ".join(problems)}, status_code=422)


# --- access control --------------------------------------------------------

_LOCAL_HOSTNAMES = {"localhost", "127.0.0.1", "::1"}
_PROXY_HEADERS = ("x-forwarded-for", "x-real-ip", "forwarded")


def _is_direct_local_request(request: Request) -> bool:
    """True only for a request made on this machine, not through a proxy.

    The Host check stops a web page using DNS tricks to reach the local app.
    """
    host = request.client.host if request.client else ""
    try:
        is_loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
    hostname = urlparse(f"//{request.headers.get('host', '')}").hostname
    return (
        is_loopback
        and hostname in _LOCAL_HOSTNAMES
        and not any(name in request.headers for name in _PROXY_HEADERS)
    )


def require_admin(
    request: Request,
    x_admin_token: Optional[str] = Header(default=None),
) -> None:
    """Guard every route that changes the knowledge base."""
    # Browsers send Origin on cross-site posts, including plain form uploads that
    # skip the CORS preflight, so another website cannot drive these routes.
    origin = request.headers.get("origin")
    if origin and urlparse(origin).netloc != request.headers.get("host"):
        raise HTTPException(status_code=403, detail="Cross-site requests are not allowed.")

    if config.ADMIN_TOKEN:
        supplied = (x_admin_token or "").encode()
        if not secrets.compare_digest(supplied, config.ADMIN_TOKEN.encode()):
            raise HTTPException(status_code=401, detail="A valid admin token is required.")
        return

    if not _is_direct_local_request(request):
        raise HTTPException(
            status_code=403,
            detail="The knowledge base can only be changed on the host machine. "
                   "Set FARMWISE_ADMIN_TOKEN to manage it remotely.",
        )


# --- request models --------------------------------------------------------

def _measurement(description: str = "") -> float:
    return Field(..., description=description, allow_inf_nan=False)


class CropIn(BaseModel):
    N: float = _measurement("nitrogen")
    P: float = _measurement("phosphorus")
    K: float = _measurement("potassium")
    temperature: float = _measurement()
    humidity: float = _measurement()
    ph: float = _measurement()
    rainfall: float = _measurement()


class QuestionIn(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    use_rag: bool = True


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=4000)


class ChatIn(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1, max_length=50)
    use_rag: bool = True

    @field_validator("messages")
    @classmethod
    def last_message_is_from_the_user(cls, messages: list[ChatMessage]) -> list[ChatMessage]:
        if messages[-1].role != "user":
            raise ValueError("The last message must come from the user.")
        return messages


class UrlIn(BaseModel):
    url: str = Field(..., min_length=4, max_length=2048)
    crawl: bool = False


class RemoveIn(BaseModel):
    source: str = Field(..., min_length=1)


def _safe_filename(name: str) -> str:
    """Keep a readable, filesystem-safe version of an uploaded file's name."""
    name = Path(name).name  # strip any path components
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    return name or "upload.txt"


# --- pages -----------------------------------------------------------------

@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/crop")
def crop_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "crop.html")


@app.get("/chat")
def chat_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "chat.html")


@app.get("/knowledge")
def knowledge_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "knowledge.html")


# --- api -------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict:
    ollama = is_ollama_running()
    index_count = 0
    try:
        index_count = ChromaVectorStore().count()
    except Exception:
        index_count = 0
    return {
        "ollama_running": ollama,
        "llm_model": config.LLM_MODEL,
        "embedding_model": config.EMBEDDING_MODEL,
        "index_chunks": index_count,
        "model_trained": config.MODEL_PATH.exists(),
    }


@app.get("/api/admin/status")
def admin_status() -> dict:
    """Tell the knowledge page whether to show the admin token field."""
    return {"token_required": bool(config.ADMIN_TOKEN)}


@app.post("/api/predict")
def predict(inp: CropIn) -> dict:
    try:
        pred = predict_crop(CropFeatures(**inp.model_dump()))
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return pred.to_dict()


@app.post("/api/ask")
def ask_endpoint(inp: QuestionIn) -> dict:
    try:
        answer = ask(inp.question) if inp.use_rag else ask_without_rag(inp.question)
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return answer.to_dict()


@app.post("/api/chat")
def chat_endpoint(inp: ChatIn) -> dict:
    try:
        answer = chat([m.model_dump() for m in inp.messages], use_rag=inp.use_rag)
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return answer.to_dict()


# --- knowledge base (ingestion) --------------------------------------------

@app.get("/api/sources")
def sources() -> dict:
    """List what is currently in the RAG index, source by source."""
    try:
        counts = ChromaVectorStore().distinct_sources()
    except Exception:
        counts = {}
    return {
        "sources": [{"source": s, "chunks": n} for s, n in sorted(counts.items())],
        "total_chunks": sum(counts.values()),
    }


@app.post("/api/sources/remove", dependencies=[Depends(require_admin)])
def remove_source(inp: RemoveIn) -> dict:
    """Remove a source (all its chunks) from the index."""
    try:
        removed = ChromaVectorStore().delete_source(inp.source)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not remove source: {exc}")
    return {"removed": removed, **sources()}


@app.post("/api/ingest/url", dependencies=[Depends(require_admin)])
def ingest_url(inp: UrlIn) -> dict:
    """Fetch a public web page or PDF by URL and add it to the RAG index.

    With ``crawl=true`` the page's relevant links (reports and PDFs) are followed
    and indexed too, so a thin landing page still contributes real content.
    """
    try:
        report = index_url(ChromaVectorStore(), inp.url, crawl=inp.crawl)
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not ingest URL: {exc}")
    return {
        "source": inp.url,
        "pages_added": report.documents,
        "added_sources": report.sources,
        "chunks_added": report.chunks,
        **sources(),
    }


def _index_and_keep(staged: Path, dest: Path) -> int:
    """Index a staged upload, then move it into the guides folder.

    The file only joins the guides once it has indexed, so a bad upload cannot
    break a later rebuild or replace a good file with the same name.
    """
    report = index_document(ChromaVectorStore(), load_file(staged))
    os.replace(staged, dest)
    return report.chunks


@app.post(UPLOAD_PATH, dependencies=[Depends(require_admin)])
async def ingest_file(file: UploadFile = File(...)) -> dict:
    """Upload a supported document, index it, and keep it in the guides folder."""
    name = _safe_filename(file.filename or "upload")
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. "
                   f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}.",
        )
    if name.lower() == "readme.md":
        raise HTTPException(status_code=400, detail="README.md is a reserved name. Rename the file.")

    config.GUIDES_DIR.mkdir(parents=True, exist_ok=True)
    dest = config.GUIDES_DIR / name

    # Stage inside the guides folder so the final move is atomic. load_guides only
    # reads files directly in that folder, so it never sees this directory.
    with tempfile.TemporaryDirectory(dir=config.GUIDES_DIR, prefix=".upload-") as staging:
        staged = Path(staging) / name
        size = 0
        with staged.open("wb") as out:
            while block := await file.read(_READ_BLOCK_BYTES):
                size += len(block)
                if size > _max_upload_bytes():
                    raise HTTPException(status_code=413, detail=_too_large_message())
                out.write(block)

        try:
            # Embedding is slow, so keep it off the event loop.
            chunks = await run_in_threadpool(_index_and_keep, staged, dest)
        except OllamaError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not index file: {exc}")
    return {"source": name, "chunks_added": chunks, **sources()}


# Serve the rest of the static assets (css/js) under /static.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
