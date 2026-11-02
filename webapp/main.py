"""FarmWise web app — FastAPI backend.

This is the Session 2 web layer. It does not reimplement any AI logic; it simply
wraps the Session 1 ``farmwise`` package:

* ``POST /api/predict``  -> ``farmwise.ml.predict.predict_crop``
* ``POST /api/ask``      -> ``farmwise.rag.assistant.ask`` (RAG) or
                            ``ask_without_rag`` (bare LLM), chosen per request

The same two entry points, now reachable from a browser.
"""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from farmwise import config
from farmwise.llm.client import OllamaError, is_ollama_running
from farmwise.ml.predict import CropFeatures, ModelNotFoundError, predict_crop
from farmwise.rag.assistant import ask, ask_without_rag, chat
from farmwise.rag.indexer import index_document, index_guides
from farmwise.rag.loader import SUPPORTED_SUFFIXES, Document, load_file, load_url
from farmwise.rag.vector_store import ChromaVectorStore

app = FastAPI(title="FarmWise", description="Smart farming assistant (Session 2 web app)")

STATIC_DIR = Path(__file__).parent / "static"


# --- request models --------------------------------------------------------

class CropIn(BaseModel):
    N: float = Field(..., description="nitrogen")
    P: float = Field(..., description="phosphorus")
    K: float = Field(..., description="potassium")
    temperature: float
    humidity: float
    ph: float
    rainfall: float


class QuestionIn(BaseModel):
    question: str = Field(..., min_length=1)
    use_rag: bool = True


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatIn(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1)
    use_rag: bool = True


class UrlIn(BaseModel):
    url: str = Field(..., min_length=4)


SUPPORTED_UPLOAD_SUFFIXES = SUPPORTED_SUFFIXES


def _safe_filename(name: str) -> str:
    """Keep a readable, filesystem-safe version of an uploaded file's name."""
    name = Path(name).name  # strip any path components
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    return name or "upload.txt"


# --- pages -----------------------------------------------------------------

@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


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
        from farmwise.rag.vector_store import ChromaVectorStore
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


@app.post("/api/ingest/url")
def ingest_url(inp: UrlIn) -> dict:
    """Fetch a web page or PDF by URL and add it to the RAG index."""
    try:
        doc = load_url(inp.url)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not load URL: {exc}")
    try:
        report = index_document(ChromaVectorStore(), doc)
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return {"source": doc.source, "chunks_added": report.chunks, **sources()}


@app.post("/api/ingest/file")
async def ingest_file(file: UploadFile = File(...)) -> dict:
    """Upload a .pdf/.txt/.md document, save it to the guides folder, and index it."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_UPLOAD_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. "
                   f"Supported: {', '.join(sorted(SUPPORTED_UPLOAD_SUFFIXES))}.",
        )

    config.GUIDES_DIR.mkdir(parents=True, exist_ok=True)
    dest = config.GUIDES_DIR / _safe_filename(file.filename or "upload")
    dest.write_bytes(await file.read())

    try:
        doc = load_file(dest)
        report = index_document(ChromaVectorStore(), doc)
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not index file: {exc}")
    return {"source": dest.name, "chunks_added": report.chunks, **sources()}


# Serve the rest of the static assets (css/js) under /static.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
