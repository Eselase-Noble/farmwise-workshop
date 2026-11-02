"""FarmWise web app — FastAPI backend.

This is the Session 2 web layer. It does not reimplement any AI logic; it simply
wraps the Session 1 ``farmwise`` package:

* ``POST /api/predict``  -> ``farmwise.ml.predict.predict_crop``
* ``POST /api/ask``      -> ``farmwise.rag.assistant.ask`` (RAG) or
                            ``ask_without_rag`` (bare LLM), chosen per request

The same two entry points, now reachable from a browser.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from farmwise import config
from farmwise.llm.client import OllamaError, is_ollama_running
from farmwise.ml.predict import CropFeatures, ModelNotFoundError, predict_crop
from farmwise.rag.assistant import ask, ask_without_rag

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


# --- pages -----------------------------------------------------------------

@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


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


# Serve the rest of the static assets (css/js) under /static.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
