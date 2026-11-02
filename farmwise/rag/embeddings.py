"""Turn text into vectors using an Ollama embedding model.

Two ways to use it:

* Functional:  :func:`embed_texts` / :func:`embed_query` — explicit, used in the
  notebook so the audience can see the vectors.
* As a class:  :class:`OllamaEmbeddingFunction` — conforms to ChromaDB's
  embedding-function interface so a vector store can embed on its own if needed.

We compute embeddings here and hand them to the vector store explicitly, which
keeps the store backend-agnostic (Session 2's PgVectorStore receives the same
vectors).
"""

from __future__ import annotations

from typing import Optional

import requests

from farmwise import config
from farmwise.llm.client import (
    OllamaModelNotFoundError,
    OllamaNotRunningError,
    _MODEL_HINT,
    _NOT_RUNNING_HINT,
)


def embed_texts(
    texts: list[str],
    *,
    model: Optional[str] = None,
    host: Optional[str] = None,
) -> list[list[float]]:
    """Embed a list of texts, returning one vector per input (order preserved)."""
    base = host or config.OLLAMA_HOST
    model_name = model or config.EMBEDDING_MODEL
    url = f"{base.rstrip('/')}/api/embeddings"

    vectors: list[list[float]] = []
    for text in texts:
        try:
            resp = requests.post(
                url, json={"model": model_name, "prompt": text}, timeout=120
            )
        except requests.exceptions.ConnectionError as exc:
            raise OllamaNotRunningError(_NOT_RUNNING_HINT.format(host=base)) from exc

        if resp.status_code == 404:
            raise OllamaModelNotFoundError(_MODEL_HINT.format(model=model_name))
        resp.raise_for_status()
        data = resp.json()
        if data.get("error"):
            raise OllamaModelNotFoundError(_MODEL_HINT.format(model=model_name))
        vectors.append(data["embedding"])
    return vectors


def embed_query(
    text: str, *, model: Optional[str] = None, host: Optional[str] = None
) -> list[float]:
    """Embed a single query string into one vector."""
    return embed_texts([text], model=model, host=host)[0]


try:  # Subclass Chroma's base so the collection treats us as a first-class EF.
    from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

    _EF_BASE = EmbeddingFunction
except Exception:  # pragma: no cover - Chroma not installed / older API
    _EF_BASE = object
    Documents = list  # type: ignore
    Embeddings = list  # type: ignore


class OllamaEmbeddingFunction(_EF_BASE):
    """ChromaDB-compatible embedding function backed by Ollama.

    Chroma calls ``self(input)`` with a list of strings and expects a list of
    vectors. We pass this to the collection so Chroma never falls back to its
    default (download-on-first-use) embedding model.
    """

    def __init__(self, model: Optional[str] = None, host: Optional[str] = None):
        self._model = model or config.EMBEDDING_MODEL
        self._host = host or config.OLLAMA_HOST

    def __call__(self, input: "Documents") -> "Embeddings":  # noqa: A002 (Chroma's param name)
        return embed_texts(list(input), model=self._model, host=self._host)

    @staticmethod
    def name() -> str:
        return "ollama"

    def get_config(self) -> dict:
        return {"model": self._model, "host": self._host}

    @staticmethod
    def build_from_config(cfg: dict) -> "OllamaEmbeddingFunction":
        return OllamaEmbeddingFunction(model=cfg.get("model"), host=cfg.get("host"))

    def default_space(self) -> str:
        return "cosine"
