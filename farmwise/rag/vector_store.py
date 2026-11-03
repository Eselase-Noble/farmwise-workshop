"""Vector store abstraction + a ChromaDB implementation.

The :class:`VectorStore` interface is intentionally tiny and backend-agnostic:
``add``, ``search``, ``count``, ``reset``, plus ``distinct_sources`` and
``delete_source`` for managing the knowledge base. ``assistant.py`` depends only
on this interface, so Session 2 can add a ``PgVectorStore`` (PostgreSQL + pgvector)
without changing a line of the assistant.

The store receives embeddings that were computed elsewhere (see ``embeddings.py``),
so swapping the backend never touches how vectors are produced.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from farmwise import config


@dataclass
class SearchResult:
    """One retrieved chunk and how well it matched the query."""

    id: str
    document: str
    metadata: dict
    score: float      # similarity in [0, 1]; higher is more similar
    distance: float   # raw backend distance; lower is more similar


class VectorStore(ABC):
    """Minimal interface every backend must implement."""

    @abstractmethod
    def add(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        """Insert (or upsert) chunks with their precomputed embeddings."""

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int) -> list[SearchResult]:
        """Return the ``top_k`` most similar chunks to the query embedding."""

    @abstractmethod
    def count(self) -> int:
        """Return how many chunks are stored."""

    @abstractmethod
    def distinct_sources(self) -> dict[str, int]:
        """Return a ``{source: chunk_count}`` map of what is stored."""

    @abstractmethod
    def delete_source(self, source: str) -> int:
        """Remove every chunk belonging to ``source`` and return how many went."""

    @abstractmethod
    def reset(self) -> None:
        """Remove everything (used when rebuilding the index)."""


class ChromaVectorStore(VectorStore):
    """A persistent, local ChromaDB-backed store using cosine distance."""

    def __init__(
        self,
        persist_dir: Optional[Path | str] = None,
        collection_name: Optional[str] = None,
        embedding_function=None,
    ):
        import chromadb

        self.persist_dir = Path(persist_dir) if persist_dir else config.CHROMA_DIR
        self.collection_name = collection_name or config.COLLECTION_NAME
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Default to the Ollama embedding function so Chroma never reaches for
        # its download-on-first-use default model (keeps us offline-friendly).
        if embedding_function is None:
            from farmwise.rag.embeddings import OllamaEmbeddingFunction

            embedding_function = OllamaEmbeddingFunction()

        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._embedding_function = embedding_function
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=embedding_function,
        )

    def add(self, ids, documents, embeddings, metadatas) -> None:
        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(self, query_embedding, top_k: int) -> list[SearchResult]:
        n = min(top_k, self.count()) or top_k
        res = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        results: list[SearchResult] = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for _id, doc, meta, dist in zip(ids, docs, metas, dists):
            results.append(
                SearchResult(
                    id=_id,
                    document=doc,
                    metadata=meta or {},
                    # cosine distance in [0, 2] -> similarity in [-1, 1], clamp to [0, 1]
                    score=max(0.0, 1.0 - float(dist)),
                    distance=float(dist),
                )
            )
        return results

    def count(self) -> int:
        return self._collection.count()

    def distinct_sources(self) -> dict[str, int]:
        """Return a ``{source: chunk_count}`` map of what is currently indexed."""
        counts: dict[str, int] = {}
        if self._collection.count() == 0:
            return counts
        res = self._collection.get(include=["metadatas"])
        for meta in res.get("metadatas", []) or []:
            source = (meta or {}).get("source", "?")
            counts[source] = counts.get(source, 0) + 1
        return counts

    def delete_source(self, source: str) -> int:
        """Remove every chunk belonging to ``source``. Returns how many were removed."""
        before = self._collection.count()
        self._collection.delete(where={"source": source})
        return before - self._collection.count()

    def reset(self) -> None:
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._embedding_function,
        )
