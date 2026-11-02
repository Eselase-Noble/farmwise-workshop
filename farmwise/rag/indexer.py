"""Ingest guides into a vector store: load -> chunk -> embed -> store.

This is the reusable pipeline behind ``scripts/build_index.py`` (and Session 2's
ingestion). It depends only on the :class:`VectorStore` interface, so it works
with any backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from farmwise import config
from farmwise.rag.chunker import Chunk, chunk_documents, chunk_text
from farmwise.rag.embeddings import embed_texts
from farmwise.rag.loader import Document, load_guides
from farmwise.rag.vector_store import VectorStore


@dataclass
class IndexReport:
    """Summary of an ingestion run."""

    documents: int
    chunks: int
    sources: list[str]


def _chunk_id(chunk: Chunk) -> str:
    """Stable id so re-indexing upserts rather than duplicates."""
    return f"{chunk.source}::{chunk.chunk_index}"


def index_guides(
    store: VectorStore,
    *,
    directory: Optional[Path | str] = None,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
    reset: bool = True,
) -> IndexReport:
    """Load guides from ``directory``, chunk, embed, and add them to ``store``.

    Args:
        store: any VectorStore implementation.
        directory: guides folder (defaults to config.GUIDES_DIR).
        chunk_size / overlap: chunking params (default from config).
        reset: clear the store first so the index matches the folder exactly.
    """
    directory = Path(directory) if directory is not None else config.GUIDES_DIR

    documents = load_guides(directory)
    chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise ValueError(
            f"No chunks produced from {directory}. Are there any .pdf/.txt/.md "
            "guides in there (other than README.md)?"
        )

    if reset:
        store.reset()

    embeddings = embed_texts([c.text for c in chunks])
    ids = [_chunk_id(c) for c in chunks]
    docs = [c.text for c in chunks]
    metadatas = [
        {"source": c.source, "chunk_index": c.chunk_index, "start": c.start, "end": c.end}
        for c in chunks
    ]
    store.add(ids=ids, documents=docs, embeddings=embeddings, metadatas=metadatas)

    return IndexReport(
        documents=len(documents),
        chunks=len(chunks),
        sources=[d.source for d in documents],
    )


def index_document(
    store: VectorStore,
    document: Document,
    *,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
) -> IndexReport:
    """Add (or refresh) a single document in the store, without resetting it.

    Chunks are upserted under stable ``source::index`` ids, so re-adding the same
    source replaces its chunks instead of duplicating them. This is how the web
    app grows the knowledge base one upload or URL at a time.
    """
    chunks = chunk_text(
        document.text, source=document.source, chunk_size=chunk_size, overlap=overlap
    )
    if not chunks:
        raise ValueError(f"No text to index from source '{document.source}'.")

    embeddings = embed_texts([c.text for c in chunks])
    store.add(
        ids=[_chunk_id(c) for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=embeddings,
        metadatas=[
            {"source": c.source, "chunk_index": c.chunk_index, "start": c.start, "end": c.end}
            for c in chunks
        ],
    )
    return IndexReport(documents=1, chunks=len(chunks), sources=[document.source])
