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
from farmwise.rag.loader import Document, crawl_and_load, load_guides, load_url
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

    # Embed first, so an Ollama failure cannot leave the existing index wiped.
    embeddings = embed_texts([c.text for c in chunks])

    if reset:
        store.reset()
    else:
        for document in documents:
            store.delete_source(document.source)

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

    Re-adding a source removes its old chunks first, so a shorter new version
    leaves nothing stale behind. This is how the web app grows the knowledge base
    one upload or URL at a time.
    """
    chunks = chunk_text(
        document.text, source=document.source, chunk_size=chunk_size, overlap=overlap
    )
    if not chunks:
        raise ValueError(f"No text to index from source '{document.source}'.")

    embeddings = embed_texts([c.text for c in chunks])
    store.delete_source(document.source)
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


def index_url(
    store: VectorStore,
    url: str,
    *,
    crawl: bool = False,
    max_pages: int = 5,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
) -> IndexReport:
    """Index a URL. With ``crawl=True`` the page's relevant links (reports, PDFs,
    same-site pages) are followed and indexed too, so a thin landing page still
    contributes the documents it points to."""
    documents = crawl_and_load(url, max_pages=max_pages) if crawl else [load_url(url)]

    total_chunks = 0
    sources: list[str] = []
    for doc in documents:
        try:
            report = index_document(store, doc, chunk_size=chunk_size, overlap=overlap)
        except ValueError:
            continue  # skip a source that produced no chunks
        total_chunks += report.chunks
        sources.append(doc.source)

    if total_chunks == 0:
        raise ValueError(f"No text could be indexed from {url}.")
    return IndexReport(documents=len(sources), chunks=total_chunks, sources=sources)
