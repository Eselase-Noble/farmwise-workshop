"""Split documents into overlapping chunks.

Deliberately simple and explainable (good for a live demo): a sliding window of
``chunk_size`` characters that advances by ``chunk_size - overlap`` each step, so
consecutive chunks share ``overlap`` characters. To avoid cutting words in half,
each chunk's end is nudged back to the nearest whitespace when one is close by.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from farmwise import config
from farmwise.rag.loader import Document


@dataclass
class Chunk:
    """A slice of a document, with enough context to cite it later."""

    text: str
    source: str
    chunk_index: int  # position of this chunk within its source document
    start: int        # character offset where the chunk starts in the source
    end: int          # character offset where the chunk ends in the source


def chunk_text(
    text: str,
    *,
    source: str = "",
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
) -> list[Chunk]:
    """Split one string into overlapping :class:`Chunk` objects."""
    size = chunk_size if chunk_size is not None else config.CHUNK_SIZE
    ov = overlap if overlap is not None else config.CHUNK_OVERLAP

    if size <= 0:
        raise ValueError("chunk_size must be positive")
    if ov < 0 or ov >= size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    text = text.strip()
    if not text:
        return []

    step = size - ov
    chunks: list[Chunk] = []
    start = 0
    index = 0
    n = len(text)

    while start < n:
        end = min(start + size, n)
        # Nudge the end back to the nearest whitespace (within the last ~15%)
        # so we don't cut a word in half — unless this is the final chunk.
        if end < n:
            window = max(start + int(size * 0.85), start + 1)
            space = text.rfind(" ", window, end)
            if space != -1:
                end = space

        piece = text[start:end].strip()
        if piece:
            chunks.append(
                Chunk(text=piece, source=source, chunk_index=index, start=start, end=end)
            )
            index += 1

        if end >= n:
            break
        start = end - ov  # step back by the overlap for the next window

    return chunks


def chunk_documents(
    documents: list[Document],
    *,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
) -> list[Chunk]:
    """Chunk a list of documents, preserving each chunk's source file name."""
    all_chunks: list[Chunk] = []
    for doc in documents:
        all_chunks.extend(
            chunk_text(
                doc.text, source=doc.source, chunk_size=chunk_size, overlap=overlap
            )
        )
    return all_chunks
