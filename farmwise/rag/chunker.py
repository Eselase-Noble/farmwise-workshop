"""Split documents into chunks.

Two strategies, picked automatically:

* Markdown with ``## `` section headings (our guides, the crop profiles) is split
  one chunk per section, so a self-contained section — e.g. a single crop's
  growing conditions — stays together and is retrieved on its own. A section
  longer than ``chunk_size`` falls back to the sliding window below.
* Everything else (PDF and web text) uses a sliding window of ``chunk_size``
  characters advancing by ``chunk_size - overlap``, nudged to whitespace so words
  are not cut in half.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from farmwise import config
from farmwise.rag.loader import Document

_HEADING = re.compile(r"(?m)^##\s")


@dataclass
class Chunk:
    """A slice of a document, with enough context to cite it later."""

    text: str
    source: str
    chunk_index: int  # position of this chunk within its source document
    start: int        # character offset where the chunk starts in the source
    end: int          # character offset where the chunk ends in the source


def _windows(text: str, size: int, ov: int) -> list[tuple[str, int, int]]:
    """Sliding character windows as (piece, start, end), nudged to whitespace."""
    out: list[tuple[str, int, int]] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            window = max(start + int(size * 0.85), start + 1)
            space = text.rfind(" ", window, end)
            if space != -1:
                end = space
        piece = text[start:end].strip()
        if piece:
            out.append((piece, start, end))
        if end >= n:
            break
        start = end - ov
    return out


def _sections(text: str) -> list[tuple[str, int]]:
    """Split markdown into (section_text, base_offset) at each ``## `` heading."""
    matches = list(_HEADING.finditer(text))
    if not matches:
        return [(text.strip(), 0)]
    out: list[tuple[str, int]] = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        out.append((preamble, 0))
    for i, m in enumerate(matches):
        s = m.start()
        e = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        seg = text[s:e].strip()
        if seg:
            out.append((seg, s))
    return out


def chunk_text(
    text: str,
    *,
    source: str = "",
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
) -> list[Chunk]:
    """Split one string into :class:`Chunk` objects (heading-aware for markdown)."""
    size = chunk_size if chunk_size is not None else config.CHUNK_SIZE
    ov = overlap if overlap is not None else config.CHUNK_OVERLAP

    if size <= 0:
        raise ValueError("chunk_size must be positive")
    if ov < 0 or ov >= size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks: list[Chunk] = []
    index = 0

    if _HEADING.search(text):
        for seg, base in _sections(text):
            if len(seg) <= size:
                chunks.append(Chunk(seg, source, index, base, base + len(seg)))
                index += 1
            else:
                for piece, s, e in _windows(seg, size, ov):
                    chunks.append(Chunk(piece, source, index, base + s, base + e))
                    index += 1
    else:
        for piece, s, e in _windows(text, size, ov):
            chunks.append(Chunk(piece, source, index, s, e))
            index += 1

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
            chunk_text(doc.text, source=doc.source, chunk_size=chunk_size, overlap=overlap)
        )
    return all_chunks
