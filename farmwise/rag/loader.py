"""Load farming guide documents from disk.

Supports ``.pdf`` (via pypdf), ``.txt`` and ``.md``. Each file becomes one
:class:`Document` whose ``source`` is the file name (used later for citations).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from farmwise import config

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


@dataclass
class Document:
    """One loaded guide: its full text and the file it came from."""

    text: str
    source: str  # file name, e.g. "sample_maize.md"


def load_file(path: Path | str) -> Document:
    """Load a single ``.pdf`` / ``.txt`` / ``.md`` file into a Document."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported file type '{suffix}' for {path.name}. "
            f"Supported: {sorted(SUPPORTED_SUFFIXES)}"
        )

    if suffix == ".pdf":
        text = _read_pdf(path)
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")

    return Document(text=text.strip(), source=path.name)


def _read_pdf(path: Path) -> str:
    """Extract text from a PDF, one page after another."""
    from pypdf import PdfReader  # imported lazily so non-PDF runs don't need it

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n\n".join(pages)


def load_guides(directory: Optional[Path | str] = None) -> list[Document]:
    """Load every supported guide in ``directory`` (defaults to config.GUIDES_DIR).

    Skips README files. Returns documents sorted by file name for stable ordering.
    Documents that come out empty (e.g. a scanned PDF with no text layer) are
    skipped with a printed warning.
    """
    directory = Path(directory) if directory is not None else config.GUIDES_DIR
    if not directory.exists():
        raise FileNotFoundError(f"Guides directory not found: {directory}")

    docs: list[Document] = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if path.name.lower() == "readme.md":
            continue
        doc = load_file(path)
        if not doc.text:
            print(f"[loader] WARNING: no extractable text in {path.name}, skipping")
            continue
        docs.append(doc)
    return docs
