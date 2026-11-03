#!/usr/bin/env python
"""Ingest the farming guides into the vector store.

Usage:
    python scripts/build_index.py

Loads every guide in data/guides/, chunks them, embeds the chunks with Ollama,
and stores them in a persistent local ChromaDB folder. Re-running rebuilds the
index so it matches the folder exactly, which removes any sources added by link.
To rebuild those as well, use scripts/rebuild_kb.py --reset.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from farmwise import config  # noqa: E402
from farmwise.llm.client import OllamaError, is_ollama_running  # noqa: E402
from farmwise.rag.indexer import index_guides  # noqa: E402
from farmwise.rag.vector_store import ChromaVectorStore  # noqa: E402


def main() -> int:
    if not is_ollama_running():
        print(
            f"ERROR: Ollama is not reachable at {config.OLLAMA_HOST}.\n"
            "Start it with:  ollama serve\n"
            "Then pull the embedding model:  ollama pull " + config.EMBEDDING_MODEL,
            file=sys.stderr,
        )
        return 1

    print(f"Guides dir : {config.GUIDES_DIR}")
    print(f"Chroma dir : {config.CHROMA_DIR}")
    print(f"Embedding  : {config.EMBEDDING_MODEL}")
    print(f"Chunk size : {config.CHUNK_SIZE} (overlap {config.CHUNK_OVERLAP})\n")

    store = ChromaVectorStore()
    try:
        report = index_guides(store)
    except OllamaError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Indexed {report.documents} document(s) -> {report.chunks} chunk(s)")
    for src in report.sources:
        print(f"  - {src}")
    print(f"\nTotal chunks now in store: {store.count()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
