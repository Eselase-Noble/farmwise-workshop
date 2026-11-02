#!/usr/bin/env python
"""Rebuild the RAG knowledge base from the committed source manifest.

The curated sources live in data/knowledge_sources.txt (in git). The actual
vector index (chroma/) is machine-local and git-ignored, so on a fresh clone run
this to recreate the knowledge base:

    python scripts/rebuild_kb.py --reset     # sample guides + all manifest sources
    python scripts/rebuild_kb.py             # add manifest sources to what's there

Requires Ollama running with the embedding model pulled.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from farmwise import config  # noqa: E402
from farmwise.llm.client import OllamaError, is_ollama_running  # noqa: E402
from farmwise.rag.indexer import index_guides, index_url  # noqa: E402
from farmwise.rag.vector_store import ChromaVectorStore  # noqa: E402

MANIFEST = config.DATA_DIR / "knowledge_sources.txt"


def read_manifest(path: Path) -> list[str]:
    if not path.exists():
        return []
    urls = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            urls.append(line)
    return urls


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild the RAG knowledge base.")
    parser.add_argument(
        "--reset", action="store_true",
        help="clear the index and start from the sample guides before adding sources",
    )
    args = parser.parse_args()

    if not is_ollama_running():
        print(f"ERROR: Ollama not reachable at {config.OLLAMA_HOST}. Run: ollama serve",
              file=sys.stderr)
        return 1

    store = ChromaVectorStore()

    if args.reset:
        print("Resetting index and loading sample guides...")
        rep = index_guides(store, reset=True)
        print(f"  {rep.documents} sample guide(s) -> {rep.chunks} chunks")

    urls = read_manifest(MANIFEST)
    if not urls:
        print(f"No sources listed in {MANIFEST}")
        return 0

    print(f"Ingesting {len(urls)} source(s) from {MANIFEST.name}...")
    ok = 0
    for url in urls:
        try:
            rep = index_url(store, url)
            ok += 1
            print(f"  OK   {rep.chunks:>4} chunks  {url[:72]}")
        except OllamaError as exc:
            print(f"  STOP  Ollama error: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"  FAIL            {url[:72]}  ({exc})")

    print(f"\nDone: {ok}/{len(urls)} sources ingested. Index now holds {store.count()} chunks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
