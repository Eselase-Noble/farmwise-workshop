"""Build a compact language primer for the local-language answer feature.

FarmWise can answer in Ghanaian languages (Twi, Fante, Ewe, Ga). ``llama3.2:3b``
has only shaky coverage of these low-resource languages, so the assistant shows
it a few short aligned English -> language sentence pairs as in-context style
examples (orthography, special characters, sentence shape). This script extracts
those pairs from a parallel corpus and writes them to a small JSON file that is
committed with the project, so the app has no runtime dependency on the corpus.

The source corpus is Bible-aligned, so the register is not agronomic; the pairs
are used only to anchor spelling and style, never as factual content.

Usage:
    python scripts/build_language_primers.py \
        --corpus /path/to/bible_parallel.jsonl \
        --per-language 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from farmwise import config  # noqa: E402

LANGUAGES = ("twi", "fante", "ewe", "ga")
# A short, self-contained sentence makes a clearer style example than a long one.
MIN_LEN, MAX_LEN = 20, 70


def build(corpus: Path, per_language: int) -> dict:
    rows = [json.loads(line) for line in corpus.read_text().splitlines() if line.strip()]
    primers: dict[str, list[dict]] = {}
    for lang in LANGUAGES:
        picked = []
        for r in rows:
            en = (r.get("en") or "").strip()
            tgt = (r.get(lang) or "").strip()
            if not en or not tgt:
                continue
            if not (MIN_LEN <= len(en) <= MAX_LEN):
                continue
            picked.append({"en": en, "text": tgt})
            if len(picked) >= per_language:
                break
        primers[lang] = picked
    return primers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path, help="Parallel corpus .jsonl")
    parser.add_argument("--per-language", type=int, default=8)
    parser.add_argument(
        "--out", type=Path, default=config.DATA_DIR / "language_primers.json"
    )
    args = parser.parse_args()

    primers = build(args.corpus, args.per_language)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(primers, ensure_ascii=False, indent=2))
    counts = ", ".join(f"{k}:{len(v)}" for k, v in primers.items())
    print(f"Wrote {args.out}  ({counts})")


if __name__ == "__main__":
    main()
