#!/usr/bin/env python
"""Turn the crop-recommendation dataset into a knowledge-base document.

This connects the two FarmWise features: the same data the ML model is trained on
becomes text the RAG assistant can retrieve. For each crop it summarises the
typical soil and weather conditions (the ranges the model actually learned), so
the assistant can explain what a crop needs and why a recommendation was made.

Writes data/guides/crop_profiles.md, which the normal index build then ingests.

    python scripts/build_crop_profiles.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from farmwise import config  # noqa: E402
from farmwise.ml.data import FEATURE_COLUMNS, TARGET_COLUMN, load_crop_dataset  # noqa: E402

FEATURE_LABELS = {
    "N": "nitrogen (N)",
    "P": "phosphorus (P)",
    "K": "potassium (K)",
    "temperature": "temperature (°C)",
    "humidity": "humidity (%)",
    "ph": "soil pH",
    "rainfall": "rainfall (mm)",
}


def build_markdown() -> str:
    df = load_crop_dataset()
    lines: list[str] = []
    lines.append("# Crop growing-condition profiles")
    lines.append("")
    lines.append(
        "> Generated from the Crop Recommendation dataset that trains the FarmWise "
        "model. The dataset's origin is undocumented and widely taken to reflect "
        "Indian conditions, so treat these ranges as what the model learned, not as "
        "validated figures for Ghana. They explain the model's recommendations and "
        "its out-of-range warnings."
    )
    lines.append("")
    crops = sorted(df[TARGET_COLUMN].unique())
    lines.append(
        f"The dataset covers {len(crops)} crops: " + ", ".join(crops) + "."
    )
    lines.append("")

    for crop in crops:
        sub = df[df[TARGET_COLUMN] == crop]
        parts = []
        for col in FEATURE_COLUMNS:
            lo, hi, mean = sub[col].min(), sub[col].max(), sub[col].mean()
            parts.append(f"{FEATURE_LABELS[col]} about {mean:.1f} (range {lo:.1f}-{hi:.1f})")
        lines.append(f"## {crop.title()}")
        lines.append(
            f"In the dataset, {crop} is grown with " + "; ".join(parts) + "."
        )
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    try:
        md = build_markdown()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    out = config.GUIDES_DIR / "crop_profiles.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(f"Wrote {out} ({len(md)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
