#!/usr/bin/env python
"""Turn the crop-recommendation dataset into a knowledge-base document.

This connects the two FarmWise features: the same data the ML model is trained on
becomes text the RAG assistant can retrieve. For each crop it summarises the
typical soil and weather conditions (the ranges the model actually learned), so
the assistant can explain what a crop needs and why a recommendation was made.

Writes data/reference/crop_profiles.md. It is kept out of data/guides/ on purpose:
the dataset is not Ghanaian (it is generally taken to reflect Indian conditions),
and the knowledge base holds Ghana-based sources only. The crop page still uses
data/crop_conditions.json for its ranges.

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
    lines.append("# Recommended growing conditions by crop")
    lines.append("")
    lines.append(
        "> Indicative soil and climate conditions for common crops, summarised for "
        "the FarmWise assistant. These ranges are general guidance learned from "
        "historical crop records; they are not calibrated specifically for Ghana, "
        "so use them alongside local extension advice."
    )
    lines.append("")
    crops = sorted(df[TARGET_COLUMN].unique())
    lines.append("This guide covers " + ", ".join(crops) + ".")
    lines.append("")

    def rng(stat, col, decimals):
        lo, hi, mean = stat[col]
        if decimals == 0:
            return f"around {mean:.0f} (typically {lo:.0f}–{hi:.0f})"
        return f"around {mean:.1f} (typically {lo:.1f}–{hi:.1f})"

    for crop in crops:
        sub = df[df[TARGET_COLUMN] == crop]
        stat = {c: (sub[c].min(), sub[c].max(), sub[c].mean()) for c in FEATURE_COLUMNS}
        name = crop.title()
        sentence = (
            f"{name} is best suited to soil supplying nitrogen (N) {rng(stat, 'N', 0)}, "
            f"phosphorus (P) {rng(stat, 'P', 0)}, and potassium (K) {rng(stat, 'K', 0)}. "
            f"It prefers temperatures {rng(stat, 'temperature', 1)} °C, "
            f"relative humidity {rng(stat, 'humidity', 1)} percent, "
            f"a soil pH {rng(stat, 'ph', 1)}, and seasonal rainfall "
            f"{rng(stat, 'rainfall', 0)} mm."
        )
        lines.append(f"## {name}")
        lines.append(sentence)
        lines.append("")

    return "\n".join(lines)


UNITS = {"N": "", "P": "", "K": "", "temperature": "°C", "humidity": "%", "ph": "", "rainfall": "mm"}


def build_conditions() -> dict:
    """Machine-readable per-crop ranges + overall ranges, for the web UI.

    Committed as data/crop_conditions.json so the web app can draw sliders and the
    'your input vs this crop' chart without needing the raw CSV at runtime.
    """
    df = load_crop_dataset()
    crops = sorted(df[TARGET_COLUMN].unique())
    ranges = {
        col: {"min": round(float(df[col].min()), 2), "max": round(float(df[col].max()), 2)}
        for col in FEATURE_COLUMNS
    }
    crop_stats = {}
    for crop in crops:
        sub = df[df[TARGET_COLUMN] == crop]
        crop_stats[crop] = {
            col: {
                "min": round(float(sub[col].min()), 2),
                "max": round(float(sub[col].max()), 2),
                "mean": round(float(sub[col].mean()), 2),
            }
            for col in FEATURE_COLUMNS
        }
    return {"features": FEATURE_COLUMNS, "units": UNITS, "ranges": ranges, "crops": crop_stats}


def main() -> int:
    import json

    try:
        md = build_markdown()
        conditions = build_conditions()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    md_out = config.DATA_DIR / "reference" / "crop_profiles.md"
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(md)
    print(f"Wrote {md_out} ({len(md)} chars)")

    json_out = config.DATA_DIR / "crop_conditions.json"
    json_out.write_text(json.dumps(conditions, indent=2))
    print(f"Wrote {json_out} ({len(conditions['crops'])} crops)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
