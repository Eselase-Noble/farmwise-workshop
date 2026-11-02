#!/usr/bin/env python
"""Train and save the crop-recommendation model.

Usage:
    python scripts/train_model.py

Reads the dataset from data/raw/ (see data/raw/README.md), trains a RandomForest
(plus a LogisticRegression baseline for comparison), prints a short report, and
saves the model + metadata into models/.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the farmwise package importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from farmwise.ml.data import DatasetNotFoundError  # noqa: E402
from farmwise.ml.train import train_and_save  # noqa: E402


def main() -> int:
    print("Training crop-recommendation model...\n")
    try:
        result, model_path, metadata_path = train_and_save()
    except DatasetNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"RandomForest accuracy : {result.accuracy:.4f}")
    if result.baseline_accuracy is not None:
        print(f"LogReg baseline       : {result.baseline_accuracy:.4f}")
    print(f"Classes               : {result.metadata['n_classes']}")
    print(f"Train / test rows     : {result.metadata['n_train']} / {result.metadata['n_test']}")
    print()
    print("Top features by importance:")
    ranked = sorted(result.feature_importance.items(), key=lambda kv: kv[1], reverse=True)
    for feat, imp in ranked:
        print(f"  {feat:<12} {imp:.3f}")
    print()
    print(f"Saved model    -> {model_path}")
    print(f"Saved metadata -> {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
