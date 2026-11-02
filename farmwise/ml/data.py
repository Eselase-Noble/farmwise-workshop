"""Load and validate the Crop Recommendation dataset.

The dataset is a CSV with seven numeric feature columns and one text label:

    N, P, K, temperature, humidity, ph, rainfall, label

See ``data/raw/README.md`` for where to download it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from farmwise import config

# The feature order is fixed and shared across training and prediction.
FEATURE_COLUMNS: list[str] = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET_COLUMN: str = "label"
REQUIRED_COLUMNS: list[str] = FEATURE_COLUMNS + [TARGET_COLUMN]


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the crop dataset CSV cannot be found on disk."""


class DatasetValidationError(ValueError):
    """Raised when the crop dataset is present but malformed."""


def load_crop_dataset(path: Optional[Path | str] = None) -> pd.DataFrame:
    """Load the crop dataset as a validated DataFrame.

    Args:
        path: CSV location. Defaults to ``config.CROP_DATASET_PATH``.

    Returns:
        A DataFrame with exactly the required columns, numeric features coerced,
        and rows with missing values dropped.

    Raises:
        DatasetNotFoundError: if the file does not exist.
        DatasetValidationError: if required columns are missing or non-numeric.
    """
    csv_path = Path(path) if path is not None else config.CROP_DATASET_PATH

    if not csv_path.exists():
        raise DatasetNotFoundError(
            f"Crop dataset not found at: {csv_path}\n"
            "Download the public 'Crop Recommendation' dataset and place the CSV "
            "there (see data/raw/README.md). The model cannot be trained without "
            "it — real data is required; do not substitute synthetic data."
        )

    df = pd.read_csv(csv_path)
    validate_crop_dataset(df)

    # Keep only the columns we need, in the canonical order.
    df = df[REQUIRED_COLUMNS].copy()

    # Coerce features to numeric and drop any rows that fail (defensive).
    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    before = len(df)
    df = df.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"[data] dropped {dropped} row(s) with missing/invalid values")

    if df.empty:
        raise DatasetValidationError("Crop dataset has no usable rows after cleaning.")

    return df


def validate_crop_dataset(df: pd.DataFrame) -> None:
    """Check that the DataFrame has the columns we need.

    Raises DatasetValidationError on any problem.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DatasetValidationError(
            f"Dataset is missing required column(s): {missing}. "
            f"Expected columns: {REQUIRED_COLUMNS}. Found: {list(df.columns)}."
        )
    if df[TARGET_COLUMN].isna().all():
        raise DatasetValidationError(f"Target column '{TARGET_COLUMN}' is entirely empty.")


def feature_ranges(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Return per-feature ``{'min': ..., 'max': ...}`` observed in the data.

    Used to flag out-of-distribution prediction inputs (Responsible-AI segment).
    """
    ranges: dict[str, dict[str, float]] = {}
    for col in FEATURE_COLUMNS:
        ranges[col] = {"min": float(df[col].min()), "max": float(df[col].max())}
    return ranges


def class_balance(df: pd.DataFrame) -> pd.Series:
    """Return the count of rows per crop label, highest first."""
    return df[TARGET_COLUMN].value_counts()
