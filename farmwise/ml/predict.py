"""Load the saved crop model and make predictions.

The public entry point is :func:`predict_crop`, a plain function with typed
input (:class:`CropFeatures`) and typed output (:class:`CropPrediction`) so a
FastAPI route in Session 2 can wrap it directly.

It also flags **out-of-distribution** inputs: if a soil/weather value falls
outside the range the model was trained on, the prediction is marked untrustworthy
(the Responsible-AI point — the dataset was not collected in Ghana).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Optional

import joblib
import pandas as pd

from farmwise import config
from farmwise.ml.data import FEATURE_COLUMNS


class ModelNotFoundError(FileNotFoundError):
    """Raised when the trained model artifact is missing."""


@dataclass(frozen=True)
class CropFeatures:
    """Soil and weather inputs for one farm (the model's seven features)."""

    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

    @classmethod
    def from_mapping(cls, data: Mapping[str, float]) -> "CropFeatures":
        """Build from a dict-like object, ignoring extra keys."""
        try:
            return cls(**{k: float(data[k]) for k in FEATURE_COLUMNS})
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Missing feature: {exc}. Need all of {FEATURE_COLUMNS}.")

    def as_row(self) -> list[float]:
        """Return values in the canonical feature order used for training."""
        return [getattr(self, name) for name in FEATURE_COLUMNS]


@dataclass(frozen=True)
class CropScore:
    """One candidate crop and its predicted probability."""

    crop: str
    probability: float


@dataclass(frozen=True)
class CropPrediction:
    """The model's recommendation plus trust signals."""

    crop: str
    confidence: float
    top3: list[CropScore]
    out_of_distribution: bool
    ood_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """JSON-serializable form (handy for FastAPI responses / notebooks)."""
        return {
            "crop": self.crop,
            "confidence": self.confidence,
            "top3": [asdict(s) for s in self.top3],
            "out_of_distribution": self.out_of_distribution,
            "ood_warnings": self.ood_warnings,
        }


@lru_cache(maxsize=1)
def _load_model_and_metadata(
    model_path: str, metadata_path: str
) -> tuple[object, dict]:
    """Load (and cache) the model and its metadata by path.

    Paths are passed as strings so the function is hashable/cacheable.
    """
    mpath = Path(model_path)
    if not mpath.exists():
        raise ModelNotFoundError(
            f"Trained model not found at: {mpath}\n"
            "Train it first with:  python scripts/train_model.py"
        )
    model = joblib.load(mpath)

    meta: dict = {}
    mdpath = Path(metadata_path)
    if mdpath.exists():
        meta = json.loads(mdpath.read_text())
    return model, meta


def _check_out_of_distribution(
    features: CropFeatures, feature_ranges: dict
) -> list[str]:
    """Return a warning per feature that falls outside the training range."""
    warnings: list[str] = []
    for name in FEATURE_COLUMNS:
        bounds = feature_ranges.get(name)
        if not bounds:
            continue
        value = getattr(features, name)
        lo, hi = bounds["min"], bounds["max"]
        if value < lo or value > hi:
            warnings.append(
                f"{name}={value:g} is outside the training range [{lo:g}, {hi:g}]"
            )
    return warnings


def predict_crop(
    features: CropFeatures | Mapping[str, float],
    *,
    model_path: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
) -> CropPrediction:
    """Recommend a crop for the given soil/weather inputs.

    Args:
        features: a :class:`CropFeatures` or a mapping with the seven feature keys.
        model_path / metadata_path: override the saved artifacts (defaults from config).

    Returns:
        A :class:`CropPrediction` with the top crop, its confidence, the top-3
        candidates, and any out-of-distribution warnings.

    Raises:
        ModelNotFoundError: if the model has not been trained yet.
    """
    if not isinstance(features, CropFeatures):
        features = CropFeatures.from_mapping(features)

    model, meta = _load_model_and_metadata(
        str(model_path or config.MODEL_PATH),
        str(metadata_path or config.MODEL_METADATA_PATH),
    )

    # Build a one-row DataFrame with named columns so the model sees the same
    # feature names it was trained with (avoids sklearn's feature-name warning).
    row = pd.DataFrame([features.as_row()], columns=FEATURE_COLUMNS)
    probabilities = model.predict_proba(row)[0]
    classes = list(model.classes_)

    ranked = sorted(zip(classes, probabilities), key=lambda cp: cp[1], reverse=True)
    top3 = [CropScore(crop=c, probability=float(p)) for c, p in ranked[:3]]

    ood_warnings = _check_out_of_distribution(
        features, meta.get("feature_ranges", {})
    )

    return CropPrediction(
        crop=top3[0].crop,
        confidence=top3[0].probability,
        top3=top3,
        out_of_distribution=bool(ood_warnings),
        ood_warnings=ood_warnings,
    )
