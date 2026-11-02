"""Shared pytest fixtures.

The ML fixtures build a TINY synthetic dataset purely to exercise code paths.
This is NOT the real Crop Recommendation dataset and is never written into
data/raw/ — the real model is always trained on the real downloaded data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from farmwise.ml.data import FEATURE_COLUMNS, TARGET_COLUMN


@pytest.fixture
def crop_df() -> pd.DataFrame:
    """A small, well-separated 3-class fixture (test-only, not real data)."""
    rng = np.random.RandomState(0)
    centers = {
        "rice": (90, 40, 40, 24, 80, 6.5, 220),
        "maize": (80, 50, 20, 27, 65, 6.2, 90),
        "lentil": (20, 65, 20, 25, 60, 7.0, 60),
    }
    rows = []
    for crop, (N, P, K, t, h, ph, r) in centers.items():
        for _ in range(40):
            rows.append(
                [
                    N + rng.randn() * 3,
                    P + rng.randn() * 3,
                    K + rng.randn() * 3,
                    t + rng.randn(),
                    h + rng.randn() * 2,
                    ph + rng.randn() * 0.1,
                    r + rng.randn() * 5,
                    crop,
                ]
            )
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS + [TARGET_COLUMN])


@pytest.fixture
def trained_model_paths(crop_df, tmp_path):
    """Train on the fixture and save artifacts to a temp dir; yield the paths."""
    from farmwise.ml.predict import _load_model_and_metadata
    from farmwise.ml.train import save_model, train_crop_model

    result = train_crop_model(crop_df, n_estimators=50)
    model_path = tmp_path / "crop_model.joblib"
    metadata_path = tmp_path / "crop_model_metadata.json"
    save_model(result, model_path=model_path, metadata_path=metadata_path)

    _load_model_and_metadata.cache_clear()
    yield model_path, metadata_path
    _load_model_and_metadata.cache_clear()
