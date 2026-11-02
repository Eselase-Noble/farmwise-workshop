"""Tests for the crop-recommendation ML pipeline."""

from __future__ import annotations

import pytest

from farmwise.ml import data as data_module
from farmwise.ml.data import (
    FEATURE_COLUMNS,
    DatasetNotFoundError,
    DatasetValidationError,
    class_balance,
    feature_ranges,
    load_crop_dataset,
    validate_crop_dataset,
)
from farmwise.ml.predict import CropFeatures, CropPrediction, predict_crop
from farmwise.ml.train import train_crop_model


# --- data.py ---------------------------------------------------------------

def test_load_missing_dataset_raises(tmp_path):
    missing = tmp_path / "nope.csv"
    with pytest.raises(DatasetNotFoundError):
        load_crop_dataset(missing)


def test_validate_rejects_missing_columns(crop_df):
    bad = crop_df.drop(columns=["ph"])
    with pytest.raises(DatasetValidationError):
        validate_crop_dataset(bad)


def test_feature_ranges_keys_and_order(crop_df):
    ranges = feature_ranges(crop_df)
    assert list(ranges.keys()) == FEATURE_COLUMNS
    for name in FEATURE_COLUMNS:
        assert ranges[name]["min"] <= ranges[name]["max"]


def test_class_balance_counts(crop_df):
    balance = class_balance(crop_df)
    assert set(balance.index) == {"rice", "maize", "lentil"}
    assert balance.sum() == len(crop_df)


def test_load_roundtrip_from_csv(crop_df, tmp_path):
    csv = tmp_path / "Crop_recommendation.csv"
    crop_df.to_csv(csv, index=False)
    loaded = load_crop_dataset(csv)
    assert list(loaded.columns) == FEATURE_COLUMNS + ["label"]
    assert len(loaded) == len(crop_df)


# --- train.py --------------------------------------------------------------

def test_training_metadata_shape(crop_df):
    result = train_crop_model(crop_df, n_estimators=50)
    md = result.metadata
    assert md["feature_names"] == FEATURE_COLUMNS
    assert sorted(md["classes"]) == ["lentil", "maize", "rice"]
    assert 0.0 <= result.accuracy <= 1.0
    assert set(md["feature_importance"]) == set(FEATURE_COLUMNS)
    assert "training_date" in md and "dataset_source" in md
    assert result.baseline_accuracy is not None  # baseline compared by default


# --- predict.py ------------------------------------------------------------

def test_predict_in_range(trained_model_paths):
    model_path, metadata_path = trained_model_paths
    pred = predict_crop(
        CropFeatures(90, 40, 40, 24, 80, 6.5, 220),
        model_path=model_path,
        metadata_path=metadata_path,
    )
    assert isinstance(pred, CropPrediction)
    assert pred.crop in {"rice", "maize", "lentil"}
    assert 0.0 <= pred.confidence <= 1.0
    assert len(pred.top3) == 3
    # top3 is sorted descending and the top matches the headline crop.
    assert pred.top3[0].crop == pred.crop
    probs = [s.probability for s in pred.top3]
    assert probs == sorted(probs, reverse=True)
    assert pred.out_of_distribution is False


def test_predict_accepts_mapping(trained_model_paths):
    model_path, metadata_path = trained_model_paths
    pred = predict_crop(
        {"N": 90, "P": 40, "K": 40, "temperature": 24, "humidity": 80, "ph": 6.5, "rainfall": 220},
        model_path=model_path,
        metadata_path=metadata_path,
    )
    assert pred.crop in {"rice", "maize", "lentil"}


def test_predict_flags_out_of_distribution(trained_model_paths):
    model_path, metadata_path = trained_model_paths
    pred = predict_crop(
        CropFeatures(N=500, P=40, K=40, temperature=24, humidity=80, ph=6.5, rainfall=220),
        model_path=model_path,
        metadata_path=metadata_path,
    )
    assert pred.out_of_distribution is True
    assert any("N=" in w for w in pred.ood_warnings)


def test_predict_to_dict_is_json_friendly(trained_model_paths):
    model_path, metadata_path = trained_model_paths
    pred = predict_crop(
        CropFeatures(90, 40, 40, 24, 80, 6.5, 220),
        model_path=model_path,
        metadata_path=metadata_path,
    )
    d = pred.to_dict()
    assert set(d) == {"crop", "confidence", "top3", "out_of_distribution", "ood_warnings"}
    assert isinstance(d["top3"], list) and d["top3"][0].keys() == {"crop", "probability"}
