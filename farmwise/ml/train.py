"""Train, evaluate, and save the crop-recommendation model.

The production model is a ``RandomForestClassifier``. A ``LogisticRegression``
baseline is trained alongside it (for the demo) so the audience can see that the
tree ensemble is worth the extra complexity.

The trained model is saved with ``joblib`` and a sidecar JSON metadata file that
records feature names, classes, accuracy, training date, and the (important!)
dataset provenance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from farmwise import config
from farmwise.ml import data as data_module
from farmwise.ml.data import FEATURE_COLUMNS, TARGET_COLUMN


@dataclass
class TrainingResult:
    """Everything produced by a training run (model + evaluation artifacts)."""

    model: RandomForestClassifier
    metadata: dict
    accuracy: float
    report_text: str
    report_dict: dict
    confusion: np.ndarray
    classes: list[str]
    feature_importance: dict[str, float]
    baseline_accuracy: Optional[float] = None
    # Held-out split kept around so notebooks can plot without re-splitting.
    X_test: Optional[pd.DataFrame] = field(default=None, repr=False)
    y_test: Optional[pd.Series] = field(default=None, repr=False)


def train_crop_model(
    df: Optional[pd.DataFrame] = None,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 200,
    compare_baseline: bool = True,
) -> TrainingResult:
    """Train a RandomForest crop classifier and evaluate it on a held-out split.

    Args:
        df: pre-loaded dataset; if None, loads via ``data.load_crop_dataset()``.
        test_size: fraction held out for evaluation.
        random_state: seed for a reproducible split and forest.
        n_estimators: number of trees in the forest.
        compare_baseline: also train a LogisticRegression baseline for comparison.

    Returns:
        A :class:`TrainingResult`.
    """
    if df is None:
        df = data_module.load_crop_dataset()

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    classes = sorted(y.unique().tolist())
    report_text = classification_report(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    confusion = confusion_matrix(y_test, y_pred, labels=classes)

    importance = {
        feat: float(imp)
        for feat, imp in zip(FEATURE_COLUMNS, model.feature_importances_)
    }

    baseline_accuracy: Optional[float] = None
    if compare_baseline:
        baseline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(max_iter=1000, random_state=random_state),
                ),
            ]
        )
        baseline.fit(X_train, y_train)
        baseline_accuracy = float(accuracy_score(y_test, baseline.predict(X_test)))

    metadata = {
        "model_type": "RandomForestClassifier",
        "feature_names": FEATURE_COLUMNS,
        "classes": classes,
        "n_classes": len(classes),
        "accuracy": accuracy,
        "baseline_model": "LogisticRegression" if compare_baseline else None,
        "baseline_accuracy": baseline_accuracy,
        "n_samples": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_estimators": n_estimators,
        "random_state": random_state,
        "test_size": test_size,
        "feature_ranges": data_module.feature_ranges(df),
        "feature_importance": importance,
        "training_date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "dataset_source": config.CROP_DATASET_SOURCE,
    }

    return TrainingResult(
        model=model,
        metadata=metadata,
        accuracy=accuracy,
        report_text=report_text,
        report_dict=report_dict,
        confusion=confusion,
        classes=classes,
        feature_importance=importance,
        baseline_accuracy=baseline_accuracy,
        X_test=X_test,
        y_test=y_test,
    )


def save_model(
    result: TrainingResult,
    *,
    model_path: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
) -> tuple[Path, Path]:
    """Persist the model (joblib) and its metadata (JSON). Returns both paths."""
    model_path = Path(model_path) if model_path else config.MODEL_PATH
    metadata_path = Path(metadata_path) if metadata_path else config.MODEL_METADATA_PATH

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(result.model, model_path)
    metadata_path.write_text(json.dumps(result.metadata, indent=2))

    return model_path, metadata_path


def train_and_save(**kwargs) -> tuple[TrainingResult, Path, Path]:
    """Convenience: train with defaults and save to the configured paths."""
    result = train_crop_model(**kwargs)
    model_path, metadata_path = save_model(result)
    return result, model_path, metadata_path
