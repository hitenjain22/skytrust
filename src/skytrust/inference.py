"""Run the exported blend: load JSON artifacts and compute P(usable) with numpy only.

This is everything the app and CLI need at run time. It deliberately imports no
scikit-learn, so the app starts fast and runs exactly the evaluated model.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from skytrust.config import REPO_ROOT
from skytrust.design import design_matrix, model_columns
from skytrust.errors import LeakageError

ARTIFACTS = REPO_ROOT / "artifacts"
METRICS_PATH = ARTIFACTS / "metrics.json"
SCHEMA_VERSION = 1


def load_metrics(path: Path = METRICS_PATH) -> dict:
    """Backtest metrics written by `skytrust evaluate` (read by the app and CLI)."""
    return json.loads(path.read_text())


def artifact_path(label: str, lead: int, root: Path = ARTIFACTS) -> Path:
    """Primary-label models are the shipped ones (SPEC 8.7 path); the ASOS/ERA5 versions exist
    only for the label-sensitivity analysis."""
    if label == "primary":
        return root / f"model_lead{lead}.json"
    return root / "sensitivity" / f"model_{label}_lead{lead}.json"


def load_artifact(path: Path) -> dict:
    artifact = json.loads(path.read_text())
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported schema_version {artifact.get('schema_version')}")
    return artifact


def assert_trained_before(artifact: dict, test_start: dt.date | str) -> None:
    """The model we evaluate must have been trained only on nights before the test period."""
    trained_to = pd.Timestamp(artifact["training"]["period"][1])
    if trained_to >= pd.Timestamp(test_start):
        raise LeakageError(f"model trained through {trained_to:%Y-%m-%d}, test starts {test_start}")


def raw_inputs(rows: pd.DataFrame, artifact: dict) -> np.ndarray:
    """Build the model's input matrix (columns in the artifact's order) from dataset rows."""
    X = design_matrix(rows, artifact["model_feature_columns"], artifact["site_ids"])
    return X[artifact["inputs"]].to_numpy(dtype=float)


def predict_proba(artifact: dict, X: np.ndarray) -> np.ndarray:
    """P(usable) from the JSON artifact, reproducing the sklearn pipeline step by step:
    1. missing-value indicators (1 where a feature is NaN), for features that had NaN in training
    2. replace NaN with the training median
    3. standardize with the training mean / scale
    4. logistic function of the weighted sum
    5. optional isotonic calibration (piecewise-linear interpolation, clipped at the ends)
    """
    X = np.asarray(X, dtype=float)
    medians = np.asarray(artifact["imputer"]["medians"])
    indicator_cols = artifact["imputer"]["indicator_features"]
    indicators = np.isnan(X[:, indicator_cols]).astype(float)
    imputed = np.where(np.isnan(X), medians, X)
    Z = np.hstack([imputed, indicators])
    Z = (Z - np.asarray(artifact["scaler"]["mean"])) / np.asarray(artifact["scaler"]["scale"])
    z = Z @ np.asarray(artifact["logistic"]["coef"]) + artifact["logistic"]["intercept"]
    p = 1.0 / (1.0 + np.exp(-z))
    calib = artifact["calibration"]
    if calib.get("applied"):
        p = np.interp(p, calib["isotonic_x"], calib["isotonic_y"])
    return p


def weight_shares(artifact: dict) -> dict[str, float]:
    """Rough 'how much does the blend lean on each model': share of the absolute standardized
    coefficients on each model's three features. (Features are correlated, so treat this as a
    guide, not a precise attribution.)"""
    coef = dict(
        zip(artifact["logistic"]["output_features"], artifact["logistic"]["coef"], strict=True)
    )
    totals = {m: sum(abs(coef.get(c, 0.0)) for c in model_columns(m)) for m in artifact["models"]}
    grand = sum(totals.values()) or 1.0
    return {m: v / grand for m, v in totals.items()}


def summary(artifact: dict) -> dict:
    """What evaluate/report need about a blend, without the full parameter arrays."""
    coef = dict(
        zip(artifact["logistic"]["output_features"], artifact["logistic"]["coef"], strict=True)
    )
    return {
        "C": artifact["logistic"]["C"],
        "cv_log_loss": artifact["training"]["cv_log_loss"],
        "n_train": artifact["training"]["n_rows"],
        "training_period": artifact["training"]["period"],
        "spread_threshold": artifact["training"].get("spread_threshold"),
        "calibration": {
            k: v for k, v in artifact["calibration"].items() if not k.startswith("isotonic")
        },
        "coef_standardized": coef,
        "intercept": artifact["logistic"]["intercept"],
        "weight_shares": weight_shares(artifact),
        "models": artifact["models"],
    }
