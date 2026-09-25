"""The blend (SPEC 8.3): one calibrated logistic regression per lead that combines every
available model's forecast features with the model spread and context features.

Pipeline: median imputation + missing indicators -> standardize -> L2 logistic regression,
C tuned by date-based TimeSeriesSplit on training data only. Each fitted model is exported
to plain JSON; `predict_proba` rebuilds predictions from that JSON with numpy alone, so the
app runs exactly the model that was evaluated, with no pickle and no sklearn at inference.

Calibration: out-of-fold (OOF) predictions over the training folds are binned; if their
expected calibration error exceeds `calibration.ece_threshold` (config), an isotonic map
fitted on those OOF predictions is applied on top (this is what sklearn's
CalibratedClassifierCV(ensemble=False) does; it's done by hand because TimeSeriesSplit folds
aren't a partition of the data, which that class requires). Otherwise no calibration is used.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.isotonic import IsotonicRegression

from skytrust.baselines import LABELS, models_at_lead
from skytrust.config import REPO_ROOT, Settings
from skytrust.modeling import (
    LeakageError,
    date_folds,
    design_matrix,
    fit_tuned_logistic,
    log_loss,
    make_pipeline,
    model_columns,
    split_train_test,
)

log = logging.getLogger(__name__)

ARTIFACTS = REPO_ROOT / "artifacts"
SCHEMA_VERSION = 1


def artifact_path(label: str, lead: int, root: Path = ARTIFACTS) -> Path:
    """Primary-label models are the shipped ones (SPEC 8.7 path); the ASOS/ERA5 versions exist
    only for the label-sensitivity analysis."""
    if label == "primary":
        return root / f"model_lead{lead}.json"
    return root / "sensitivity" / f"model_{label}_lead{lead}.json"


def blend_feature_columns(settings: Settings, lead: int) -> list[str]:
    """Each model that forecasts this lead contributes its three features; the spread across
    models is added when there are at least two models."""
    models = models_at_lead(settings, lead)
    cols = [c for m in models for c in model_columns(m.short)]
    return cols + (["spread_frac_clear"] if len(models) >= 2 else [])


def training_rows(train: pd.DataFrame, label_col: str, lead: int) -> pd.DataFrame:
    """Labeled training rows at this lead with at least one model available (the first weeks
    of 2024 predate the forecast archive and carry no information)."""
    rows = train[
        (train["lead"] == lead) & train[label_col].notna() & (train["n_models_available"] > 0)
    ]
    return rows.sort_values(["night_date", "site"])


# ---------- calibration check ----------


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Average |observed frequency − mean forecast| over 10 equal-width bins, weighted by the
    number of forecasts in each bin. 0 = perfectly calibrated."""
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            total += mask.sum() * abs(y[mask].mean() - p[mask].mean())
    return float(total / len(p))


def out_of_fold(
    X: pd.DataFrame, y: np.ndarray, dates: pd.Series, C: float, settings: Settings
) -> tuple[np.ndarray, np.ndarray]:
    """Predictions for each validation fold from a model trained only on earlier folds."""
    idx, preds = [], []
    for tr, va in date_folds(dates, settings.raw["split"]["cv_folds"]):
        model = make_pipeline(C, impute=True).fit(X.iloc[tr], y[tr])
        idx.append(va)
        preds.append(model.predict_proba(X.iloc[va])[:, 1])
    return np.concatenate(idx), np.concatenate(preds)


def calibration_decision(
    X: pd.DataFrame, y: np.ndarray, dates: pd.Series, C: float, settings: Settings
) -> dict:
    idx, p = out_of_fold(X, y, dates, C, settings)
    threshold = float(settings.raw["calibration"]["ece_threshold"])
    ece = expected_calibration_error(y[idx], p)
    decision = {
        "oof_ece": ece,
        "oof_log_loss": log_loss(y[idx], p),
        "n_oof": int(len(idx)),
        "ece_threshold": threshold,
        "applied": bool(ece > threshold),
    }
    if decision["applied"]:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y[idx])
        decision["isotonic_x"] = iso.X_thresholds_.tolist()
        decision["isotonic_y"] = iso.y_thresholds_.tolist()
    return decision


# ---------- training + export ----------


def git_commit() -> str | None:
    from skytrust.evaluate import git_commit as _git_commit

    return _git_commit()


def fit_blend(train: pd.DataFrame, label: str, lead: int, settings: Settings) -> dict:
    """Fit one lead's blend on training rows and return its JSON-ready artifact."""
    label_col = LABELS[label]
    rows = training_rows(train, label_col, lead)
    feature_cols = blend_feature_columns(settings, lead)
    site_ids = sorted(rows["site"].unique())
    X = design_matrix(rows, feature_cols, site_ids)
    y = rows[label_col].astype(bool).to_numpy().astype(int)
    tuned = fit_tuned_logistic(X, pd.Series(y), rows["night_date"], settings, impute=True)
    calib = calibration_decision(X, y, rows["night_date"], tuned.C, settings)

    imputer = tuned.pipeline.named_steps["impute"]
    scaler = tuned.pipeline.named_steps["scale"]
    lr = tuned.pipeline.named_steps["lr"]
    if np.isnan(imputer.statistics_).any():  # a feature with no training values at all
        raise ValueError(f"lead {lead}: a blend feature has no training data")
    inputs = list(X.columns)
    indicators = imputer.indicator_.features_.tolist() if imputer.indicator_ is not None else []
    output_features = inputs + [f"missing_{inputs[i]}" for i in indicators]
    dates = pd.to_datetime(rows["night_date"])
    return {
        "schema_version": SCHEMA_VERSION,
        "label": label,
        "lead": lead,
        "models": [m.short for m in models_at_lead(settings, lead)],
        "model_feature_columns": feature_cols,
        "site_ids": site_ids,
        "inputs": inputs,
        "imputer": {"medians": imputer.statistics_.tolist(), "indicator_features": indicators},
        "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
        "logistic": {
            "output_features": output_features,
            "coef": lr.coef_[0].tolist(),
            "intercept": float(lr.intercept_[0]),
            "C": tuned.C,
        },
        "calibration": calib,
        "training": {
            "period": [f"{dates.min():%Y-%m-%d}", f"{dates.max():%Y-%m-%d}"],
            "n_rows": int(len(rows)),
            "base_rate": float(y.mean()),
            "cv_log_loss": tuned.cv_log_loss,
            "cv_scores": {f"{c:g}": v for c, v in tuned.cv_scores.items()},
            # Threshold for the live "models agree / split" badge (SPEC 9), from training rows.
            "spread_threshold": (
                float(
                    rows["spread_frac_clear"].quantile(settings.raw["live"]["agreement_quantile"])
                )
                if "spread_frac_clear" in feature_cols
                else None
            ),
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "created_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": git_commit(),
    }


def train_all(df: pd.DataFrame, settings: Settings, root: Path = ARTIFACTS) -> list[Path]:
    """Fit and save a blend for every label x lead. Touches training rows only."""
    train, _ = split_train_test(df)
    paths = []
    for label in LABELS:
        for lead in settings.raw["leads"]:
            artifact = fit_blend(train, label, lead, settings)
            path = artifact_path(label, lead, root)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(artifact, indent=1))
            paths.append(path)
            log.info(
                "%s lead %d: C=%g, CV log loss %.4f, OOF ECE %.3f, calibration %s",
                label, lead, artifact["logistic"]["C"], artifact["training"]["cv_log_loss"],
                artifact["calibration"]["oof_ece"],
                "applied" if artifact["calibration"]["applied"] else "not needed",
            )  # fmt: skip
    return paths


# ---------- numpy-only inference ----------


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
