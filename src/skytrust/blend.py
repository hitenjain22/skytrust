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
from skytrust.config import Settings
from skytrust.inference import (  # noqa: F401  (inference names re-exported for callers)
    ARTIFACTS,
    SCHEMA_VERSION,
    artifact_path,
    assert_trained_before,
    load_artifact,
    predict_proba,
    raw_inputs,
    summary,
    weight_shares,
)
from skytrust.modeling import (
    date_folds,
    design_matrix,
    fit_tuned_logistic,
    log_loss,
    make_pipeline,
    model_columns,
    split_train_test,
)

log = logging.getLogger(__name__)


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
