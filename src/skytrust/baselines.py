"""Baselines B1-B5 (SPEC 8.2), each producing test-set predictions for one (label, lead).

Every method is scored on the same **evaluation set**: test rows at that lead where the label
is known, every model that forecasts that lead has features, and the persistence outcome is
known. Same rows for everyone means the comparisons (and paired bootstraps) are fair.

- B1 climatology: training base rate for the row's (site, month).
- B2 persistence: the observed outcome d nights earlier (already known when you'd decide).
- B3 rule: a model's own forecast run through the usable rule, as a hard yes/no.
- B4 single-model logistic regression on that model's features, C tuned by CV on train.
- B5 equal-weight average of the available models' frac_clear, used directly as P(usable).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from skytrust.config import ModelSpec, Settings
from skytrust.modeling import TunedModel, fit_tuned_logistic, split_train_test

log = logging.getLogger(__name__)

LABELS = {"primary": "usable_primary", "asos": "usable_asos", "era5": "usable_era5"}
SINGLE_MODEL_FEATURES = ("frac_clear", "longest_clear_run_frac", "mean_cover")


@dataclass
class MethodPrediction:
    method: str
    kind: str  # "prob" (a probability) or "hard" (0/1 verdict)
    family: str  # climatology / persistence / rule / single_lr / equal_weight / blend
    p: np.ndarray
    info: dict = field(default_factory=dict)


@dataclass
class LeadResult:
    label: str
    lead: int
    rows: pd.DataFrame  # the evaluation set: site, night_date, season, month, y
    methods: dict[str, MethodPrediction]
    best_single: str | None  # B4 model with the lowest *training CV* log loss (never test)


def models_at_lead(settings: Settings, lead: int) -> list[ModelSpec]:
    return [m for m in settings.models if lead in m.leads]


def single_model_columns(model: ModelSpec) -> list[str]:
    return [f"{model.short}_{f}" for f in SINGLE_MODEL_FEATURES]


# ---------- B1 climatology ----------


def climatology_table(train: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Base rate per (site, month) from *training nights only* (labels repeat across leads,
    so each night is counted once)."""
    nights = train.drop_duplicates(["site", "night_date"]).dropna(subset=[label_col])
    y = nights[label_col].astype(float)
    return (
        y.groupby([nights["site"], nights["month"]])
        .agg(["mean", "size"])
        .rename(columns={"mean": "rate", "size": "n"})
    )


def predict_climatology(table: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
    keys = pd.MultiIndex.from_arrays([rows["site"], rows["month"]])
    rate = table["rate"].reindex(keys).to_numpy()
    # A (site, month) never seen in training falls back to the site's overall rate.
    site_rate = table["rate"].groupby(level=0).mean()
    fallback = site_rate.reindex(rows["site"]).to_numpy()
    return np.where(np.isnan(rate), fallback, rate)


# ---------- B2 persistence ----------


def persistence_outcome(
    df: pd.DataFrame, rows: pd.DataFrame, label_col: str, lead: int
) -> pd.Series:
    """Observed label of the same site d nights before each row's night (NA if unknown)."""
    nights = df.drop_duplicates(["site", "night_date"])[["site", "night_date", label_col]]
    lookup = nights.set_index(["site", "night_date"])[label_col]
    earlier = pd.to_datetime(rows["night_date"]) - pd.Timedelta(days=lead)
    keys = pd.MultiIndex.from_arrays([rows["site"], earlier.dt.date])
    return pd.Series(lookup.reindex(keys).to_numpy(), index=rows.index, dtype="boolean")


# ---------- evaluation set ----------


def evaluation_rows(
    df: pd.DataFrame, test: pd.DataFrame, label_col: str, lead: int, settings: Settings
) -> tuple[pd.DataFrame, pd.Series]:
    rows = test[test["lead"] == lead]
    keep = rows[label_col].notna()
    for m in models_at_lead(settings, lead):
        keep &= rows[single_model_columns(m)].notna().all(axis=1)
        keep &= rows[f"{m.short}_pred_usable"].notna()
    persist = persistence_outcome(df, rows, label_col, lead)
    keep &= persist.notna()
    return rows[keep.to_numpy()], persist[keep.to_numpy()]


# ---------- B4 single-model logistic regression ----------


def fit_single_model(
    train: pd.DataFrame, model: ModelSpec, label_col: str, lead: int, settings: Settings
) -> TunedModel:
    cols = single_model_columns(model)
    rows = train[(train["lead"] == lead) & train[label_col].notna()].dropna(subset=cols)
    rows = rows.sort_values("night_date")
    return fit_tuned_logistic(
        rows[cols], rows[label_col].astype(bool), rows["night_date"], settings, impute=False
    )


# ---------- all baselines for one (label, lead) ----------


def run_baselines(df: pd.DataFrame, label: str, lead: int, settings: Settings) -> LeadResult:
    label_col = LABELS[label]
    train, test = split_train_test(df)
    rows, persist = evaluation_rows(df, test, label_col, lead, settings)
    eval_rows = rows[["site", "night_date", "season", "month"]].copy()
    eval_rows["y"] = rows[label_col].astype(bool).to_numpy()
    methods: dict[str, MethodPrediction] = {}

    clim = climatology_table(train, label_col)
    methods["climatology"] = MethodPrediction(
        "climatology", "prob", "climatology", predict_climatology(clim, rows)
    )
    methods["persistence"] = MethodPrediction(
        "persistence", "hard", "persistence", persist.astype(float).to_numpy()
    )
    cv_losses = {}
    for m in models_at_lead(settings, lead):
        methods[f"{m.short}_rule"] = MethodPrediction(
            f"{m.short}_rule",
            "hard",
            "rule",
            rows[f"{m.short}_pred_usable"].astype(float).to_numpy(),
        )
        tuned = fit_single_model(train, m, label_col, lead, settings)
        lr = tuned.pipeline.named_steps["lr"]
        methods[f"{m.short}_lr"] = MethodPrediction(
            f"{m.short}_lr",
            "prob",
            "single_lr",
            tuned.pipeline.predict_proba(rows[tuned.features])[:, 1],
            {
                "C": tuned.C,
                "cv_log_loss": tuned.cv_log_loss,
                "n_train": tuned.n_train,
                "coef_standardized": dict(zip(tuned.features, lr.coef_[0].tolist(), strict=True)),
                "intercept": float(lr.intercept_[0]),
            },
        )
        cv_losses[f"{m.short}_lr"] = tuned.cv_log_loss
    frac = rows[[f"{m.short}_frac_clear" for m in models_at_lead(settings, lead)]]
    methods["equal_weight"] = MethodPrediction(
        "equal_weight", "prob", "equal_weight", frac.mean(axis=1).to_numpy()
    )
    best = min(cv_losses, key=cv_losses.get) if cv_losses else None
    log.info("%s lead %d: %d eval rows, best single (CV) = %s", label, lead, len(rows), best)
    return LeadResult(label, lead, eval_rows.reset_index(drop=True), methods, best)
