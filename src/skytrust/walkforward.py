"""Walk-forward (rolling-origin) evaluation: is the skill stable over time?

The main evaluation freezes one model (trained 2024-25) and tests it on 2026. Walk-forward
simulates operating SkyTrust month by month instead: for every month from `walkforward.start`
on, retrain everything (including the choice of C) on all nights *before* that month, then
forecast that month. Every prediction is out-of-sample, 2025 becomes extra evidence, and the
month-by-month scores show whether skill drifts, e.g. by season or as models are upgraded.

Methods (all refit each month, same features as the main evaluation): the blend, the
calibrated equal-weight average (B6), NOAA NBM calibrated, the raw equal-weight average (B5),
and the long-term climatology as the fixed reference.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from skytrust import baselines, blend, climatology, evaluate
from skytrust.config import REPO_ROOT, Settings
from skytrust.modeling import design_matrix, fit_tuned_logistic, model_columns

log = logging.getLogger(__name__)

WALKFORWARD_PATH = REPO_ROOT / "artifacts" / "walkforward.json"
METHODS = ("blend", "equal_weight_cal", "nbm_lr", "equal_weight", "climatology")
PAIRS = (
    ("blend", "equal_weight_cal"),
    ("blend", "nbm_lr"),
    ("blend", "equal_weight"),
    ("blend", "climatology"),
)


def _fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cols: list[str],
    label: str,
    settings: Settings,
    impute: bool,
) -> np.ndarray:
    rows = train if impute else train.dropna(subset=cols)
    rows = rows.sort_values("night_date")
    site_ids = sorted(rows["site"].unique())
    tuned = fit_tuned_logistic(
        design_matrix(rows, cols, site_ids),
        rows[label].astype(bool),
        rows["night_date"],
        settings,
        impute=impute,
    )
    return tuned.pipeline.predict_proba(design_matrix(test, cols, site_ids))[:, 1]


def month_predictions(
    df: pd.DataFrame, month: pd.Period, lead: int, settings: Settings, clim: pd.DataFrame
) -> pd.DataFrame:
    """Out-of-sample predictions for one month and lead, from models fit on earlier nights."""
    label = "usable_primary"
    dates = pd.to_datetime(df["night_date"])
    at_lead = df[df["lead"] == lead]
    past = at_lead[(dates[at_lead.index] < month.start_time) & at_lead[label].notna()]
    past = past[past["n_models_available"] > 0]
    members = baselines.models_at_lead(settings, lead)
    benches = baselines.benchmarks_at_lead(settings, lead)
    needed = [c for m in members + benches for c in model_columns(m.short)]
    now = at_lead[(dates[at_lead.index].dt.to_period("M") == month) & at_lead[label].notna()]
    now = now.dropna(subset=needed)  # same nights for every method
    if now.empty or past.empty:
        return pd.DataFrame()
    member_frac = [f"{m.short}_frac_clear" for m in members]
    out = now[["site", "night_date", "lead", "month"]].copy()
    out["y"] = now[label].astype(float).to_numpy()
    out["blend"] = _fit_predict(
        past, now, blend.blend_feature_columns(settings, lead), label, settings, impute=True
    )
    ew_past = past.assign(ew_frac_clear=past[member_frac].mean(axis=1))
    ew_now = now.assign(ew_frac_clear=now[member_frac].mean(axis=1))
    out["equal_weight_cal"] = _fit_predict(
        ew_past, ew_now, ["ew_frac_clear"], label, settings, False
    )
    out["equal_weight"] = ew_now["ew_frac_clear"].to_numpy()
    for b in benches:
        out[f"{b.short}_lr"] = _fit_predict(
            past, now, model_columns(b.short), label, settings, False
        )
    out["climatology"] = baselines.predict_climatology(clim, now)
    out["period"] = str(month)
    return out


def predictions(
    df: pd.DataFrame, settings: Settings, leads: list[int] | None = None
) -> pd.DataFrame:
    leads = leads or settings.raw["leads"]
    start = pd.Period(settings.raw["walkforward"]["start"], "M")
    end = pd.Period(pd.Timestamp(settings.raw["split"]["test_end"]), "M")
    clim = climatology.load_table("primary")
    if clim is None:  # fall back to rates from everything before the first walk-forward month
        first = df[pd.to_datetime(df["night_date"]) < start.start_time]
        clim = baselines.climatology_table(first, "usable_primary")
    frames = []
    for month in pd.period_range(start, end, freq="M"):
        for lead in leads:
            log.info("walk-forward %s lead %d", month, lead)
            frames.append(month_predictions(df, month, lead, settings, clim))
    return pd.concat([f for f in frames if not f.empty], ignore_index=True)


def summarize(preds: pd.DataFrame, settings: Settings) -> dict:
    """Pooled metrics with week-block CIs per lead, paired differences, and monthly skill."""
    threshold = settings.raw["decision_threshold"]
    rng = np.random.default_rng(int(settings.raw["seed"]))
    pooled, monthly = [], []
    for lead, g in preds.groupby("lead"):
        y, clim_p = g["y"].to_numpy(), g["climatology"].to_numpy()
        codes = evaluate.week_codes(g["night_date"])
        W = evaluate.bootstrap_weights(codes, settings.raw["bootstrap_resamples"], rng)
        entry: dict[str, Any] = {
            "lead": int(lead),
            "n": int(len(g)),
            "n_weeks": int(codes.max()) + 1,
            "months": int(g["period"].nunique()),
            "methods": {},
            "differences": [],
        }
        for m in METHODS:
            if m not in g:
                continue
            p = g[m].to_numpy(dtype=float)
            entry["methods"][m] = evaluate.point_and_ci(y, p, clim_p, W, m, threshold)
        for a, b in PAIRS:
            if a in g and b in g:
                diff = evaluate.brier_matrix(y, g[a].to_numpy(float), W) - evaluate.brier_matrix(
                    y, g[b].to_numpy(float), W
                )
                point = float(np.mean((g[a] - g["y"]) ** 2) - np.mean((g[b] - g["y"]) ** 2))
                lo, hi = evaluate._ci(diff)
                entry["differences"].append(
                    {
                        "a": a,
                        "b": b,
                        "brier_diff": point,
                        "lo": lo,
                        "hi": hi,
                        "significant": bool(hi < 0 or lo > 0),
                    }
                )
        pooled.append(entry)
        for period, mg in g.groupby("period"):
            ref = np.mean((mg["climatology"] - mg["y"]) ** 2)
            row: dict[str, Any] = {
                "lead": int(lead),
                "period": period,
                "n": int(len(mg)),
                "base_rate": float(mg["y"].mean()),
            }
            for m in METHODS:
                if m in mg and ref > 0:
                    row[f"bss_{m}"] = float(1 - np.mean((mg[m] - mg["y"]) ** 2) / ref)
            monthly.append(row)
    return {
        "start": str(settings.raw["walkforward"]["start"]),
        "end": str(settings.raw["split"]["test_end"]),
        "protocol": "expanding window, refit monthly (C re-tuned by date-based CV each time)",
        "pooled": pooled,
        "monthly": monthly,
    }


def run(df: pd.DataFrame, settings: Settings, leads: list[int] | None = None) -> dict:
    return summarize(predictions(df, settings, leads), settings)


def save(result: dict, path: Path = WALKFORWARD_PATH) -> Path:
    path.write_text(json.dumps(evaluate._clean(result), indent=1))
    return path


def load(path: Path = WALKFORWARD_PATH) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None
