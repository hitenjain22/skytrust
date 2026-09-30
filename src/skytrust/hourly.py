"""Hourly probability model: P(this dark hour is clear), per lead time.

The nightly blend answers "will tonight be usable?". Planning a session also needs "which hours?",
and hour-level post-processing is how the forecast literature treats cloud cover (e.g. logistic
/ proportional-odds regression on model output). Each dark hour becomes one example:

- target: the hour is clear under the primary truth (max of ASOS and ERA5 cover <= threshold),
  exactly the hourly quantity the nightly usable rule is built from;
- inputs: every member model's forecast cover for that hour at lead d, their mean and spread,
  where the hour sits in the night (0 = dusk, 1 = dawn), month, and site;
- model: the same tuned L2 logistic regression and JSON export as the nightly blend, so the app
  runs it with the same numpy loader.

It's evaluated on the frozen test period against the hourly climatology (training-years rate per
site and month) and the "share of models forecasting clear" baseline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from skytrust import astro, baselines, evaluate, features, labels
from skytrust.config import REPO_ROOT, Settings, Site
from skytrust.data.cache import RAW_DIR
from skytrust.dataset import default_last_night
from skytrust.inference import predict_proba, raw_inputs
from skytrust.modeling import design_matrix, fit_tuned_logistic

log = logging.getLogger(__name__)

HOURLY_DATASET = REPO_ROOT / "data" / "processed" / "hourly.parquet"
HOURLY_METRICS = REPO_ROOT / "artifacts" / "hourly_metrics.json"
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "hourly"


def model_cover_columns(settings: Settings, lead: int) -> list[str]:
    return [f"{m.short}_cover" for m in baselines.models_at_lead(settings, lead)]


def feature_columns(settings: Settings, lead: int) -> list[str]:
    return [*model_cover_columns(settings, lead), "mean_cover", "spread_cover", "night_position"]


def site_hours(site: Site, settings: Settings, first, last, root: Path = RAW_DIR) -> pd.DataFrame:
    """One row per (dark hour, lead) for one site: truth + every model's forecast cover."""
    windows = astro.dark_windows(site, first, last, settings.raw["definitions"]["sun_altitude_deg"])
    nh = astro.night_hours(windows)
    nh = nh.merge(windows[["dusk_utc", "dawn_utc"]].reset_index(), on="night_date")
    span = (nh["dawn_utc"] - nh["dusk_utc"]).dt.total_seconds()
    nh["night_position"] = ((nh["hour"] - nh["dusk_utc"]).dt.total_seconds() / span).clip(0, 1)
    observed = labels.load_observed_hourly(site.id, settings, root)
    idx = pd.DatetimeIndex(nh["hour"])
    a, e = observed["asos"].reindex(idx).to_numpy(), observed["era5"].reindex(idx).to_numpy()
    nh["truth_cover"] = np.fmax(a, e)
    forecasts = features.load_forecasts(site.id, settings.models, root)
    frames = []
    for lead in settings.raw["leads"]:
        part = nh[["night_date", "hour", "night_position", "truth_cover"]].copy()
        part["lead"] = lead
        covers = []
        for m in baselines.models_at_lead(settings, lead):
            df = forecasts.get(m.short)
            col = f"lead{lead}"
            part[f"{m.short}_cover"] = (
                df[col].reindex(idx).to_numpy() if df is not None and col in df else np.nan
            )
            covers.append(f"{m.short}_cover")
        part["mean_cover"] = part[covers].mean(axis=1)
        part["spread_cover"] = (
            part[covers].std(axis=1, ddof=0).where(part[covers].notna().sum(axis=1) >= 2)
        )
        frames.append(part)
    out = pd.concat(frames, ignore_index=True)
    out["site"] = site.id
    return out


def build_hourly(settings: Settings, sites: tuple[Site, ...], root: Path = RAW_DIR) -> pd.DataFrame:
    from skytrust.dataset import assign_split

    last = default_last_night(sites, settings, root)
    df = pd.concat(
        [site_hours(s, settings, settings.history_start, last, root) for s in sites],
        ignore_index=True,
    )
    df["month"] = pd.to_datetime(df["night_date"]).dt.month
    df["dark_hours"] = 0.0  # context columns expected by design_matrix; not informative hourly
    df["clear"] = (df["truth_cover"] <= settings.clear_threshold + 1e-9).where(
        df["truth_cover"].notna()
    )
    df["split"] = assign_split(df["night_date"], settings)
    return df


def fit_lead(train: pd.DataFrame, lead: int, settings: Settings) -> dict:
    """Fit and export (same JSON schema as the nightly blend) the hourly model for one lead."""
    from skytrust import blend

    rows = train[(train["lead"] == lead) & train["clear"].notna() & train["mean_cover"].notna()]
    rows = rows.sort_values(["night_date", "hour"])
    cols = feature_columns(settings, lead)
    site_ids = sorted(rows["site"].unique())
    X = design_matrix(rows, cols, site_ids)
    tuned = fit_tuned_logistic(
        X, rows["clear"].astype(bool), rows["night_date"], settings, impute=True
    )
    imputer, scaler, lr = (tuned.pipeline.named_steps[k] for k in ("impute", "scale", "lr"))
    inputs = list(X.columns)
    indicators = imputer.indicator_.features_.tolist() if imputer.indicator_ is not None else []
    return {
        "schema_version": blend.SCHEMA_VERSION,
        "label": "hourly_clear",
        "lead": lead,
        "variant": "hourly",
        "models": [m.short for m in baselines.models_at_lead(settings, lead)],
        "model_feature_columns": cols,
        "site_ids": site_ids,
        "inputs": inputs,
        "imputer": {"medians": imputer.statistics_.tolist(), "indicator_features": indicators},
        "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
        "logistic": {
            "output_features": inputs + [f"missing_{inputs[i]}" for i in indicators],
            "coef": lr.coef_[0].tolist(),
            "intercept": float(lr.intercept_[0]),
            "C": tuned.C,
        },
        "calibration": {"applied": False},
        "training": {
            "n_rows": int(len(rows)),
            "cv_log_loss": tuned.cv_log_loss,
            "period": [str(rows["night_date"].min()), str(rows["night_date"].max())],
        },
        "git_commit": blend.git_commit(),
    }


def evaluate_lead(
    test: pd.DataFrame,
    train: pd.DataFrame,
    artifact: dict,
    lead: int,
    settings: Settings,
    seed: int,
) -> dict:
    rows = test[(test["lead"] == lead) & test["clear"].notna()].dropna(
        subset=model_cover_columns(settings, lead)
    )
    y = rows["clear"].astype(float).to_numpy()
    tr = train[(train["lead"] == lead) & train["clear"].notna()]
    clim = tr.groupby(["site", "month"])["clear"].mean()
    p_clim = clim.reindex(pd.MultiIndex.from_arrays([rows["site"], rows["month"]])).to_numpy(
        dtype=float
    )
    p_clim = np.where(np.isnan(p_clim), tr["clear"].mean(), p_clim)
    covers = rows[model_cover_columns(settings, lead)].to_numpy()
    p_share = (covers <= settings.clear_threshold + 1e-9).mean(
        axis=1
    )  # share of models saying clear
    p_model = predict_proba(artifact, raw_inputs(rows, artifact))
    W = evaluate.bootstrap_weights(
        evaluate.week_codes(rows["night_date"]),
        settings.raw["bootstrap_resamples"],
        np.random.default_rng(seed),
    )
    out: dict[str, Any] = {
        "lead": lead,
        "n_hours": int(len(rows)),
        "base_rate": float(y.mean()),
        "methods": {},
    }
    for name, p in [
        ("hourly_model", p_model),
        ("share_of_models", p_share),
        ("climatology", p_clim),
    ]:
        out["methods"][name] = {
            **evaluate.point_and_ci(y, p, p_clim, W, name, 0.5),
            "reliability": evaluate.reliability_table(y, p),
        }
    return out


def train_and_evaluate(df: pd.DataFrame, settings: Settings, out_dir: Path = ARTIFACT_DIR) -> dict:
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for lead in settings.raw["leads"]:
        artifact = fit_lead(train, lead, settings)
        (out_dir / f"model_hourly_lead{lead}.json").write_text(json.dumps(artifact, indent=1))
        results.append(
            evaluate_lead(test, train, artifact, lead, settings, int(settings.raw["seed"]) + lead)
        )
        log.info("hourly lead %d: BSS %.3f", lead, results[-1]["methods"]["hourly_model"]["bss"])
    return {"leads": results}


def save(result: dict, path: Path = HOURLY_METRICS) -> Path:
    path.write_text(json.dumps(evaluate._clean(result), indent=1))
    return path


def load(path: Path = HOURLY_METRICS) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None
