"""Forecast features per (night, lead, model) from Previous Runs data (SPEC 4.7).

For lead d, each dark hour H uses the value the model predicted ~24*d hours before H. Over
a night's dark hours we compute, per model: frac_clear, longest_clear_run(_frac), mean_cover,
pred_usable (the usable-night rule applied to the forecast), and missing_frac. Across
models: spread_frac_clear (how much they disagree) and n_models_available.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from skytrust.config import ModelSpec
from skytrust.data import openmeteo
from skytrust.data.cache import RAW_DIR
from skytrust.nightly import NightRules, summarize_nights

FEATURES = [
    "frac_clear",
    "longest_clear_run",
    "longest_clear_run_frac",
    "mean_cover",
    "pred_usable",
    "missing_frac",
]


def load_forecasts(
    site_id: str, models: tuple[ModelSpec, ...], root: Path = RAW_DIR
) -> dict[str, pd.DataFrame]:
    """Cached Previous Runs data per model short name, columns lead1..leadN."""
    return {m.short: openmeteo.load_prevruns(site_id, m, root) for m in models}


def model_features(
    night_hours: pd.DataFrame, nights: pd.Index, hourly: pd.Series | None, rules: NightRules
) -> pd.DataFrame:
    """The six features for one model at one lead. A lead the model doesn't have (e.g. HRRR
    beyond day 1) gives all-NaN features with missing_frac = 1."""
    if hourly is None:
        out = pd.DataFrame(np.nan, index=nights, columns=FEATURES)
        out["missing_frac"] = 1.0
        out["pred_usable"] = pd.array([pd.NA] * len(nights), dtype="boolean")
        return out
    s = summarize_nights(night_hours, hourly, rules).reindex(nights)
    out = s.rename(columns={"usable": "pred_usable"})[FEATURES].copy()
    out["missing_frac"] = out["missing_frac"].fillna(1.0)  # nights with no dark hours
    return out


def build_features(
    night_hours: pd.DataFrame,
    nights: pd.Index,
    forecasts: dict[str, pd.DataFrame],
    models: tuple[ModelSpec, ...],
    leads: list[int],
    rules: NightRules,
) -> pd.DataFrame:
    """Wide feature table indexed by (night_date, lead): `{model}_{feature}` columns plus
    cross-model `spread_frac_clear` and `n_models_available`."""
    per_lead = []
    for lead in leads:
        cols = {}
        for m in models:
            df = forecasts.get(m.short)
            col = f"lead{lead}"
            hourly = df[col] if df is not None and lead in m.leads and col in df else None
            feats = model_features(night_hours, nights, hourly, rules)
            for name in FEATURES:
                cols[f"{m.short}_{name}"] = feats[name]
        wide = pd.DataFrame(cols, index=nights)
        frac = wide[[f"{m.short}_frac_clear" for m in models]]
        wide["n_models_available"] = frac.notna().sum(axis=1)
        # Population std (ddof=0) across the models that have a value. Undefined with < 2.
        spread = frac.std(axis=1, ddof=0)
        wide["spread_frac_clear"] = spread.where(wide["n_models_available"] >= 2)
        wide.insert(0, "lead", lead)
        per_lead.append(wide)
    return pd.concat(per_lead).reset_index().set_index(["night_date", "lead"]).sort_index()
