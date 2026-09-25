"""Model input design shared by training (B4, blend) and live inference.

Pure pandas/numpy on purpose: the app imports this, and must not pull in scikit-learn.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MODEL_FEATURES = ("frac_clear", "longest_clear_run_frac", "mean_cover")


def model_columns(model_short: str) -> list[str]:
    """The three per-model forecast features used by B4 and the blend."""
    return [f"{model_short}_{f}" for f in MODEL_FEATURES]


def context_features(rows: pd.DataFrame, site_ids: list[str]) -> pd.DataFrame:
    """Context shared by B4 and the blend (SPEC 8.3): site one-hot, month as a point on a
    circle (so December and January are neighbours), and the night's dark-hour count.

    Giving B4 exactly the same context as the blend means the only difference between
    "best single model" and "blend" is how many weather models they see.
    """
    out = pd.DataFrame(index=rows.index)
    for site in site_ids:
        out[f"site_{site}"] = (rows["site"] == site).astype(float)
    angle = 2 * np.pi * rows["month"].astype(float) / 12
    out["month_sin"] = np.sin(angle)
    out["month_cos"] = np.cos(angle)
    out["dark_hours"] = rows["dark_hours"].astype(float)
    return out


def design_matrix(rows: pd.DataFrame, feature_cols: list[str], site_ids: list[str]) -> pd.DataFrame:
    """Forecast feature columns (as given) followed by the context features."""
    return pd.concat([rows[feature_cols].astype(float), context_features(rows, site_ids)], axis=1)
