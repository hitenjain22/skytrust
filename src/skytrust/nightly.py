"""The night-level rules from SPEC 4.4-4.5, implemented once and shared by labels
(observed cloud), features (forecast cloud), and the live forecast.

- A dark hour is **clear** if cloud fraction <= clear_threshold. A missing hour is never clear.
- A night is **usable** if it has a run of >= min_run_hours consecutive clear dark hours.
  Dark hours are a contiguous hourly UTC range, so "consecutive" = adjacent rows, and a
  missing hour breaks a run.
- If more than max_missing_frac of a night's dark hours are missing, its summary values are
  NaN: we don't know, and "don't know" must never be recorded as "cloudy" or "clear".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Cloud fractions come from integer percents / 100 or okta midpoints; the tolerance makes
# "exactly 0.20" count as clear even if float arithmetic lands at 0.20000000000000004.
EPS = 1e-9


@dataclass(frozen=True)
class NightRules:
    clear_threshold: float
    min_run_hours: int
    max_missing_frac: float


def is_clear(cover: np.ndarray | pd.Series, threshold: float) -> np.ndarray:
    """Boolean per hour; NaN (missing) compares False, i.e. not clear."""
    return np.asarray(cover, dtype=float) <= threshold + EPS


def longest_clear_run(cover: np.ndarray | pd.Series, threshold: float) -> int:
    """Longest streak of consecutive clear hours in an hourly sequence (simple loop version,
    used by the live forecast and as the reference implementation in tests)."""
    longest = current = 0
    for clear in is_clear(cover, threshold):
        current = current + 1 if clear else 0
        longest = max(longest, current)
    return longest


def is_usable(cover: np.ndarray | pd.Series, rules: NightRules) -> bool | None:
    """Usable-night rule for one night's dark-hour sequence.

    Returns None (excluded) when the dark window is empty or too much data is missing.
    """
    values = np.asarray(cover, dtype=float)
    if len(values) == 0 or np.isnan(values).mean() > rules.max_missing_frac:
        return None
    return longest_clear_run(values, rules.clear_threshold) >= rules.min_run_hours


SUMMARY_COLUMNS = [
    "n_hours",
    "missing_frac",
    "frac_clear",
    "longest_clear_run",
    "longest_clear_run_frac",
    "mean_cover",
    "usable",
]


def summarize_nights(
    night_hours: pd.DataFrame, cover: pd.Series, rules: NightRules
) -> pd.DataFrame:
    """Vectorised per-night summary of an hourly cloud series over each night's dark hours.

    `night_hours`: rows (night_date, hour) in time order (from astro.night_hours).
    `cover`: hourly cloud fraction indexed by UTC hour (may have gaps / NaN).

    Returns one row per night: n_hours, missing_frac, frac_clear (share of *available* dark
    hours that are clear), longest_clear_run, longest_clear_run_frac (run / all dark hours),
    mean_cover, usable (nullable boolean). All but n_hours/missing_frac are NaN/<NA> when
    missing_frac > max_missing_frac.
    """
    df = night_hours[["night_date", "hour"]].copy()
    df["cover"] = cover.reindex(pd.DatetimeIndex(df["hour"])).to_numpy()
    df["missing"] = df["cover"].isna()
    df["clear"] = is_clear(df["cover"], rules.clear_threshold)

    # Run-length trick: every non-clear hour starts a new block, so within a night all clear
    # hours in the same block are consecutive. Block size (counting clear hours) = run length.
    df["block"] = (~df["clear"]).groupby(df["night_date"]).cumsum()
    runs = df[df["clear"]].groupby(["night_date", "block"]).size()
    longest = runs.groupby(level="night_date").max()

    g = df.groupby("night_date", sort=True)
    out = pd.DataFrame(
        {
            "n_hours": g.size(),
            "missing_frac": g["missing"].mean(),
            "n_clear": g["clear"].sum(),
            "n_available": g["cover"].count(),
            "mean_cover": g["cover"].mean(),
        }
    )
    out["longest_clear_run"] = longest.reindex(out.index).fillna(0).astype(float)
    out["frac_clear"] = out["n_clear"] / out["n_available"].replace(0, np.nan)
    out["longest_clear_run_frac"] = out["longest_clear_run"] / out["n_hours"]
    out["usable"] = (out["longest_clear_run"] >= rules.min_run_hours).astype("boolean")

    too_missing = out["missing_frac"] > rules.max_missing_frac
    for col in ["frac_clear", "longest_clear_run", "longest_clear_run_frac", "mean_cover"]:
        out.loc[too_missing, col] = np.nan
    out.loc[too_missing, "usable"] = pd.NA
    return out[SUMMARY_COLUMNS]
