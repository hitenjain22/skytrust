"""Observed ("ground truth") labels per night: ASOS-only, ERA5-only, primary (SPEC 4.6), and
GOES satellite.

- ASOS sees real clouds but only below 12,000 ft (except human-augmented FAT).
- ERA5 sees every layer, including cirrus, but it's a model (reanalysis) on a ~28 km grid.
- Primary: per hour take max(asos, era5), so an hour is clear only if *both* agree it's
  clear. If one source is missing that hour, the other is used and the hour is flagged. A
  night is excluded from the primary label if either source misses > 25 % of its dark hours.
- GOES: the GOES-18 Clear Sky Mask over a ~10 km box (data/goes.py). A real observation that
  also sees high cloud, used as an independent check. It never changes the primary label.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from skytrust.config import Settings
from skytrust.data import goes, iem, openmeteo
from skytrust.data.cache import RAW_DIR
from skytrust.nightly import NightRules, summarize_nights

log = logging.getLogger(__name__)

NO_DARKNESS = "no_darkness"
ERA5_LAYERS = ("low", "mid", "high")


def load_observed_hourly(
    site_id: str, settings: Settings, root: Path = RAW_DIR
) -> dict[str, pd.Series]:
    """Hourly cloud fraction from the raw cache: ASOS (configured aggregation), ASOS with
    the 'max' aggregation (sensitivity only), ERA5 total cloud cover, and GOES (empty if the
    satellite extracts haven't been fetched)."""
    obs = iem.load_asos(site_id, root)
    era5 = openmeteo.load_era5(site_id, root)
    mapping = settings.sky_cover_mapping
    return {
        "asos": iem.hourly_cover(obs, mapping, settings.asos_hour_aggregation),
        "asos_max": iem.hourly_cover(obs, mapping, "max"),
        "era5": era5["cloud_cover"],
        "goes": goes.goes_hourly(site_id, root / "goes"),
        # Layers are stored for analysis only (SPEC 4.6b), e.g. "is the ASOS/ERA5 gap cirrus?"
        **{f"era5_{layer}": era5[f"cloud_cover_{layer}"] for layer in ERA5_LAYERS},
    }


def _single_source_reason(no_dark: pd.Series, bad: pd.Series, reason: str) -> pd.Series:
    out = pd.Series(pd.NA, index=no_dark.index, dtype="string")
    out[bad.fillna(False).astype(bool)] = reason
    out[no_dark] = NO_DARKNESS
    return out


def _exclusion_reason(no_dark: pd.Series, asos_bad: pd.Series, era5_bad: pd.Series) -> pd.Series:
    reason = pd.Series(pd.NA, index=no_dark.index, dtype="string")
    reason[asos_bad & ~era5_bad] = "asos_missing"
    reason[era5_bad & ~asos_bad] = "era5_missing"
    reason[asos_bad & era5_bad] = "asos_and_era5_missing"
    reason[no_dark] = NO_DARKNESS
    return reason


def build_labels(
    windows: pd.DataFrame,
    night_hours: pd.DataFrame,
    hourly: dict[str, pd.Series],
    rules: NightRules,
) -> pd.DataFrame:
    """One row per night in `windows` (from astro.dark_windows) with every label.

    Columns: usable_primary / usable_asos / usable_era5 / usable_asos_max / usable_goes
    (nullable boolean),
    asos_missing_frac, era5_missing_frac, n_single_source_hours, longest runs, mean truth
    cover, and exclusion reasons for each label (NA = included).
    """
    nights = windows.index
    asos = summarize_nights(night_hours, hourly["asos"], rules).reindex(nights)
    era5 = summarize_nights(night_hours, hourly["era5"], rules).reindex(nights)
    asos_max = summarize_nights(night_hours, hourly["asos_max"], rules).reindex(nights)

    # Primary hourly truth = max of the two sources; np.fmax ignores a NaN on one side.
    idx = pd.DatetimeIndex(night_hours["hour"])
    # GOES: same rules, independent of the others; no input at all = every night missing.
    goes_cover = hourly.get("goes")
    if goes_cover is None:
        goes_cover = pd.Series(np.nan, index=idx)
    sat = summarize_nights(night_hours, goes_cover, rules).reindex(nights)
    a = hourly["asos"].reindex(idx).to_numpy()
    e = hourly["era5"].reindex(idx).to_numpy()
    truth = pd.Series(np.fmax(a, e), index=idx)
    single = pd.Series(np.isnan(a) ^ np.isnan(e), index=idx)
    primary = summarize_nights(night_hours, truth, rules).reindex(nights)
    n_single = single.groupby(night_hours["night_date"].to_numpy()).sum().reindex(nights)

    no_dark = windows["dusk_utc"].isna()
    too_missing_asos = asos["missing_frac"] > rules.max_missing_frac
    too_missing_era5 = era5["missing_frac"] > rules.max_missing_frac

    out = pd.DataFrame(index=nights)
    out["usable_primary"] = primary["usable"].astype("boolean")
    out.loc[too_missing_asos | too_missing_era5, "usable_primary"] = pd.NA
    out["usable_asos"] = asos["usable"].astype("boolean")
    out["usable_era5"] = era5["usable"].astype("boolean")
    out["usable_asos_max"] = asos_max["usable"].astype("boolean")
    out["asos_missing_frac"] = asos["missing_frac"]
    out["era5_missing_frac"] = era5["missing_frac"]
    out["n_single_source_hours"] = n_single.fillna(0).astype(int)
    out["truth_longest_clear_run"] = primary["longest_clear_run"]
    out["truth_mean_cover"] = primary["mean_cover"]
    out["asos_mean_cover"] = asos["mean_cover"]
    out["era5_mean_cover"] = era5["mean_cover"]
    out["usable_goes"] = sat["usable"].astype("boolean")
    out["goes_missing_frac"] = sat["missing_frac"]
    out["goes_mean_cover"] = sat["mean_cover"]
    for layer in ERA5_LAYERS:
        key = f"era5_{layer}"
        if key in hourly:
            out[f"era5_{layer}_mean_cover"] = summarize_nights(
                night_hours, hourly[key], rules
            ).reindex(nights)["mean_cover"]
    out["exclusion_primary"] = _exclusion_reason(no_dark, too_missing_asos, too_missing_era5)
    none = pd.Series(False, index=nights)
    out["exclusion_asos"] = _exclusion_reason(no_dark, too_missing_asos, none)
    out["exclusion_era5"] = _exclusion_reason(no_dark, none, too_missing_era5)
    too_missing_goes = sat["missing_frac"] > rules.max_missing_frac
    out["exclusion_goes"] = _single_source_reason(no_dark, too_missing_goes, "goes_missing")
    # Nights with no darkness have no summary rows at all; make sure they read as excluded.
    for col in ["usable_primary", "usable_asos", "usable_era5", "usable_asos_max", "usable_goes"]:
        out.loc[no_dark, col] = pd.NA

    counts = out["exclusion_primary"].value_counts()
    if len(counts):
        log.info("primary label exclusions: %s", counts.to_dict())
    return out
