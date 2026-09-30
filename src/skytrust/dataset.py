"""Join astronomy, labels, and forecast features into the modeling dataset (SPEC 7).

One row per (site, night_date, lead). Labels and astronomy are per night, so they repeat
across the 7 leads; features differ by lead. Saved to data/processed/dataset.parquet and
committed, so the app and later phases never need the raw cache.
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import pandas as pd

from skytrust import astro, features, labels
from skytrust.config import REPO_ROOT, Settings, Site
from skytrust.data import iem, openmeteo
from skytrust.data.cache import RAW_DIR
from skytrust.nightly import NightRules

log = logging.getLogger(__name__)

DATASET_PATH = REPO_ROOT / "data" / "processed" / "dataset.parquet"
KEYS = ["site", "night_date", "lead"]
SEASONS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
           6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}  # fmt: skip
LABEL_COLUMNS = ["usable_primary", "usable_asos", "usable_era5", "usable_goes"]


class DatasetValidationError(Exception):
    """The built dataset violates an invariant; it must not be saved."""


def night_rules(settings: Settings) -> NightRules:
    return NightRules(settings.clear_threshold, settings.min_run_hours, settings.max_missing_frac)


def assign_split(night_dates: pd.Series, settings: Settings) -> pd.Series:
    """'train' / 'test' from the fixed time-based split (SPEC 8.1); anything else 'none'
    (before training, or after the frozen end of the test period)."""
    split = settings.raw["split"]
    d = pd.to_datetime(night_dates)

    def between(start, end) -> pd.Series:
        return (d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))

    out = pd.Series("none", index=night_dates.index, dtype="string")
    out[between(split["train_start"], split["train_end"])] = "train"
    out[between(split["test_start"], split["test_end"])] = "test"
    return out


def build_site_dataset(
    site: Site,
    settings: Settings,
    first_night: dt.date,
    last_night: dt.date,
    observed: dict[str, pd.Series],
    forecasts: dict[str, pd.DataFrame],
    astro_tables: tuple[pd.DataFrame, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Pure function: all inputs are in memory, so it's testable without the cache.
    `astro_tables` (from astro.night_table) can be passed in to reuse them across runs that only
    change the cloud definitions (the sensitivity grid)."""
    rules = night_rules(settings)
    nights_astro, night_hours = astro_tables or astro.night_table(
        site,
        first_night,
        last_night,
        settings.raw["definitions"]["sun_altitude_deg"],
        settings.raw["astro"]["moon_up_altitude_deg"],
    )
    nights = nights_astro.index
    lab = labels.build_labels(nights_astro[["dusk_utc", "dawn_utc"]], night_hours, observed, rules)
    feats = features.build_features(
        night_hours, nights, forecasts, settings.models, settings.raw["leads"], rules,
        benchmarks=settings.benchmarks,
    )  # fmt: skip
    per_night = nights_astro.join(lab)
    df = feats.reset_index().merge(per_night.reset_index(), on="night_date", how="left")
    months = pd.to_datetime(df["night_date"]).dt.month
    df["month"] = months
    df["season"] = months.map(SEASONS).astype("string")
    df["split"] = assign_split(df["night_date"], settings)
    df["site"] = site.id
    first = ["site", "night_date", "lead", "season", "month", "split", "dusk_utc", "dawn_utc",
             "dark_hours", "moon_illum_mean", "moon_free_dark_hours"]  # fmt: skip
    rest = [c for c in df.columns if c not in first]
    return df[first + rest].sort_values(["night_date", "lead"]).reset_index(drop=True)


def cached_last_night(sites: tuple[Site, ...], settings: Settings, root: Path = RAW_DIR) -> dt.date:
    """Latest night for which ASOS and every model's forecasts are cached at every site.

    ERA5 lags by ~6 days and is deliberately NOT part of this: nights after its end are kept
    with an 'era5_missing' exclusion, so the report shows the gap honestly.
    """
    ends = []
    for site in sites:
        obs = iem.load_asos(site.id, root)
        if len(obs):
            ends.append(obs["valid"].max())
        for m in settings.models:
            prev = openmeteo.load_prevruns(site.id, m, root)
            if len(prev):
                ends.append(prev.index.max())
    if not ends:
        raise DatasetValidationError("raw cache is empty; run `skytrust fetch` first")
    # The last night's dark hours end on the following morning (UTC).
    return min(ends).date() - dt.timedelta(days=1)


def default_last_night(
    sites: tuple[Site, ...], settings: Settings, root: Path = RAW_DIR
) -> dt.date:
    """The dataset stops at the frozen test end, so rebuilding on a later day gives the same
    dataset (and the same results) even though the raw cache keeps growing."""
    return min(cached_last_night(sites, settings, root), settings.raw["split"]["test_end"])


def build_dataset(
    settings: Settings,
    sites: tuple[Site, ...],
    first_night: dt.date | None = None,
    last_night: dt.date | None = None,
    root: Path = RAW_DIR,
) -> pd.DataFrame:
    first_night = first_night or settings.history_start
    last_night = last_night or default_last_night(sites, settings, root)
    parts = []
    for site in sites:
        log.info("building %s nights %s..%s", site.id, first_night, last_night)
        observed = labels.load_observed_hourly(site.id, settings, root)
        forecasts = features.load_forecasts(site.id, settings.forecast_models, root)
        parts.append(
            build_site_dataset(site, settings, first_night, last_night, observed, forecasts)
        )
    return pd.concat(parts, ignore_index=True)


def _fraction_columns(df: pd.DataFrame) -> list[str]:
    """Columns that must be cloud/missing fractions in [0, 1]."""
    suffixes = ("_frac_clear", "_longest_clear_run_frac", "_mean_cover", "_missing_frac")
    return [c for c in df.columns if c.endswith(suffixes)]


def validate_dataset(df: pd.DataFrame) -> None:
    """Invariants from SPEC 12.1 'Dataset validation'. Raises with every problem found."""
    problems = []
    if df.duplicated(KEYS).any():
        problems.append(f"{int(df.duplicated(KEYS).sum())} duplicate (site, night_date, lead) rows")
    for col in _fraction_columns(df):
        values = df[col].dropna()
        if len(values) and not values.between(0, 1).all():
            problems.append(f"{col} has values outside [0, 1]")
    dark = df.loc[df["dark_hours"] > 0, "dark_hours"]
    if len(dark) and not dark.between(4, 13).all():
        problems.append(f"dark_hours outside 4-13: {sorted(dark[~dark.between(4, 13)].unique())}")
    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_datetime64_dtype(dtype) and not isinstance(dtype, pd.DatetimeTZDtype):
            problems.append(f"{col} is a naive timestamp column")
    for col in ["dusk_utc", "dawn_utc"]:
        if col in df and str(df[col].dt.tz) != "UTC":
            problems.append(f"{col} is not UTC")
    for col in LABEL_COLUMNS:
        if str(df[col].dtype) != "boolean":
            problems.append(f"{col} should be nullable boolean, got {df[col].dtype}")
    if problems:
        raise DatasetValidationError("; ".join(problems))


def save_dataset(df: pd.DataFrame, path: Path = DATASET_PATH) -> Path:
    validate_dataset(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)
