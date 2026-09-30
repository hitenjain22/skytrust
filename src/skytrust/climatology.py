"""Long-term climatology: the reference forecast every skill score is measured against.

The first version used the two training years (~60 nights per site-month), which is noisy, and
a noisy reference flatters every forecast's skill score. Forecast verification normally uses a
long-term climatology, so this module labels every night of a multi-decade reference period
(2004-2023, entirely before the training and test periods) with exactly the same label code, keeps
station-years with good coverage, and stores P(usable) per (site, month).
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

import pandas as pd

from skytrust import astro, labels
from skytrust.config import REPO_ROOT, Settings, Site
from skytrust.data.cache import RAW_DIR
from skytrust.dataset import night_rules
from skytrust.errors import LeakageError

log = logging.getLogger(__name__)

TABLE_PATH = REPO_ROOT / "artifacts" / "climatology.json"
LABELS_PATH = REPO_ROOT / "data" / "processed" / "climatology_labels.parquet"
LABEL_COLUMNS = {"primary": "usable_primary", "asos": "usable_asos", "era5": "usable_era5"}


def reference_period(settings: Settings) -> tuple[dt.date, dt.date]:
    cfg = settings.raw["climatology"]
    first, last = cfg["start"], cfg["end"]
    if last >= settings.raw["split"]["train_start"]:
        raise LeakageError(f"climatology period ends {last}, after training starts")
    return first, last


def label_reference_nights(
    settings: Settings, sites: tuple[Site, ...], root: Path = RAW_DIR
) -> pd.DataFrame:
    """Every reference-period night at every site, labelled exactly like the dataset."""
    first, last = reference_period(settings)
    rules, frames = night_rules(settings), []
    for site in sites:
        log.info("climatology labels for %s %s..%s", site.id, first, last)
        windows = astro.dark_windows(
            site, first, last, settings.raw["definitions"]["sun_altitude_deg"]
        )
        observed = labels.load_observed_hourly(site.id, settings, root)
        lab = labels.build_labels(windows, astro.night_hours(windows), observed, rules)
        frames.append(lab[list(LABEL_COLUMNS.values())].reset_index().assign(site=site.id))
    df = pd.concat(frames, ignore_index=True)
    dates = pd.to_datetime(df["night_date"])
    return df.assign(year=dates.dt.year, month=dates.dt.month)


def usable_years(df: pd.DataFrame, min_coverage: float) -> dict[str, list[int]]:
    """Per site, the years in which at least `min_coverage` of nights have a primary label.
    Station outages (e.g. AUN had no reports for stretches of 2014) would otherwise tilt a
    month's rate toward whatever part of it happened to be observed."""
    cov = df.groupby(["site", "year"])["usable_primary"].apply(lambda s: s.notna().mean())
    good = cov[cov >= min_coverage].reset_index()
    return {site: sorted(g["year"].tolist()) for site, g in good.groupby("site")}


def rate_tables(df: pd.DataFrame, years: dict[str, list[int]]) -> dict:
    """{label: {site: {month: {"rate", "n"}}}} from the usable years only."""
    keep = pd.concat([df[(df["site"] == s) & df["year"].isin(ys)] for s, ys in years.items()])
    tables: dict = {}
    for name, col in LABEL_COLUMNS.items():
        sub = keep.dropna(subset=[col])
        grouped = sub.groupby(["site", "month"])[col].agg(
            rate=lambda s: float(s.astype(float).mean()), n="size"
        )
        tables[name] = {
            site: {
                int(m): {"rate": float(r.rate), "n": int(r.n)} for m, r in g.droplevel(0).iterrows()
            }
            for site, g in grouped.groupby(level=0)
        }
    return tables


def build(
    settings: Settings, sites: tuple[Site, ...], root: Path = RAW_DIR
) -> tuple[dict, pd.DataFrame]:
    df = label_reference_nights(settings, sites, root)
    years = usable_years(df, settings.raw["climatology"]["min_year_coverage"])
    first, last = reference_period(settings)
    return {
        "period": [str(first), str(last)],
        "min_year_coverage": settings.raw["climatology"]["min_year_coverage"],
        "years_used": years,
        "tables": rate_tables(df, years),
    }, df


def save(table: dict, labels_df: pd.DataFrame, table_path: Path = TABLE_PATH,
         labels_path: Path = LABELS_PATH) -> None:  # fmt: skip
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(json.dumps(table, indent=1))
    labels_df.to_parquet(labels_path, index=False)


def load_table(label: str = "primary", path: Path = TABLE_PATH) -> pd.DataFrame | None:
    """Rates indexed by (site, month) in the same shape as baselines.climatology_table, or None
    if the long-term table hasn't been built."""
    if not path.exists():
        return None
    tables = json.loads(path.read_text())["tables"][label]
    rows = [
        (site, int(m), v["rate"], v["n"])
        for site, months in tables.items()
        for m, v in months.items()
    ]
    df = pd.DataFrame(rows, columns=["site", "month", "rate", "n"])
    return df.set_index(["site", "month"])
