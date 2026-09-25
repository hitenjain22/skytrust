"""Generated documents. Every number here is computed from the dataset/metrics in code;
nothing in these files is typed by hand (SPEC 0, rule 3).

Phase 2: docs/DATA_QUALITY.md. (RESULTS.md and figures are added in Phase 3.)
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from skytrust.config import REPO_ROOT, Settings

DOCS = REPO_ROOT / "docs"
SEASON_ORDER = ["DJF", "MAM", "JJA", "SON"]
MIN_TEST_NIGHTS_LEAD1 = 300  # SPEC 17: flag if fewer labeled test nights at lead 1


# ---------- small formatting helpers ----------


def pct(x: float) -> str:
    return "–" if pd.isna(x) else f"{x:.1%}"


def md_table(df: pd.DataFrame, index: bool = True) -> str:
    """Render a DataFrame of already-formatted values as a GitHub markdown table."""
    frame = df.reset_index() if index else df
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    sep = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.to_numpy()]
    return "\n".join([header, sep, *rows])


def rate_with_n(values: pd.Series) -> str:
    """'63.0 % (n=978)' for a nullable boolean label; excluded (NA) nights are not counted."""
    known = values.dropna()
    return "–" if known.empty else f"{known.astype(float).mean():.1%} (n={len(known)})"


def one_row_per_night(df: pd.DataFrame) -> pd.DataFrame:
    """Labels/astronomy repeat across leads; take lead 1 for night-level statistics."""
    return df[df["lead"] == df["lead"].min()].copy()


# ---------- DATA_QUALITY sections ----------


def overview_table(nights: pd.DataFrame) -> pd.DataFrame:
    g = nights.groupby("site", sort=False)
    dark = nights[nights["dark_hours"] > 0].groupby("site", sort=False)["dark_hours"]
    labeled = nights[nights["usable_primary"].notna()]
    return pd.DataFrame(
        {
            "nights": g.size(),
            "first": g["night_date"].min(),
            "last": g["night_date"].max(),
            "no darkness": g["dark_hours"].apply(lambda s: int((s == 0).sum())),
            "dark hours min/median/max": dark.agg(
                lambda s: f"{s.min()} / {s.median():.0f} / {s.max()}"
            ),
            "labeled train (primary)": labeled[labeled["split"] == "train"].groupby("site").size(),
            "labeled test (primary)": labeled[labeled["split"] == "test"].groupby("site").size(),
        }
    ).fillna(0)


def observation_coverage_table(nights: pd.DataFrame) -> pd.DataFrame:
    g = nights[nights["dark_hours"] > 0].groupby("site", sort=False)
    return pd.DataFrame(
        {
            "ASOS dark hours missing": g["asos_missing_frac"].mean().map(pct),
            "ASOS nights excluded": g["exclusion_asos"].apply(lambda s: int(s.notna().sum())),
            "ERA5 dark hours missing": g["era5_missing_frac"].mean().map(pct),
            "ERA5 nights excluded": g["exclusion_era5"].apply(lambda s: int(s.notna().sum())),
            "hours filled from one source": (
                g["n_single_source_hours"].sum() / g["dark_hours"].sum()
            ).map(pct),
        }
    )


def forecast_coverage_table(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Share of site-nights with usable features, per model x lead, counted from each model's
    first available night (so the pre-archive weeks of Jan 2024 don't dilute it)."""
    table = {}
    for m in settings.models:
        col = f"{m.short}_frac_clear"
        row = {}
        for lead in settings.raw["leads"]:
            if lead not in m.leads:
                row[f"L{lead}"] = "–"
                continue
            sub = df[df["lead"] == lead]
            available = sub[col].notna()
            start = sub.loc[available, "night_date"].min() if available.any() else None
            in_range = sub[sub["night_date"] >= start] if start else sub.iloc[0:0]
            row[f"L{lead}"] = pct(in_range[col].notna().mean()) if len(in_range) else "0"
        first = df.loc[df[col].notna(), "night_date"].min()
        row["first night"] = first if pd.notna(first) else "–"
        table[f"{m.short} ({m.id})"] = row
    out = pd.DataFrame(table).T
    out.index.name = "model"
    return out


def exclusions_table(nights: pd.DataFrame) -> pd.DataFrame:
    counts = (
        nights.dropna(subset=["exclusion_primary"])
        .groupby(["site", "exclusion_primary"], sort=False)
        .size()
        .unstack(fill_value=0)
    )
    counts["total excluded"] = counts.sum(axis=1)
    counts["share of nights"] = (counts["total excluded"] / nights.groupby("site").size()).map(pct)
    return counts


def base_rate_by_season(nights: pd.DataFrame, label: str) -> pd.DataFrame:
    table = nights.pivot_table(
        index="site", columns="season", values=label, aggfunc=rate_with_n, sort=False
    )
    table = table.reindex(columns=[s for s in SEASON_ORDER if s in table.columns])
    table["all"] = nights.groupby("site", sort=False)[label].apply(rate_with_n)
    return table


def label_comparison_table(nights: pd.DataFrame) -> pd.DataFrame:
    labels = {
        "primary": "usable_primary",
        "ASOS only": "usable_asos",
        "ERA5 only": "usable_era5",
        "ASOS (max-per-hour rule)": "usable_asos_max",
    }
    g = nights.groupby("site", sort=False)
    return pd.DataFrame(
        {
            name: g[col].apply(lambda s: pct(s.dropna().astype(float).mean()))
            for name, col in labels.items()
        }
    )


def agreement_table(nights: pd.DataFrame) -> pd.DataFrame:
    """How the two truth sources disagree, and whether ERA5's extra cloud is high cloud."""
    both = nights.dropna(subset=["usable_asos", "usable_era5"])
    a, e = both["usable_asos"].astype(bool), both["usable_era5"].astype(bool)
    both = both.assign(
        cat=np.select([a & e, a & ~e, ~a & e], ["both", "asos_only", "era5_only"], "neither")
    )
    g = both.groupby("site", sort=False)
    out = pd.DataFrame(
        {
            "agree": g.apply(lambda d: pct((d["cat"].isin(["both", "neither"])).mean())),
            "ASOS usable, ERA5 not": g.apply(lambda d: pct((d["cat"] == "asos_only").mean())),
            "ERA5 usable, ASOS not": g.apply(lambda d: pct((d["cat"] == "era5_only").mean())),
        }
    )
    if {"era5_high_mean_cover", "era5_low_mean_cover"} <= set(both.columns):
        dis = both[both["cat"] == "asos_only"].groupby("site", sort=False)
        out["…those nights: ERA5 mean high cloud"] = dis["era5_high_mean_cover"].mean().map(pct)
        out["…those nights: ERA5 mean low cloud"] = dis["era5_low_mean_cover"].mean().map(pct)
    return out


def train_test_shift_table(nights: pd.DataFrame) -> pd.DataFrame:
    """Base rate in train vs test: a big shift means 2026 weather differs from 2024-25, which
    matters when reading test metrics."""
    sub = nights[nights["split"].isin(["train", "test"])]
    return sub.pivot_table(
        index="site", columns="split", values="usable_primary", aggfunc=rate_with_n, sort=False
    )[["train", "test"]]


def test_sufficiency(df: pd.DataFrame) -> tuple[int, str]:
    lead1 = df[(df["lead"] == 1) & (df["split"] == "test")]
    n = int((lead1["usable_primary"].notna() & (lead1["n_models_available"] > 0)).sum())
    if n >= MIN_TEST_NIGHTS_LEAD1:
        verdict = f"OK: {n} ≥ {MIN_TEST_NIGHTS_LEAD1}"
    else:
        verdict = f"⚠ FLAG: only {n} < {MIN_TEST_NIGHTS_LEAD1} (SPEC 17)"
    return n, verdict


def data_quality_markdown(df: pd.DataFrame, settings: Settings, generated: dt.datetime) -> str:
    nights = one_row_per_night(df)
    n_test, verdict = test_sufficiency(df)
    defs = settings.raw["definitions"]
    parts = [
        "# Data Quality Report",
        "",
        f"_Generated by `python -m skytrust build-dataset` at {generated:%Y-%m-%d %H:%M} UTC. "
        "Do not edit by hand; every number is computed from `data/processed/dataset.parquet`._",
        "",
        f"Dataset: **{len(df):,} rows** = {nights['site'].nunique()} sites × "
        f"{len(nights):,} site-nights × {df['lead'].nunique()} leads. "
        f"Definitions: clear ≤ {defs['clear_threshold']:.2f}, usable = ≥ {defs['min_run_hours']} "
        f"consecutive clear dark hours, > {defs['max_missing_frac']:.0%} missing → excluded, "
        f"ASOS hourly value = `{defs['asos_hour_aggregation']}` report.",
        "",
        "## 1. Overview",
        "",
        md_table(overview_table(nights)),
        "",
        "## 2. Observation coverage over true dark windows",
        "",
        "Share of astronomical-dark hours with no value, and nights excluded (> 25 % missing). "
        "ERA5's trailing exclusions are its ~6-day publication lag, not data loss.",
        "",
        md_table(observation_coverage_table(nights)),
        "",
        "## 3. Forecast coverage (share of site-nights with features)",
        "",
        'Counted from each model/lead\'s first available night. "–" = the model has no such lead.',
        "",
        md_table(forecast_coverage_table(df, settings)),
        "",
        "## 4. Nights excluded from the primary label, and why",
        "",
        md_table(exclusions_table(nights)),
        "",
        "## 5. Base rate of usable nights (primary label) by site × season",
        "",
        "Rate among labeled nights, with the number of labeled nights. This is also what the "
        "climatology baseline will predict.",
        "",
        md_table(base_rate_by_season(nights, "usable_primary")),
        "",
        "## 6. Base rate under each label definition (sensitivity)",
        "",
        "ASOS can't see cloud above 12,000 ft, so ASOS-only rates are expected to be higher. "
        "The last column applies SPEC's original max-over-the-hour ASOS rule.",
        "",
        md_table(label_comparison_table(nights)),
        "",
        "## 7. Do ASOS and ERA5 agree?",
        "",
        "Nights where both labels exist. If ASOS's blind spot is cirrus, the nights where ASOS "
        "says usable but ERA5 doesn't should show mostly *high* ERA5 cloud.",
        "",
        md_table(agreement_table(nights)),
        "",
        "## 8. Train vs test base rate",
        "",
        md_table(train_test_shift_table(nights)),
        "",
        "## 9. Enough test data?",
        "",
        f"Labeled test nights at lead 1 with ≥ 1 model available (all sites): **{verdict}**.",
        "",
    ]
    return "\n".join(parts)


def write_data_quality(df: pd.DataFrame, settings: Settings, path: Path | None = None) -> Path:
    path = path or DOCS / "DATA_QUALITY.md"
    path.write_text(data_quality_markdown(df, settings, dt.datetime.now(dt.UTC)))
    return path
