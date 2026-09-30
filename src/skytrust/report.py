"""Generated documents. Every number here is computed from the dataset/metrics in code;
nothing in these files is typed by hand (SPEC 0, rule 3).

docs/DATA_QUALITY.md (from the dataset) and docs/RESULTS.md + docs/figures (from metrics.json).
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


def display_name(method: str) -> str:
    fixed = {
        "climatology": "Climatology (B1)",
        "climatology_train": "Climatology, training years (B1b)",
        "persistence": "Persistence (B2)",
        "equal_weight": "Equal-weight average (B5)",
        "equal_weight_cal": "Equal-weight, calibrated (B6)",
        "blend": "Blend",
        "blend_nbm": "Blend + NBM input (research)",
        "blend_primary": "Shipped blend (trained on primary)",
    }
    if method in fixed:
        return fixed[method]
    model, kind = method.rsplit("_", 1)
    if model == "nbm":  # NOAA's own blend: a benchmark, not one of the blend's inputs
        return {"rule": "NOAA NBM rule", "raw": "NOAA NBM (raw)", "lr": "NOAA NBM calibrated"}[kind]
    return f"{model.upper()} {'rule (B3)' if kind == 'rule' else 'calibrated (B4)'}"


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
    for m in settings.forecast_models:
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
        role = " [benchmark]" if m in settings.benchmarks else ""
        table[f"{m.short} ({m.id}){role}"] = row
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
        "GOES satellite": "usable_goes",
    }
    labels = {k: c for k, c in labels.items() if c in nights and nights[c].notna().any()}
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
    if "usable_goes" in both and both["usable_goes"].notna().any():
        # The satellite as referee on the disagreement nights: which source does it side with?
        pairs = [("asos_only", "ASOS usable, ERA5 not"), ("era5_only", "ERA5 usable, ASOS not")]
        for cat, name in pairs:
            sub = both[(both["cat"] == cat) & both["usable_goes"].notna()]
            out[f"…{name}: GOES says usable"] = (
                sub["usable_goes"].astype(float).groupby(sub["site"], sort=False).mean().map(pct)
            )
    if {"era5_high_mean_cover", "era5_low_mean_cover"} <= set(both.columns):
        dis = both[both["cat"] == "asos_only"].groupby("site", sort=False)
        out["…those nights: ERA5 mean high cloud"] = dis["era5_high_mean_cover"].mean().map(pct)
        out["…those nights: ERA5 mean low cloud"] = dis["era5_low_mean_cover"].mean().map(pct)
    return out


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    """Agreement beyond chance (1 = perfect, 0 = what two coins with these base rates would
    manage). Raw agreement flatters labels that are both usually "clear"."""
    a, b = a.astype(bool).to_numpy(), b.astype(bool).to_numpy()
    if len(a) == 0:
        return float("nan")
    po = float(np.mean(a == b))
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def goes_agreement_table(nights: pd.DataFrame) -> pd.DataFrame | None:
    """How often the independent satellite label agrees with each ground-truth label, per site
    and overall: raw agreement and Cohen's kappa."""
    if "usable_goes" not in nights or nights["usable_goes"].notna().sum() == 0:
        return None
    groups = [(s, g) for s, g in nights.groupby("site", sort=False)] + [("all sites", nights)]
    rows = {}
    for site, g in groups:
        row = {}
        for name, col in [("primary", "usable_primary"), ("ASOS", "usable_asos"),
                          ("ERA5", "usable_era5")]:  # fmt: skip
            both = g.dropna(subset=[col, "usable_goes"])
            same = both[col].astype(bool) == both["usable_goes"].astype(bool)
            row[f"agree w/ {name}"] = pct(same.mean()) if len(both) else "–"
            row[f"κ w/ {name}"] = _fmt(cohen_kappa(both[col], both["usable_goes"]), "num")
        row["GOES-labeled nights"] = int(g["usable_goes"].notna().sum())
        rows[site] = row
    out = pd.DataFrame(rows).T
    out.index.name = "site"
    return out


def goes_quality_section(nights: pd.DataFrame) -> list[str]:
    table = goes_agreement_table(nights)
    if table is None:
        return []
    return [
        "## 7b. The satellite check (GOES-18 Clear Sky Mask)",
        "",
        "A third, independent truth: the GOES-18 satellite's cloud mask averaged over a ~10 km "
        "box around each airport, for the scan nearest the top of each dark hour, with the same "
        "clear / usable rules. It is a real observation (unlike ERA5) and sees high cloud "
        "(unlike ASOS); at night it relies on infrared channels only, so very thin cirrus can "
        "still slip through. κ (Cohen's kappa) is agreement beyond chance.",
        "",
        md_table(table),
        "",
    ]


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


def climatology_quality_section(nights: pd.DataFrame) -> list[str]:
    """Long-term reference: years used per site, and 20-year vs training-years base rates."""
    import json

    from skytrust import climatology

    if not climatology.TABLE_PATH.exists():
        return []
    table = json.loads(climatology.TABLE_PATH.read_text())
    train = nights[nights["split"] == "train"]
    rows = {}
    for site, years in table["years_used"].items():
        months = table["tables"]["primary"].get(site, {})
        n = sum(v["n"] for v in months.values())
        long_rate = sum(v["rate"] * v["n"] for v in months.values()) / n if n else float("nan")
        train_rate = (
            train.loc[train["site"] == site, "usable_primary"].dropna().astype(float).mean()
        )
        rows[site] = {
            "usable years": f"{len(years)} ({min(years)}–{max(years)})" if years else "0",
            "labeled nights": n,
            "base rate, 20-yr": pct(long_rate),
            "base rate, training years": pct(train_rate),
        }
    out = pd.DataFrame(rows).T
    out.index.name = "site"
    return [
        "## 10. Long-term climatology (the skill-score reference)",
        "",
        f"Nights {table['period'][0]} → {table['period'][1]} labeled with the same code; station-"
        f"years with < {table['min_year_coverage']:.0%} labeled nights dropped.",
        "",
        md_table(out),
        "",
    ]


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
        *goes_quality_section(nights),
        "## 8. Train vs test base rate",
        "",
        md_table(train_test_shift_table(nights)),
        "",
        "## 9. Enough test data?",
        "",
        f"Labeled test nights at lead 1 with ≥ 1 model available (all sites): **{verdict}**.",
        "",
        *climatology_quality_section(nights),
    ]
    return "\n".join(parts)


def write_data_quality(df: pd.DataFrame, settings: Settings, path: Path | None = None) -> Path:
    path = path or DOCS / "DATA_QUALITY.md"
    path.write_text(data_quality_markdown(df, settings, dt.datetime.now(dt.UTC)))
    return path


# =====================================================================================
# RESULTS.md (Phase 3+): generated only from artifacts/metrics.json
# =====================================================================================

RESULTS_LEAD = 1  # the "night before" headline lead
CAVEATS = [
    '**ASOS can\'t see above 12,000 ft.** "CLR" at an automated station means no cloud *below* '
    "12,000 ft; cirrus is invisible to it. The primary label adds ERA5 to catch it (FAT's human "
    "observers are the exception and do report high cloud).",
    "**ERA5 is a reanalysis, not an observation,** on a ~28 km grid, and it is produced by ECMWF. "
    "Where ECMWF looks best under an ERA5-based label, part of that may be shared model physics.",
    "**Point vs grid.** Forecasts and ERA5 are grid-cell values (3–28 km); ASOS is a single point. "
    "Mountain and valley sites (TRK, BIH, AUN) are where these differ most.",
    '**Lead 1 vs the live "tonight" forecast.** Lead-1 values were issued ~24 h before each '
    "hour; the live app shows fresher forecasts, so its tonight probability is slightly "
    "conservative relative to what the backtest measured.",
    "**Interpolated hours.** ECMWF 0.25° (and GFS beyond ~5 days) are 3-hourly upstream; "
    'Open-Meteo interpolates to hourly, which affects "consecutive clear hours" for those models.',
    "**Uncalibrated equal-weight average.** B5 uses the mean forecast clear fraction directly as "
    "a probability. It's a deliberately naive reference, yet it is hard to beat (see Summary).",
]


def reference_phrase(meta: dict) -> str:
    ref = meta.get("climatology_reference") or {}
    if ref.get("source") == "long_term":
        years = ref["years_per_site"].values()
        return (
            f"Skill is measured against a long-term climatology ({ref['period'][0][:4]}–"
            f"{ref['period'][1][:4]}, {min(years)}–{max(years)} usable years per site)."
        )
    return "Skill is measured against the training-years climatology."


def data_caveats(v: MetricsView) -> list[str]:
    """Caveats whose numbers come from metrics.json."""
    meta = v.m["meta"]
    weeks = int(v.leads.loc[v.leads["label"] == "primary", "n_weeks"].max())
    line = (
        f"**One test period.** Testing covers {test_period_name(meta)} only ({weeks} weeks "
        "of labeled nights), so intervals are wide-ish and a different year could rank close "
        "methods differently."
    )
    rates = meta.get("primary_base_rate_by_split") or {}
    both = {s: r for s, r in rates.items() if "train" in r and "test" in r}
    if both:
        lower = sorted(s for s, r in both.items() if r["test"] < r["train"])
        line += (
            f" The test-period base rate of usable nights is lower than in training at "
            f"{len(lower)} of {len(both)} sites" + (f" ({', '.join(lower)})." if lower else ".")
        )
    return [line]


def test_period_name(meta: dict) -> str:
    """'Jan–Aug 2026' from metrics.json's test period (so wording can't drift from the data)."""
    start, end = (pd.Timestamp(d) for d in meta["test_period"])
    if start.year == end.year:
        return f"{start:%b}–{end:%b %Y}"
    return f"{start:%b %Y}–{end:%b %Y}"


def _fmt(value, kind: str) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "–"
    return f"{value:.1%}" if kind == "pct" else f"{value:.3f}"


def with_ci(rec: pd.Series | None, metric: str, kind: str = "num") -> str:
    if rec is None or pd.isna(rec.get(metric)):
        return "–"
    if pd.isna(rec.get(f"{metric}_lo")):
        return f"{_fmt(rec[metric], kind)} (too few weeks for a CI)"
    return (
        f"{_fmt(rec[metric], kind)} [{_fmt(rec.get(f'{metric}_lo'), kind)}, "
        f"{_fmt(rec.get(f'{metric}_hi'), kind)}]"
    )


class MetricsView:
    """Convenience lookups over metrics.json."""

    def __init__(self, metrics: dict):
        self.m = metrics
        self.records = pd.DataFrame(metrics["records"])
        self.diffs = pd.DataFrame(metrics["differences"])
        self.leads = pd.DataFrame(metrics["leads"])
        self.name = display_name

    def rec(self, label, lead, method, subset_type="overall", subset="all") -> pd.Series | None:
        r = self.records
        hit = r[
            (r["label"] == label)
            & (r["lead"] == lead)
            & (r["method"] == method)
            & (r["subset_type"] == subset_type)
            & (r["subset"] == subset)
        ]
        return None if hit.empty else hit.iloc[0]

    def best_single(self, label, lead) -> str | None:
        hit = self.leads[(self.leads["label"] == label) & (self.leads["lead"] == lead)]
        return None if hit.empty else hit.iloc[0]["best_single"]

    def lead_list(self, label="primary") -> list[int]:
        return sorted(self.records.loc[self.records["label"] == label, "lead"].unique())

    def methods(self, label, lead, kind=None) -> list[str]:
        r = self.records
        sub = r[(r["label"] == label) & (r["lead"] == lead) & (r["subset_type"] == "overall")]
        if kind:
            sub = sub[sub["kind"] == kind]
        return list(dict.fromkeys(sub["method"]))

    def diff(self, label, lead, a, b) -> pd.Series | None:
        d = self.diffs
        if d.empty:
            return None
        hit = d[(d["label"] == label) & (d["lead"] == lead) & (d["a"] == a) & (d["b"] == b)]
        return None if hit.empty else hit.iloc[0]


def headline_table(v: MetricsView, label: str, lead: int) -> pd.DataFrame:
    rows = {}
    for method in v.methods(label, lead):
        r = v.rec(label, lead, method)
        if r["family"] == "rule":
            continue  # shown in its own table (confusion metrics only)
        rows[v.name(method)] = {
            "Brier ↓": with_ci(r, "brier"),
            "BSS ↑": with_ci(r, "bss"),
            "log loss ↓": _fmt(r.get("log_loss"), "num"),
            "AUC ↑": _fmt(r.get("auc"), "num"),
            "false-clear ↓": with_ci(r, "false_clear_rate", "pct"),
            "miss rate ↓": _fmt(r.get("miss_rate"), "pct"),
        }
    out = pd.DataFrame(rows).T
    out.index.name = "method"
    return out


def rule_table(v: MetricsView, label: str, lead: int) -> pd.DataFrame:
    rows = {}
    for method in v.methods(label, lead, kind="hard"):
        r = v.rec(label, lead, method)
        rows[v.name(method)] = {
            "false-clear ↓": with_ci(r, "false_clear_rate", "pct"),
            "miss rate ↓": with_ci(r, "miss_rate", "pct"),
            "accuracy ↑": with_ci(r, "accuracy", "pct"),
        }
    out = pd.DataFrame(rows).T
    out.index.name = "method (hard yes/no)"
    return out


def lead_table(v: MetricsView, label: str, metric: str, kind: str = "num") -> pd.DataFrame:
    leads = v.lead_list(label)
    methods = list(dict.fromkeys(m for lead in leads for m in v.methods(label, lead, "prob")))
    rows = {}
    for method in methods:
        rows[v.name(method)] = {
            f"L{lead}": (lambda r: _fmt(r[metric], kind) if r is not None else "–")(
                v.rec(label, lead, method)
            )
            for lead in leads
        }
    out = pd.DataFrame(rows).T
    out.index.name = "method"
    return out


def differences_table(v: MetricsView, label: str) -> pd.DataFrame:
    d = v.diffs[v.diffs["label"] == label] if not v.diffs.empty else v.diffs
    rows = []
    for _, r in d.iterrows():
        rows.append(
            {
                "lead": int(r["lead"]),
                "A − B": f"{v.name(r['a'])} − {v.name(r['b'])}",
                "Brier difference [95% CI]": (
                    f"{r['brier_diff']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]"
                ),
                "A better in": f"{r['share_a_better']:.0%} of resamples",
                "significant": "yes" if r["significant"] else "no",
            }
        )
    return pd.DataFrame(rows)


def breakdown_table(v: MetricsView, label: str, lead: int, subset_type: str) -> pd.DataFrame:
    best = v.best_single(label, lead)
    r = v.records
    subsets = r[(r["label"] == label) & (r["lead"] == lead) & (r["subset_type"] == subset_type)]
    order = SEASON_ORDER if subset_type == "season" else sorted(subsets["subset"].unique())
    rows = {}
    for s in [x for x in order if x in set(subsets["subset"])]:
        ew = v.rec(label, lead, "equal_weight", subset_type, s)
        bs = v.rec(label, lead, best, subset_type, s) if best else None
        cl = v.rec(label, lead, "climatology", subset_type, s)
        bl = v.rec(label, lead, "blend", subset_type, s)
        row = {
            "n": int(ew["n"]),
            "weeks": int(ew["n_weeks"]) if pd.notna(ew.get("n_weeks")) else "–",
            "base rate": _fmt(ew["base_rate"], "pct"),
        }
        if bl is not None:
            row["BSS blend"] = with_ci(bl, "bss")
        row["BSS equal-weight"] = with_ci(ew, "bss")
        row[f"BSS {v.name(best) if best else 'best single'}"] = with_ci(bs, "bss")
        if bl is not None:
            row["false-clear blend"] = _fmt(bl["false_clear_rate"], "pct")
        row["false-clear equal-weight"] = _fmt(ew["false_clear_rate"], "pct")
        row["false-clear climatology"] = _fmt(cl["false_clear_rate"], "pct")
        rows[s] = row
    out = pd.DataFrame(rows).T
    out.index.name = subset_type
    return out


def labels_in(v: MetricsView) -> list[str]:
    present = set(v.records["label"])
    return [lab for lab in ["primary", "asos", "era5", "goes"] if lab in present]


def label_sensitivity_table(v: MetricsView, lead: int) -> pd.DataFrame:
    labels = labels_in(v)
    methods = [m for lab in labels for m in v.methods(lab, lead, "prob") if m != "blend_primary"]
    methods = list(dict.fromkeys(methods))
    rows = {}
    for method in methods:
        rows[v.name(method)] = {lab: with_ci(v.rec(lab, lead, method), "bss") for lab in labels}
    out = pd.DataFrame(rows).T
    base = {lab: _fmt(v.rec(lab, lead, "climatology")["base_rate"], "pct") for lab in labels}
    out.loc["(test base rate)"] = base
    out.index.name = f"BSS at lead {lead}"
    return out


def cross_truth_table(v: MetricsView, lead: int) -> pd.DataFrame | None:
    """The shipped (primary-trained) blend scored against every other truth it never saw."""
    rows = {}
    for lab in labels_in(v):
        r = v.rec(lab, lead, "blend_primary")
        if r is None:
            continue
        nbm, clim = v.rec(lab, lead, "nbm_lr"), v.rec(lab, lead, "climatology")
        rows[LABEL_NAMES[lab]] = {
            "BSS shipped blend": with_ci(r, "bss"),
            "BSS NOAA NBM calibrated": with_ci(nbm, "bss"),
            "false-clear shipped blend": _fmt(r["false_clear_rate"], "pct"),
            "false-clear climatology": _fmt(clim["false_clear_rate"], "pct"),
            "shipped blend vs NBM": _diff_phrase(v.diff(lab, lead, "blend_primary", "nbm_lr")),
        }
    if not rows:
        return None
    out = pd.DataFrame(rows).T
    out.index.name = f"judged by (lead {lead})"
    return out


def cross_truth_lines(v: MetricsView, lead: int) -> list[str]:
    table = cross_truth_table(v, lead)
    if table is None:
        return []
    lines = [
        "**Cross-truth check.** The shipped blend is trained on the primary label only. Scoring "
        "that same model against truths it never trained on tests whether its skill is an "
        "artifact of how the primary label is built.",
    ]
    if "goes" in labels_in(v):
        lines[0] += (
            " GOES is the most independent: a satellite observation, not a model, that also "
            "sees the high cloud ASOS can't. Its climatology is its own 2024–25 training-years "
            "rate (there is no 20-year satellite record)."
        )
    return [*lines, "", md_table(table), ""]


def model_info_table(v: MetricsView, label: str) -> pd.DataFrame:
    rows = []
    for _, lead_row in v.leads[v.leads["label"] == label].iterrows():
        for method, info in (lead_row["model_info"] or {}).items():
            if method == "blend" or "C" not in info:
                continue  # the blend has its own section; only tuned models belong here
            rows.append(
                {
                    "lead": int(lead_row["lead"]),
                    "model": v.name(method),
                    "C": f"{info['C']:.3g}",
                    "CV log loss (train only)": f"{info['cv_log_loss']:.4f}",
                    "train rows": info["n_train"],
                    "chosen as best single": "✓" if method == lead_row["best_single"] else "",
                }
            )
    return pd.DataFrame(rows)


def _diff_phrase(d: pd.Series | None) -> str:
    if d is None:
        return "not compared"
    ci = f"Brier difference {d['brier_diff']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]"
    if d["significant"] and d["brier_diff"] < 0:
        return f"better ({ci}, CI excludes zero)"
    if d["significant"]:
        return f"worse ({ci}, CI excludes zero)"
    return f"not significantly different ({ci})"


def blend_verdicts(v: MetricsView, label: str = "primary") -> dict[str, dict[str, list[int]]]:
    """Per comparison target, which leads the blend is significantly better / not better."""
    out: dict[str, dict[str, list[int]]] = {}
    for target in ["best_single", "equal_weight", "equal_weight_cal", "nbm_lr"]:
        better, not_better = [], []
        for ld in v.lead_list(label):
            if (
                v.rec(label, ld, "blend") is None
                or v.rec(label, ld, target) is None
                and (target != "best_single")
            ):
                continue
            b = v.best_single(label, ld) if target == "best_single" else target
            d = v.diff(label, ld, "blend", b)
            (
                better if d is not None and d["significant"] and d["brier_diff"] < 0 else not_better
            ).append(ld)
        out[target] = {"better": better, "not_better": not_better}
    return out


def significant_wins(v: MetricsView, a: str, b: str, label: str = "primary") -> list[int]:
    """Leads where A's Brier score is significantly lower than B's (paired CI excludes zero)."""
    wins = []
    for ld in v.lead_list(label):
        d = v.diff(label, ld, a, b)
        if d is not None and d["significant"] and d["brier_diff"] < 0:
            wins.append(ld)
    return wins


def attribution_line(v: MetricsView) -> str:
    """Where the blend's skill comes from, measured step by step (each step = paired test)."""
    if v.rec("primary", RESULTS_LEAD, "equal_weight_cal") is None:
        return ""
    n = len(v.lead_list("primary"))
    averaging = [
        ld
        for ld in v.lead_list("primary")
        if ld in significant_wins(v, "equal_weight", v.best_single("primary", ld) or "")
    ]
    calibration = significant_wins(v, "equal_weight_cal", "equal_weight")
    weights = significant_wins(v, "blend", "equal_weight_cal")
    line = (
        f"- **Where does the skill come from?** Measured step by step: averaging the models beats "
        f"the best single model significantly at {len(averaging)} of {n} leads; calibrating that "
        f"average with site/season context (B6 vs B5) at {len(calibration)} of {n}; learning a "
        f"separate weight per model (blend vs B6) at {len(weights)} of {n}."
    )
    if len(averaging) > max(len(calibration), len(weights)):
        line += (
            " Most of the value is simply combining several models, the *forecast combination "
            "puzzle*: a plain average is hard to beat with learned weights."
        )
    return line


def blend_summary_lines(v: MetricsView) -> list[str]:
    """The honesty rule (SPEC 8.8): say plainly where the blend does and doesn't win."""
    lab, lead = "primary", RESULTS_LEAD
    blend = v.rec(lab, lead, "blend")
    if blend is None:
        return []
    best = v.best_single(lab, lead)
    n_leads = len([ld for ld in v.lead_list(lab) if v.rec(lab, ld, "blend") is not None])
    verdict = blend_verdicts(v, lab)
    lines = [
        f"- **The learned blend** has a Brier Skill Score of **{with_ci(blend, 'bss')}** at lead "
        f"{lead} and a false-clear rate of **{with_ci(blend, 'false_clear_rate', 'pct')}**. "
        f"Versus the best single model ({v.name(best)}): "
        f"{_diff_phrase(v.diff(lab, lead, 'blend', best))}. Versus the equal-weight average: "
        f"{_diff_phrase(v.diff(lab, lead, 'blend', 'equal_weight'))}.",
    ]
    nbm, raw = v.rec(lab, lead, "nbm_lr"), v.rec(lab, lead, "nbm_raw")
    if nbm is not None:
        lines.append(
            f"- **Versus NOAA's National Blend of Models (NBM)**, NOAA's own statistical blend, "
            f"calibrated here with the same site/season context (BSS {with_ci(nbm, 'bss')}): at "
            f"lead {lead} SkyTrust's blend is {_diff_phrase(v.diff(lab, lead, 'blend', 'nbm_lr'))}."
            + (
                f" NBM's raw forecast used directly as a probability scores "
                f"{with_ci(raw, 'bss')}; it isn't a calibrated probability, so the calibrated "
                "version is the fair comparison."
                if raw is not None
                else ""
            )
        )
    research = v.rec(lab, lead, "blend_nbm")
    if research is not None:
        lines.append(
            f"- **Would NBM help as an input?** A research variant of the blend that also sees NBM "
            f"(trained only on the ~15 months NBM's archive covers) is "
            f"{_diff_phrase(v.diff(lab, lead, 'blend_nbm', 'blend'))} compared with the shipped "
            "blend at the same lead. The shipped model is unchanged either way, so the live "
            "forward test keeps scoring one fixed model."
        )
    for target, name in [
        ("best_single", "the best single model"),
        ("equal_weight", "the simple equal-weight average"),
        (
            "equal_weight_cal",
            "the calibrated equal-weight average (B6: same model, one shared weight)",
        ),
        ("nbm_lr", "NOAA's calibrated NBM"),
    ]:
        if not (verdict[target]["better"] or verdict[target]["not_better"]):
            continue
        wins, others = verdict[target]["better"], verdict[target]["not_better"]
        sentence = (
            f"- Across leads, the blend beats {name} with a 95% CI excluding zero at "
            f"**{len(wins)} of {n_leads} leads**."
        )
        if others:
            sentence += (
                f" **At lead{'s' if len(others) > 1 else ''} {', '.join(map(str, others))} it does "
                f"not beat {name}.**"
            )
        lines.append(sentence)
    attribution = attribution_line(v)
    if attribution:
        lines.append(attribution)
    return lines


def blend_lead_table(v: MetricsView, label: str = "primary") -> pd.DataFrame:
    rows = {}
    for ld in v.lead_list(label):
        r = v.rec(label, ld, "blend")
        if r is None:
            continue
        best = v.best_single(label, ld)
        info = v.leads[(v.leads["label"] == label) & (v.leads["lead"] == ld)].iloc[0]["model_info"][
            "blend"
        ]
        cal = info["calibration"]
        rows[f"L{ld}"] = {
            "BSS": with_ci(r, "bss"),
            "false-clear": with_ci(r, "false_clear_rate", "pct"),
            "vs best single": _diff_phrase(v.diff(label, ld, "blend", best)).split(" (")[0]
            + f" ({v.name(best).split(' ')[0]})",
            "vs equal-weight": _diff_phrase(v.diff(label, ld, "blend", "equal_weight")).split(" (")[
                0
            ],
            "models": ", ".join(m.upper() for m in info["models"]),
            "C": f"{info['C']:.3g}",
            "OOF calibration error": f"{cal['oof_ece']:.3f}",
            "isotonic applied": "yes" if cal["applied"] else "no",
        }
    out = pd.DataFrame(rows).T
    out.index.name = "lead"
    return out


def weight_share_table(v: MetricsView, label: str = "primary") -> pd.DataFrame:
    rows = {}
    for _, lr in v.leads[v.leads["label"] == label].iterrows():
        info = (lr["model_info"] or {}).get("blend")
        if info:
            rows[f"L{int(lr['lead'])}"] = {
                m.upper(): pct(s) for m, s in info["weight_shares"].items()
            }
    out = pd.DataFrame(rows).T.fillna("–")
    out.index.name = "lead"
    return out


def coefficient_table(v: MetricsView, label: str, lead: int, top: int = 12) -> pd.DataFrame:
    lr = v.leads[(v.leads["label"] == label) & (v.leads["lead"] == lead)].iloc[0]
    coef = pd.Series(lr["model_info"]["blend"]["coef_standardized"])
    coef = coef.reindex(coef.abs().sort_values(ascending=False).index).head(top)
    out = pd.DataFrame({"standardized coefficient": coef.map(lambda c: f"{c:+.3f}")})
    out.index.name = "feature (largest effects first)"
    return out


def spread_sentence(v: MetricsView) -> str:
    """Plain-English reading of the model-spread coefficient, from the lead-1 blend."""
    lr = v.leads[(v.leads["label"] == "primary") & (v.leads["lead"] == RESULTS_LEAD)].iloc[0]
    coef = lr["model_info"]["blend"]["coef_standardized"].get("spread_frac_clear")
    if coef is None:
        return ""
    direction = "lowers" if coef < 0 else "raises"
    return (
        f"The model-spread coefficient at lead {RESULTS_LEAD} is {coef:+.3f}: when the models "
        f"disagree, the blend {direction} its probability of a usable night."
    )


VALUE_ALPHAS = (0.1, 0.2, 0.3, 0.5)


def value_table(v: MetricsView, label: str, lead: int, methods: list[str]) -> pd.DataFrame:
    rows = {}
    for c in v.m.get("value_curves", []):
        if c["label"] == label and c["lead"] == lead and c["method"] in methods:
            lookup = dict(zip([round(a, 2) for a in c["alpha"]], c["value"], strict=True))
            rows[v.name(c["method"])] = {
                f"α = {a}": _fmt(lookup.get(a), "pct") for a in VALUE_ALPHAS
            }
    out = pd.DataFrame(rows).T
    out.index.name = "share of a perfect forecast's value"
    return out.reindex([v.name(m) for m in methods if v.name(m) in out.index])


def decomposition_table(v: MetricsView, label: str, lead: int) -> pd.DataFrame:
    rows = {}
    for method in v.methods(label, lead, kind="prob"):
        r = v.rec(label, lead, method)
        if r is None or pd.isna(r.get("brier_resolution")):
            continue
        rows[v.name(method)] = {
            "Brier": _fmt(r["brier"], "num"),
            "reliability ↓": f"{r['brier_reliability']:.4f}",
            "resolution ↑": f"{r['brier_resolution']:.4f}",
            "uncertainty": f"{r['brier_uncertainty']:.4f}",
        }
    out = pd.DataFrame(rows).T
    out.index.name = f"lead {lead}"
    return out


def value_sentence(v: MetricsView, label: str, lead: int, alpha: float = 0.2) -> str:
    def at(method: str):
        for c in v.m.get("value_curves", []):
            if c["label"] == label and c["lead"] == lead and c["method"] == method:
                return dict(zip([round(a, 2) for a in c["alpha"]], c["value"], strict=True)).get(
                    alpha
                )
        return None

    blend, nbm = at("blend"), at("nbm_lr")
    if blend is None:
        return ""
    return (
        f"For an astrophotographer whose good night is worth {1 / alpha:.0f}× the setup effort "
        f"(α = {alpha}), acting on SkyTrust's night-before probability captures "
        f"{blend:.0%} of the value of a perfect forecast"
        + (f", versus {nbm:.0%} for NOAA's calibrated NBM" if nbm is not None else "")
        + ". Where a curve drops below zero, following that forecast is worse than a fixed habit."
    )


def blend_section(v: MetricsView) -> list[str]:
    if v.rec("primary", RESULTS_LEAD, "blend") is None:
        return []
    shares = {}
    for ld in [v.lead_list()[0], v.lead_list()[-1]]:
        info = v.leads[(v.leads["label"] == "primary") & (v.leads["lead"] == ld)].iloc[0][
            "model_info"
        ]["blend"]
        top = max(info["weight_shares"], key=info["weight_shares"].get)
        shares[ld] = (top.upper(), info["weight_shares"][top])
    first, last = v.lead_list()[0], v.lead_list()[-1]
    return [
        "## 8. The blend: what it learned",
        "",
        "One logistic regression per lead on every available model's features, the model spread, "
        "site, month, and dark hours. Tuned and calibration-checked on training years only, "
        "exported to JSON, and scored here through the numpy loader, i.e. exactly the file the app "
        "uses. Calibration is added only if the out-of-fold calibration error on the training "
        "folds exceeds the configured threshold.",
        "",
        md_table(blend_lead_table(v)),
        "",
        "**How much the blend leans on each model** (share of absolute standardized "
        "coefficients on each model's three features; features are correlated, so read this "
        "as a rough guide):",
        "",
        md_table(weight_share_table(v)),
        "",
        f"At lead {first} the blend leans most on {shares[first][0]} ({shares[first][1]:.0%}); at "
        f"lead {last}, on {shares[last][0]} ({shares[last][1]:.0%}). " + spread_sentence(v),
        "",
        f"**Largest standardized coefficients at lead {RESULTS_LEAD}** "
        "(positive = more likely usable):",
        "",
        md_table(coefficient_table(v, "primary", RESULTS_LEAD)),
        "",
    ]


def robustness_table(sens: dict, lead: int) -> pd.DataFrame:
    def verdict(d: dict | None) -> str:
        if not d:
            return "–"
        return "better" if d["significant"] and d["brier_diff"] < 0 else "not significant"

    rows = {}
    for cell in sens["cells"]:
        e = next((x for x in cell["leads"] if x["lead"] == lead), None)
        if e is None:
            continue
        name = f"clear ≤ {cell['clear_threshold']:.0%}, run ≥ {cell['min_run_hours']} h"
        rows[name + (" ★" if cell["is_default"] else "")] = {
            "base rate": _fmt(e["base_rate"], "pct"),
            "blend BSS": with_ci(pd.Series(e), "bss"),
            "false-clear": _fmt(e.get("false_clear_rate"), "pct"),
            "vs best single": verdict(e.get("vs_best_single")),
            "vs NOAA NBM": verdict(e.get("vs_nbm_lr")),
            "vs equal-weight": verdict(e.get("vs_equal_weight")),
        }
    out = pd.DataFrame(rows).T
    out.index.name = f"definition (lead {lead}; ★ = default)"
    return out


def robustness_counts(sens: dict, lead: int, key: str) -> tuple[int, int]:
    hits = total = 0
    for cell in sens["cells"]:
        e = next((x for x in cell["leads"] if x["lead"] == lead), None)
        d = e and e.get(key)
        if d:
            total += 1
            hits += int(d["significant"] and d["brier_diff"] < 0)
    return hits, total


def robustness_section(sens: dict | None) -> list[str]:
    if not sens:
        return []
    lead = RESULTS_LEAD
    best_hits, n = robustness_counts(sens, lead, "vs_best_single")
    nbm_hits, n2 = robustness_counts(sens, lead, "vs_nbm_lr")
    return [
        "## 10. Robustness to the definitions",
        "",
        "The whole pipeline (labels, features, blend, baselines) re-run for every combination of "
        "the clear-sky threshold and the minimum clear run, each cell scored against its own "
        "training-years climatology.",
        "",
        md_table(robustness_table(sens, lead)),
        "",
        f"At lead {lead}, the blend beats the best single model (CI excluding zero) under "
        f"{best_hits} of {n} definitions, and NOAA's calibrated NBM under {nbm_hits} of {n2}.",
        "",
    ]


def walkforward_section(wf: dict | None, figure_paths: dict[str, str]) -> list[str]:
    if not wf:
        return []
    names = {
        "blend": "Blend",
        "nbm_lr": "NOAA NBM calibrated",
        "equal_weight_cal": "Equal-weight, calibrated (B6)",
        "equal_weight": "Equal-weight average (B5)",
    }
    rows = {}
    for e in wf["pooled"]:
        row = {"nights": e["n"], "months": e["months"]}
        for m, name in names.items():
            if m in e["methods"]:
                row[f"BSS {name}"] = with_ci(pd.Series(e["methods"][m]), "bss")
        for d in e["differences"]:
            if d["b"] in ("nbm_lr", "equal_weight_cal"):
                verdict = (
                    "better" if d["significant"] and d["brier_diff"] < 0 else "not significant"
                )
                row[f"blend vs {names[d['b']]}"] = verdict
        rows[f"L{e['lead']}"] = row
    table = pd.DataFrame(rows).T
    table.index.name = "lead"
    lead1 = [m for m in wf["monthly"] if m["lead"] == RESULTS_LEAD and "bss_blend" in m]
    sentence = ""
    if lead1:
        worst = min(lead1, key=lambda m: m["bss_blend"])
        beats = sum(
            1 for m in lead1 if m.get("bss_nbm_lr") is not None and m["bss_blend"] > m["bss_nbm_lr"]
        )
        with_nbm = sum(1 for m in lead1 if m.get("bss_nbm_lr") is not None)
        sentence = (
            f"At lead {RESULTS_LEAD}, the blend's monthly skill stays positive in "
            f"{sum(m['bss_blend'] > 0 for m in lead1)} of {len(lead1)} months (lowest: "
            f"{worst['bss_blend']:.2f} in {worst['period']}), and it scores above NOAA's "
            f"calibrated NBM in {beats} of {with_nbm} months."
        )
    return [
        "## 11. Walk-forward evaluation (stability over time)",
        "",
        f"Operational simulation from {wf['start'][:7]} to {wf['end'][:7]}: every month, every "
        f"model is refit on all earlier nights only ({wf['protocol']}), then forecasts that month. "
        "All predictions are out-of-sample, and 2025 becomes additional evidence.",
        "",
        f"![Walk-forward monthly skill]({figure_paths['walkforward']})"
        if "walkforward" in figure_paths
        else "",
        "",
        md_table(table),
        "",
        sentence,
        "",
    ]


def spatial_section(sp: dict | None) -> list[str]:
    if not sp:
        return []
    names = {
        "geo_unseen": "Geo blend, site never seen",
        "blend_site": "Shipped blend (site-aware)",
        "equal_weight": "Equal-weight average (B5)",
    }
    rows = {}
    for e in sp["leads"]:
        row = {name: with_ci(pd.Series(e["methods"][m]), "bss") for m, name in names.items()}
        d = e["geo_minus_site"]
        row["geo − site-aware (Brier)"] = f"{d['brier_diff']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]"
        rows[f"L{e['lead']}"] = row
    table = pd.DataFrame(rows).T
    table.index.name = "BSS by lead"
    lead1 = next((e for e in sp["leads"] if e["lead"] == RESULTS_LEAD), None)
    sentence = ""
    if lead1:
        worse = [
            s
            for s, v in lead1["per_site"].items()
            if v["brier_geo_unseen"] > v["brier_equal_weight"]
        ]
        sentence = (
            f"At lead {RESULTS_LEAD}, a blend that has never seen the site scores BSS "
            f"{with_ci(pd.Series(lead1['methods']['geo_unseen']), 'bss')} versus "
            f"{with_ci(pd.Series(lead1['methods']['blend_site']), 'bss')} for the site-aware "
            "blend; "
            + (
                f"it trails the untrained equal-weight average at {', '.join(worse)}."
                if worse
                else "it beats the untrained equal-weight average at every held-out site."
            )
            + " This is the evidence behind the app's custom-location forecasts."
        )
    return [
        "## 12. Unseen locations (leave-one-site-out)",
        "",
        "Each site in turn is held out: a site-agnostic blend (no site identity in its inputs) is "
        "trained on the other sites' training years only, then scored on the held-out site's test "
        "period, on the same nights as the shipped site-aware blend.",
        "",
        md_table(table),
        "",
        sentence,
        "",
    ]


def hourly_section(h: dict | None) -> list[str]:
    if not h:
        return []
    names = {
        "hourly_model": "Hourly model",
        "share_of_models": "Share of models saying clear",
        "climatology": "Hourly climatology",
    }
    rows = {}
    for e in h["leads"]:
        row = {"dark hours": e["n_hours"], "clear rate": _fmt(e["base_rate"], "pct")}
        for m, name in names.items():
            row[f"BSS {name}"] = with_ci(pd.Series(e["methods"][m]), "bss")
        rows[f"L{e['lead']}"] = row
    table = pd.DataFrame(rows).T
    table.index.name = "lead"
    return [
        "## 13. Hourly probabilities",
        "",
        "Planning *when* to image needs hour-level forecasts. A second logistic model per lead "
        "predicts P(this dark hour is clear) from every model's forecast cover for that hour, "
        "their mean and spread, the hour's position in the night, month and site; the app shows "
        "these as "
        "hourly bars. Test period, scored per dark hour (BSS vs the hourly training-years "
        "climatology; CIs resample weeks).",
        "",
        md_table(table),
        "",
    ]


def decision_section(v: MetricsView, figure_paths: dict[str, str]) -> list[str]:
    lead = RESULTS_LEAD
    best = v.best_single("primary", lead)
    methods = [m for m in ["blend", "nbm_lr", "equal_weight_cal", "equal_weight", best] if m]
    if not v.m.get("value_curves"):
        return []
    return [
        "## 9. Decision value and forecast quality",
        "",
        "Skill scores don't say whether a forecast is worth acting on. The cost-loss model does: "
        "setting up costs effort C; skipping a usable night loses L. With α = C/L, a calibrated "
        "user goes out when P ≥ α. *Relative value* is the share of a perfect forecast's benefit "
        "(over the best fixed habit: always go, or never go) that acting on the forecast delivers.",
        "",
        f"![Decision value]({figure_paths['value']})" if "value" in figure_paths else "",
        "",
        md_table(value_table(v, "primary", lead, methods)),
        "",
        value_sentence(v, "primary", lead),
        "",
        "**Brier score decomposition** (Murphy 1973): reliability measures how honest the "
        "probabilities are (lower is better), resolution how well they separate good nights from "
        "bad (higher is better); uncertainty is the same for every method.",
        "",
        md_table(decomposition_table(v, "primary", lead)),
        "",
    ]


def summary_lines(v: MetricsView) -> list[str]:
    lab, lead = "primary", RESULTS_LEAD
    leads = v.lead_list(lab)
    best = v.best_single(lab, lead)
    ew, clim = v.rec(lab, lead, "equal_weight"), v.rec(lab, lead, "climatology")
    bs = v.rec(lab, lead, best)
    n_models = sum(1 for m in v.m["meta"]["models"] if lead in m["leads"])
    lines = blend_summary_lines(v)
    lines += [
        f"- At lead {lead} (the night-before forecast), the **equal-weight average of {n_models} "
        f"models** has a Brier Skill Score of **{with_ci(ew, 'bss')}** relative to climatology. "
        f"The best single calibrated model ({v.name(best)}, chosen by *training* cross-validation, "
        f"never by test results) scores {with_ci(bs, 'bss')}.",
    ]
    wins, ties = [], []
    for ld in leads:
        d = v.diff(lab, ld, "equal_weight", v.best_single(lab, ld))
        if d is not None:
            (wins if d["significant"] and d["brier_diff"] < 0 else ties).append(ld)
    lines.append(
        f"- The equal-weight average beats the best single model with a 95% CI excluding zero at "
        f"**{len(wins)} of {len(leads)} leads**"
        + (f" (not significant at lead {', '.join(map(str, ties))})." if ties else ".")
    )
    lines.append(
        f'- False-clear rate at lead {lead} (of nights called "go" at P ≥ '
        f"{v.m['meta']['decision_threshold']}, the share that were not usable): "
        f"**{with_ci(ew, 'false_clear_rate', 'pct')}** for the equal-weight average vs "
        f"{with_ci(clim, 'false_clear_rate', 'pct')} for climatology."
    )
    first, last = v.rec(lab, leads[0], "equal_weight"), v.rec(lab, leads[-1], "equal_weight")
    lines.append(
        f"- Skill decays with lead time: equal-weight BSS falls from {_fmt(first['bss'], 'num')} "
        f"at lead {leads[0]} to {_fmt(last['bss'], 'num')} at lead {leads[-1]}, and its "
        f"false-clear rate rises from {_fmt(first['false_clear_rate'], 'pct')} to "
        f"{_fmt(last['false_clear_rate'], 'pct')}."
    )
    asos = v.rec("asos", lead, "equal_weight")
    if asos is not None and asos["bss"] < 0:
        lines.append(
            f"- Under the **ASOS-only label** the picture changes: equal-weight BSS at lead {lead} "
            f"is {with_ci(asos, 'bss')}. The models forecast *total* cloud (including cirrus), "
            "while ASOS only reports cloud below 12,000 ft, so they're scored against a truth that "
            "ignores part of what they predict."
            + (
                f" A blend trained on that label learns the relationship directly and scores "
                f"{with_ci(v.rec('asos', lead, 'blend'), 'bss')}."
                if v.rec("asos", lead, "blend") is not None
                else ""
            )
            + " See the label sensitivity section."
        )
    if v.rec(lab, lead, "blend") is None:
        lines.append("- The learned blend is trained and evaluated in Phase 4 (not yet run).")
    return lines


def ecmwf_flags(v: MetricsView) -> list[str]:
    flags = []
    for lab in ["era5", "primary"]:
        hit = v.leads[(v.leads["label"] == lab) & (v.leads["best_single"] == "ecmwf_lr")]
        if len(hit):
            leads = ", ".join(str(int(x)) for x in hit["lead"])
            flags.append(
                f"- ⚠ ECMWF is the best single model (by training CV) under the **{lab}** label at "
                f"lead(s) {leads}. ERA5 is produced by ECMWF, so part of this may be shared model "
                "physics rather than real-world skill."
            )
    return flags


def hourly_results() -> dict | None:
    from skytrust import hourly

    return hourly.load()


def spatial_results() -> dict | None:
    from skytrust import spatial

    return spatial.load()


def walkforward_results() -> dict | None:
    from skytrust import walkforward

    return walkforward.load()


def sensitivity_results() -> dict | None:
    from skytrust import sensitivity

    return sensitivity.load()


def results_markdown(metrics: dict, figure_paths: dict[str, str]) -> str:
    v = MetricsView(metrics)
    meta = metrics["meta"]
    lead = RESULTS_LEAD
    n_eval = v.leads[(v.leads["label"] == "primary") & (v.leads["lead"] == lead)].iloc[0]
    parts = [
        "# Results",
        "",
        f"_Generated by `python -m skytrust report` from `artifacts/metrics.json` "
        f"(evaluation run {meta['created_utc']}, commit `{meta['git_commit']}`). "
        "Do not edit by hand._",
        "",
        f"**Setup.** Train on nights {meta['train_period'][0]} → {meta['train_period'][1]}; "
        f"test on {meta['test_period'][0]} → {meta['test_period'][1]} "
        "(never used for fitting or tuning). " + reference_phrase(meta) + " "
        f"Each method is scored on the same test nights per lead (at lead {lead}: "
        f"{int(n_eval['n_eval'])} site-nights over {int(n_eval['n_weeks'])} weeks). 95% CIs from a "
        f"block bootstrap that resamples whole {meta['bootstrap_block']}s "
        f"({meta['bootstrap_resamples']} resamples, seed {meta['seed']}).",
        "",
        "## Summary",
        "",
        *summary_lines(v),
        *ecmwf_flags(v),
        "",
        f"## 1. Headline: primary label, lead {lead}",
        "",
        md_table(headline_table(v, "primary", lead)),
        "",
        "BSS = 1 − Brier / Brier(climatology). Persistence is a hard yes/no, so it has no log loss "
        "or AUC.",
        "",
        f"### Hard yes/no forecasts: persistence (B2) and each model's own rule (B3), lead {lead}",
        "",
        md_table(rule_table(v, "primary", lead)),
        "",
        "## 2. Skill vs lead time",
        "",
        f"![Lead-time curves]({figure_paths['lead_curves']})",
        "",
        "**Brier Skill Score by lead (primary label)**",
        "",
        md_table(lead_table(v, "primary", "bss")),
        "",
        "**False-clear rate by lead (primary label)**",
        "",
        md_table(lead_table(v, "primary", "false_clear_rate", "pct")),
        "",
        "## 3. Calibration",
        "",
        "When a method says 70%, does it happen ~70% of the time? Points on the diagonal are "
        "well calibrated; bars show how many nights fall in each bin.",
        "",
        f"![Reliability diagram]({figure_paths['reliability']})",
        "",
        "## 4. Paired comparisons (Brier difference, negative = A better)",
        "",
        md_table(differences_table(v, "primary"), index=False),
        "",
        f"## 5. By site and season (primary label, lead {lead})",
        "",
        f"![Skill by site]({figure_paths['site_skill']})",
        "",
        md_table(breakdown_table(v, "primary", lead, "site")),
        "",
        md_table(breakdown_table(v, "primary", lead, "season")),
        "",
        f"Test seasons are partial (the test period is {test_period_name(v.m['meta'])}). "
        "Subsets spanning fewer than `min_weeks_for_ci` weeks (config) get no CI, because a "
        "bootstrap over so few weekly blocks is unreliable.",
        "",
        "## 6. Sensitivity to the truth label",
        "",
        md_table(label_sensitivity_table(v, lead)),
        "",
        *cross_truth_lines(v, lead),
        "## 7. Single-model tuning (training data only)",
        "",
        md_table(model_info_table(v, "primary"), index=False),
        "",
        *blend_section(v),
        *decision_section(v, figure_paths),
        *robustness_section(sensitivity_results()),
        *walkforward_section(walkforward_results(), figure_paths),
        *spatial_section(spatial_results()),
        *hourly_section(hourly_results()),
        "## Caveats",
        "",
        *[f"- {c}" for c in data_caveats(v) + CAVEATS],
        "",
    ]
    return "\n".join(parts)


def write_results(metrics: dict, docs: Path = DOCS) -> Path:
    from skytrust import figures

    fig_dir = docs / "figures"
    records = pd.DataFrame(metrics["records"])
    best = MetricsView(metrics).best_single("primary", RESULTS_LEAD)
    present = set(records["method"])
    shown = ["climatology", best, "equal_weight"] + [m for m in ["nbm_lr", "blend"] if m in present]
    paths = {
        "lead_curves": figures.lead_curves(records, "primary", fig_dir / "lead_curves_primary.png"),
        "reliability": figures.reliability(
            metrics["reliability"],
            [m for m in shown if m],
            "primary",
            RESULTS_LEAD,
            fig_dir / "reliability_primary_lead1.png",
        ),
        "site_skill": figures.site_skill(
            records,
            [m for m in shown if m and m != "climatology"],
            "primary",
            RESULTS_LEAD,
            fig_dir / "site_skill_primary_lead1.png",
        ),
    }
    if metrics.get("value_curves"):
        value_methods = ["blend", "nbm_lr", "equal_weight_cal", "equal_weight", best, "climatology"]
        paths["value"] = figures.value_curves(
            metrics["value_curves"],
            [m for m in value_methods if m],
            "primary",
            RESULTS_LEAD,
            fig_dir / "value_primary_lead1.png",
        )
    wf = walkforward_results()
    if wf:
        paths["walkforward"] = figures.monthly_skill(
            wf["monthly"],
            RESULTS_LEAD,
            ["blend", "nbm_lr", "equal_weight_cal", "equal_weight"],
            fig_dir / "walkforward_primary_lead1.png",
        )
    rel = {k: str(p.relative_to(docs)) for k, p in paths.items()}
    path = docs / "RESULTS.md"
    path.write_text(results_markdown(metrics, rel))
    return path


# ---------- README headline block ----------

README_START, README_END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def readme_block(metrics: dict) -> str:
    """Short headline table for the README (primary label, lead 1), from metrics.json."""
    v = MetricsView(metrics)
    lead = RESULTS_LEAD
    best = v.best_single("primary", lead)
    methods = [m for m in ["blend", "nbm_lr", "equal_weight", best, "climatology"] if m]
    rows = {}
    for method in methods:
        r = v.rec("primary", lead, method)
        if r is None:
            continue
        rows[v.name(method)] = {
            "Brier Skill Score ↑": with_ci(r, "bss"),
            "False-clear rate ↓": with_ci(r, "false_clear_rate", "pct"),
            "AUC ↑": _fmt(r.get("auc"), "num"),
        }
    table = pd.DataFrame(rows).T
    table.index.name = (
        f"Night-before forecast (lead {lead}), test {test_period_name(metrics['meta'])}"
    )
    note = (
        f"_Auto-generated from `artifacts/metrics.json` by `python -m skytrust report` "
        f"(commit `{metrics['meta']['git_commit']}`). 95% CIs from a week-block bootstrap. "
        "Full results: [docs/RESULTS.md](docs/RESULTS.md)._"
    )
    return "\n".join([README_START, "", md_table(table), "", note, "", README_END])


def update_readme(metrics: dict, path: Path = REPO_ROOT / "README.md") -> bool:
    """Replace the text between the RESULTS markers. Returns False if the markers are absent."""
    text = path.read_text()
    if README_START not in text or README_END not in text:
        return False
    before, rest = text.split(README_START, 1)
    _, after = rest.split(README_END, 1)
    path.write_text(before + readme_block(metrics) + after)
    return True


# ---------- plain-English takeaways (app Track Record page) ----------

LABEL_NAMES = {
    "primary": "primary (ASOS + ERA5)",
    "asos": "ASOS-only",
    "era5": "ERA5-only",
    "goes": "GOES satellite",
}


def takeaways(v: MetricsView, label: str, lead: int) -> list[str]:
    """Template sentences filled from metrics.json for any label and lead (never hand-typed)."""
    main = "blend" if v.rec(label, lead, "blend") is not None else "equal_weight"
    r, clim = v.rec(label, lead, main), v.rec(label, lead, "climatology")
    if r is None or clim is None:
        return ["No metrics for this label and lead."]
    best = v.best_single(label, lead)
    name = v.name(main)
    lines = [
        f"Judged against the {LABEL_NAMES[label]} label, {lead} day(s) ahead, the {name.lower()} "
        f'says "go" on nights that turn out cloudy {with_ci(r, "false_clear_rate", "pct")} of the '
        f"time, versus {with_ci(clim, 'false_clear_rate', 'pct')} for the seasonal base rate.",
        f"Its Brier Skill Score is {with_ci(r, 'bss')} "
        "(0 = no better than the base rate, 1 = perfect).",
    ]
    if main == "blend":
        vs_best = _diff_phrase(v.diff(label, lead, "blend", best))
        vs_equal = _diff_phrase(v.diff(label, lead, "blend", "equal_weight"))
        lines.append(f"Versus the best single model ({v.name(best)}): {vs_best}.")
        lines.append(f"Versus the simple equal-weight average: {vs_equal}.")
    leads = [ld for ld in v.lead_list(label) if v.rec(label, ld, main) is not None]
    first, last = v.rec(label, leads[0], main), v.rec(label, leads[-1], main)
    lines.append(
        f"Skill fades with lead time: from {_fmt(first['bss'], 'num')} at {leads[0]} day(s) ahead "
        f"to {_fmt(last['bss'], 'num')} at {leads[-1]} days."
    )
    return lines


# ---------- resume bullets (SPEC 13, Phase 6): every number from metrics.json ----------


def resume_bullets(metrics: dict) -> list[str]:
    """2-3 resume bullets whose numbers all come from metrics.json, so they can't drift."""
    v = MetricsView(metrics)
    lead = RESULTS_LEAD
    blend, clim = v.rec("primary", lead, "blend"), v.rec("primary", lead, "climatology")
    if blend is None or clim is None:
        return []
    n_models = len(metrics["meta"]["models"])
    sites = sorted(v.records.loc[v.records["subset_type"] == "site", "subset"].unique())
    lead_info = v.leads[(v.leads["label"] == "primary") & (v.leads["lead"] == lead)].iloc[0]
    verdict = blend_verdicts(v, "primary")
    n_leads = len(v.lead_list("primary"))
    wins = len(verdict["best_single"]["better"])
    period = test_period_name(metrics["meta"])
    nbm_wins = verdict["nbm_lr"]["better"]
    nbm_clause = (
        f" and NOAA's National Blend of Models at {len(nbm_wins)} of {n_leads}" if nbm_wins else ""
    )
    value_clause = ""
    got = {
        c["method"]: dict(zip([round(a, 2) for a in c["alpha"]], c["value"], strict=True)).get(0.2)
        for c in metrics.get("value_curves", [])
        if c["label"] == "primary" and c["lead"] == lead
    }
    if got.get("blend") is not None and got.get("nbm_lr") is not None:
        value_clause = (
            f" In a cost-loss decision analysis it captured {got['blend']:.0%} of a perfect "
            f"forecast's value vs {got['nbm_lr']:.0%} for NOAA's blend."
        )
    return [
        f"Built SkyTrust, an astronomy cloud forecast that backtests {n_models} weather models "
        f"against airport ceilometer observations and ERA5 reanalysis at {len(sites)} California "
        f"sites; on a held-out {period} test period ({int(lead_info['n_eval']):,} site-nights), a "
        f"per-lead logistic-regression blend cut the night-before false-clear rate to "
        f"{blend['false_clear_rate']:.1%} vs {clim['false_clear_rate']:.1%} for climatology.",
        "Designed a leakage-safe evaluation (time-based split, date-grouped cross-validation, "
        "archived fixed-lead forecasts to avoid look-ahead bias) with week-block bootstrap CIs; "
        f"the blend reached a Brier Skill Score of {blend['bss']:.2f} "
        f"(95% CI {blend['bss_lo']:.2f}–{blend['bss_hi']:.2f}) and beat the best single model at "
        f"{wins} of {n_leads} lead times" + nbm_clause + "." + value_clause,
        "Shipped a Streamlit app with live 7-night outlooks, per-site track records, and graceful "
        "API-failure fallback; models are exported to JSON and served with numpy, with CI running "
        "an offline test suite on every push.",
    ]


def write_resume_bullets(metrics: dict, path: Path | None = None) -> Path:
    path = path or DOCS / "RESUME_BULLETS.md"
    lines = [
        "# Resume bullets",
        "",
        "_Generated by `python -m skytrust report` from `artifacts/metrics.json` "
        f"(commit `{metrics['meta']['git_commit']}`). Every number below comes from that file; "
        "edit the wording freely, but re-run the report instead of hand-editing numbers._",
        "",
        "Links: live app https://skytrust.streamlit.app/ · code https://github.com/hitenjain22/skytrust",
        "",
        *[f"- {b}" for b in resume_bullets(metrics)],
        "",
    ]
    path.write_text("\n".join(lines))
    return path
