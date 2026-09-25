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


def data_caveats(v: MetricsView) -> list[str]:
    """Caveats whose numbers come from metrics.json."""
    meta = v.m["meta"]
    weeks = int(v.leads.loc[v.leads["label"] == "primary", "n_weeks"].max())
    line = (
        f"**One test year.** The test period is {meta['test_period'][0][:4]} only ({weeks} weeks "
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
        from skytrust.figures import display_name

        self.m = metrics
        self.records = pd.DataFrame(metrics["records"])
        self.diffs = pd.DataFrame(metrics["differences"])
        self.leads = pd.DataFrame(metrics["leads"])
        self.name = display_name

    def rec(self, label, lead, method, subset_type="overall", subset="all") -> pd.Series | None:
        r = self.records
        hit = r[(r["label"] == label) & (r["lead"] == lead) & (r["method"] == method)
                & (r["subset_type"] == subset_type) & (r["subset"] == subset)]  # fmt: skip
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


def label_sensitivity_table(v: MetricsView, lead: int) -> pd.DataFrame:
    labels = ["primary", "asos", "era5"]
    methods = list(dict.fromkeys(m for lab in labels for m in v.methods(lab, lead, "prob")))
    rows = {}
    for method in methods:
        rows[v.name(method)] = {lab: with_ci(v.rec(lab, lead, method), "bss") for lab in labels}
    out = pd.DataFrame(rows).T
    base = {lab: _fmt(v.rec(lab, lead, "climatology")["base_rate"], "pct") for lab in labels}
    out.loc["(test base rate)"] = base
    out.index.name = f"BSS at lead {lead}"
    return out


def model_info_table(v: MetricsView, label: str) -> pd.DataFrame:
    rows = []
    for _, lead_row in v.leads[v.leads["label"] == label].iterrows():
        for method, info in (lead_row["model_info"] or {}).items():
            if method == "blend":
                continue  # the blend has its own section
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
    for target in ["best_single", "equal_weight"]:
        better, not_better = [], []
        for ld in v.lead_list(label):
            if v.rec(label, ld, "blend") is None:
                continue
            b = v.best_single(label, ld) if target == "best_single" else "equal_weight"
            d = v.diff(label, ld, "blend", b)
            (
                better if d is not None and d["significant"] and d["brier_diff"] < 0 else not_better
            ).append(ld)
        out[target] = {"better": better, "not_better": not_better}
    return out


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
    for target, name in [
        ("best_single", "the best single model"),
        ("equal_weight", "the simple equal-weight average"),
    ]:
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
        "(never used for fitting or tuning). "
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
        "Test seasons are partial (the test year starts in January and ends at the latest labeled "
        "night). Subsets spanning fewer than `min_weeks_for_ci` weeks (config) get no CI, because "
        "a bootstrap over so few weekly blocks is unreliable.",
        "",
        "## 6. Sensitivity to the truth label",
        "",
        md_table(label_sensitivity_table(v, lead)),
        "",
        "## 7. Single-model tuning (training data only)",
        "",
        md_table(model_info_table(v, "primary"), index=False),
        "",
        *blend_section(v),
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
    shown = ["climatology", best, "equal_weight"] + (
        ["blend"] if "blend" in set(records["method"]) else []
    )
    paths = {
        "lead_curves": figures.lead_curves(records, "primary", fig_dir / "lead_curves_primary.png"),
        "reliability": figures.reliability(
            metrics["reliability"],
            [m for m in shown if m],
            "primary",
            RESULTS_LEAD,
            fig_dir / "reliability_primary_lead1.png",
        ),  # fmt: skip
        "site_skill": figures.site_skill(
            records,
            [m for m in shown if m and m != "climatology"],
            "primary",
            RESULTS_LEAD,
            fig_dir / "site_skill_primary_lead1.png",
        ),  # fmt: skip
    }
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
    methods = [m for m in ["blend", "equal_weight", best, "climatology"] if m]
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
    table.index.name = f"Night-before forecast (lead {lead}), test year"
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

LABEL_NAMES = {"primary": "primary (ASOS + ERA5)", "asos": "ASOS-only", "era5": "ERA5-only"}


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
