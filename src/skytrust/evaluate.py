"""Test-set metrics with block-bootstrap confidence intervals (SPEC 8.5-8.6).

Why a *block* bootstrap: consecutive nights share weather systems, so they aren't independent.
Resampling single nights would pretend we have more independent evidence than we do and give
too-narrow intervals. We resample whole calendar weeks (all sites' nights in a week move
together, which also respects the correlation between sites on the same night).

Implementation: each bootstrap replicate is a vector of row *weights* (how many times each
row's week was drawn). Every metric is then a weighted average, computed for all 1,000
replicates at once with matrix products. Using the same weights for every method makes the
comparisons *paired*.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from skytrust import baselines
from skytrust.baselines import LeadResult, MethodPrediction
from skytrust.config import REPO_ROOT, Settings
from skytrust.inference import METRICS_PATH, load_metrics  # noqa: F401  (re-exported)
from skytrust.modeling import PROB_CLIP

log = logging.getLogger(__name__)

PROB_METRICS = ["brier", "bss", "log_loss", "auc", "false_clear_rate", "miss_rate", "accuracy"]
HARD_METRICS = ["brier", "bss", "false_clear_rate", "miss_rate", "accuracy"]
RULE_METRICS = ["false_clear_rate", "miss_rate", "accuracy"]  # SPEC: B3 confusion metrics only
N_BINS = 10


# ---------- bootstrap weights ----------


def week_codes(night_dates: pd.Series) -> np.ndarray:
    """Integer code per row for its ISO calendar week (the resampling block)."""
    iso = pd.to_datetime(night_dates).dt.isocalendar()
    keys = iso["year"].astype(int) * 100 + iso["week"].astype(int)
    return pd.factorize(keys, sort=True)[0]


def bootstrap_weights(codes: np.ndarray, n_resamples: int, rng: np.random.Generator) -> np.ndarray:
    """(n_resamples x n_rows) matrix: row weight = times its week was drawn in that replicate."""
    n_weeks = int(codes.max()) + 1
    draws = rng.integers(0, n_weeks, size=(n_resamples, n_weeks))
    week_counts = np.stack([np.bincount(d, minlength=n_weeks) for d in draws])
    return week_counts[:, codes].astype(float)


# ---------- metrics, vectorized over replicates (rows of W) ----------


def _wmean(W: np.ndarray, values: np.ndarray) -> np.ndarray:
    total = W.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (W @ values) / total


def brier_matrix(y: np.ndarray, p: np.ndarray, W: np.ndarray) -> np.ndarray:
    return _wmean(W, (p - y) ** 2)


def log_loss_matrix(y: np.ndarray, p: np.ndarray, W: np.ndarray) -> np.ndarray:
    q = np.clip(p, PROB_CLIP, 1 - PROB_CLIP)
    return _wmean(W, -(y * np.log(q) + (1 - y) * np.log(1 - q)))


def auc_matrix(y: np.ndarray, score: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Weighted ROC AUC = P(score of a usable night > score of an unusable night), ties = 1/2.

    Sort by score once; for each distinct score value sum the positive/negative weights; each
    positive then "beats" all negative weight at lower scores plus half the tied negatives.
    """
    order = np.argsort(score, kind="mergesort")
    s, yy, Wo = score[order], y[order], W[:, order]
    starts = np.r_[0, np.flatnonzero(np.diff(s)) + 1]
    pos = np.add.reduceat(Wo * yy, starts, axis=1)
    neg = np.add.reduceat(Wo * (1 - yy), starts, axis=1)
    neg_below = np.cumsum(neg, axis=1) - neg
    with np.errstate(invalid="ignore", divide="ignore"):
        return (pos * (neg_below + 0.5 * neg)).sum(axis=1) / (pos.sum(axis=1) * neg.sum(axis=1))


def confusion_matrices(
    y: np.ndarray, p: np.ndarray, W: np.ndarray, threshold: float
) -> dict[str, np.ndarray]:
    """False-clear rate = FP / (TP + FP): of the nights we said 'go', the share that weren't
    usable (the #1 user complaint). Miss rate = FN / (TP + FN). Plus accuracy."""
    go = p >= threshold
    tp, fp = W @ (go & (y == 1)), W @ (go & (y == 0))
    fn = W @ (~go & (y == 1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "false_clear_rate": fp / (tp + fp),
            "miss_rate": fn / (tp + fn),
            "accuracy": _wmean(W, (go == (y == 1)).astype(float)),
        }


def all_metrics(
    y: np.ndarray, p: np.ndarray, p_clim: np.ndarray, W: np.ndarray, kind: str, family: str,
    threshold: float,
) -> dict[str, np.ndarray]:  # fmt: skip
    out: dict[str, np.ndarray] = {}
    wanted = RULE_METRICS if family == "rule" else PROB_METRICS if kind == "prob" else HARD_METRICS
    if "brier" in wanted:
        out["brier"] = brier_matrix(y, p, W)
        with np.errstate(invalid="ignore", divide="ignore"):
            out["bss"] = 1 - out["brier"] / brier_matrix(y, p_clim, W)
    if "log_loss" in wanted:
        out["log_loss"] = log_loss_matrix(y, p, W)
        out["auc"] = auc_matrix(y, p, W)
    out.update(confusion_matrices(y, p, W, threshold))
    return {k: v for k, v in out.items() if k in wanted}


def brier_decomposition(y: np.ndarray, p: np.ndarray, n_bins: int = N_BINS) -> dict[str, float]:
    """Murphy (1973) decomposition over 10 equal-width bins: Brier ≈ reliability − resolution +
    uncertainty. Reliability (lower is better) = how far each bin's forecasts sit from what
    happened; resolution (higher is better) = how much the forecasts separate good nights from
    bad; uncertainty = base-rate difficulty (same for every method). The residual is the part
    lost by binning."""
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    base = y.mean()
    rel = res = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            n_k, f_k, o_k = mask.sum(), p[mask].mean(), y[mask].mean()
            rel += n_k * (f_k - o_k) ** 2
            res += n_k * (o_k - base) ** 2
    rel, res, unc = rel / len(y), res / len(y), base * (1 - base)
    brier = float(np.mean((p - y) ** 2))
    return {"reliability": rel, "resolution": res, "uncertainty": unc,
            "binning_residual": brier - (rel - res + unc)}  # fmt: skip


ALPHAS = tuple(round(a, 2) for a in np.arange(0.05, 0.96, 0.05))
THRESHOLDS = tuple(round(t, 2) for t in np.arange(0.05, 0.96, 0.05))


def relative_value(y: np.ndarray, p: np.ndarray, alphas=ALPHAS) -> list[float | None]:
    """Relative economic value (Richardson 2000) of acting on the forecast, per cost/loss ratio.

    The decision: set up (costs the effort C) or stay home (lose L if the night was usable).
    With alpha = C/L, a calibrated forecast says "go" when P >= alpha. Value is the share of the
    gap between the best fixed policy (always go, or never go: min(alpha, base rate)) and a
    perfect forecast (alpha * base rate) that the forecast closes. 1 = perfect, 0 = no better
    than climatology, negative = worse.
    """
    s = float(y.mean())
    out: list[float | None] = []
    for a in alphas:
        go = p >= a
        expense = (a * go.sum() + (~go & (y == 1)).sum()) / len(y)
        e_clim, e_perfect = min(a, s), a * s
        out.append(float((e_clim - expense) / (e_clim - e_perfect)) if e_clim > e_perfect else None)
    return out


def threshold_curve(y: np.ndarray, p: np.ndarray, thresholds=THRESHOLDS) -> list[dict]:
    """What a user would have experienced at each 'go' threshold: how often it said go, how
    many go calls were cloudy (false-clear rate), and how many usable nights it skipped."""
    rows = []
    for t in thresholds:
        go = p >= t
        tp, fp, fn = (
            int((go & (y == 1)).sum()),
            int((go & (y == 0)).sum()),
            int((~go & (y == 1)).sum()),
        )
        rows.append({
            "threshold": t,
            "go_rate": float(go.mean()),
            "false_clear_rate": fp / (tp + fp) if tp + fp else None,
            "miss_rate": fn / (tp + fn) if tp + fn else None,
        })  # fmt: skip
    return rows


def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int = N_BINS) -> list[dict]:
    """10 equal-width probability bins: count, mean forecast, observed frequency."""
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        rows.append(
            {
                "bin": b,
                "lo": float(edges[b]),
                "hi": float(edges[b + 1]),
                "n": int(mask.sum()),
                "mean_p": float(p[mask].mean()) if mask.any() else None,
                "observed": float(y[mask].mean()) if mask.any() else None,
            }
        )
    return rows


# ---------- evaluating one (label, lead) ----------


def _ci(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(finite, [2.5, 97.5])
    return float(lo), float(hi)


def subsets(rows: pd.DataFrame) -> list[tuple[str, str, np.ndarray]]:
    out = [("overall", "all", np.ones(len(rows), dtype=bool))]
    out += [("site", s, (rows["site"] == s).to_numpy()) for s in sorted(rows["site"].unique())]
    out += [("season", s, (rows["season"] == s).to_numpy()) for s in ["DJF", "MAM", "JJA", "SON"]
            if (rows["season"] == s).any()]  # fmt: skip
    return out


def evaluate_lead(result: LeadResult, settings: Settings, seed: int) -> dict:
    """Metrics + CIs for every method and subset, paired differences, reliability tables."""
    rows = result.rows
    y = rows["y"].to_numpy().astype(float)
    rng = np.random.default_rng(seed)
    codes = week_codes(rows["night_date"])
    W_boot = bootstrap_weights(codes, settings.raw["bootstrap_resamples"], rng)
    threshold = settings.raw["decision_threshold"]
    p_clim = result.methods["climatology"].p
    records = []
    for method in result.methods.values():
        for subset_type, subset, mask in subsets(rows):
            yy, pp, cc = y[mask], method.p[mask], p_clim[mask]
            point = all_metrics(
                yy, pp, cc, np.ones((1, mask.sum())), method.kind, method.family, threshold
            )
            boot = all_metrics(yy, pp, cc, W_boot[:, mask], method.kind, method.family, threshold)
            rec = {
                "label": result.label, "lead": result.lead, "method": method.method,
                "family": method.family, "kind": method.kind, "subset_type": subset_type,
                "subset": subset, "n": int(mask.sum()),
                "n_weeks": int(len(np.unique(codes[mask]))), "base_rate": float(yy.mean()),
            }  # fmt: skip
            enough_weeks = rec["n_weeks"] >= settings.raw["min_weeks_for_ci"]
            if subset_type == "overall" and method.kind == "prob":
                decomposition = brier_decomposition(yy, pp)
                rec.update({f"brier_{k}": v for k, v in decomposition.items()})
            for name, value in point.items():
                # Too few weeks -> a bootstrap interval would be unreliable; report none.
                lo, hi = _ci(boot[name]) if enough_weeks else (float("nan"), float("nan"))
                rec[name] = float(value[0])
                rec[f"{name}_lo"], rec[f"{name}_hi"] = lo, hi
            records.append(rec)
    reliability = [
        {"label": result.label, "lead": result.lead, "method": m.method,
         "bins": reliability_table(y, m.p)}
        for m in result.methods.values()
        if m.kind == "prob"
    ]  # fmt: skip
    value_curves = [
        {"label": result.label, "lead": result.lead, "method": m.method, "alpha": list(ALPHAS),
         "value": relative_value(y, m.p)}
        for m in result.methods.values()
        if m.kind == "prob"
    ]  # fmt: skip
    main = "blend" if "blend" in result.methods else "equal_weight"
    thresholds = {"label": result.label, "lead": result.lead, "method": main,
                  "curve": threshold_curve(y, result.methods[main].p)}  # fmt: skip
    return {
        "records": records,
        "reliability": reliability,
        "value_curves": value_curves,
        "threshold_curves": [thresholds],
        "differences": paired_differences(result, y, W_boot),
        "model_info": {m.method: m.info for m in result.methods.values() if m.info},
        "best_single": result.best_single,
        "n_eval": len(rows),
        "n_weeks": int(codes.max()) + 1,
    }


def comparison_pairs(result: LeadResult) -> list[tuple[str, str]]:
    """(A, B) pairs whose Brier difference A - B we bootstrap. Negative = A is better."""
    pairs = []
    best = result.best_single
    if best:
        pairs += [(best, "climatology"), ("equal_weight", best)]
    pairs += [("equal_weight", "climatology")]
    if "blend" in result.methods:
        head = [("blend", best), ("blend", "climatology"), ("blend", "equal_weight")]
        # NOAA's own blend: calibrated head-to-head, and against its raw forecast.
        head += [("blend", "nbm_lr"), ("blend", "nbm_raw"), ("nbm_lr", "climatology")]
        head += [("blend_nbm", "blend"), ("blend_nbm", "nbm_lr")]
        # Weights vs calibration: blend - B6 = learned weights; B6 - B5 = calibration.
        head += [("blend", "equal_weight_cal"), ("equal_weight_cal", "equal_weight")]
        pairs = head + pairs
    return [(a, b) for a, b in pairs if a in result.methods and b in result.methods]


def paired_differences(result: LeadResult, y: np.ndarray, W_boot: np.ndarray) -> list[dict]:
    out = []
    ones = np.ones((1, len(y)))
    for a, b in comparison_pairs(result):
        pa, pb = result.methods[a].p, result.methods[b].p
        point = float((brier_matrix(y, pa, ones) - brier_matrix(y, pb, ones))[0])
        boot = brier_matrix(y, pa, W_boot) - brier_matrix(y, pb, W_boot)
        lo, hi = _ci(boot)
        out.append(
            {
                "label": result.label,
                "lead": result.lead,
                "a": a,
                "b": b,
                "brier_diff": point,
                "lo": lo,
                "hi": hi,
                "share_a_better": float(np.mean(boot < 0)),
                "significant": bool(hi < 0 or lo > 0),
            }  # fmt: skip
        )
    return out


# ---------- the whole evaluation ----------


# Paths whose changes would change the results (everything else is an output or docs).
RESULT_INPUTS = ("src", "config", "pyproject.toml", "uv.lock")


def git_commit() -> str | None:
    """Short commit hash, with '-dirty' if the *code or config that produces results* has
    uncommitted changes. Generated outputs (artifacts/, docs/, data/) are excluded: `make train`
    rewrites the model files just before `evaluate` records this."""
    try:
        run = lambda *args: subprocess.run(  # noqa: E731
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = run("status", "--porcelain", "--untracked-files=no", "--", *RESULT_INPUTS)
        return run("rev-parse", "--short", "HEAD") + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def climatology_reference(settings: Settings) -> dict:
    """Which climatology the skill scores are measured against (recorded in metrics.json)."""
    from skytrust import climatology

    if not settings.raw.get("climatology", {}).get("use_long_term", True):
        return {"source": "training_years"}
    if not climatology.TABLE_PATH.exists():
        return {"source": "training_years"}
    table = json.loads(climatology.TABLE_PATH.read_text())
    return {"source": "long_term", "period": table["period"],
            "years_per_site": {s: len(y) for s, y in table["years_used"].items()}}  # fmt: skip


def base_rates_by_split(df: pd.DataFrame) -> dict:
    """Primary-label base rate per site in train vs test (context for reading test metrics)."""
    nights = df.drop_duplicates(["site", "night_date"])
    nights = nights[nights["split"].isin(["train", "test"]) & nights["usable_primary"].notna()]
    rates = nights.groupby(["site", "split"])["usable_primary"].apply(
        lambda s: s.astype(float).mean()
    )
    return {site: rates[site].to_dict() for site in rates.index.get_level_values(0).unique()}


def add_blend(result: LeadResult, settings: Settings, artifacts_dir: Path) -> bool:
    """Score the exported blend for this (label, lead) on the evaluation rows, through the
    numpy JSON loader: exactly the model file the app ships. Returns False if not trained."""
    from skytrust import blend

    path = blend.artifact_path(result.label, result.lead, artifacts_dir)
    if not path.exists():
        log.warning("no blend artifact at %s; run `skytrust train`", path)
        return False
    artifact = blend.load_artifact(path)
    blend.assert_trained_before(artifact, settings.raw["split"]["test_start"])
    p = blend.predict_proba(artifact, blend.raw_inputs(result.features, artifact))
    result.methods["blend"] = MethodPrediction("blend", "prob", "blend", p, blend.summary(artifact))
    return True


def add_blend_with_benchmarks(result: LeadResult, df: pd.DataFrame, settings: Settings) -> None:
    """Research variant (not shipped): the blend with NOAA NBM's features as extra inputs.

    Trained exactly like the blend, but only on training nights where NBM exists (its archive
    starts 2024-10-09, so ~15 months instead of ~23). Answers "would NBM help as an input?"
    without changing the shipped model the live forward test is scoring.
    """
    from skytrust import blend
    from skytrust.modeling import design_matrix, fit_tuned_logistic, model_columns, split_train_test

    benches = baselines.benchmarks_at_lead(settings, result.lead)
    if not benches:
        return
    train, _ = split_train_test(df)
    cols = blend.blend_feature_columns(settings, result.lead)
    cols += [c for b in benches for c in model_columns(b.short)]
    label_col = baselines.LABELS[result.label]
    rows = blend.training_rows(train, label_col, result.lead)
    rows = rows.dropna(subset=[c for b in benches for c in model_columns(b.short)])
    site_ids = sorted(rows["site"].unique())
    X = design_matrix(rows, cols, site_ids)
    tuned = fit_tuned_logistic(
        X, rows[label_col].astype(bool), rows["night_date"], settings, impute=True
    )
    p = tuned.pipeline.predict_proba(design_matrix(result.features, cols, site_ids))[:, 1]
    result.methods["blend_nbm"] = MethodPrediction(
        "blend_nbm", "prob", "research", p,
        {"C": tuned.C, "cv_log_loss": tuned.cv_log_loss, "n_train": tuned.n_train,
         "train_start": str(min(rows["night_date"]))},
    )  # fmt: skip


def lead_results(
    df: pd.DataFrame, settings: Settings, artifacts_dir: Path | None = None
) -> list[LeadResult]:
    """Baselines for every label x lead, plus the blend when `artifacts_dir` has it."""
    results = []
    for label in baselines.LABELS:
        for lead in settings.raw["leads"]:
            result = baselines.run_baselines(df, label, lead, settings)
            if artifacts_dir is not None and add_blend(result, settings, artifacts_dir):
                add_blend_with_benchmarks(result, df, settings)
            results.append(result)
    return results


def run_evaluation(df: pd.DataFrame, settings: Settings, artifacts_dir: Path | None = None) -> dict:
    base_seed = int(settings.raw["seed"])
    out = {"records": [], "reliability": [], "differences": [], "leads": [], "value_curves": [],
           "threshold_curves": []}  # fmt: skip
    for i, result in enumerate(lead_results(df, settings, artifacts_dir)):
        # A fixed seed per (label, lead) keeps each result reproducible on its own.
        ev = evaluate_lead(result, settings, seed=base_seed + i)
        out["records"] += ev["records"]
        out["reliability"] += ev["reliability"]
        out["differences"] += ev["differences"]
        out["value_curves"] += ev["value_curves"]
        out["threshold_curves"] += ev["threshold_curves"]
        out["leads"].append(
            {
                "label": result.label,
                "lead": result.lead,
                "n_eval": ev["n_eval"],
                "n_weeks": ev["n_weeks"],
                "best_single": ev["best_single"],
                "model_info": ev["model_info"],
            }  # fmt: skip
        )
    split = settings.raw["split"]
    test_nights = df.loc[df["split"] == "test", "night_date"]
    out["meta"] = {
        "created_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": git_commit(),
        "train_period": [str(split["train_start"]), str(split["train_end"])],
        "test_period": [
            str(split["test_start"]),
            str(max(test_nights)) if len(test_nights) else None,
        ],
        "decision_threshold": settings.raw["decision_threshold"],
        "bootstrap_resamples": settings.raw["bootstrap_resamples"],
        "bootstrap_block": "ISO calendar week",
        "seed": base_seed,
        "prob_clip_for_log_loss": PROB_CLIP,
        "primary_base_rate_by_split": base_rates_by_split(df),
        "climatology_reference": climatology_reference(settings),
        "models": [{"short": m.short, "id": m.id, "leads": list(m.leads)} for m in settings.models],
    }
    return out


def _clean(obj):
    """JSON can't hold NaN; write null instead so any JSON reader can load the file."""
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


def save_metrics(metrics: dict, path: Path = METRICS_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(metrics), indent=1))
    return path
