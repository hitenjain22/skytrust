"""Metric correctness vs sklearn / hand calculation, bootstrap behaviour, and an end-to-end
evaluation on the synthetic dataset (SPEC 12.1 'Metrics')."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from skytrust import evaluate

Y = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=float)
P = np.array([0.9, 0.3, 0.6, 0.8, 0.1, 0.55, 0.4, 0.2])
ONE = np.ones((1, len(Y)))


def test_brier_log_loss_auc_match_sklearn():
    assert evaluate.brier_matrix(Y, P, ONE)[0] == pytest.approx(brier_score_loss(Y, P))
    assert evaluate.log_loss_matrix(Y, P, ONE)[0] == pytest.approx(log_loss(Y, P))
    assert evaluate.auc_matrix(Y, P, ONE)[0] == pytest.approx(roc_auc_score(Y, P))


def test_weighted_auc_with_ties_matches_sklearn():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300).astype(float)
    score = np.round(rng.random(300), 1)  # lots of ties
    W = rng.integers(0, 4, size=(5, 300)).astype(float)
    got = evaluate.auc_matrix(y, score, W)
    for i in range(5):
        assert got[i] == pytest.approx(roc_auc_score(y, score, sample_weight=W[i]))


def test_bss_hand_computed():
    clim = np.full(len(Y), 0.5)
    m = evaluate.all_metrics(Y, P, clim, ONE, "prob", "single_lr", 0.5)
    bs = np.mean((P - Y) ** 2)
    assert m["bss"][0] == pytest.approx(1 - bs / 0.25)


def test_confusion_metrics_on_tiny_example():
    # go = p >= 0.5 -> rows 0, 2, 3, 5 ; truly usable among them: 0, 2, 3 -> TP=3, FP=1
    # usable rows = 0, 2, 3, 6 ; missed: 6 -> FN=1
    c = evaluate.confusion_matrices(Y, P, ONE, 0.5)
    assert c["false_clear_rate"][0] == pytest.approx(1 / 4)
    assert c["miss_rate"][0] == pytest.approx(1 / 4)
    assert c["accuracy"][0] == pytest.approx(6 / 8)


def test_rule_family_reports_confusion_only():
    m = evaluate.all_metrics(Y, (P > 0.5).astype(float), P, ONE, "hard", "rule", 0.5)
    assert set(m) == set(evaluate.RULE_METRICS)
    h = evaluate.all_metrics(Y, (P > 0.5).astype(float), P, ONE, "hard", "persistence", 0.5)
    assert "brier" in h and "log_loss" not in h


def test_reliability_bin_counts_sum_to_n():
    rng = np.random.default_rng(0)
    p = rng.random(500)
    p[:3] = [0.0, 1.0, 0.1]  # edges land in the first/last bins
    bins = evaluate.reliability_table((rng.random(500) < p).astype(float), p)
    assert len(bins) == 10 and sum(b["n"] for b in bins) == 500
    assert bins[-1]["n"] > 0 and bins[0]["n"] > 0


def test_bootstrap_is_reproducible_and_week_blocked():
    dates = pd.Series(pd.date_range("2026-01-05", periods=28).date)  # exactly 4 ISO weeks
    codes = evaluate.week_codes(dates)
    assert sorted(set(codes)) == [0, 1, 2, 3] and (np.bincount(codes) == 7).all()
    w1 = evaluate.bootstrap_weights(codes, 200, np.random.default_rng(42))
    w2 = evaluate.bootstrap_weights(codes, 200, np.random.default_rng(42))
    assert np.array_equal(w1, w2)
    # Every night in a week gets the same weight (the week moves as a block)...
    for week in range(4):
        cols = w1[:, codes == week]
        assert (cols == cols[:, :1]).all()
    # ...and each replicate draws exactly n_weeks weeks.
    assert (w1.sum(axis=1) == 28).all()


def test_ci_ignores_non_finite_values():
    lo, hi = evaluate._ci(np.array([np.nan, 1.0, 2.0, 3.0, np.inf]))
    assert 1.0 <= lo <= hi <= 3.0


@pytest.fixture(scope="module")
def metrics(synthetic_built, fast_settings):
    return evaluate.run_evaluation(synthetic_built[0], fast_settings)


def test_run_evaluation_covers_every_label_and_lead(metrics, fast_settings):
    rec = pd.DataFrame(metrics["records"])
    assert set(rec["label"]) == {"primary", "asos", "era5"}
    assert set(rec["lead"]) == set(fast_settings.raw["leads"])
    overall = rec[rec["subset_type"] == "overall"]
    clim = overall[overall["method"] == "climatology"]
    assert (clim["bss"].abs() < 1e-12).all()  # climatology has zero skill vs itself
    probs = overall[overall["kind"] == "prob"]
    assert (probs["brier_lo"] <= probs["brier"] + 1e-12).all()
    assert (probs["brier"] <= probs["brier_hi"] + 1e-12).all()
    assert {"site", "season"} <= set(rec["subset_type"])


def test_paired_differences_and_best_single(metrics):
    diffs = pd.DataFrame(metrics["differences"])
    assert {"equal_weight"} <= set(diffs["a"])
    assert ((diffs["lo"] <= diffs["brier_diff"]) & (diffs["brier_diff"] <= diffs["hi"])).all()
    leads = pd.DataFrame(metrics["leads"])
    assert leads["best_single"].str.endswith("_lr").all()


def test_metrics_json_is_strict_json(metrics, tmp_path):
    path = evaluate.save_metrics(metrics, tmp_path / "m.json")
    text = path.read_text()
    assert "NaN" not in text and "Infinity" not in text
    back = json.loads(text)
    assert back["meta"]["bootstrap_block"] == "ISO calendar week"
    assert evaluate.load_metrics(path)["records"]


def test_evaluation_is_deterministic(synthetic_built, fast_settings):
    from skytrust import baselines

    result = baselines.run_baselines(synthetic_built[0], "primary", 2, fast_settings)
    a = pd.DataFrame(evaluate.evaluate_lead(result, fast_settings, seed=5)["records"])
    b = pd.DataFrame(evaluate.evaluate_lead(result, fast_settings, seed=5)["records"])
    pd.testing.assert_frame_equal(a, b)


# ---------- RESULTS.md, figures, CLI ----------


def test_write_results_renders_every_section_and_figure(metrics, tmp_path):
    from skytrust import report

    path = report.write_results(metrics, docs=tmp_path)
    md = path.read_text()
    for heading in ["## Summary", "## 1. Headline", "## 2. Skill vs lead time", "## 3. Calibration",
                    "## 4. Paired comparisons", "## 5. By site", "## 6. Sensitivity",
                    "## 7. Single-model tuning", "## Caveats"]:  # fmt: skip
        assert heading in md, heading
    for fig in [
        "lead_curves_primary.png",
        "reliability_primary_lead1.png",
        "site_skill_primary_lead1.png",
    ]:
        assert (tmp_path / "figures" / fig).stat().st_size > 1000
        assert f"figures/{fig}" in md
    assert "weeks of labeled nights" in md  # generated caveat, not hand-typed
    assert "Phase 4" in md  # no blend yet -> says so plainly


def test_summary_counts_significant_wins(metrics):
    from skytrust import report

    v = report.MetricsView(metrics)
    lines = "\n".join(report.summary_lines(v))
    assert f"of {len(v.lead_list())} leads" in lines and "Brier Skill Score" in lines


def test_ecmwf_flag_appears_only_when_ecmwf_is_best(metrics):
    from skytrust import report

    v = report.MetricsView(metrics)
    v.leads["best_single"] = "gfs_lr"
    assert report.ecmwf_flags(v) == []
    v.leads.loc[v.leads["label"] == "era5", "best_single"] = "ecmwf_lr"
    flags = report.ecmwf_flags(v)
    assert len(flags) == 1 and "ERA5 is produced by ECMWF" in flags[0]


def test_figures_are_byte_reproducible(metrics, tmp_path):
    from skytrust import figures

    rec = pd.DataFrame(metrics["records"])
    a = figures.lead_curves(rec, "primary", tmp_path / "a.png").read_bytes()
    b = figures.lead_curves(rec, "primary", tmp_path / "b.png").read_bytes()
    assert a == b


def test_cli_evaluate_and_report(monkeypatch, metrics, tmp_path, capsys):
    from skytrust import __main__ as cli
    from skytrust import dataset, report

    monkeypatch.setattr(dataset, "load_dataset", lambda: pd.DataFrame())
    monkeypatch.setattr(evaluate, "run_evaluation", lambda df, s, **k: metrics)
    monkeypatch.setattr(evaluate, "save_metrics", lambda m: tmp_path / "m.json")
    assert cli.main(["evaluate"]) == 0
    monkeypatch.setattr(evaluate, "load_metrics", lambda: metrics)
    monkeypatch.setattr(report, "write_results", lambda m: tmp_path / "RESULTS.md")
    monkeypatch.setattr(report, "update_readme", lambda m: True)
    assert cli.main(["report"]) == 0
    out = capsys.readouterr().out
    assert "metric records" in out and "RESULTS.md" in out


def test_git_commit_marks_dirty_tree(monkeypatch):
    import subprocess

    def fake_run(args, **kw):
        out = " M src/x.py" if "status" in args else "abc1234"
        return subprocess.CompletedProcess(args, 0, stdout=out)

    monkeypatch.setattr(evaluate.subprocess, "run", fake_run)
    assert evaluate.git_commit() == "abc1234-dirty"


def test_readme_block_replaces_only_between_markers(metrics, tmp_path):
    from skytrust import report

    readme = tmp_path / "README.md"
    readme.write_text(f"# T\n\nintro\n{report.README_START}\nold\n{report.README_END}\n\ntail\n")
    assert report.update_readme(metrics, readme)
    text = readme.read_text()
    assert text.startswith("# T\n\nintro\n") and text.endswith("\n\ntail\n")
    assert "old" not in text and "Brier Skill Score" in text and "Climatology (B1)" in text
    assert report.update_readme(metrics, readme)  # idempotent: markers survive
    no_markers = tmp_path / "plain.md"
    no_markers.write_text("# nothing\n")
    assert not report.update_readme(metrics, no_markers)


def test_every_method_in_a_figure_gets_a_distinct_colour(metrics, monkeypatch, tmp_path):
    """Regression: the best single model was once drawn in the same blue as equal-weight."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex

    from skytrust import figures

    captured = []
    monkeypatch.setattr(plt, "close", lambda fig: captured.append(fig))
    methods = ["climatology", "icon_lr", "equal_weight"]
    figures.reliability(metrics["reliability"], methods, "primary", 1, tmp_path / "r.png")
    lines = [
        ln for ln in captured[0].axes[0].get_lines() if ln.get_label() != "Perfectly calibrated"
    ]
    colours = [to_hex(ln.get_color()) for ln in lines]  # normalise tuple vs '#hex'
    assert len(lines) == 3 and len(set(colours)) == 3
    rec = pd.DataFrame(metrics["records"])
    figures.site_skill(rec, ["icon_lr", "equal_weight"], "primary", 1, tmp_path / "s.png")
    bars = {to_hex(p.get_facecolor()) for p in captured[1].axes[0].patches}
    assert len(bars) == 2


def test_subsets_with_too_few_weeks_get_no_ci(synthetic_built, fast_settings):
    import copy
    import dataclasses

    from skytrust import baselines

    raw = copy.deepcopy(fast_settings.raw)
    raw["min_weeks_for_ci"] = 999
    strict = dataclasses.replace(fast_settings, raw=raw)
    result = baselines.run_baselines(synthetic_built[0], "primary", 1, strict)
    rec = pd.DataFrame(evaluate.evaluate_lead(result, strict, seed=1)["records"])
    assert rec["brier"].notna().any()  # point estimates are still reported
    assert rec["brier_lo"].isna().all() and rec["false_clear_rate_hi"].isna().all()


def test_git_commit_dirty_check_ignores_generated_outputs(monkeypatch):
    """Regression: `make train` rewrites artifacts/*.json right before `evaluate` records the
    commit, which falsely marked every pipeline run '-dirty'. Only code/config count."""
    import subprocess

    seen = {}

    def fake_run(args, **kw):
        if "status" in args:
            seen["status_args"] = args
            return subprocess.CompletedProcess(args, 0, stdout="")
        return subprocess.CompletedProcess(args, 0, stdout="abc1234")

    monkeypatch.setattr(evaluate.subprocess, "run", fake_run)
    assert evaluate.git_commit() == "abc1234"
    pathspec = seen["status_args"][seen["status_args"].index("--") + 1 :]
    assert "src" in pathspec and "config" in pathspec
    assert "artifacts" not in pathspec and "docs" not in pathspec


def test_test_period_name():
    from skytrust import report

    assert report.test_period_name({"test_period": ["2026-01-01", "2026-08-31"]}) == "Jan–Aug 2026"
    assert (
        report.test_period_name({"test_period": ["2025-11-01", "2026-02-28"]})
        == "Nov 2025–Feb 2026"
    )
