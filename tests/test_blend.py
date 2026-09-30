"""The blend: JSON export round-trip (SPEC 8.7), leakage guarantees, calibration rule, and the
end-to-end integration test (SPEC 12.2: build-dataset -> train -> evaluate -> report, offline)."""

from __future__ import annotations

import copy
import dataclasses
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from skytrust import blend, evaluate, modeling, report
from skytrust.baselines import LABELS


@pytest.fixture(scope="module")
def trained(synthetic_built, fast_settings, tmp_path_factory):
    """Train every blend once on the synthetic dataset and evaluate once; shared by tests."""
    root = tmp_path_factory.mktemp("artifacts")
    paths = blend.train_all(synthetic_built[0], fast_settings, root=root)
    metrics = evaluate.run_evaluation(synthetic_built[0], fast_settings, artifacts_dir=root)
    return root, paths, metrics


@pytest.fixture(scope="module")
def train_rows(synthetic_built):
    train, _ = modeling.split_train_test(synthetic_built[0])
    return train


def refit_sklearn(train, artifact, settings):
    """Refit the same pipeline with the artifact's chosen C, for comparison."""
    rows = blend.training_rows(train, LABELS[artifact["label"]], artifact["lead"])
    X = modeling.design_matrix(rows, artifact["model_feature_columns"], artifact["site_ids"])
    y = rows[LABELS[artifact["label"]]].astype(bool).to_numpy().astype(int)
    return modeling.make_pipeline(artifact["logistic"]["C"], impute=True).fit(X, y), X


def test_json_round_trip_matches_sklearn_to_1e9(train_rows, synthetic_built, fast_settings):
    artifact = json.loads(json.dumps(blend.fit_blend(train_rows, "primary", 1, fast_settings)))
    pipe, X_train = refit_sklearn(train_rows, artifact, fast_settings)
    df = synthetic_built[0]
    rows = df[df["lead"] == 1].copy()
    # Knock out some inputs so the imputer + missing-indicator path is exercised too.
    rows.loc[rows.index[::7], "ecmwf_frac_clear"] = np.nan
    rows.loc[rows.index[::11], "gfs_mean_cover"] = np.nan
    X = modeling.design_matrix(rows, artifact["model_feature_columns"], artifact["site_ids"])
    ours = blend.predict_proba(artifact, blend.raw_inputs(rows, artifact))
    theirs = pipe.predict_proba(X[artifact["inputs"]])[:, 1]
    assert np.max(np.abs(ours - theirs)) < 1e-9


def test_json_round_trip_with_isotonic_calibration(train_rows, synthetic_built, fast_settings):
    raw = copy.deepcopy(fast_settings.raw)
    raw["calibration"]["ece_threshold"] = -1.0  # force the calibration branch
    forced = dataclasses.replace(fast_settings, raw=raw)
    artifact = json.loads(json.dumps(blend.fit_blend(train_rows, "primary", 1, forced)))
    assert artifact["calibration"]["applied"]
    pipe, _ = refit_sklearn(train_rows, artifact, forced)
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.X_thresholds_ = np.array(artifact["calibration"]["isotonic_x"])
    iso.y_thresholds_ = np.array(artifact["calibration"]["isotonic_y"])
    rows = synthetic_built[0][synthetic_built[0]["lead"] == 1]
    X = modeling.design_matrix(rows, artifact["model_feature_columns"], artifact["site_ids"])
    expected = np.interp(
        pipe.predict_proba(X[artifact["inputs"]])[:, 1], iso.X_thresholds_, iso.y_thresholds_
    )
    ours = blend.predict_proba(artifact, blend.raw_inputs(rows, artifact))
    assert np.max(np.abs(ours - expected)) < 1e-9
    assert ((ours >= 0) & (ours <= 1)).all()


def test_no_test_rows_reach_blend_fit(synthetic_built, fast_settings, monkeypatch, tmp_path):
    df = synthetic_built[0]
    test_index = set(df.index[df["split"] == "test"])
    seen = []
    real_fit = Pipeline.fit

    def spy(self, X, y=None, **kw):
        seen.append(X.index)
        return real_fit(self, X, y, **kw)

    monkeypatch.setattr(Pipeline, "fit", spy)
    raw = copy.deepcopy(fast_settings.raw)
    raw["leads"] = [1]  # one lead is enough to exercise tuning, OOF calibration, and refit
    blend.train_all(df, dataclasses.replace(fast_settings, raw=raw), root=tmp_path)
    assert seen
    assert all(not (set(idx) & test_index) for idx in seen)


def test_imputer_medians_come_from_training_rows_only(train_rows, fast_settings):
    artifact = blend.fit_blend(train_rows, "primary", 1, fast_settings)
    rows = blend.training_rows(train_rows, "usable_primary", 1)
    X = modeling.design_matrix(rows, artifact["model_feature_columns"], artifact["site_ids"])
    expected = X[artifact["inputs"]].median(skipna=True).to_numpy()
    assert np.allclose(artifact["imputer"]["medians"], expected)
    assert artifact["training"]["period"][1] < "2026-01-01"


def test_artifact_schema_and_contents(train_rows, fast_settings):
    a = blend.fit_blend(train_rows, "primary", 7, fast_settings)
    for key in ["inputs", "imputer", "scaler", "logistic", "calibration", "training", "versions",
                "created_utc", "git_commit", "site_ids", "model_feature_columns"]:  # fmt: skip
        assert key in a, key
    assert a["models"] == ["gfs", "ecmwf", "gem"]  # only models that forecast 7 days ahead
    assert "hrrr_frac_clear" not in a["inputs"] and "spread_frac_clear" in a["inputs"]
    assert len(a["logistic"]["coef"]) == len(a["logistic"]["output_features"])
    assert len(a["scaler"]["mean"]) == len(a["logistic"]["output_features"])
    assert abs(sum(blend.weight_shares(a).values()) - 1) < 1e-9


def test_trained_before_guard():
    art = {"training": {"period": ["2024-01-01", "2026-01-03"]}}
    with pytest.raises(modeling.LeakageError):
        blend.assert_trained_before(art, "2026-01-01")
    blend.assert_trained_before(
        {"training": {"period": ["2024-01-01", "2025-12-31"]}}, "2026-01-01"
    )


def test_expected_calibration_error_hand_example():
    y = np.array([1, 0, 1, 1])
    p = np.array([0.95, 0.95, 0.05, 0.05])  # two bins, each 50% observed
    assert blend.expected_calibration_error(y, p) == pytest.approx(0.5 * 0.45 + 0.5 * 0.95)
    assert blend.expected_calibration_error(np.array([1, 0]), np.array([0.5, 0.5])) == 0


def test_context_features():
    rows = pd.DataFrame({"site": ["SAC", "BIH"], "month": [12, 6], "dark_hours": [11, 5]})
    ctx = modeling.context_features(rows, ["BIH", "SAC"])
    assert ctx["site_SAC"].tolist() == [1.0, 0.0] and ctx["site_BIH"].tolist() == [0.0, 1.0]
    assert ctx["month_cos"].iloc[0] == pytest.approx(1.0)  # December sits next to January
    assert ctx["month_cos"].iloc[1] == pytest.approx(-1.0)
    assert ctx["dark_hours"].tolist() == [11.0, 5.0]


def test_bad_schema_version_rejected(tmp_path):
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"schema_version": 99}))
    with pytest.raises(ValueError):
        blend.load_artifact(path)


def test_end_to_end_offline_pipeline(trained, synthetic_built, fast_settings, tmp_path):
    """SPEC 12.2: dataset (built from the fake raw cache) -> train -> evaluate -> report."""
    artifacts, paths, metrics = trained
    assert len(paths) == len(LABELS) * len(fast_settings.raw["leads"])
    assert (artifacts / "model_lead1.json").exists()
    assert (artifacts / "sensitivity" / "model_asos_lead1.json").exists()

    rec = pd.DataFrame(metrics["records"])
    blend_rows = rec[(rec["method"] == "blend") & (rec["subset_type"] == "overall")]
    assert set(blend_rows["label"]) == set(LABELS)
    assert blend_rows["brier"].between(0, 1).all()
    diffs = pd.DataFrame(metrics["differences"])
    assert (diffs["a"] == "blend").any()
    # NOAA NBM head-to-head and the research blend+NBM variant are evaluated too.
    assert {("blend", "nbm_lr"), ("blend_nbm", "blend")} <= set(
        zip(diffs["a"], diffs["b"], strict=True)
    )
    leads = pd.DataFrame(metrics["leads"])
    assert all("blend" in info for info in leads["model_info"])

    evaluate.save_metrics(metrics, tmp_path / "metrics.json")
    docs = tmp_path / "docs"
    docs.mkdir()
    md = report.write_results(
        evaluate.load_metrics(tmp_path / "metrics.json"), docs=docs
    ).read_text()
    assert "## 8. The blend: what it learned" in md
    assert "The learned blend" in md and "Phase 4 (not yet run)" not in md
    assert "of 3 leads" in md


def test_report_blend_details(trained):
    v = report.MetricsView(trained[2])
    assert "Blend" not in set(report.model_info_table(v, "primary")["model"])
    site = report.breakdown_table(v, "primary", 1, "site")
    assert "BSS blend" in site.columns and "false-clear blend" in site.columns
    assert "model-spread coefficient" in report.spread_sentence(v)


def test_resume_bullets_use_only_metrics_numbers(trained, tmp_path):
    import re

    metrics = trained[2]
    bullets = report.resume_bullets(metrics)
    assert len(bullets) == 3
    v = report.MetricsView(metrics)
    blend_rec = v.rec("primary", 1, "blend")
    assert f"{blend_rec['false_clear_rate']:.1%}" in bullets[0]
    assert f"{blend_rec['bss']:.2f}" in bullets[1]
    text = report.write_resume_bullets(metrics, tmp_path / "r.md").read_text()
    assert "Every number below comes from that file" in text
    assert re.search(r"\d+ weather models", bullets[0])
