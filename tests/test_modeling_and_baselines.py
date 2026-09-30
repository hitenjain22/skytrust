"""Split / leakage guarantees (SPEC 12.1 'Split and leakage') and baseline behaviour."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from skytrust import baselines, modeling


def test_train_max_date_before_test_min_date(synthetic_built):
    train, test = modeling.split_train_test(synthetic_built[0])
    assert pd.to_datetime(train["night_date"]).max() < pd.to_datetime(test["night_date"]).min()


def test_overlap_assertion_raises():
    a = pd.Series([dt.date(2025, 12, 30), dt.date(2026, 1, 2)])
    b = pd.Series([dt.date(2026, 1, 1)])
    with pytest.raises(modeling.LeakageError):
        modeling.assert_no_date_overlap(a, b)


def test_split_refuses_mislabelled_split_column(synthetic_built):
    df = synthetic_built[0].copy()
    df.loc[df.index[-1], "split"] = "train"  # a January (test-period) row sneaks into train
    with pytest.raises(modeling.LeakageError):
        modeling.split_train_test(df)


def test_date_folds_are_forward_only_and_keep_nights_together():
    dates = pd.Series(np.repeat(pd.date_range("2025-01-01", periods=40).date, 5))  # 5 sites/night
    for tr, va in modeling.date_folds(dates, 4):
        d_tr, d_va = pd.to_datetime(dates.iloc[tr]), pd.to_datetime(dates.iloc[va])
        assert d_tr.max() < d_va.min()  # always train on the past
        assert not set(d_tr) & set(d_va)  # no night split across train/validation
        assert len(va) % 5 == 0


def test_no_test_rows_reach_fit(synthetic_built, fast_settings, monkeypatch):
    """Spy on every Pipeline.fit call made while running the baselines."""
    df = synthetic_built[0]
    test_index = set(df.index[df["split"] == "test"])
    seen: list[pd.Index] = []
    real_fit = Pipeline.fit

    def spy(self, X, y=None, **kw):
        seen.append(X.index)
        return real_fit(self, X, y, **kw)

    monkeypatch.setattr(Pipeline, "fit", spy)
    baselines.run_baselines(df, "primary", 1, fast_settings)
    assert seen, "expected the single-model baselines to fit something"
    for idx in seen:
        assert not (set(idx) & test_index)


def test_climatology_uses_training_nights_only(synthetic_built, fast_settings):
    df = synthetic_built[0].copy()
    train = df["split"] == "train"
    df.loc[train, "usable_primary"] = False  # train: never usable
    df.loc[~train, "usable_primary"] = True  # test: always usable
    table = baselines.climatology_table(df[train], "usable_primary")
    rows = df[~train]
    assert (baselines.predict_climatology(table, rows) == 0).all()


def test_climatology_falls_back_to_site_rate_for_unseen_month():
    table = pd.DataFrame(
        {"rate": [0.2, 0.6], "n": [30, 30]},
        index=pd.MultiIndex.from_tuples([("SAC", 12), ("SAC", 11)], names=["site", "month"]),
    )
    rows = pd.DataFrame({"site": ["SAC", "SAC"], "month": [12, 1]})
    assert baselines.predict_climatology(table, rows).tolist() == pytest.approx([0.2, 0.4])


def test_persistence_looks_back_exactly_lead_nights():
    d = pd.date_range("2026-01-01", periods=5).date
    df = pd.DataFrame(
        {
            "site": "SAC",
            "night_date": d,
            "usable_primary": pd.array([True, False, True, True, False], dtype="boolean"),
        }
    )
    rows = df.iloc[2:]
    out = baselines.persistence_outcome(df, rows, "usable_primary", lead=2)
    assert out.tolist() == [True, False, True]  # nights 0, 1, 2
    first = baselines.persistence_outcome(df, df.iloc[:1], "usable_primary", lead=1)
    assert first.isna().all()


def test_run_baselines_shapes_and_common_eval_set(synthetic_built, fast_settings):
    df = synthetic_built[0]
    result = baselines.run_baselines(df, "primary", 1, fast_settings)
    n = len(result.rows)
    assert n > 0 and set(result.rows.columns) >= {"site", "night_date", "y"}
    assert (pd.to_datetime(result.rows["night_date"]) >= "2026-01-01").all()
    expected = {"climatology", "persistence", "equal_weight", "equal_weight_cal"}
    for m in fast_settings.models:
        expected |= {f"{m.short}_rule", f"{m.short}_lr"}
    for b in fast_settings.benchmarks:
        expected |= {f"{b.short}_rule", f"{b.short}_raw", f"{b.short}_lr"}
    assert set(result.methods) == expected
    for method in result.methods.values():
        assert len(method.p) == n and np.isfinite(method.p).all()
        assert ((method.p >= 0) & (method.p <= 1)).all()
        if method.kind == "hard":
            assert set(np.unique(method.p)) <= {0.0, 1.0}
    assert result.best_single in {f"{m.short}_lr" for m in fast_settings.models}
    info = result.methods["gfs_lr"].info
    assert info["C"] in modeling.c_grid(fast_settings) and info["n_train"] > 0


def test_models_at_long_lead_exclude_short_range_models(synthetic_built, fast_settings):
    result = baselines.run_baselines(synthetic_built[0], "primary", 7, fast_settings)
    assert "hrrr_lr" not in result.methods and "icon_lr" not in result.methods
    assert "ecmwf_lr" in result.methods


def test_log_loss_matches_sklearn():
    from sklearn.metrics import log_loss as sk_log_loss

    y = np.array([1, 0, 1, 1, 0])
    p = np.array([0.9, 0.2, 0.6, 0.99, 0.4])
    assert modeling.log_loss(y, p) == pytest.approx(sk_log_loss(y, p))


def test_choose_c_prefers_regularized_among_near_ties():
    scores = {0.01: 0.40, 0.1: 0.37260, 1.0: 0.37255, 1000.0: 0.37254}
    assert modeling.choose_C(scores, 1e-4) == 0.1
    assert modeling.choose_C(scores, 0.0) == 1000.0


def test_benchmarks_are_scored_but_never_chosen_as_best_single(synthetic_built, fast_settings):
    result = baselines.run_baselines(synthetic_built[0], "primary", 1, fast_settings)
    assert result.methods["nbm_lr"].family == "benchmark"
    assert result.methods["nbm_raw"].kind == "prob"
    assert not result.best_single.startswith("nbm")  # "best single" = best blend *member*


def test_benchmark_features_never_change_blend_inputs(synthetic_built, fast_settings):
    """Adding NBM must not alter the spread / model count the shipped blend was trained on."""
    df = synthetic_built[0]
    members = [f"{m.short}_frac_clear" for m in fast_settings.models]
    lead1 = df[df["lead"] == 1]
    assert (lead1["n_models_available"] == lead1[members].notna().sum(axis=1)).all()
    assert "nbm_frac_clear" in df and lead1["nbm_frac_clear"].notna().any()


def test_calibrated_equal_weight_uses_one_shared_weight(synthetic_built, fast_settings):
    result = baselines.run_baselines(synthetic_built[0], "primary", 1, fast_settings)
    b6 = result.methods["equal_weight_cal"]
    assert b6.kind == "prob" and ((b6.p > 0) & (b6.p < 1)).all()
    # Monotone in the average clear fraction at fixed context: more clear -> higher P.
    assert np.corrcoef(b6.p, result.methods["equal_weight"].p)[0, 1] > 0.8


def test_labels_available_skips_labels_without_data(synthetic_built):
    df = synthetic_built[0]
    assert baselines.labels_available(df) == ["primary", "asos", "era5", "goes"]
    no_goes = df.assign(usable_goes=pd.array([pd.NA] * len(df), dtype="boolean"))
    assert "goes" not in baselines.labels_available(no_goes)
    only_train = df.assign(usable_goes=df["usable_goes"].where(df["split"] == "train"))
    assert "goes" not in baselines.labels_available(only_train)
    assert "goes" in baselines.labels_available(only_train, splits=("train",))
