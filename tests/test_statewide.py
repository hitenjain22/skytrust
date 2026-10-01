"""Statewide evaluation (leave-one-region-out / leave-one-station-out), offline on the synthetic
two-station dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skytrust import statewide

REGIONS = {"SAC": "STO", "BIH": "VEF"}


def test_held_out_fits_never_see_their_own_group():
    train = pd.DataFrame({"g": ["a", "a", "b", "b", "c"], "x": [1, 2, 3, 4, 5]})
    rows = pd.DataFrame({"g": ["a", "b", "c", "c"]})
    seen = {}

    def fit(tr):
        return set(tr["g"])

    def predict(model, r):
        seen[r["g"].iloc[0]] = model
        return np.zeros(len(r))

    p = statewide.held_out(train, rows, train["g"], rows["g"], fit, predict)
    assert len(p) == 4 and not np.isnan(p).any()
    assert seen == {"a": {"b", "c"}, "b": {"a", "c"}, "c": {"a", "b"}}


def test_leave_one_region_out_never_trains_on_the_held_out_region(
    synthetic_built, fast_settings, monkeypatch
):
    """Spy on the blend fit: when a region's nights are forecast, none of its stations were in
    the training rows."""
    df = synthetic_built[0]
    calls = []
    real = statewide.blend.fit_blend

    def spy(train, label, lead, settings, use_site=True):
        calls.append(set(train["site"]))
        assert not use_site  # the statewide blend never knows which station it is
        return real(train, label, lead, settings, use_site=use_site)

    monkeypatch.setattr(statewide.blend, "fit_blend", spy)
    preds = statewide.predictions(df, fast_settings, "primary", 1, REGIONS, loso=False)
    # regions in order: STO (SAC) held out -> trained on BIH only; then VEF (BIH) -> SAC only
    assert calls == [{"BIH"}, {"SAC"}]
    assert preds["statewide_loro"].notna().all()
    assert set(preds["region"]) == {"STO", "VEF"}


def test_predictions_and_scores_have_every_method(synthetic_built, fast_settings):
    df = synthetic_built[0]
    preds = statewide.predictions(df, fast_settings, "primary", 1, REGIONS, loso=True)
    for col in ["statewide_loro", "statewide_loso", "equal_weight", "climatology", "y"]:
        assert preds[col].notna().all(), col
    assert any(c.startswith("single_") for c in preds.columns)
    # every row is a test night with every member present
    assert (pd.to_datetime(preds["night_date"]) >= "2026-01-01").all()
    result = statewide.score(preds, fast_settings, airports={"SAC"})
    assert set(result["subsets"]) == {"all", "new_stations"}
    assert result["subsets"]["new_stations"]["n_stations"] == 1
    m = result["subsets"]["all"]["methods"]["statewide_loro"]
    assert {"brier", "bss", "false_clear_rate", "brier_lo", "brier_hi"} <= set(m)
    assert set(result["stations"]) == {"SAC", "BIH"} and set(result["regions"]) == {"STO", "VEF"}
    paired = result["subsets"]["all"]["paired"]
    assert {"statewide_minus_equal_weight", "statewide_minus_best_single", "loro_minus_loso"} <= set(
        paired
    )


def test_evaluation_rows_need_every_member(synthetic_built, fast_settings):
    df = synthetic_built[0]
    _, test = statewide.split_train_test(df)
    rows = statewide.evaluation_rows(test, "usable_primary", 1, fast_settings)
    test1 = test[(test["lead"] == 1) & test["usable_primary"].notna()].copy()
    test1.loc[test1.index[0], "gfs_frac_clear"] = np.nan  # one night without GFS
    fewer = statewide.evaluation_rows(test1, "usable_primary", 1, fast_settings)
    assert len(fewer) == len(rows) - 1


@pytest.mark.parametrize("label", ["asos", "era5"])
def test_other_labels_score_too(synthetic_built, fast_settings, label):
    preds = statewide.predictions(synthetic_built[0], fast_settings, label, 1, REGIONS, loso=False)
    assert "statewide_loso" not in preds and preds["statewide_loro"].notna().all()
