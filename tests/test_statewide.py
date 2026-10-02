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
    assert {
        "statewide_minus_equal_weight",
        "statewide_minus_best_single",
        "loro_minus_loso",
    } <= set(paired)


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


def synthetic_result(df, settings) -> dict:
    """A statewide.json in the shape `statewide.run` writes, from the synthetic dataset (one
    lead, primary label) - for testing the readers without the real network."""
    from skytrust import network

    preds = statewide.predictions(df, settings, "primary", 1, REGIONS, loso=True)
    stations = [s for s in network.load_network_table() if s["id"] in REGIONS]
    return {
        "stations": stations,
        "region_names": network.REGION_NAMES,
        "labels": {"primary": [{"lead": 1, **statewide.score(preds, settings, {"SAC"})}]},
    }


def test_place_record_lists_the_nearest_stations(synthetic_built, fast_settings, tmp_path):
    import json

    from skytrust import inference

    path = tmp_path / "statewide.json"
    result = synthetic_result(synthetic_built[0], fast_settings)
    path.write_text(json.dumps(statewide.evaluate._clean(result)))
    # Davis: 0.038° of latitude (4.2 km) and 0.2455° of longitude (21.4 km at 38.5°N) from
    # Sacramento Executive, so 21.8 km; Bishop is ~300 km away
    davis = (38.5449, -121.7405)
    rec = inference.place_record(*davis, lead=1, path=path)
    assert rec["kind"] == "statewide" and rec["n_stations"] == 2
    assert [s["id"] for s in rec["near"]] == ["SAC", "BIH"]
    assert rec["near"][0]["distance_km"] == pytest.approx(21.8, abs=0.2)
    assert {"bss", "false_clear_rate", "n"} <= set(rec["near"][0])
    # no statewide results yet: the five-airport leave-one-site-out record
    assert inference.place_record(*davis, lead=1, path=tmp_path / "absent.json") == (
        inference.unseen_site_record(1)
    )


def _verdicts(*diffs):
    """A statewide.json skeleton with the paired statewide-minus-airport-geo Brier difference
    (point, lo, hi) at the new stations for each lead."""
    return {"labels": {"primary": [
        {"lead": i + 1, "subsets": {"new_stations": {"paired": {"statewide_minus_airport_geo": {
            "brier_diff": d, "lo": lo, "hi": hi, "significant": bool(hi < 0 or lo > 0)}}}}}
        for i, (d, lo, hi) in enumerate(diffs)]}}  # fmt: skip


def test_the_shipping_rule_is_the_one_fixed_before_the_results():
    """DECISIONS 2026-10-01: the statewide blend ships unless it is significantly *worse* than
    the five-airport geo blend at the new stations (at any lead); ties and wins ship it."""
    tie_and_wins = _verdicts((0.0005, -0.001, 0.002), (-0.002, -0.003, -0.0001))
    assert statewide.shipping_decision(tie_and_wins)["method"] == "statewide_loro"
    worse = _verdicts((0.0005, -0.001, 0.002), (0.003, 0.001, 0.005))
    d = statewide.shipping_decision(worse)
    assert d["method"] == "geo_airports" and d["worse_at_leads"] == [2]


def test_shipping_copies_the_winner_and_records_it(tmp_path):
    import json

    cand, live = tmp_path / "geo_statewide", tmp_path / "geo"
    cand.mkdir()
    live.mkdir()
    (cand / "model_geo_lead1.json").write_text('{"who": "statewide"}')
    (live / "model_geo_lead1.json").write_text('{"who": "airports"}')
    path = tmp_path / "statewide.json"
    path.write_text(json.dumps(_verdicts((-0.001, -0.002, -0.0005))))
    decision = statewide.ship(path, candidates=cand, live=live)
    assert decision["method"] == "statewide_loro"
    assert json.loads((live / "model_geo_lead1.json").read_text())["who"] == "statewide"
    assert json.loads(path.read_text())["shipped_method"] == "statewide_loro"
