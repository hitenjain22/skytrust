from __future__ import annotations

import copy
import dataclasses
import datetime as dt

import numpy as np
import pytest

from skytrust import hourly
from skytrust.inference import predict_proba, raw_inputs
from synthetic import SYNTH_SITES

pytestmark = pytest.mark.slow  # builds hours for two sites and fits a model per lead


@pytest.fixture(scope="module")
def hourly_df(fast_settings, synthetic_built):
    _, root = synthetic_built
    raw = copy.deepcopy(fast_settings.raw)
    raw["sources"]["history_start"] = dt.date(2025, 12, 1)
    s = dataclasses.replace(fast_settings, raw=raw, history_start=dt.date(2025, 12, 1))
    return hourly.build_hourly(s, SYNTH_SITES, root), s


def test_hourly_rows_match_dark_hours_and_truth_rule(hourly_df):
    df, s = hourly_df
    assert set(df["lead"]) == set(s.raw["leads"])
    assert df["night_position"].between(0, 1).all()
    labeled = df.dropna(subset=["truth_cover"])
    assert (labeled["clear"] == (labeled["truth_cover"] <= 0.2 + 1e-9)).all()
    assert df.loc[df["lead"] == 2, "hrrr_cover"].isna().all() if "hrrr_cover" in df else True


def test_hourly_model_round_trip_and_beats_climatology(hourly_df, tmp_path):
    df, s = hourly_df
    result = hourly.train_and_evaluate(df, s, tmp_path)
    lead1 = next(e for e in result["leads"] if e["lead"] == 1)
    assert lead1["methods"]["hourly_model"]["bss"] > 0  # synthetic forecasts are informative
    import json

    artifact = json.loads((tmp_path / "model_hourly_lead1.json").read_text())
    rows = df[(df["lead"] == 1) & (df["split"] == "test")].dropna(subset=["mean_cover"]).head(50)
    p = predict_proba(artifact, raw_inputs(rows, artifact))
    assert ((p > 0) & (p < 1)).all() and np.isfinite(p).all()
    assert hourly.save(result, tmp_path / "h.json").exists()
