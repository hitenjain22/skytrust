from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from skytrust.config import ModelSpec
from skytrust.features import build_features
from skytrust.nightly import NightRules

RULES = NightRules(clear_threshold=0.20, min_run_hours=3, max_missing_frac=0.25)
GFS = ModelSpec("gfs", "gfs_global", "GFS", (1, 2))
HRRR = ModelSpec("hrrr", "ncep_hrrr_conus", "HRRR", (1,))
N1 = dt.date(2025, 1, 1)


def setup():
    hrs = pd.date_range("2025-01-02 03:00", periods=6, freq="h", tz="UTC")
    nh = pd.DataFrame({"night_date": N1, "hour": hrs})
    return nh, pd.Index([N1], name="night_date"), hrs


def test_known_feature_values_from_synthetic_series():
    nh, nights, hrs = setup()
    gfs = pd.DataFrame({"lead1": [0.0, 0.1, 0.2, 0.9, 0.0, 0.0], "lead2": [1.0] * 6}, index=hrs)
    hrrr = pd.DataFrame({"lead1": [0.0] * 6}, index=hrs)
    out = build_features(nh, nights, {"gfs": gfs, "hrrr": hrrr}, (GFS, HRRR), [1, 2], RULES)
    r1 = out.loc[(N1, 1)]
    assert r1.gfs_frac_clear == pytest.approx(5 / 6)
    assert r1.gfs_longest_clear_run == 3
    assert r1.gfs_longest_clear_run_frac == pytest.approx(0.5)
    assert r1.gfs_mean_cover == pytest.approx(1.2 / 6)
    assert r1.gfs_pred_usable and r1.hrrr_pred_usable
    assert r1.n_models_available == 2
    assert r1.spread_frac_clear == pytest.approx(np.std([5 / 6, 1.0]))
    r2 = out.loc[(N1, 2)]
    assert r2.gfs_frac_clear == 0 and not r2.gfs_pred_usable


def test_lead_a_model_does_not_have_is_nan_with_full_missing():
    nh, nights, hrs = setup()
    gfs = pd.DataFrame({"lead1": [0.0] * 6, "lead2": [0.0] * 6}, index=hrs)
    hrrr = pd.DataFrame({"lead1": [0.0] * 6}, index=hrs)
    out = build_features(nh, nights, {"gfs": gfs, "hrrr": hrrr}, (GFS, HRRR), [1, 2], RULES)
    r2 = out.loc[(N1, 2)]
    assert np.isnan(r2.hrrr_frac_clear) and r2.hrrr_missing_frac == 1.0
    assert pd.isna(r2.hrrr_pred_usable)
    assert r2.n_models_available == 1
    assert np.isnan(r2.spread_frac_clear)  # spread undefined with a single model


def test_more_than_25pct_missing_sets_features_nan():
    nh, nights, hrs = setup()
    gfs = pd.DataFrame({"lead1": [0.0, np.nan, np.nan, 0.0, 0.0, 0.0]}, index=hrs)  # 33 %
    out = build_features(nh, nights, {"gfs": gfs}, (GFS,), [1], RULES)
    r = out.loc[(N1, 1)]
    assert r.gfs_missing_frac == pytest.approx(2 / 6)
    assert np.isnan(r.gfs_frac_clear) and np.isnan(r.gfs_mean_cover) and pd.isna(r.gfs_pred_usable)
    assert r.n_models_available == 0


def test_model_with_no_cached_data_at_all():
    nh, nights, _ = setup()
    out = build_features(nh, nights, {}, (GFS,), [1], RULES)
    assert out.loc[(N1, 1)].gfs_missing_frac == 1.0
