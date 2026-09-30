from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from skytrust.labels import build_labels
from skytrust.nightly import NightRules

RULES = NightRules(clear_threshold=0.20, min_run_hours=3, max_missing_frac=0.25)
N1, N2, N3 = dt.date(2025, 1, 1), dt.date(2025, 1, 2), dt.date(2025, 1, 3)


def setup(n_hours: int = 4):
    """Two dark nights of `n_hours` each, plus N3 with no darkness."""
    hours, rows = [], []
    for i, night in enumerate([N1, N2]):
        start = pd.Timestamp("2025-01-02 03:00", tz="UTC") + pd.Timedelta(days=i)
        hrs = pd.date_range(start, periods=n_hours, freq="h")
        hours.append(pd.DataFrame({"night_date": night, "hour": hrs}))
        rows.append({"dusk_utc": hrs[0], "dawn_utc": hrs[-1]})
    rows.append({"dusk_utc": pd.NaT, "dawn_utc": pd.NaT})
    windows = pd.DataFrame(rows, index=pd.Index([N1, N2, N3], name="night_date"))
    windows = windows.astype("datetime64[ns, UTC]")
    night_hours = pd.concat(hours, ignore_index=True)
    return windows, night_hours, pd.DatetimeIndex(night_hours["hour"])


def series(idx, values):
    return pd.Series(values, index=idx, dtype=float)


def test_primary_is_per_hour_max_so_era5_catches_cirrus():
    windows, nh, idx = setup()
    asos = series(idx, [0.0] * 8)  # ASOS: clear every hour (can't see cirrus)
    era5 = series(idx, [0.0, 0.0, 0.0, 0.0] + [0.0, 0.9, 0.0, 0.0])  # cirrus on night 2
    out = build_labels(windows, nh, {"asos": asos, "asos_max": asos, "era5": era5}, RULES)
    assert out.loc[N1, "usable_primary"] and out.loc[N1, "usable_asos"]
    assert out.loc[N2, "usable_asos"]  # ASOS alone says usable...
    assert not out.loc[N2, "usable_primary"]  # ...but max(asos, era5) breaks the run
    assert not out.loc[N2, "usable_era5"]


def test_single_source_hour_uses_the_other_and_is_flagged():
    windows, nh, idx = setup()
    asos = series(idx, [0.0, np.nan, 0.0, 0.0] + [0.0] * 4)  # 1 of 4 missing = 25 %, allowed
    era5 = series(idx, [0.0] * 8)
    out = build_labels(windows, nh, {"asos": asos, "asos_max": asos, "era5": era5}, RULES)
    assert out.loc[N1, "n_single_source_hours"] == 1
    assert out.loc[N1, "truth_longest_clear_run"] == 4  # the gap was filled by ERA5
    assert out.loc[N1, "usable_primary"]
    assert not out.loc[N1, "usable_asos"]  # ASOS-only: the gap breaks its run (1 + 2)


def test_missing_coverage_exclusion_and_reason():
    windows, nh, idx = setup()
    asos = series(idx, [0.0, np.nan, np.nan, 0.0] + [0.0] * 4)  # night 1: 50 % ASOS missing
    era5 = series(idx, [0.0] * 4 + [np.nan] * 4)  # night 2: ERA5 entirely missing (lag)
    out = build_labels(windows, nh, {"asos": asos, "asos_max": asos, "era5": era5}, RULES)
    assert pd.isna(out.loc[N1, "usable_primary"])
    assert out.loc[N1, "exclusion_primary"] == "asos_missing"
    assert pd.isna(out.loc[N1, "usable_asos"]) and out.loc[N1, "usable_era5"]
    assert out.loc[N2, "exclusion_primary"] == "era5_missing"
    assert out.loc[N2, "usable_asos"] and pd.isna(out.exclusion_asos[N2])
    assert out.loc[N3, "exclusion_primary"] == "no_darkness"
    assert pd.isna(out.loc[N3, "usable_primary"]) and pd.isna(out.loc[N3, "usable_asos"])


def test_asos_max_sensitivity_column_is_independent():
    windows, nh, idx = setup()
    near = series(idx, [0.0] * 8)
    mx = series(idx, [0.0, 0.4375, 0.0, 0.0] + [0.0] * 4)
    out = build_labels(
        windows, nh, {"asos": near, "asos_max": mx, "era5": series(idx, [0.0] * 8)}, RULES
    )
    assert out.loc[N1, "usable_asos"] and not out.loc[N1, "usable_asos_max"]


def test_era5_layer_means_are_stored_when_available():
    windows, nh, idx = setup()
    zeros = series(idx, [0.0] * 8)
    high = series(idx, [0.5] * 4 + [1.0] * 4)
    hourly = {"asos": zeros, "asos_max": zeros, "era5": high, "era5_high": high}
    out = build_labels(windows, nh, hourly, RULES)
    assert out.loc[N1, "era5_high_mean_cover"] == 0.5
    assert out.loc[N2, "era5_high_mean_cover"] == 1.0
    assert "era5_low_mean_cover" not in out  # absent input -> no column


def test_goes_label_follows_the_same_rules_independently():
    windows, nh, idx = setup()
    zeros = series(idx, [0.0] * 8)
    goes = series(idx, [0.0] * 4 + [0.0, 0.6, 0.0, np.nan])  # night 2: a cloudy hour + a gap
    hourly = {"asos": zeros, "asos_max": zeros, "era5": zeros, "goes": goes}
    out = build_labels(windows, nh, hourly, RULES)
    assert out.loc[N1, "usable_goes"] and not out.loc[N2, "usable_goes"]
    assert out.loc[N2, "goes_missing_frac"] == 0.25
    assert out.loc[N2, "usable_primary"]  # GOES never changes the primary label
    assert pd.isna(out.loc[N3, "usable_goes"]) and out.loc[N3, "exclusion_goes"] == "no_darkness"


def test_goes_label_is_excluded_when_the_satellite_data_is_missing():
    windows, nh, idx = setup()
    zeros = series(idx, [0.0] * 8)
    goes = series(idx, [0.0] * 4 + [0.0, np.nan, np.nan, 0.0])  # night 2: 50 % missing
    out = build_labels(
        windows, nh, {"asos": zeros, "asos_max": zeros, "era5": zeros, "goes": goes}, RULES
    )
    assert out.loc[N1, "usable_goes"] and pd.isna(out.loc[N1, "exclusion_goes"])
    assert pd.isna(out.loc[N2, "usable_goes"]) and out.loc[N2, "exclusion_goes"] == "goes_missing"


def test_no_goes_input_gives_an_all_missing_goes_label():
    windows, nh, idx = setup()
    zeros = series(idx, [0.0] * 8)
    out = build_labels(windows, nh, {"asos": zeros, "asos_max": zeros, "era5": zeros}, RULES)
    assert out["usable_goes"].isna().all() and str(out["usable_goes"].dtype) == "boolean"
    assert (out["exclusion_goes"].dropna() != "").all()
