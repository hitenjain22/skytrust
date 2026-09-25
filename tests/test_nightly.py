"""Usable-night rule (SPEC 12.1) and feature arithmetic on synthetic series."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from skytrust.nightly import NightRules, is_usable, longest_clear_run, summarize_nights

RULES = NightRules(clear_threshold=0.20, min_run_hours=3, max_missing_frac=0.25)
NAN = np.nan


@pytest.mark.parametrize(
    ("cover", "expected"),
    [
        ([0.0, 0.1, 0.2, 0.9], True),  # exactly 3 consecutive clear -> usable
        ([0.0, 0.1, 0.9, 0.0, 0.1], False),  # two runs of 2 -> not usable
        ([0.1, 0.1, NAN, 0.1, 0.1, 0.9, 0.9, 0.9], False),  # a missing hour breaks the run
        ([0.20, 0.20, 0.20, 0.5], True),  # cover exactly at the threshold counts as clear
        ([0.21, 0.21, 0.21, 0.21], False),
    ],
)
def test_usable_night_rule(cover, expected):
    assert is_usable(cover, RULES) is expected


def test_empty_dark_window_is_excluded_not_false():
    assert is_usable([], RULES) is None


def test_too_much_missing_is_excluded():
    # 2 of 7 hours missing = 28.6 % > 25 % -> excluded even though a 3-hour run exists
    assert is_usable([0.0, 0.0, 0.0, NAN, NAN, 0.9, 0.9], RULES) is None
    # 1 of 7 = 14 % -> allowed
    assert is_usable([0.0, 0.0, 0.0, NAN, 0.9, 0.9, 0.9], RULES) is True


def test_float_threshold_edge():
    # 20 / 100 must count as clear even with float noise.
    assert longest_clear_run(np.array([20, 20, 20]) / 100, 0.20) == 3
    assert longest_clear_run([0.2 + 1e-12] * 3, 0.20) == 3


def night_frame(nights: dict[dt.date, int], start: str = "2025-01-01 03:00") -> pd.DataFrame:
    """Synthetic (night_date, hour) table: consecutive nights with the given dark-hour counts."""
    rows, t = [], pd.Timestamp(start, tz="UTC")
    for night, n in nights.items():
        rows += [{"night_date": night, "hour": t + pd.Timedelta(hours=i)} for i in range(n)]
        t += pd.Timedelta(days=1)
    return pd.DataFrame(rows)


def test_summary_known_values():
    d1, d2 = dt.date(2025, 1, 1), dt.date(2025, 1, 2)
    hours = night_frame({d1: 5, d2: 4})
    values = [0.0, 0.1, 0.5, 0.0, 0.0] + [1.0, 1.0, 0.0, 1.0]
    cover = pd.Series(values, index=pd.DatetimeIndex(hours["hour"]))
    out = summarize_nights(hours, cover, RULES)
    n1 = out.loc[d1]
    assert n1.n_hours == 5 and n1.missing_frac == 0
    assert n1.frac_clear == pytest.approx(4 / 5)
    assert n1.longest_clear_run == 2
    assert n1.longest_clear_run_frac == pytest.approx(2 / 5)
    assert n1.mean_cover == pytest.approx(0.6 / 5)
    assert not n1.usable
    assert out.loc[d2].frac_clear == pytest.approx(1 / 4)


def test_summary_missing_rule_sets_nan():
    d1 = dt.date(2025, 1, 1)
    hours = night_frame({d1: 4})
    cover = pd.Series(
        [0.0, 0.0, 0.0], index=pd.DatetimeIndex(hours["hour"][:3])
    )  # last hour absent
    ok = summarize_nights(hours, cover, RULES).loc[d1]
    assert ok.missing_frac == 0.25 and ok.usable  # exactly 25 % is still allowed
    cover2 = cover.iloc[:2]  # 50 % missing
    bad = summarize_nights(hours, cover2, RULES).loc[d1]
    assert bad.missing_frac == 0.5
    assert np.isnan(bad.frac_clear) and np.isnan(bad.mean_cover) and pd.isna(bad.usable)


def test_vectorised_summary_matches_reference_loop():
    """Property check: on random data (with gaps) the fast groupby implementation agrees with
    the simple loop for every night."""
    rng = np.random.default_rng(0)
    nights = {
        dt.date(2025, 1, 1) + dt.timedelta(days=i): int(rng.integers(4, 12)) for i in range(200)
    }
    hours = night_frame(nights)
    values = rng.choice(
        [0.0, 0.1, 0.2, 0.3, 0.8, NAN], size=len(hours), p=[0.25, 0.2, 0.1, 0.15, 0.25, 0.05]
    )
    cover = pd.Series(values, index=pd.DatetimeIndex(hours["hour"]))
    out = summarize_nights(hours, cover, RULES)
    for night, group in hours.groupby("night_date"):
        seq = cover.loc[pd.DatetimeIndex(group["hour"])].to_numpy()
        expected = is_usable(seq, RULES)
        got = out.loc[night, "usable"]
        assert (pd.isna(got) and expected is None) or bool(got) == expected, night
        if expected is not None:
            assert out.loc[night, "longest_clear_run"] == longest_clear_run(seq, 0.20)
