"""astro.py tests. Independent reference: the `astral` library (test-only dependency) with an
18 degree depression. Astral uses its own NOAA-style solar formulas, not Skyfield/JPL, so
agreement within 2 minutes is a genuine cross-check."""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
from astral import Observer
from astral import moon as astral_moon
from astral.sun import dawn, dusk

from skytrust import astro
from skytrust.config import Site

LA = ZoneInfo("America/Los_Angeles")
SAC = Site("SAC", "Sacramento Exec", 38.5069, -121.495, 8.0, "valley", "America/Los_Angeles")
ONE_NIGHT = dt.timedelta(days=1)


@pytest.mark.parametrize("night", [dt.date(2025, 6, 21), dt.date(2025, 12, 21)])
def test_dusk_dawn_match_astral_within_2_minutes(night):
    w = astro.dark_windows(SAC, night, night).loc[night]
    obs = Observer(latitude=SAC.lat, longitude=SAC.lon, elevation=0)
    ref_dusk = dusk(obs, night, tzinfo=LA, depression=18)
    ref_dawn = dawn(obs, night + ONE_NIGHT, tzinfo=LA, depression=18)
    assert abs(w.dusk_utc - pd.Timestamp(ref_dusk)) < pd.Timedelta(minutes=2)
    assert abs(w.dawn_utc - pd.Timestamp(ref_dawn)) < pd.Timedelta(minutes=2)


def test_night_is_keyed_by_local_evening_date():
    night = dt.date(2025, 1, 10)
    w = astro.dark_windows(SAC, night, night).loc[night]
    assert w.dusk_utc.tz_convert(LA).date() == night  # dusk ~18:30 PST on Jan 10
    assert w.dawn_utc.tz_convert(LA).date() == night + ONE_NIGHT
    assert w.dusk_utc.date() == night + ONE_NIGHT  # but already Jan 11 in UTC


# Spring forward happens at 02:00 on 2025-03-09 (inside the night of 03-08); fall back at
# 02:00 on 2025-11-02 (inside the night of 11-01). Test both nights on each side.
@pytest.mark.parametrize(
    "night",
    [dt.date(2025, 3, 8), dt.date(2025, 3, 9), dt.date(2025, 11, 1), dt.date(2025, 11, 2)],
)
def test_dst_nights_have_no_duplicated_or_missing_hours(night):
    w = astro.dark_windows(SAC, night, night).loc[night]
    hours = astro.dark_hours(w.dusk_utc, w.dawn_utc)
    assert str(hours.tz) == "UTC"
    assert hours.is_unique
    assert (np.diff(hours.asi8) == 3600 * 10**9).all()  # exactly 1 h apart, no gaps
    assert hours[0] >= w.dusk_utc and hours[-1] <= w.dawn_utc
    assert hours[0] - w.dusk_utc < pd.Timedelta(hours=1)
    assert w.dawn_utc - hours[-1] < pd.Timedelta(hours=1)


def test_spring_forward_night_skips_2am_local():
    night = dt.date(2025, 3, 8)
    w = astro.dark_windows(SAC, night, night).loc[night]
    local = astro.dark_hours(w.dusk_utc, w.dawn_utc).tz_convert(LA)
    assert 2 not in local.hour  # 02:00 local doesn't exist that night
    assert {1, 3}.issubset(set(local.hour))


def test_dark_hours_includes_exact_endpoints():
    dusk_t = pd.Timestamp("2025-01-11 03:00", tz="UTC")
    dawn_t = pd.Timestamp("2025-01-11 06:00", tz="UTC")
    assert list(astro.dark_hours(dusk_t, dawn_t).hour) == [3, 4, 5, 6]


def test_empty_dark_window_gives_no_hours():
    assert len(astro.dark_hours(pd.NaT, pd.NaT)) == 0


def test_night_without_darkness_is_nat_and_logged(caplog):
    # At 65 N there is no astronomical darkness around the June solstice.
    north = Site("FAI", "Fairbanks", 64.8, -147.7, 130.0, "test", "America/Anchorage")
    night = dt.date(2025, 6, 21)
    w = astro.dark_windows(north, night, night)
    assert w.loc[night].isna().all()
    assert "no astronomical darkness" in caplog.text


def test_moon_illumination_full_and_new():
    full = pd.DatetimeIndex([pd.Timestamp("2025-01-13 22:27", tz="UTC")])
    new = pd.DatetimeIndex([pd.Timestamp("2025-01-29 12:36", tz="UTC")])
    assert astro.moon_illumination(full)[0] > 0.98
    assert astro.moon_illumination(new)[0] < 0.02
    # Independent check that these really are full/new moon dates (astral: 0=new, ~14=full).
    assert 13 <= astral_moon.phase(dt.date(2025, 1, 13)) <= 15
    assert (
        astral_moon.phase(dt.date(2025, 1, 29)) < 1 or astral_moon.phase(dt.date(2025, 1, 29)) > 27
    )


def test_naive_times_rejected():
    with pytest.raises(ValueError):
        astro.moon_illumination(pd.DatetimeIndex([pd.Timestamp("2025-01-13 22:00")]))


def test_night_table_shape_and_ranges():
    tab = astro.night_table(SAC, dt.date(2025, 1, 1), dt.date(2025, 1, 14))
    assert len(tab) == 14
    assert tab["dark_hours"].between(4, 13).all()  # plausible for California latitudes
    assert (tab["moon_free_dark_hours"] <= tab["dark_hours"]).all()
    assert tab["moon_illum_mean"].between(0, 1).all()
    assert str(tab["dusk_utc"].dt.tz) == "UTC"
    # Around the Jan 13 full moon the Moon is up nearly all night; around new moon it isn't.
    assert tab.loc[dt.date(2025, 1, 13), "moon_free_dark_hours"] <= 2
    assert tab.loc[dt.date(2025, 1, 13), "moon_illum_mean"] > 0.95


def test_moon_events_alternate_rise_and_set():
    ev = astro.moon_events(
        SAC, pd.Timestamp("2025-01-10", tz="UTC"), pd.Timestamp("2025-01-13", tz="UTC")
    )
    assert len(ev) >= 4
    assert set(ev["event"]) == {"rise", "set"}
    assert (ev["event"].to_numpy()[1:] != ev["event"].to_numpy()[:-1]).all()
