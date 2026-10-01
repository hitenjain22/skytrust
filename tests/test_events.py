"""Sky events against published values: the IMO 2026 calendar's lunar phases and shower peak
times, Saturn's 2026 opposition, the 2026 lunar eclipses; and the meteor-rate formula."""

from __future__ import annotations

import pandas as pd
import pytest

from skytrust import events as ev
from skytrust.config import Site

LA = Site("LA", "Los Angeles", 34.0522, -118.2437, 96.0, "city", "America/Los_Angeles")

# IMO 2026 Meteor Shower Calendar, Table 4 (dates as printed; they follow Central European time)
IMO_NEW = [
    "01-18",
    "02-17",
    "03-19",
    "04-17",
    "05-16",
    "06-15",
    "07-14",
    "08-12",
    "09-11",
    "10-10",
    "11-09",
    "12-09",
]
IMO_FULL = [
    "01-03",
    "02-01",
    "03-03",
    "04-02",
    "05-01",
    "05-31",
    "06-30",
    "07-29",
    "08-28",
    "09-26",
    "10-26",
    "11-24",
    "12-24",
]


def test_lunar_phases_match_the_imo_table():
    found = ev.moon_phase_events(
        pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2027-01-01", tz="UTC")
    )
    dates = {"New Moon": [], "Full Moon": []}
    for e in found:
        dates[e.title].append(e.utc.tz_convert("Europe/Berlin").strftime("%m-%d"))
    assert dates["New Moon"] == IMO_NEW
    assert dates["Full Moon"] == IMO_FULL


@pytest.mark.parametrize(
    ("code", "imo_peak"),
    [
        ("GEM", "2026-12-14 14:00"),
        ("LEO", "2026-11-17 23:45"),
        ("DRA", "2026-10-09 01:00"),
        ("QUA", "2026-01-03 21:00"),
    ],
)
def test_shower_peaks_from_solar_longitude_match_the_imo_times(code, imo_peak):
    shower = next(s for s in ev.load_showers() if s.code == code)
    expect = pd.Timestamp(imo_peak, tz="UTC")
    got = ev.time_of_solar_longitude(shower.peak_sol, expect - pd.Timedelta(days=3))
    assert abs(got - expect) < pd.Timedelta(minutes=30)  # λ⊙ given to 0.01-0.1° ≈ 15 min-2 h


def test_solar_longitude_round_trip():
    t = ev.time_of_solar_longitude(262.2, pd.Timestamp("2026-12-10", tz="UTC"))
    assert float(ev.solar_longitude(pd.DatetimeIndex([t]))[0]) == pytest.approx(262.2, abs=1e-4)


def test_saturn_opposition_2026_and_eclipses():
    found = ev.opposition_events(
        pd.Timestamp("2026-09-20", tz="UTC"), pd.Timestamp("2026-10-20", tz="UTC"), LA
    )
    saturn = next(e for e in found if e.title.startswith("Saturn"))
    # published: 2026 October 4, ~12:00 UTC (definitions differ by tens of minutes)
    assert abs(saturn.utc - pd.Timestamp("2026-10-04 12:00", tz="UTC")) < pd.Timedelta(hours=1)
    ecl = ev.eclipse_events(
        pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-12-31", tz="UTC"), LA
    )
    assert [e.title for e in ecl] == ["Total lunar eclipse", "Partial lunar eclipse"]
    # NASA (Espenak): greatest eclipse 04:13 UTC on the 28th, umbral magnitude 0.9299 (93% of
    # the Moon's diameter in the umbra; the "96%" in news reports is by area)
    assert abs(ecl[1].utc - pd.Timestamp("2026-08-28 04:13", tz="UTC")) < pd.Timedelta(minutes=2)
    assert ecl[1].details["umbral_magnitude"] == pytest.approx(0.930, abs=0.005)


def test_expected_meteor_rate():
    assert float(ev.expected_rate(100, 2.5, 90, 6.5)) == pytest.approx(100)
    assert float(ev.expected_rate(100, 2.5, 30, 6.5)) == pytest.approx(50)
    assert float(ev.expected_rate(100, 2.5, 90, 7.5)) == pytest.approx(100)  # capped at ZHR
    assert float(ev.expected_rate(100, 2.5, 90, 5.5)) == pytest.approx(40)  # one mag: ÷ r
    assert float(ev.expected_rate(100, 2.5, -5, 6.5)) == 0.0


def test_quadrantid_activity_wraps_the_new_year():
    qua = next(s for s in ev.load_showers() if s.code == "QUA")
    start, end = ev.active_window(qua, pd.Timestamp("2027-01-03 20:00", tz="UTC"))
    assert start == pd.Timestamp("2026-12-28", tz="UTC") and end.year == 2027


def test_moon_pairings_on_one_night_are_merged():
    def pairing(name, when, sep):
        view = {"utc": pd.Timestamp(when, tz="UTC"), "alt": 20.0, "az": 90.0}
        return ev._pairing("Moon", name, view["utc"], sep, view, LA)

    merged = ev.merge_moon_pairings(
        [
            pairing("Jupiter", "2026-10-06 11:30", 0.3),
            pairing("Regulus", "2026-10-06 12:45", 0.6),
            pairing("Antares", "2026-10-15 02:00", 0.4),
        ],
        LA,
    )
    titles = sorted(e.title for e in merged)
    assert titles == ["The Moon near Antares", "The Moon near Jupiter and Regulus"]
