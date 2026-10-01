"""Statewide verification network: the station-selection rule (offline)."""

from __future__ import annotations

import datetime as dt

import pytest

from skytrust import network
from skytrust.network import Station

CFG = {"min_per_region": 2, "stations_per_quota": 6, "km_per_m": 0.1, "exclude": ["ISL"]}


def st_(sid, lat, lon, region="AAA", elev=0.0, begin="2000-01-01", online=True):
    return Station(sid, sid, lat, lon, elev, region, "County", "America/Los_Angeles", begin, online)


def test_quota_rounds_halves_up_and_has_a_floor():
    stations = [st_(f"A{i}", 34, -118, "AAA") for i in range(27)]  # 27 / 6 = 4.5 -> 5
    stations += [st_(f"B{i}", 34, -118, "BBB") for i in range(3)]  # 0.5 -> floor of 2
    stations += [st_("C0", 34, -118, "CCC", online=False)]  # offline stations don't count
    assert network.quotas(stations, CFG) == {"AAA": 5, "BBB": 2}


def test_candidates_need_full_history_mainland_and_online():
    start = dt.date(2024, 1, 1)
    pool = network.candidates(
        [
            st_("OK", 34, -118),
            st_("NEW", 34, -118, begin="2025-01-23"),  # archive starts after the history start
            st_("OFF", 34, -118, online=False),
            st_("ISL", 33.4, -118.4),  # excluded island
            st_("NOWFO", 34, -118, region=""),
        ],
        CFG,
        start,
    )
    assert [s.id for s in pool] == ["OK"]


def test_design_distance_counts_height():
    a, b = st_("A", 34.0, -118.0, elev=0), st_("B", 34.0, -118.0, elev=2000)
    assert network.design_distance_km(a, b, 0.1) == pytest.approx(200.0)
    # one degree of latitude is ~111 km on the ground
    c = st_("C", 35.0, -118.0)
    assert network.design_distance_km(a, c, 0.1) == pytest.approx(111.2, abs=0.2)


def test_maximin_picks_the_farthest_station_and_respects_quotas():
    seed = st_("SEED", 34.0, -118.0, "AAA")
    pool = [
        st_("NEAR", 34.1, -118.0, "AAA"),
        st_("FAR", 36.0, -118.0, "AAA"),
        st_("MID", 35.0, -118.0, "AAA"),
        st_("OTHER", 34.0, -116.0, "BBB"),
        st_("OTHER2", 34.0, -115.0, "BBB"),
    ]
    chosen = network.select(pool, [seed], {"AAA": 3, "BBB": 1}, 0.1)
    # OTHER2 is 3° of longitude away (276 km at 34°N), then FAR (222 km north), then MID
    # (111 km from both SEED and FAR) beats NEAR (11 km). BBB's quota of 1 is used by OTHER2.
    assert [s.id for s in chosen] == ["SEED", "OTHER2", "FAR", "MID"]


def test_station_failing_coverage_is_replaced_by_the_next_farthest():
    seed = st_("SEED", 34.0, -118.0)
    pool = [st_("FAR", 36.0, -118.0), st_("MID", 35.0, -118.0), st_("NEAR", 34.2, -118.0)]
    asked = []

    def passes(s):
        asked.append(s.id)
        return s.id != "FAR"

    chosen = network.select(pool, [seed], {"AAA": 2}, 0.1, passes)
    assert [s.id for s in chosen] == ["SEED", "MID"]
    assert asked == ["FAR", "MID"]  # the check runs only on stations about to be picked


def test_selection_is_deterministic_on_ties():
    seed = st_("SEED", 34.0, -118.0)
    pool = [st_("ZZZ", 35.0, -118.0), st_("AAA", 33.0, -118.0)]  # same distance from the seed
    chosen = network.select(pool, [seed], {"AAA": 2}, 0.1)
    assert chosen[1].id == "AAA"
