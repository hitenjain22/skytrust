"""Every astronomical number the app shows, checked against an independent source or method.

- Planet and Moon positions (Skyfield + the DE421 excerpt in the repo) vs NASA/JPL Horizons
  (DE441), recorded in tests/fixtures/horizons_2026-10-02.json.
- The Moon seen from a place, parallax included (sky.Observer) vs Horizons for that spot.
- Moon illumination (Skyfield) vs Meeus, Astronomical Algorithms ch. 48, and vs Horizons; the
  phase angle that drives the moonlight model vs Horizons.
- Sidereal time, which turns the star field (Skyfield GAST), vs Meeus eq. 12.4 (IAU 1982).
- Dark windows (astro.dark_windows and the app's fast events._dark_windows) vs astral (NOAA's
  algorithm) at ten places from the Mexican border to Oregon.
- Elsewhere: the galactic frame vs the IAU definition, meteor peaks vs the IMO calendar, the 2026
  eclipses vs NASA (tests/test_sky.py, tests/test_events.py).
"""

from __future__ import annotations

import datetime as dt
import json
import math
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
from astral import Observer as AstralObserver
from astral.sun import dawn, dusk

from conftest import FIXTURES
from skytrust import astro, events, sky
from skytrust.config import Site

HORIZONS = json.loads((FIXTURES / "horizons_2026-10-02.json").read_text())
TZ = "America/Los_Angeles"
LA = Site("LA", "Los Angeles", 34.0522, -118.2437, 96.0, "city", TZ)


def _arcsec(ra1, dec1, ra2, dec2) -> float:
    return float(sky.separation_deg(ra1, dec1, ra2, dec2)) * 3600


def test_planets_and_moon_match_jpl_horizons():
    """Apparent geocentric RA/Dec of date from the repo's DE421 excerpt agree with JPL's DE441
    to well under an arcsecond (planets are taken at their system barycentres in DE421, which
    sit within ~0.05" of the planet's centre)."""
    eph, earth = astro._ephemeris(), astro._ephemeris()["earth"]
    for body in HORIZONS["bodies"].values():
        for row in body["rows"]:
            t = astro._to_skyfield(pd.DatetimeIndex([pd.Timestamp(row["utc"])]))
            ra, dec, _ = earth.at(t).observe(eph[body["skyfield_key"]]).apparent().radec("date")
            err = _arcsec(ra._degrees[0], dec.degrees[0], row["ra_deg"], row["dec_deg"])
            assert err < 1.0, (body["skyfield_key"], row["utc"], err)


def test_the_moon_seen_from_los_angeles_matches_horizons():
    """The Moon is close enough for parallax to move it by ~1°; the app's topocentric position
    (Earth shape, the place's elevation) must agree with Horizons for that exact spot."""
    rows = HORIZONS["moon_from_los_angeles"]["rows"]
    obs = sky.Observer(LA)
    times = pd.DatetimeIndex([pd.Timestamp(r["utc"]) for r in rows])
    ra, dec = obs.radec_of_date("moon", times)
    body = obs.body("moon", times)
    for k, r in enumerate(rows):
        assert _arcsec(ra[k], dec[k], r["ra_deg"], r["dec_deg"]) < 2.0
        # Horizons' elevation is airless (no refraction), like the app's
        assert body["alt"][k] == pytest.approx(r["alt_deg"], abs=0.002)
        dz = (body["az"][k] - r["az_deg"] + 180) % 360 - 180
        assert abs(dz) * math.cos(math.radians(r["alt_deg"])) < 0.002


def meeus_moon(jd: float) -> tuple[float, float]:
    """Phase angle (deg) and illuminated fraction from Meeus, Astronomical Algorithms (2nd ed.),
    ch. 48, eqs. 48.4 and 48.1: a short analytic series, independent of any ephemeris."""
    t = (jd - 2451545.0) / 36525
    d = 297.8501921 + 445267.1114034 * t - 0.0018819 * t**2 + t**3 / 545868 - t**4 / 113065000
    m = 357.5291092 + 35999.0502909 * t - 0.0001536 * t**2 + t**3 / 24490000
    mp = 134.9633964 + 477198.8675055 * t + 0.0087414 * t**2 + t**3 / 69699 - t**4 / 14712000
    r = math.radians
    i = (
        180
        - (d % 360)
        - 6.289 * math.sin(r(mp))
        + 2.100 * math.sin(r(m))
        - 1.274 * math.sin(r(2 * d - mp))
        - 0.658 * math.sin(r(2 * d))
        - 0.214 * math.sin(r(2 * mp))
        - 0.110 * math.sin(r(d))
    )
    i = abs((i + 180) % 360 - 180)
    return i, (1 + math.cos(r(i))) / 2


def test_moon_illumination_matches_meeus_and_horizons():
    """Illuminated fraction vs Meeus's short series (independent of any ephemeris) and vs JPL.
    Meeus's eq. 48.4 leaves out the Moon's latitude, so near full Moon its *phase angle* is off
    by up to ~5° (the true angle can't drop below the Moon's ecliptic latitude); the fraction,
    (1 + cos i) / 2, is insensitive to that there, so only the fraction is checked against it."""
    times = pd.date_range("2026-01-01", "2026-12-31", freq="61h", tz="UTC")
    moon = sky.Observer(LA).moon(times)
    jds = astro._to_skyfield(times).tt  # Meeus' series is in dynamical time
    for k, jd in enumerate(jds):
        _, frac = meeus_moon(float(jd))
        assert moon["illum"][k] == pytest.approx(frac, abs=0.006)


def test_moon_phase_angle_matches_horizons():
    """The phase angle drives the moonlight model. The app uses 180° - elongation, which differs
    from the true Sun-Moon-Earth angle by at most (Moon distance / Sun distance) rad ≈ 0.15°."""
    rows = HORIZONS["moon_phase_2026"]["rows"]
    times = pd.DatetimeIndex([pd.Timestamp(r["utc"]) for r in rows])
    moon = sky.Observer(LA).moon(times)
    for k, r in enumerate(rows):
        assert abs(float(moon["phase_angle"][k])) == pytest.approx(r["phase_angle_deg"], abs=0.2)
        assert moon["illum"][k] == pytest.approx(r["illuminated_pct"] / 100, abs=0.002)


def test_sidereal_time_matches_the_iau_formula():
    """Local sidereal time (what turns the star field) vs Meeus eq. 12.4 for mean sidereal time.
    They differ only by the equation of the equinoxes (< 0.005°) and UT1 - UTC (< 0.004°)."""
    times = pd.date_range("2026-01-01", "2026-12-31", freq="7D", tz="UTC")
    lst = sky.Observer(LA).lst_deg(times)
    for k, t in enumerate(times):
        jd = t.to_julian_date()
        T = (jd - 2451545.0) / 36525
        gmst = (
            280.46061837 + 360.98564736629 * (jd - 2451545.0) + 0.000387933 * T**2
            - T**3 / 38710000
        )  # fmt: skip
        expected = (gmst + LA.lon) % 360
        assert ((lst[k] - expected + 180) % 360 - 180) == pytest.approx(0, abs=0.012)


PLACES = [  # south to north, coast to desert and mountains (town centres from the gazetteer)
    Site("SD", "San Diego", 32.7157, -117.1611, 20.0, "", TZ),
    Site("EC", "El Centro", 32.792, -115.5631, -12.0, "", TZ),
    Site("LA", "Los Angeles", 34.0522, -118.2437, 96.0, "", TZ),
    Site("ND", "Needles", 34.8481, -114.6142, 150.0, "", TZ),
    Site("BP", "Bishop", 37.3635, -118.3951, 1260.0, "", TZ),
    Site("SF", "San Francisco", 37.7749, -122.4194, 20.0, "", TZ),
    Site("SA", "Sacramento", 38.5816, -121.4944, 10.0, "", TZ),
    Site("RD", "Redding", 40.5865, -122.3917, 170.0, "", TZ),
    Site("AL", "Alturas", 41.4871, -120.5424, 1330.0, "", TZ),
    Site("CC", "Crescent City", 41.7558, -124.2026, 10.0, "", TZ),
]
DATES = [dt.date(2026, m, d) for m, d in [(1, 15), (3, 20), (6, 21), (9, 22), (11, 1), (12, 21)]]


@pytest.mark.slow  # ~40 s: 60 place-nights through three algorithms
def test_dark_windows_across_california_match_astral():
    """Astronomical dusk/dawn (sun 18° below the horizon) at ten places from the Mexican border
    to Oregon on six dates (solstices, equinoxes, DST change), against astral's NOAA-based
    algorithm, within 2 minutes; the app's fast 10-minute-grid version within 1 minute of the
    exact one."""
    for site in PLACES:
        w = astro.dark_windows(site, DATES[0], DATES[-1], -18.0).dropna()
        for day in DATES:
            obs = AstralObserver(latitude=site.lat, longitude=site.lon, elevation=0)
            local = ZoneInfo(TZ)  # a night is keyed by the local date of its evening
            ref_dusk = pd.Timestamp(dusk(obs, day, depression=18, tzinfo=local)).tz_convert("UTC")
            ref_dawn = pd.Timestamp(
                dawn(obs, day + dt.timedelta(days=1), depression=18, tzinfo=local)
            ).tz_convert("UTC")
            row = w.loc[pd.Timestamp(day).date()] if pd.Timestamp(day).date() in w.index else None
            assert row is not None, (site.name, day)
            assert abs((row["dusk_utc"] - ref_dusk).total_seconds()) < 120, (site.name, day)
            assert abs((row["dawn_utc"] - ref_dawn).total_seconds()) < 120, (site.name, day)
            fast = events._dark_windows(site, day, -18.0).loc[day]  # (the night before, too)
            assert abs((fast["dusk_utc"] - row["dusk_utc"]).total_seconds()) < 60
            assert abs((fast["dawn_utc"] - row["dawn_utc"]).total_seconds()) < 60


def test_meteor_rate_follows_the_imo_formula_by_hand():
    """ZHR 150 (Geminids), radiant 40° up, sky limiting magnitude 5.5, population index 2.6:
    rate = 150 · sin 40° / 2.6^(6.5 - 5.5) = 37.1 an hour (IMO's definition)."""
    rate = float(events.expected_rate(150.0, 2.6, 40.0, 5.5))
    assert rate == pytest.approx(150 * math.sin(math.radians(40)) / 2.6, rel=1e-12)
    assert rate == pytest.approx(37.08, abs=0.01)
    # a perfect sky with the radiant overhead gives the ZHR itself, and never more
    assert float(events.expected_rate(150.0, 2.6, 90.0, 7.2)) == pytest.approx(150.0)


def test_light_pollution_tier_boundaries_are_the_bortle_sqm_limits():
    """Plain-word tiers start exactly at the Bortle classes' SQM boundaries, and '× natural'
    is the flux ratio 10^(0.4 Δmag) (5 magnitudes = 100×)."""
    assert [t[0] for t in sky.TIERS[:4]] == [21.30, 20.30, 19.25, 18.00]
    assert float(sky.times_natural(sky.NATURAL_SQM - 5)) == pytest.approx(100.0)
    assert np.allclose(sky.nl_to_mag(sky.mag_to_nl([17.0, 19.0, 21.5])), [17.0, 19.0, 21.5])
