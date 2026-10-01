"""The sky guide's physics and geometry: limiting magnitude, moonlight, coordinates, and what a
person can see from a bright city versus a dark desert."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from skytrust import astro, sky
from skytrust.config import Site
from skytrust.skycatalog import constellation_of

LA = Site("LA", "Los Angeles", 34.0522, -118.2437, 96.0, "city", "America/Los_Angeles")
DV = Site("DV", "Death Valley", 36.2396, -116.8113, -70.0, "desert", "America/Los_Angeles")


def test_limiting_magnitude_formula_and_tiers():
    # Schaefer/Carlin: a natural 22.0 sky gives ~6.6 to a typical observer; brighter skies less
    assert float(sky.limiting_magnitude(22.0)) == pytest.approx(6.62, abs=0.01)
    assert float(sky.limiting_magnitude(18.0)) == pytest.approx(3.97, abs=0.02)
    assert np.all(np.diff(sky.limiting_magnitude(np.linspace(16, 22, 13))) > 0)
    assert sky.darkness(21.98).key == "very-dark" and sky.darkness(17.29).key == "city"
    assert sky.darkness(20.5).key == "dark" and sky.darkness(19.5).key == "suburban"
    assert float(sky.times_natural(22.0)) == pytest.approx(1.0)
    assert float(sky.times_natural(19.5)) == pytest.approx(10.0)
    assert sky.brightness_words(76.4) == "76× brighter than a natural sky"


def test_unit_conversions_round_trip():
    assert float(sky.nl_to_mag(sky.mag_to_nl(21.3))) == pytest.approx(21.3)
    # no Moon, zenith: the formula returns the zenith brightness unchanged (airmass 1)
    assert float(sky.sky_brightness(90.0, 20.5)) == pytest.approx(20.5)
    # towards the horizon the dark sky brightens (van Rhijn), so the number goes down
    assert float(sky.sky_brightness(15.0, 21.5)) < 21.5


def test_moonlight_is_brightest_near_the_moon_and_scales_with_phase():
    seps = np.array([5.0, 30.0, 60.0, 90.0, 120.0])
    b = sky.moon_sky_nl(seps, zenith_deg=40.0, moon_zenith_deg=40.0, phase_angle_deg=10.0)
    assert b[0] > b[1] > b[2] > b[3]  # falls away from the Moon...
    assert b[4] / b[3] < 1.3  # ...and roughly levels off beyond 90° (Rayleigh minimum)
    full = sky.moon_sky_nl(60.0, 40.0, 40.0, 0.0)
    crescent = sky.moon_sky_nl(60.0, 40.0, 40.0, 120.0)
    assert full > 10 * crescent
    # a full Moon high up turns a dark site into a bright one overhead
    moon = {"alt": 60.0, "phase_angle": 5.0, "sep": 30.0}
    assert float(sky.sky_brightness(90.0, 21.9, moon)) < 19.5


def test_galactic_frame_matches_the_iau_definition():
    ra, dec = sky.galactic_to_equatorial(np.array([0.0, 0.0]), np.array([0.0, 90.0]))
    assert ra[0] == pytest.approx(266.405, abs=0.01) and dec[0] == pytest.approx(-28.936, abs=0.01)
    assert ra[1] == pytest.approx(192.859, abs=0.01) and dec[1] == pytest.approx(27.128, abs=0.01)


def test_fast_altaz_agrees_with_skyfield():
    from skyfield.api import Star

    times = pd.date_range("2026-10-01 03:00", periods=4, freq="2h", tz="UTC")
    ra, dec = np.array([279.2341, 101.2885, 37.9461]), np.array([38.783, -16.7131, 89.2641])
    alt, az = sky.Observer(LA).altaz(ra, dec, times)
    eph = astro._ephemeris()
    obs = eph["earth"] + astro._topos(LA)
    for k in range(3):
        star = Star(ra_hours=ra[k] / 15, dec_degrees=dec[k])
        a, z, _ = obs.at(astro._to_skyfield(times)).observe(star).apparent().altaz()
        assert np.allclose(alt[:, k], a.degrees, atol=0.02)
        dz = (az[:, k] - z.degrees + 180) % 360 - 180
        assert np.all(np.abs(dz * np.cos(np.radians(alt[:, k]))) < 0.05)


def test_plain_words_for_directions_and_heights():
    assert sky.compass(0) == "N" and sky.compass(359) == "N" and sky.compass(202) == "SSW"
    assert sky.compass_words("SSW") == "south-southwest" and sky.compass_words("NE") == "northeast"
    assert sky.height_words(80) == "nearly overhead" and sky.height_words(8) == "low"
    assert sky.where_words(30, 135) == "halfway up in the SE"


def test_catalogue_is_complete_and_consistent():
    cat = sky.load_catalog()
    assert len(cat.stars) > 8000 and cat.stars["mag"].max() <= 6.5
    sirius = cat.stars.loc[cat.stars["mag"].idxmin()]
    assert sirius["name"] == "Sirius" and sirius["con"] == "CMa"
    assert len(cat.constellations) == 88 and len(cat.milky_way) == 5
    d = cat.dsos
    assert constellation_of(d["ra"], d["dec"]) == d["con"].tolist()
    # wide clusters are judged by their stars, compact ones as a glow
    pleiades = d[d["name"] == "Pleiades"].iloc[0]
    assert not pleiades["extended"] and pleiades["vis_mag"] > pleiades["mag"]
    assert bool(d[d["name"] == "Andromeda Galaxy"].iloc[0]["extended"])


@pytest.mark.slow
def test_city_versus_dark_desert_on_one_night():
    night = dt.date(2026, 10, 13)  # three days after New Moon: no Moon in the evening
    out = {}
    for site, sqm in [(LA, 17.29), (DV, 21.98)]:
        w = astro.dark_windows(site, night, night).iloc[0]
        out[site.id] = sky.tonight(site, w["dusk_utc"], w["dawn_utc"], sqm)
    la, dv = out["LA"], out["DV"]
    assert dv["stars_visible"]["here"] > 20 * la["stars_visible"]["here"]
    assert dv["milky_way"]["visible"] and not la["milky_way"]["visible"]
    names = {i.name: i for i in dv["items"]}
    assert names["Saturn"].visible and names["Andromeda Galaxy"].visible
    assert not {i.name: i for i in la["items"]}["Andromeda Galaxy"].visible
    # the Milky Way's arc is described in compass words and named constellations
    band = dv["milky_way"]["band"]
    assert len(band["ends"]) == 2 and "Cygnus" in band["through"]
