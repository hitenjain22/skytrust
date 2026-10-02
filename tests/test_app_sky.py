"""The sky guide's and events page's view logic: wording, dates of night events, the chart."""

from __future__ import annotations

import re
import sys

import numpy as np
import pandas as pd
import pytest

from skytrust import sky
from skytrust.config import REPO_ROOT, Site
from skytrust.events import Event

sys.path.insert(0, str(REPO_ROOT / "app"))

from views import events_view, lookup, skychart, skylive  # noqa: E402

TZ = "America/Los_Angeles"
LA = Site("los-angeles", "Los Angeles", 34.0522, -118.2437, 96.0, "Southern California", TZ)
DV = Site("death-valley", "Death Valley", 36.2396, -116.8113, -70.0, "Mojave Desert", TZ)
DUSK = pd.Timestamp("2026-10-01 03:00", tz="UTC")  # 8 PM PDT
DAWN = pd.Timestamp("2026-10-01 12:20", tz="UTC")  # 5:20 AM PDT


def test_capitalisation_keeps_compass_points_and_names():
    assert lookup.cap("high in the SE") == "High in the SE"
    assert lookup.cap("vega + Deneb + Altair") == "Vega + Deneb + Altair"
    assert lookup.uncap("The Milky Way is hidden") == "the Milky Way is hidden"


def test_clock_rounds_to_five_minutes():
    assert lookup.clock(pd.Timestamp("2026-10-01 04:01:51", tz="UTC"), TZ) == "9 PM"
    assert lookup.clock(pd.Timestamp("2026-10-01 07:38", tz="UTC"), TZ) == "12:40 AM"


def item(up_from, up_until, best) -> sky.SkyItem:
    it = sky.SkyItem("Saturn", "planet", 0.3, 0.3, False, np.array([10.0]), np.array([180.0]))
    it.up_from, it.up_until, it.best_utc = up_from, up_until, best
    return it


def test_when_words():
    best = pd.Timestamp("2026-10-01 07:40", tz="UTC")
    assert (
        lookup.when_words(item(DUSK, DAWN, best), DUSK, DAWN, TZ)
        == "All night, highest at 12:40 AM"
    )
    late = pd.Timestamp("2026-10-01 10:25", tz="UTC")
    assert lookup.when_words(item(late, DAWN, DAWN), DUSK, DAWN, TZ) == "From 3:25 AM until dawn"
    early_end = pd.Timestamp("2026-10-01 05:00", tz="UTC")
    assert (
        lookup.when_words(item(DUSK, early_end, DUSK), DUSK, DAWN, TZ) == "After dark until 10 PM"
    )


def test_nearest_darker_place():
    tahoe = Site("lake-tahoe", "Lake Tahoe", 38.93, -119.98, 1902.0, "Sierra Nevada", TZ)
    found = lookup.nearest_darker(LA, [(LA, 17.3), (DV, 21.98), (tahoe, 21.37)], 20.3)
    assert found is not None and found[0].id == "death-valley" and 250 < found[1] < 320


def test_night_events_are_dated_by_the_evening():
    """A planet at its best at 12:40 AM on Sunday belongs to Saturday night, like NASA writes."""
    best = pd.Timestamp("2026-10-04 07:40", tz="UTC")  # Sun 12:40 AM PDT
    e = Event("planet", "Saturn at opposition", best, "", best_start=best - pd.Timedelta(hours=2),
              best_end=best + pd.Timedelta(hours=2))  # fmt: skip
    day = events_view.viewing_date(e, TZ)
    assert day == pd.Timestamp("2026-10-03")
    evening = pd.Timestamp("2026-10-15 01:55", tz="UTC")  # Wed 6:55 PM PDT
    e2 = Event("pairing", "Moon near Antares", evening, "", best_start=evening,
               best_end=evening + pd.Timedelta(minutes=10))  # fmt: skip
    assert events_view.viewing_date(e2, TZ) == pd.Timestamp("2026-10-14")


def test_chart_shows_fewer_stars_under_city_light_and_has_unique_ids():
    t = pd.Timestamp("2026-10-13 04:30", tz="UTC")  # moonless evening
    city, city_info = skychart.sky_svg(LA, t, 17.29)
    dark, dark_info = skychart.sky_svg(DV, t, 21.98)
    assert city_info["stars"] < 100 < 2000 < dark_info["stars"]
    assert dark_info["milky_way"] and not city_info["milky_way"]
    ids = lambda svg: set(re.findall(r'id="([^"]+)"', svg))  # noqa: E731
    assert ids(city).isdisjoint(ids(dark))  # two charts on one page must not share ids
    natural, info = skychart.sky_svg(LA, t, 17.29, mode="dark")
    assert info["stars"] > 2000  # "a perfectly dark sky" ignores the city


# ---------- the live chart (views/skylive.py + skylive.js) ----------


@pytest.fixture(scope="module")
def live_payload():
    guide = sky.tonight(LA, DUSK, DAWN, 17.8)
    return guide, skylive.payload(LA, guide, 17.8, DUSK + pd.Timedelta(hours=2))


def _decode(b64: str, dtype: str) -> np.ndarray:
    import base64

    return np.frombuffer(base64.b64decode(b64), dtype=dtype)


def browser_place(lat: float, ra: np.ndarray, dec: np.ndarray, lst: float):
    """A line-by-line Python copy of skylive.js `place()` (hour angle from the angle-difference
    identities, azimuth from the north/east components), to prove its maths."""
    d2r = np.pi / 180
    cos_l, sin_l = np.cos(lst * d2r), np.sin(lst * d2r)
    cr, sr, sd, cd = np.cos(ra * d2r), np.sin(ra * d2r), np.sin(dec * d2r), np.cos(dec * d2r)
    sin_lat, cos_lat = np.sin(lat * d2r), np.cos(lat * d2r)
    cos_h = cos_l * cr + sin_l * sr
    sin_h = sin_l * cr - cos_l * sr
    sin_alt = sin_lat * sd + cos_lat * cd * cos_h
    north = sd * cos_lat - cd * sin_lat * cos_h
    east = -cd * sin_h
    alt = np.degrees(np.arcsin(np.clip(sin_alt, -1, 1)))
    az = np.degrees(np.arctan2(east, north)) % 360
    return alt, az


def test_browser_geometry_matches_the_python_reference(live_payload):
    _, data = live_payload
    obs = sky.Observer(LA)
    ra = _decode(data["stars"]["ra"], "<u2") / 100
    dec = _decode(data["stars"]["dec"], "<i2") / 100
    for minutes in (0, 137, 400):
        t_ms = data["t0"] + minutes * 60_000
        lst = (data["lst0"] + data["lstRate"] * (t_ms - data["t0"])) % 360
        alt, az = browser_place(LA.lat, ra, dec, lst)
        when = pd.DatetimeIndex([pd.Timestamp(t_ms, unit="ms", tz="UTC")])
        ref_alt, ref_az = obs.altaz_from_radec(ra, dec, when)
        assert np.abs(alt - ref_alt[0]).max() < 0.002  # degrees
        up = ref_alt[0] > 1
        dz = (az - ref_az[0] + 180) % 360 - 180
        assert np.abs(dz[up] * np.cos(np.radians(alt[up]))).max() < 0.002


def test_live_payload_is_complete_and_light(live_payload):
    import json

    guide, data = live_payload
    n = len(sky.load_catalog().stars)
    nt = len(data["steps"])
    assert len(_decode(data["stars"]["ra"], "<u2")) == n
    assert len(_decode(data["stars"]["tint"], "u1")) == n
    assert nt == len(guide["times"]) == len(_decode(data["moonK"], "<f4"))
    assert len(_decode(data["twK"], "<f4")) == len(data["sun"]["ra"]) == nt
    assert len(data["moon"]["ra"]) == nt
    assert {p["name"] for p in data["planets"]} == set(sky.PLANETS)
    na, nz = len(sky.LIMIT_ALTS), len(sky.GLOW_AZ)
    assert len(_decode(data["vis"]["glow"], "<f4")) == na * nz
    assert len(_decode(data["vis"]["moonSep"], "<f4")) == len(sky.LIMIT_SEPS)
    assert data["t0"] <= data["start"] <= data["t1"]
    assert len(json.dumps(data)) < 400_000  # whole night, sent once


def _tables(data) -> dict:
    t = {k: _decode(v, "<f4").astype(float) for k, v in data["vis"].items()}
    t["glow"] = t["glow"].reshape(len(sky.LIMIT_ALTS), len(sky.GLOW_AZ))
    return t | {"k": data["k"], "kRef": data["kRef"]}


def test_the_shipped_numbers_reproduce_the_guides_visibility(live_payload):
    """With the numbers sent to the browser (float32) and its arithmetic, the faintest visible
    magnitude matches the exact model at the real Moon and Sun positions of the night."""
    guide, data = live_payload
    t = _tables(data)
    moon_k, tw_k = _decode(data["moonK"], "<f4"), _decode(data["twK"], "<f4")
    obs = sky.Observer(LA)
    times = pd.DatetimeIndex(guide["times"])
    sun = obs.body("sun", times)
    rng = np.random.default_rng(1)
    alt, az = rng.uniform(1, 90, 2000), rng.uniform(0, 360, 2000)
    for i in range(0, len(times), 5):
        moon = sky.moon_at(guide["moon"], i, alt, az)
        ssep = sky.separation_deg(az, alt, sun["az"][i], sun["alt"][i])
        exact = sky.faintest_visible(alt, 17.8, moon, sun_alt=float(sun["alt"][i]), az=az,
                                     sun_sep=ssep)  # fmt: skip
        mk = float(moon_k[i]) if moon["alt"] > 0 else 0.0
        got = sky.limit_from_tables(t, alt, az, moon["sep"], mk, ssep, float(tw_k[i]))
        err = np.abs(got - exact)[exact > -1.5]
        assert np.percentile(err, 99) < 0.05 and err.max() < 0.2


def test_star_tints_match_the_python_chart():
    bvs = [-0.3, 0.0, 0.2, 0.5, 0.8, 1.2, 1.6, float("nan")]
    tints = [c for _, c in skychart.BV_TINTS]
    assert [tints[i] for i in skylive.tint_index(bvs)] == [skychart.star_tint(b) for b in bvs]


def test_live_chart_covers_twilight():
    """From civil dusk the chart shows the sky darkening: twilight is on at civil dusk (Sun 6°
    down), gone by the middle of the night, and the fully dark part is marked."""
    civil_dusk, civil_dawn = DUSK - pd.Timedelta(minutes=55), DAWN + pd.Timedelta(minutes=55)
    guide = sky.tonight(LA, civil_dusk, civil_dawn, 17.8)
    data = skylive.payload(LA, guide, 17.8, DUSK, dark=(DUSK, DAWN))
    assert data["t0"] < data["dark0"] < data["dark1"] < data["t1"]
    tw = _decode(data["twK"], "<f4")
    assert tw[0] > 0 and tw[len(tw) // 2] == 0 and tw[-1] > 0
