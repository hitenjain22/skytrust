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

from views import events_view, lookup, skychart, skyguide  # noqa: E402

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


def test_time_options_cover_the_night_hour_by_hour():
    opts = skyguide.time_options(DUSK, DAWN, TZ)
    labels = list(opts)
    assert labels[0].startswith("Dusk") and labels[-1].startswith("Dawn")
    assert labels[1:4] == ["9 PM", "10 PM", "11 PM"]
    assert all(DUSK <= t <= DAWN for t in opts.values())
    assert skyguide.default_choice(opts, None) == "10 PM"  # about two hours after dark


@pytest.mark.slow
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
