"""App smoke tests (SPEC 12.3): render every page with Streamlit's AppTest, offline, including
with the live API down (with and without a saved last-good copy)."""

from __future__ import annotations

import json
import re

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from conftest import FIXTURES
from skytrust import live
from skytrust.config import REPO_ROOT
from skytrust.data import openmeteo
from skytrust.data.http import SourceUnavailableError

# Each AppTest spins up the whole app, so these run in `make test` / CI, not in quick `pytest`.
pytestmark = pytest.mark.slow

APP = str(REPO_ROOT / "app" / "streamlit_app.py")
PAGES = ["Tonight", "Sky Guide", "Events", "Where to Go", "Accuracy"]
NOW = pd.Timestamp("2026-09-25 01:00", tz="UTC")


@pytest.fixture
def offline(monkeypatch, tmp_path):
    """Freeze time, isolate the last-good cache, and start with empty Streamlit caches."""
    monkeypatch.setattr(live, "utcnow", lambda: NOW)
    # exact coordinates look up the terrain height online; the tests stay offline (a slow
    # elevation service on GitHub's runners once timed a test out)
    monkeypatch.setattr(live, "custom_site", _offline_custom_site)
    monkeypatch.setenv("SKYTRUST_LIVE_CACHE", str(tmp_path))
    # No network for the live-verification record: point it at a file that doesn't exist.
    monkeypatch.setenv("SKYTRUST_FORWARD_SUMMARY", str(tmp_path / "no_summary.json"))
    st.cache_data.clear()
    yield tmp_path
    st.cache_data.clear()


def _offline_custom_site(lat, lon, name, settings, client=None):
    import math

    from skytrust.config import Site

    (la0, la1), (lo0, lo1) = live.CUSTOM_BOUNDS["lat"], live.CUSTOM_BOUNDS["lon"]
    if not (la0 <= lat <= la1 and lo0 <= lon <= lo1):
        raise ValueError(f"custom locations must be within lat {la0}-{la1}, lon {lo0}-{lo1}")
    return Site(f"CUSTOM_{lat:.3f}_{lon:.3f}", name, lat, lon, math.nan, "custom",
                "America/Los_Angeles")  # fmt: skip


def api_up(monkeypatch):
    payload = json.loads((FIXTURES / "openmeteo_forecast_SAC_live.json").read_text())
    monkeypatch.setattr(openmeteo, "fetch_live", lambda c, s, site: payload)


def api_down(monkeypatch):
    def down(*a, **k):
        raise SourceUnavailableError("HTTP 503")

    monkeypatch.setattr(openmeteo, "fetch_live", down)


def visit(page: str) -> AppTest:
    """Open the app directly on `page` (one script run instead of two)."""
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["page"] = page
    return at.run()


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders(page, offline, monkeypatch):
    api_up(monkeypatch)
    at = visit(page)
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]


def test_tonight_shows_probability_and_verdict(offline, monkeypatch):
    api_up(monkeypatch)
    at = visit("Tonight")
    html = " ".join(m.value for m in at.markdown)
    assert "clear hours in a row after dark" in html
    assert "Look up tonight" in html and "The week ahead" in html
    assert "<svg" in html  # the picture of tonight's sky
    assert any(v in html for v in ["GO", "MAYBE", "SKIP"])
    assert "Open-Meteo" in " ".join(c.value for c in at.caption)


@pytest.mark.parametrize("page", PAGES)
def test_api_down_without_saved_copy_never_crashes(page, offline, monkeypatch):
    api_down(monkeypatch)
    at = visit(page)
    assert not at.exception
    if page == "Tonight":
        assert any("unavailable" in w.value for w in at.warning)


def test_api_down_with_saved_copy_shows_as_of_banner(offline, monkeypatch):
    api_up(monkeypatch)
    visit("Tonight")  # populates the last-good copy on disk
    st.cache_data.clear()
    api_down(monkeypatch)
    at = visit("Tonight")
    assert not at.exception
    banners = [w.value for w in at.warning]
    assert any("showing the saved forecast from" in b for b in banners), banners


def test_track_record_label_toggle(offline, monkeypatch):
    api_up(monkeypatch)
    at = visit("Accuracy")
    at.radio(key="label").set_value("asos").run()
    assert not at.exception
    assert any("ASOS-only" in m.value for m in at.markdown)
    at.radio(key="label").set_value("goes").run()  # the satellite truth
    assert not at.exception
    assert any("GOES satellite" in m.value for m in at.markdown)


def test_site_switch(offline, monkeypatch):
    api_up(monkeypatch)
    at = visit("Tonight")
    at.selectbox(key="site").set_value("death-valley").run()
    assert not at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "Death Valley" in html and "Very dark" in html  # light pollution in plain words
    assert "Bortle" not in html


def test_deep_link_query_params(offline, monkeypatch):
    api_up(monkeypatch)
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params["page"] = "sky-guide"
    at.query_params["site"] = "death-valley"
    at.run()
    assert not at.exception
    assert at.session_state["page"] == "Sky Guide"
    assert at.selectbox(key="site").value == "death-valley"


@pytest.mark.parametrize(
    ("page", "site", "want_page", "want_site"),
    [("methodology", "sac", "Accuracy", "sacramento"), ("track-record", "trk", "Accuracy",
     "lake-tahoe"), ("7-nights", "bih", "Tonight", "los-angeles")],
)  # fmt: skip
def test_old_links_still_work(page, site, want_page, want_site, offline, monkeypatch):
    """Bookmarks from before the redesign (old page names, airport codes) still open sensibly."""
    api_up(monkeypatch)
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params["page"] = page
    at.query_params["site"] = site
    at.run()
    assert not at.exception
    assert at.session_state["page"] == want_page
    assert at.selectbox(key="site").value == want_site


def test_moon_band_when_moon_is_still_up_at_the_chart_edge():
    """Regression: the moon-up run wasn't closed if the Moon was above the horizon at the end
    of the window (found by screenshotting the real app on 2026-09-29)."""
    from skytrust import astro
    from skytrust.config import load_sites

    sac = next(s for s in load_sites() if s.id == "SAC")
    start = pd.Timestamp("2026-09-30 02:19", tz="UTC")
    end = pd.Timestamp("2026-09-30 13:32", tz="UTC")
    assert astro.moon_altitude(sac, pd.DatetimeIndex([end]))[0] > 0  # precondition: moon up at edge
    runs = astro.moon_up_intervals(sac, start, end)
    assert runs and runs[-1][1] >= end - pd.Timedelta(minutes=10)


def test_true_runs_edge_cases():
    from skytrust.astro import true_runs

    t = pd.date_range("2026-01-01", periods=5, freq="10min", tz="UTC")
    assert true_runs(t, [False, True, True, False, False]) == [(t[1], t[3])]
    assert true_runs(t, [True, True, False, True, True]) == [(t[0], t[2]), (t[3], t[4])]
    assert true_runs(t, [False] * 5) == []
    assert true_runs(t, [True] * 5) == [(t[0], t[4])]


def _summary(n_verified: int) -> dict:
    blend = {
        "n": 40,
        "brier": 0.09,
        "bss": 0.6,
        "false_clear_rate": 0.1,
        "go_calls": 20,
        "go_calls_usable": 18,
    }
    entry = {
        "n": 40,
        "n_weeks": 2,
        "methods": {"blend": blend, "nbm_frac_clear": {"brier": 0.2}, "ens_ecmwf": {"brier": 0.12}},
        "backtest": {"false_clear_rate": 0.098, "bss": 0.64},
    }
    return {
        "forward_start": "2026-09-30",
        "first_issue": "2026-09-30",
        "n_logged": 350,
        "n_verified": n_verified,
        "last_verified_night": "2026-10-10",
        "verify_after_days": 9,
        "by_lead": [] if not n_verified else [{**entry, "lead": "all"}, {**entry, "lead": 1}],
    }


@pytest.mark.parametrize(("n_verified", "expected"), [(0, "forecasts saved so far"), (40, "18/20")])
def test_live_verification_panel(offline, monkeypatch, n_verified, expected):
    api_up(monkeypatch)
    path = offline / "summary.json"
    path.write_text(json.dumps(_summary(n_verified)))
    monkeypatch.setenv("SKYTRUST_FORWARD_SUMMARY", str(path))
    at = visit("Accuracy")
    assert not at.exception
    shown = " ".join([i.value for i in at.info] + [str(m.value) for m in at.metric])
    assert expected in shown


def test_live_verification_unreachable_is_graceful(offline, monkeypatch):
    api_up(monkeypatch)
    at = visit("Accuracy")
    assert not at.exception
    assert any("isn't reachable" in i.value for i in at.info)


def test_risk_slider_changes_the_call(offline, monkeypatch):
    api_up(monkeypatch)
    at = visit("Tonight")
    assert not at.exception
    for bar, words in [(0, "go out every night"), (30, "in 10"), (85, "in 10"),
                       (100, "never go")]:  # fmt: skip
        at.slider(key="risk").set_value(bar).run()
        assert not at.exception
        shown = " ".join(m.value for m in at.markdown)
        assert f"at {bar}% the call is" in shown and words in shown


def test_custom_location_via_deep_link(offline, monkeypatch):
    import math

    from skytrust.config import Site

    api_up(monkeypatch)
    monkeypatch.setattr(
        live,
        "custom_site",
        lambda lat, lon, name, settings, client=None: Site(
            f"CUSTOM_{lat:.3f}_{lon:.3f}", name, lat, lon, math.nan, "custom", "America/Los_Angeles"
        ),
    )
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params.update(
        {"page": "tonight", "lat": "37.7306", "lon": "-119.5738", "name": "Glacier Point"}
    )
    at.run()
    assert not at.exception, at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "Glacier Point" in html and "never seen" in html


def test_custom_location_out_of_bounds_is_friendly(offline, monkeypatch):
    api_up(monkeypatch)
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params.update({"page": "tonight", "lat": "37.7", "lon": "-119.6"})
    at.run()
    at.number_input(key="lat").set_value(48.9).run()
    assert not at.exception


@pytest.mark.parametrize(
    ("lat", "lon"), [("55", "-100"), ("abc", "1"), ("nan", "-119"), ("37.7", "inf")]
)
def test_bad_coordinates_in_a_shared_link_never_crash(lat, lon, offline, monkeypatch):
    """Regression: `?lat=55` (outside the West) or `?lat=abc` took the whole app down."""
    api_up(monkeypatch)
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params.update({"page": "how-it-works", "lat": lat, "lon": lon})
    at.run()
    assert not at.exception, at.exception
    assert any("couldn't be used" in w.value for w in at.warning)


def test_where_to_go_ranks_every_place(offline, monkeypatch):
    from skytrust.config import load_places

    api_up(monkeypatch)
    at = visit("Where to Go")
    assert not at.exception, at.exception
    assert len(at.dataframe) == 1 and len(at.dataframe[0].value) == len(load_places())
    html = " ".join(m.value for m in at.markdown)
    assert "Where should I go tonight?" in html and "How dark is it at Los Angeles?" in html
    assert "Darkest spot within 50 km" in html and "Nearest very dark sky" in html
    assert "Dark sky" in html and "Moon down" in html  # the three conditions
    assert "darkest part of the horizon" in html
    assert "Biggest glows" in html  # the city-glow details, folded away
    assert "Since 2015" in html
    # places are described in plain words; "Bortle" appears only in the method notes
    assert not re.search(r"Bortle \d", html) and not re.search(r"\bB\d(\.5)?\b", html)


def live_chart_data(at: AppTest) -> dict:
    """The data the Sky Guide's live chart (a browser component) receives."""
    found = [e for e in at.main if type(e).__name__ == "UnknownElement" and "json" in
             {f.name for f, _ in e.proto.ListFields()}]  # fmt: skip
    assert len(found) == 1, "the live sky chart is missing"
    return json.loads(found[0].proto.json)


def test_sky_guide_draws_the_sky_and_lists_what_is_up(offline, monkeypatch):
    api_up(monkeypatch)
    at = visit("Sky Guide")
    assert not at.exception, at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "The sky tonight over Los Angeles" in html and "The Milky Way" in html
    data = live_chart_data(at)
    assert data["place"] == "Los Angeles"
    assert data["t0"] <= data["start"] <= data["t1"]
    assert {"Moon", "Jupiter", "Saturn"} <= {i["n"] for i in data["items"]}
    # switching place sends the new place's sky
    at.selectbox(key="site").set_value("death-valley").run()
    assert not at.exception
    assert live_chart_data(at)["place"] == "Death Valley"


def test_any_california_place_can_be_chosen(offline, monkeypatch):
    """Regression for "you have to press Enter and it doesn't change": every town, community and
    ZIP code in California is in the location menu, and choosing one switches the app."""
    api_up(monkeypatch)
    at = visit("Tonight")
    at.selectbox(key="site").set_value("davis").run()
    assert not at.exception, at.exception
    assert at.query_params["site"] == ["davis"]
    html = " ".join(m.value for m in at.markdown)
    assert "Davis" in html and "clear hours in a row after dark" in html
    at.session_state["page"] = "Sky Guide"
    at.run()
    assert not at.exception
    assert live_chart_data(at)["place"] == "Davis"


def test_zip_code_link(offline, monkeypatch):
    api_up(monkeypatch)
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params.update({"page": "where-to-go", "site": "zip-95616"})
    at.run()
    assert not at.exception, at.exception
    assert at.selectbox(key="site").value == "zip-95616"
    html = " ".join(m.value for m in at.markdown)
    assert "How dark is it at ZIP 95616?" in html


def test_coordinates_without_a_name_are_named_after_the_nearest_place(offline, monkeypatch):
    import math

    from skytrust.config import Site

    api_up(monkeypatch)
    monkeypatch.setattr(
        live,
        "custom_site",
        lambda lat, lon, name, settings, client=None: Site(
            f"CUSTOM_{lat:.3f}_{lon:.3f}", name, lat, lon, math.nan, "custom", "America/Los_Angeles"
        ),
    )
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params.update({"page": "sky-guide", "lat": "36.4500", "lon": "-117.6000"})
    at.run()
    assert not at.exception, at.exception
    assert live_chart_data(at)["place"].startswith("Near ")
    assert not any(w.key == "name" for w in at.text_input)  # no name box to press Enter in


def test_events_page_lists_showers_with_places(offline, monkeypatch):
    api_up(monkeypatch)
    at = visit("Events")
    assert not at.exception, at.exception
    html = " ".join(m.value for m in at.markdown)
    # late September 2026: the Draconids (Oct 9) and Orionids (Oct 21) are coming up
    assert "Draconids peak" in html and "Orionids peak" in html
    assert "Best places in California" in html and "Saturn at opposition" in html
    at.segmented_control(key="ev_filter").set_value("Meteor showers").run()
    assert not at.exception
    assert "Saturn at opposition" not in " ".join(m.value for m in at.markdown)


def test_modules_left_over_from_before_a_deploy_are_reloaded(offline, monkeypatch):
    """Regression (live site, 2026-09-30): Streamlit Cloud pulled new code but kept the old
    copies of already-imported modules in memory, so the new main script failed with
    `ImportError: cannot import name ... from views.common`."""
    import sys
    import types

    api_up(monkeypatch)
    visit("Accuracy")  # imports views.* normally
    ours = ("views", "skytrust")
    saved = {k: m for k, m in sys.modules.items() if k.split(".")[0] in ours}
    real = sys.modules["views.common"]
    stale = types.ModuleType("views.common")  # an older version, missing newer helpers
    stale.__file__ = real.__file__
    stale.Context = real.Context
    stale.__skytrust_stamp__ = (0, 0)  # loaded from a file that has since changed
    sys.modules["views.common"] = stale
    try:
        at = visit("Accuracy")
        assert not at.exception
        assert sys.modules["views.common"] is not stale
    finally:  # put back the modules the other tests' monkeypatches are attached to
        for k in [k for k in sys.modules if k.split(".")[0] in ours]:
            del sys.modules[k]
        sys.modules.update(saved)


def test_statewide_accuracy_section(offline, monkeypatch, tmp_path, synthetic_built, fast_settings):
    """With statewide results the Accuracy page shows them and Tonight, for a place without its
    own record, lists the nearest stations; without them it says the test is still running."""
    from skytrust import inference
    from test_statewide import synthetic_result

    api_up(monkeypatch)
    monkeypatch.setattr(inference, "STATEWIDE_PATH", tmp_path / "absent.json")
    at = visit("Accuracy")
    assert not at.exception, at.exception
    assert any("statewide test is still running" in i.value for i in at.info)

    path = tmp_path / "statewide.json"
    result = synthetic_result(synthetic_built[0], fast_settings)
    path.write_text(json.dumps(inference_clean(result)))
    monkeypatch.setattr(inference, "STATEWIDE_PATH", path)
    st.cache_data.clear()
    at = visit("Accuracy")
    assert not at.exception, at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "Across California" in html and "“Go” calls that held" in html
    assert "Nearest stations to Los Angeles" in html

    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params.update({"page": "tonight", "site": "davis"})
    at.run()
    assert not at.exception, at.exception
    assert any("never seen" in m.value and "SAC" in m.value for m in at.markdown)


def inference_clean(obj):
    from skytrust import evaluate

    return evaluate._clean(obj)
