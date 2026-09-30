"""App smoke tests (SPEC 12.3): render every page with Streamlit's AppTest, offline, including
with the live API down (with and without a saved last-good copy)."""

from __future__ import annotations

import json

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
PAGES = ["Tonight", "7 Nights", "Where to Go", "Track Record", "How It Works"]
NOW = pd.Timestamp("2026-09-25 01:00", tz="UTC")


@pytest.fixture
def offline(monkeypatch, tmp_path):
    """Freeze time, isolate the last-good cache, and start with empty Streamlit caches."""
    monkeypatch.setattr(live, "utcnow", lambda: NOW)
    monkeypatch.setenv("SKYTRUST_LIVE_CACHE", str(tmp_path))
    # No network for the live-verification record: point it at a file that doesn't exist.
    monkeypatch.setenv("SKYTRUST_FORWARD_SUMMARY", str(tmp_path / "no_summary.json"))
    st.cache_data.clear()
    yield tmp_path
    st.cache_data.clear()


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
    assert "chance of a usable night" in html
    assert any(v in html for v in ["GO", "MAYBE", "SKIP"])
    assert "Open-Meteo" in " ".join(c.value for c in at.caption)


@pytest.mark.parametrize("page", PAGES)
def test_api_down_without_saved_copy_never_crashes(page, offline, monkeypatch):
    api_down(monkeypatch)
    at = visit(page)
    assert not at.exception
    if page in ("Tonight", "7 Nights"):
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
    at = visit("Track Record")
    at.radio(key="label").set_value("asos").run()
    assert not at.exception
    assert any("ASOS-only" in m.value for m in at.markdown)
    at.radio(key="label").set_value("goes").run()  # the satellite truth
    assert not at.exception
    assert any("GOES satellite" in m.value for m in at.markdown)


def test_site_switch(offline, monkeypatch):
    api_up(monkeypatch)
    at = visit("Tonight")
    at.selectbox(key="site").set_value("BIH").run()
    assert not at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "Bishop" in html and "Bortle" in html  # light pollution shows on Tonight


def test_deep_link_query_params(offline, monkeypatch):
    api_up(monkeypatch)
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params["page"] = "track-record"
    at.query_params["site"] = "bih"
    at.run()
    assert not at.exception
    assert at.session_state["page"] == "Track Record"
    assert at.selectbox(key="site").value == "BIH"


def test_old_page_links_still_work(offline, monkeypatch):
    api_up(monkeypatch)
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params["page"] = "methodology"  # renamed to "How It Works"
    at.run()
    assert not at.exception
    assert at.session_state["page"] == "How It Works"


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
    at = visit("Track Record")
    assert not at.exception
    shown = " ".join([i.value for i in at.info] + [str(m.value) for m in at.metric])
    assert expected in shown


def test_live_verification_unreachable_is_graceful(offline, monkeypatch):
    api_up(monkeypatch)
    at = visit("Track Record")
    assert not at.exception
    assert any("isn't reachable" in i.value for i in at.info)


def test_risk_slider_changes_the_call(offline, monkeypatch):
    api_up(monkeypatch)
    at = visit("Tonight")
    assert not at.exception
    for setting in ["Adventurous: go at 30%+", "Cautious: go at 70%+"]:
        at.select_slider(key="risk").set_value(setting).run()
        assert not at.exception
        assert any("At this setting tonight's call is" in m.value for m in at.markdown)


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


def test_where_tonight_ranks_every_site(offline, monkeypatch):
    api_up(monkeypatch)
    at = visit("Where to Go")
    assert not at.exception, at.exception
    assert any("Where should I go tonight?" in h.value for h in at.header)
    assert len(at.dataframe) == 1 and len(at.dataframe[0].value) == 5
    html = " ".join(m.value for m in at.markdown)
    assert "Darkest within 50 km" in html and "Nearest Bortle 1–3 sky" in html
    assert "Dark site" in html and "Moon down" in html  # the three conditions
    assert any("Light pollution at Sacramento" in h.value for h in at.subheader)
