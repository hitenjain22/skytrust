"""SkyTrust Streamlit app.

The app only reads precomputed artifacts (the blend JSON models and metrics.json) plus the
live Open-Meteo forecast. It never trains or backfills (SPEC 10). Any failure on a page is
caught and shown as a message; the app itself must never crash.
"""

from __future__ import annotations

import logging

import streamlit as st

from skytrust import inference, live
from skytrust.config import load_settings, load_sites
from views import methodology, outlook, tonight, track_record
from views.common import Context, footer, inject_night_vision_css, palette

log = logging.getLogger("skytrust.app")

PAGES = {
    "Tonight": tonight.render,
    "7-Night Outlook": outlook.render,
    "Track Record": track_record.render,
    "Methodology": methodology.render,
}


@st.cache_resource(show_spinner=False)
def load_static():
    """Settings, sites, and backtest metrics: read once per server process."""
    settings = load_settings()
    sites = load_sites()
    try:
        metrics = inference.load_metrics()
    except (FileNotFoundError, ValueError):
        metrics = None
    return settings, sites, metrics


@st.cache_data(ttl=60 * 60, show_spinner="Fetching the latest forecasts…")
def cached_forecast(site_id: str):
    """Live forecast, cached for 60 minutes per site. Exceptions are not cached, so a failed
    fetch is retried on the next page load (with the last-good disk copy as fallback)."""
    settings, sites, metrics = load_static()
    site = next(s for s in sites if s.id == site_id)
    return live.get_forecast(site, settings, metrics)


def get_forecast(site_id: str) -> tuple[object | None, str | None]:
    try:
        return cached_forecast(site_id), None
    except live.LiveUnavailableError as exc:
        return None, str(exc)
    except Exception as exc:  # never let a forecast problem take down the app
        log.exception("forecast failed for %s", site_id)
        return None, f"unexpected error: {exc}"


def main() -> None:
    st.set_page_config(page_title="SkyTrust", page_icon="🔭", layout="centered")
    settings, sites, metrics = load_static()

    with st.sidebar:
        st.title("🔭 SkyTrust")
        st.caption("An astronomy cloud forecast that tells you how often it's been wrong.")
        page = st.radio("Page", list(PAGES), key="page")
        site_ids = [s.id for s in sites]
        site_id = st.selectbox(
            "Site",
            site_ids,
            index=site_ids.index("SAC") if "SAC" in site_ids else 0,
            key="site",
            format_func=lambda sid: (
                f"{sid} · {next(s.terrain_class for s in sites if s.id == sid)}"
            ),
        )
        night_vision = st.toggle(
            "Night vision (red)",
            key="night_vision",
            help="Dim red display that preserves dark adaptation at the telescope.",
        )

    if night_vision:
        inject_night_vision_css()
    site = next(s for s in sites if s.id == site_id)
    needs_live = page in ("Tonight", "7-Night Outlook")
    forecast, error = get_forecast(site_id) if needs_live else (None, None)
    ctx = Context(settings, site, sites, metrics, forecast, error, palette(night_vision))
    try:
        PAGES[page](ctx)
    except Exception as exc:  # show a message instead of a stack trace
        log.exception("page %s failed", page)
        st.error(
            f"Something went wrong drawing this page ({type(exc).__name__}). "
            "Other pages should still work."
        )
    footer()


main()
