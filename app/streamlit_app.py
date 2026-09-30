"""SkyTrust Streamlit app.

The app only reads precomputed artifacts (the blend JSON models and metrics.json) plus the
live Open-Meteo forecast. It never trains or backfills (SPEC 10). Any failure on a page is
caught and shown as a message; the app itself must never crash.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import streamlit as st

try:  # installed via `uv sync` locally; on a host that only installs dependencies, use src/
    import skytrust  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skytrust import inference, live  # noqa: E402
from skytrust.config import load_settings, load_sites  # noqa: E402
from views import methodology, outlook, tonight, track_record  # noqa: E402
from views.common import Context, footer, inject_night_vision_css, palette  # noqa: E402

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


@st.cache_data(ttl=60 * 60, show_spinner=False)
def cached_forward_summary():
    """Live verification record from the forward-data branch (refreshed hourly)."""
    settings, _, _ = load_static()
    local = Path(__file__).resolve().parents[1] / "forward" / "summary.json"
    return inference.load_forward_summary(settings.raw["forward"]["summary_url"], local)


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
        # Optional deep links: ?page=track-record&site=BIH
        query = st.query_params
        slugs = {name.lower().replace(" ", "-"): name for name in PAGES}
        start_page = slugs.get(query.get("page", ""), "Tonight")
        page = st.radio("Page", list(PAGES), index=list(PAGES).index(start_page), key="page")
        site_ids = [s.id for s in sites]
        wanted_site = query.get("site", "SAC").upper()
        site_id = st.selectbox(
            "Site",
            site_ids,
            index=site_ids.index(wanted_site) if wanted_site in site_ids else 0,
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
    summary = cached_forward_summary() if page == "Track Record" else None
    ctx = Context(settings, site, sites, metrics, forecast, error, palette(night_vision), summary)
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
