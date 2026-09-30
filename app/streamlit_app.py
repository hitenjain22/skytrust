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
from views import methodology, outlook, tonight, track_record, where  # noqa: E402
from views.common import Context, footer, inject_night_vision_css, palette  # noqa: E402

log = logging.getLogger("skytrust.app")

PAGES = {
    "Tonight": tonight.render,
    "7-Night Outlook": outlook.render,
    "Where Tonight": where.render,
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
def cached_forecast(
    site_id: str, lat: float | None = None, lon: float | None = None, name: str = ""
):
    """Live forecast, cached for 60 minutes per location. Exceptions are not cached, so a failed
    fetch is retried on the next page load (with the last-good disk copy as fallback)."""
    settings, sites, metrics = load_static()
    site = resolve_site(site_id, lat, lon, name)
    return live.get_forecast(site, settings, metrics)


@st.cache_data(show_spinner=False)
def cached_custom_site(lat: float, lon: float, name: str):
    settings, _, _ = load_static()
    return live.custom_site(lat, lon, name, settings)


def resolve_site(site_id: str, lat: float | None, lon: float | None, name: str):
    _, sites, _ = load_static()
    if site_id == live.CUSTOM_ID:
        return cached_custom_site(round(lat, 3), round(lon, 3), name)
    return next(s for s in sites if s.id == site_id)


@st.cache_data(ttl=60 * 60, show_spinner=False)
def cached_forward_summary():
    """Live verification record from the forward-data branch (refreshed hourly)."""
    settings, _, _ = load_static()
    local = Path(__file__).resolve().parents[1] / "forward" / "summary.json"
    return inference.load_forward_summary(settings.raw["forward"]["summary_url"], local)


def get_forecast(
    site_id: str, lat=None, lon=None, name: str = ""
) -> tuple[object | None, str | None]:
    try:
        return cached_forecast(site_id, lat, lon, name), None
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
        options = [*site_ids, live.CUSTOM_ID]
        if "lat" in query and "lon" in query:
            wanted_site = live.CUSTOM_ID
        site_id = st.selectbox(
            "Site",
            options,
            index=options.index(wanted_site) if wanted_site in options else 0,
            key="site",
            format_func=lambda sid: (
                "📍 Custom location"
                if sid == live.CUSTOM_ID
                else f"{sid} · {next(s.terrain_class for s in sites if s.id == sid)}"
            ),
        )
        lat = lon = None
        name = ""
        if site_id == live.CUSTOM_ID:
            (la0, la1), (lo0, lo1) = live.CUSTOM_BOUNDS["lat"], live.CUSTOM_BOUNDS["lon"]
            lat = st.number_input(
                "Latitude", la0, la1, float(query.get("lat", 37.7306)), 0.01, key="lat"
            )
            lon = st.number_input(
                "Longitude", lo0, lo1, float(query.get("lon", -119.5738)), 0.01, key="lon"
            )
            name = st.text_input("Name", query.get("name", "Glacier Point"), key="name")
            st.caption(
                "Uses the site-agnostic blend; its reliability comes from the unseen-site test."
            )
        with st.expander("About"):
            st.markdown(
                "Built by **Hiten Jain** (UC Davis, statistics / operations research). Every "
                "number is generated by code from real data and verified on a held-out period "
                "plus a live forward test.  \n"
                "[Code on GitHub](https://github.com/hitenjain22/skytrust) · "
                "[Full results](https://github.com/hitenjain22/skytrust/blob/main/docs/RESULTS.md)"
            )
        night_vision = st.toggle(
            "Night vision (red)",
            key="night_vision",
            help="Dim red display that preserves dark adaptation at the telescope.",
        )

    if night_vision:
        inject_night_vision_css()
    try:
        site = resolve_site(site_id, lat, lon, name)
    except ValueError as exc:  # out-of-bounds custom location
        st.error(str(exc))
        return
    needs_live = page in ("Tonight", "7-Night Outlook")
    forecast, error = get_forecast(site_id, lat, lon, name) if needs_live else (None, None)
    summary = cached_forward_summary() if page == "Track Record" else None
    ctx = Context(
        settings,
        site,
        sites,
        metrics,
        forecast,
        error,
        palette(night_vision),
        summary,
        get_forecast,
    )
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
