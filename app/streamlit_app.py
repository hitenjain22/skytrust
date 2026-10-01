"""SkyTrust Streamlit app.

The app only reads precomputed artifacts (the blend JSON models and metrics.json) plus the
live Open-Meteo forecast. It never trains or backfills (SPEC 10). Any failure on a page is
caught and shown as a message; the app itself must never crash.

Navigation and the location picker live at the top of the page (not in a sidebar), so they're
one tap away on a phone. The URL always reflects the current page and location, so any view can
be bookmarked or shared.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import streamlit as st

try:  # installed via `uv sync` locally; on a host that only installs dependencies, use src/
    import skytrust  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skytrust import inference, lightpollution, live, skyglow  # noqa: E402
from skytrust.config import CONFIG_DIR, load_settings, load_sites  # noqa: E402
from views import components as ui  # noqa: E402
from views import methodology, outlook, theme, tonight, track_record, where  # noqa: E402
from views.common import Context, fingerprint, footer, site_option_label  # noqa: E402

log = logging.getLogger("skytrust.app")

PAGES = {
    "Tonight": tonight.render,
    "7 Nights": outlook.render,
    "Where to Go": where.render,
    "Track Record": track_record.render,
    "How It Works": methodology.render,
}
SLUGS = {name.lower().replace(" ", "-"): name for name in PAGES}
# Old page names keep working in bookmarked links.
SLUGS |= {
    "7-night-outlook": "7 Nights",
    "where-tonight": "Where to Go",
    "methodology": "How It Works",
}
DEFAULT_CUSTOM = {"lat": 37.7306, "lon": -119.5738, "name": "Glacier Point"}


STATIC_FILES = (CONFIG_DIR / "settings.yaml", CONFIG_DIR / "sites.yaml", inference.METRICS_PATH)


@st.cache_resource(show_spinner=False)
def _load_static(version: str):
    """Settings, sites, and backtest metrics, cached per version of the files they come from."""
    settings = load_settings()
    sites = load_sites()
    try:
        metrics = inference.load_metrics()
    except (FileNotFoundError, ValueError):
        metrics = None
    return settings, sites, metrics


def load_static():
    return _load_static(fingerprint(*STATIC_FILES))


LIGHT_FILES = (
    lightpollution.GRID_PATH, lightpollution.BASE_PATH, skyglow.SOURCES_PATH,
    skyglow.CITIES_PATH, skyglow.MODEL_PATH,
)  # fmt: skip


@st.cache_resource(show_spinner=False)
def _load_light_pollution(version: str):
    model = json.loads(skyglow.MODEL_PATH.read_text()) if skyglow.MODEL_PATH.exists() else None
    return {
        "grid": lightpollution.load(),
        "base": lightpollution.load(lightpollution.BASE_PATH),
        "sources": skyglow.load_sources(),
        "model": model,
    }


def load_light_pollution() -> dict:
    """Light-pollution grids (latest + 2015 atlas), light sources and the model card, re-read
    whenever any of those files changes."""
    return _load_light_pollution(fingerprint(*LIGHT_FILES))


@st.cache_data(ttl=60 * 60, show_spinner="Reading the latest forecasts…")
def cached_forecast(
    site_id: str,
    lat: float | None = None,
    lon: float | None = None,
    name: str = "",
    hour: str = "",
):
    """Live forecast per location, refreshed every clock hour (`hour` is part of the cache key)
    and at most 60 minutes old. Exceptions are not cached, so a failed fetch is retried on the
    next page load (with the last-good disk copy as fallback)."""
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
    hour = live.utcnow().floor("h").isoformat()
    try:
        return cached_forecast(site_id, lat, lon, name, hour), None
    except live.LiveUnavailableError as exc:
        return None, str(exc)
    except Exception as exc:  # never let a forecast problem take down the app
        log.exception("forecast failed for %s", site_id)
        return None, f"unexpected error: {exc}"


def top_bar(sites) -> tuple[str, str, float | None, float | None, str]:
    """One sticky row: brand, page navigation, location. On a phone the columns
    stack and the navigation wraps, so everything stays one tap away."""
    query = st.query_params
    site_ids = [s.id for s in sites]
    options = [*site_ids, live.CUSTOM_ID]
    wanted = query.get("site", "SAC").upper()
    if "lat" in query and "lon" in query:
        wanted = live.CUSTOM_ID
    labels = {s.id: site_option_label(s) for s in sites} | {live.CUSTOM_ID: "Custom location…"}
    start = SLUGS.get(query.get("page", ""), "Tonight")

    with st.container(key="topbar"):
        brand, nav, where = st.columns([1.0, 3.6, 1.4], vertical_alignment="center", gap="small")
        brand.markdown(
            ui.block(
                '<div class="sk-brand">',
                ui.moon_svg(300.0, 18),
                "<span>SkyTrust</span>",
                "</div>",
            ),
            unsafe_allow_html=True,
        )
        page = nav.segmented_control(
            "Page",
            list(PAGES),
            default=None if "page" in st.session_state else start,
            required=True,
            key="page",
            label_visibility="collapsed",
            wrap=True,  # on a phone the five pages flow onto two lines instead of off-screen
        )
        site_id = where.selectbox(
            "Location",
            options,
            index=options.index(wanted) if wanted in options else 0,
            key="site",
            format_func=labels.get,
            label_visibility="collapsed",
        )
    lat = lon = None
    name = ""
    if site_id == live.CUSTOM_ID:
        (la0, la1), (lo0, lo1) = live.CUSTOM_BOUNDS["lat"], live.CUSTOM_BOUNDS["lon"]
        with st.container(key="panel_custom"):
            c1, c2, c3 = st.columns([1, 1, 1.4])
            lat = c1.number_input(
                "Latitude", la0, la1, float(query.get("lat", DEFAULT_CUSTOM["lat"])), 0.01,
                key="lat",
            )  # fmt: skip
            lon = c2.number_input(
                "Longitude", lo0, lo1, float(query.get("lon", DEFAULT_CUSTOM["lon"])), 0.01,
                key="lon",
            )  # fmt: skip
            name = c3.text_input("Name", query.get("name", DEFAULT_CUSTOM["name"]), key="name")
            st.caption(
                "Anywhere in the Pacific-time West. Uses the site-agnostic blend, whose accuracy "
                "was measured at airports it had never seen."
            )
    return page or "Tonight", site_id, lat, lon, name


def sync_url(page: str, site_id: str, lat, lon, name: str) -> None:
    """Keep the address bar in step with what's shown, so the view can be shared or bookmarked."""
    params = {"page": next(s for s, p in SLUGS.items() if p == page)}
    if site_id == live.CUSTOM_ID and lat is not None:
        params |= {"lat": f"{lat:.4f}", "lon": f"{lon:.4f}", "name": name}
    else:
        params["site"] = site_id.lower()
    if dict(st.query_params) != params:
        st.query_params.from_dict(params)


def main() -> None:
    st.set_page_config(
        page_title="SkyTrust · clear-sky forecasts for stargazers",
        page_icon=":material/dark_mode:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    settings, sites, metrics = load_static()
    page, site_id, lat, lon, name = top_bar(sites)
    pal = theme.palette()
    theme.inject(pal)
    try:
        site = resolve_site(site_id, lat, lon, name)
    except ValueError as exc:  # out-of-bounds custom location
        st.error(str(exc))
        return
    sync_url(page, site_id, lat, lon, name)
    needs_live = page in ("Tonight", "7 Nights")
    forecast, error = get_forecast(site_id, lat, lon, name) if needs_live else (None, None)
    summary = cached_forward_summary() if page == "Track Record" else None
    ctx = Context(
        settings,
        site,
        sites,
        metrics,
        forecast,
        error,
        pal,
        summary,
        get_forecast,
        live.utcnow(),
        load_light_pollution(),
    )
    try:
        PAGES[page](ctx)
    except Exception as exc:  # show a message instead of a stack trace
        log.exception("page %s failed", page)
        st.error(
            f"Something went wrong drawing this page ({type(exc).__name__}: {exc}). "
            "Other pages should still work; reloading usually helps after an update."
        )
    footer()


main()
