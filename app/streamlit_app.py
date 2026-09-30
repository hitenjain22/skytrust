"""SkyTrust Streamlit app.

The app only reads precomputed artifacts (the blend JSON models and metrics.json) plus the
live Open-Meteo forecast. It never trains or backfills (SPEC 10). Any failure on a page is
caught and shown as a message; the app itself must never crash.

Navigation and the location picker live at the top of the page (not in a sidebar), so they're
one tap away on a phone. The URL always reflects the current page and location, so any view can
be bookmarked or shared.
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
from views import components as ui  # noqa: E402
from views import methodology, outlook, theme, tonight, track_record, where  # noqa: E402
from views.common import Context, footer, site_option_label  # noqa: E402

log = logging.getLogger("skytrust.app")

PAGES = {
    "Tonight": tonight.render,
    "7 Nights": outlook.render,
    "Where to Go": where.render,
    "Track Record": track_record.render,
    "How It Works": methodology.render,
}
ICONS = {
    "Tonight": ":material/bedtime:",
    "7 Nights": ":material/calendar_month:",
    "Where to Go": ":material/explore:",
    "Track Record": ":material/verified:",
    "How It Works": ":material/menu_book:",
}
SLUGS = {name.lower().replace(" ", "-"): name for name in PAGES}
# Old page names keep working in bookmarked links.
SLUGS |= {
    "7-night-outlook": "7 Nights",
    "where-tonight": "Where to Go",
    "methodology": "How It Works",
}
DEFAULT_CUSTOM = {"lat": 37.7306, "lon": -119.5738, "name": "Glacier Point"}


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


def top_bar(sites) -> tuple[str, str, float | None, float | None, str, bool]:
    """Brand, night-vision switch, location picker, and page navigation."""
    query = st.query_params
    brand, switch = st.columns([4, 1.3], vertical_alignment="center")
    brand.markdown(
        ui.block(
            '<div class="sk-brand">',
            ui.moon_svg(200.0, 34),
            '<div><div class="sk-brand-name">SkyTrust</div>',
            '<div class="sk-brand-tag">Clear-sky forecasts for stargazers, with an honest '
            "track record</div></div></div>",
        ),
        unsafe_allow_html=True,
    )
    if "night_vision" not in st.session_state:  # first load: honour ?nv=1 from a bookmark
        st.session_state["night_vision"] = query.get("nv") == "1"
    night_vision = switch.toggle(
        "Night vision",
        key="night_vision",
        help="Dim red display that keeps your eyes dark-adapted at the telescope.",
    )

    site_ids = [s.id for s in sites]
    options = [*site_ids, live.CUSTOM_ID]
    wanted = query.get("site", "SAC").upper()
    if "lat" in query and "lon" in query:
        wanted = live.CUSTOM_ID
    labels = {s.id: site_option_label(s) for s in sites} | {live.CUSTOM_ID: "📍 Custom location…"}
    where_col, nav_col = st.columns([1.25, 3], vertical_alignment="bottom")
    site_id = where_col.selectbox(
        "Location",
        options,
        index=options.index(wanted) if wanted in options else 0,
        key="site",
        format_func=labels.get,
    )
    start = SLUGS.get(query.get("page", ""), "Tonight")
    page = nav_col.segmented_control(
        "Page",
        list(PAGES),
        default=None if "page" in st.session_state else start,
        required=True,
        key="page",
        format_func=lambda p: f"{ICONS[p]} {p}",
        label_visibility="hidden",
        width="stretch",
        wrap=True,  # on a phone the five pages flow onto two lines instead of off-screen
    )
    lat = lon = None
    name = ""
    if site_id == live.CUSTOM_ID:
        (la0, la1), (lo0, lo1) = live.CUSTOM_BOUNDS["lat"], live.CUSTOM_BOUNDS["lon"]
        with st.container(border=True):
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
    return page or "Tonight", site_id, lat, lon, name, night_vision


def sync_url(page: str, site_id: str, lat, lon, name: str, night_vision: bool) -> None:
    """Keep the address bar in step with what's shown, so the view can be shared or bookmarked
    (including the red night-vision display, for opening straight into it at the telescope)."""
    params = {"page": next(s for s, p in SLUGS.items() if p == page)}
    if site_id == live.CUSTOM_ID and lat is not None:
        params |= {"lat": f"{lat:.4f}", "lon": f"{lon:.4f}", "name": name}
    else:
        params["site"] = site_id.lower()
    if night_vision:
        params["nv"] = "1"
    if dict(st.query_params) != params:
        st.query_params.from_dict(params)


def main() -> None:
    st.set_page_config(
        page_title="SkyTrust · clear-sky forecasts for stargazers",
        page_icon="🌙",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    settings, sites, metrics = load_static()
    page, site_id, lat, lon, name, night_vision = top_bar(sites)
    pal = theme.palette(night_vision)
    theme.inject(pal, night_vision)
    try:
        site = resolve_site(site_id, lat, lon, name)
    except ValueError as exc:  # out-of-bounds custom location
        st.error(str(exc))
        return
    sync_url(page, site_id, lat, lon, name, night_vision)
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
