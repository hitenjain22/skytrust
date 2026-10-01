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
import os
import re
import sys
from pathlib import Path

import streamlit as st

log = logging.getLogger("skytrust.app")

# Packages whose modules must match the files on disk. Streamlit Cloud pulls new code into the
# running process; it re-reads this script but can keep the old copies of modules it already
# imported, so new code calls old code (ImportError / KeyError on the live site). Before
# importing anything of ours, drop every module of these packages if any of them changed.
OUR_PACKAGES = ("views", "skytrust")
STAMP = "__skytrust_stamp__"


def _our_modules() -> dict:
    return {
        name: module
        for name, module in list(sys.modules.items())
        if name.split(".")[0] in OUR_PACKAGES and getattr(module, "__file__", None)
    }


def _is_stale(module) -> bool:
    """Has the module's source file changed since it was loaded? Uses our own stamp when we set
    one, else the source mtime/size Python records in the .pyc header at import (PEP 552)."""
    try:
        now = os.stat(module.__file__)
    except OSError:
        return True  # file removed by the update
    stamp = getattr(module, STAMP, None)
    if stamp is not None:
        return tuple(stamp) != (now.st_mtime_ns, now.st_size)
    try:
        header = Path(module.__cached__).read_bytes()[:16]
    except (AttributeError, TypeError, OSError):
        return False  # no record: assume current
    if len(header) < 16 or int.from_bytes(header[4:8], "little") != 0:
        return False  # hash-based .pyc: no timestamp to compare
    mtime, size = int.from_bytes(header[8:12], "little"), int.from_bytes(header[12:16], "little")
    return (mtime, size) != (int(now.st_mtime) & 0xFFFFFFFF, now.st_size & 0xFFFFFFFF)


def reload_changed_modules() -> None:
    modules = _our_modules()
    changed = [name for name, module in modules.items() if _is_stale(module)]
    if not changed:
        return
    log.warning("code changed on disk (%s); reloading our modules", ", ".join(sorted(changed)))
    for name in modules:
        del sys.modules[name]
    # cached objects were built by the old code (old classes, old settings)
    st.cache_resource.clear()
    st.cache_data.clear()


def stamp_modules() -> None:
    for module in _our_modules().values():
        if getattr(module, STAMP, None) is None:
            now = os.stat(module.__file__)
            setattr(module, STAMP, (now.st_mtime_ns, now.st_size))


reload_changed_modules()

try:  # installed via `uv sync` locally; on a host that only installs dependencies, use src/
    import skytrust  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skytrust import inference, lightpollution, live, skyglow  # noqa: E402
from skytrust.config import CONFIG_DIR, load_settings, load_sites  # noqa: E402
from views import components as ui  # noqa: E402
from views import methodology, outlook, theme, tonight, track_record, where  # noqa: E402
from views.common import Context, fingerprint, footer, site_option_label  # noqa: E402

stamp_modules()

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


def link_coords(query) -> tuple[float, float, bool]:
    """Latitude/longitude from a shared link, or the default spot if they're missing, not
    numbers, or outside the area SkyTrust covers. The flag says whether the link's were used."""
    (la0, la1), (lo0, lo1) = live.CUSTOM_BOUNDS["lat"], live.CUSTOM_BOUNDS["lon"]
    try:
        lat, lon = float(query.get("lat", "")), float(query.get("lon", ""))
    except ValueError:
        return DEFAULT_CUSTOM["lat"], DEFAULT_CUSTOM["lon"], False
    if la0 <= lat <= la1 and lo0 <= lon <= lo1:  # also False for nan
        return lat, lon, True
    return DEFAULT_CUSTOM["lat"], DEFAULT_CUSTOM["lon"], False


def clean_name(name: str) -> str:
    """A place name from a link or the name box, reduced to plain text: it is shown in headings
    (markdown), so brackets, asterisks and the like could otherwise turn into links or styling."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s.,'’()&/-]", "", name)).strip()[:60]


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
        brand, nav, where = st.columns([0.9, 3.4, 1.7], vertical_alignment="center", gap="small")
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
        lat0, lon0, from_link = link_coords(query)
        with st.container(key="panel_custom"):
            if not from_link and "lat" in query and "lat" not in st.session_state:
                st.warning(
                    f"The location in this link couldn't be used (SkyTrust covers latitude "
                    f"{la0:g}–{la1:g}, longitude {lo0:g}–{lo1:g}), so it shows "
                    f"{DEFAULT_CUSTOM['name']} instead.",
                    icon=":material/wrong_location:",
                )
            c1, c2, c3 = st.columns([1, 1, 1.4])
            lat = c1.number_input("Latitude", la0, la1, lat0, 0.01, key="lat")
            lon = c2.number_input("Longitude", lo0, lo1, lon0, 0.01, key="lon")
            default_name = query.get("name", DEFAULT_CUSTOM["name"])[:60] if from_link else ""
            name = clean_name(
                c3.text_input(
                    "Name", default_name or DEFAULT_CUSTOM["name"], max_chars=60, key="name"
                )
            )
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
