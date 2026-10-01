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

import pandas as pd
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

from skytrust import events, gazetteer, inference, lightpollution, live, sky, skyglow  # noqa: E402
from skytrust.config import CONFIG_DIR, Site, load_places, load_settings  # noqa: E402
from views import (  # noqa: E402
    accuracy,
    events_view,
    skychart,
    skyguide,
    skylive,
    theme,
    tonight,
    where,
)
from views import components as ui  # noqa: E402
from views.common import Context, fingerprint, footer, site_option_label  # noqa: E402

stamp_modules()

PAGES = {
    "Tonight": tonight.render,
    "Sky Guide": skyguide.render,
    "Events": events_view.render,
    "Where to Go": where.render,
    "Accuracy": accuracy.render,
}
SLUGS = {name.lower().replace(" ", "-"): name for name in PAGES}
# Old page names keep working in bookmarked links.
SLUGS |= {
    "7-nights": "Tonight",
    "7-night-outlook": "Tonight",
    "where-tonight": "Where to Go",
    "track-record": "Accuracy",
    "how-it-works": "Accuracy",
    "methodology": "Accuracy",
}
DEFAULT_PLACE = "los-angeles"
# Links from before the places replaced the airports keep their nearest place.
OLD_SITES = {"sac": "sacramento", "trk": "lake-tahoe"}
DEFAULT_CUSTOM = {"lat": 37.7306, "lon": -119.5738, "name": "Glacier Point"}


STATIC_FILES = (
    CONFIG_DIR / "settings.yaml",
    CONFIG_DIR / "places.yaml",
    CONFIG_DIR / "meteor_showers.yaml",
    inference.METRICS_PATH,
    gazetteer.PLACES_PATH,
)


@st.cache_resource(show_spinner=False)
def _load_static(version: str):
    """Settings, places, and backtest metrics, cached per version of the files they come from."""
    settings = load_settings()
    sites = load_places()
    try:
        metrics = inference.load_metrics()
    except (FileNotFoundError, ValueError):
        metrics = None
    return settings, sites, metrics


def load_static():
    return _load_static(fingerprint(*STATIC_FILES))


LIGHT_FILES = (
    lightpollution.GRID_PATH,
    lightpollution.BASE_PATH,
    skyglow.SOURCES_PATH,
    skyglow.CITIES_PATH,
    skyglow.MODEL_PATH,
)


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
    """The Site for a menu choice: a featured place, any California place or ZIP code from the
    gazetteer, or exact coordinates."""
    _, sites, _ = load_static()
    if site_id == live.CUSTOM_ID:
        return cached_custom_site(round(lat, 3), round(lon, 3), name)
    for s in sites:
        if s.id == site_id:
            return s
    gaz = gazetteer.load()
    if gaz is not None and site_id in gaz.table.index:
        return gaz.site(site_id)
    raise KeyError(f"unknown place {site_id!r}")


@st.cache_resource(show_spinner=False)
def _place_menu(version: str) -> tuple[list[str], dict[str, str]]:
    """Menu options and labels: the featured places (with how dark each is), exact coordinates,
    then every place in California and every ZIP code, biggest first (the gazetteer's order),
    so the likeliest match is at the top while typing. Built once per version of the files."""
    _, sites, _ = load_static()
    grid = load_light_pollution()["grid"]
    labels = {
        s.id: site_option_label(s, sky.darkness(_sqm(grid, s)) if grid is not None else None)
        for s in sites
    }
    labels[live.CUSTOM_ID] = "Exact coordinates…"
    gaz = gazetteer.load()
    if gaz is not None:
        for pid, label in gaz.labels().items():
            labels.setdefault(pid, label)  # a featured place keeps its own entry
    return list(labels), labels


def place_menu() -> tuple[list[str], dict[str, str]]:
    return _place_menu(fingerprint(*STATIC_FILES, *LIGHT_FILES))


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


# ---------- cached sky computations (shared by every visitor in this process) ----------


@st.cache_data(ttl=6 * 60 * 60, max_entries=64, show_spinner="Working out tonight's sky…")
def cached_sky(site: Site, dusk: pd.Timestamp, dawn: pd.Timestamp, sqm: float) -> dict:
    return sky.tonight(site, dusk, dawn, sqm)


@st.cache_data(ttl=6 * 60 * 60, max_entries=256, show_spinner=False)
def cached_chart(
    site: Site, utc: pd.Timestamp, sqm: float, mode: str, labels: bool, lines: bool, compact: bool
) -> tuple[str, dict]:
    return skychart.sky_svg(site, utc, sqm, mode=mode, labels=labels, lines=lines, compact=compact)


@st.cache_data(ttl=6 * 60 * 60, max_entries=48, show_spinner="Drawing tonight's sky…")
def cached_live(
    site: Site, dusk: pd.Timestamp, dawn: pd.Timestamp, sqm: float, start: pd.Timestamp
) -> dict:
    """Data for the Sky Guide's live chart. The start time is rounded to 5 minutes so the
    cache isn't defeated by the clock."""
    guide = cached_sky(site, dusk, dawn, sqm)
    return skylive.payload(site, guide, sqm, pd.Timestamp(start))


def live_chart(site: Site, dusk, dawn, sqm: float, start: pd.Timestamp) -> dict:
    return cached_live(site, dusk, dawn, sqm, pd.Timestamp(start).floor("5min"))


@st.cache_data(ttl=12 * 60 * 60, max_entries=32, show_spinner="Working out the coming sky events…")
def _cached_events(site: Site, day: str, sqm: float) -> list:
    _, places, _ = load_static()
    grid = load_light_pollution()["grid"]
    rated = [(p, _sqm(grid, p)) for p in places]
    return events.upcoming(live.utcnow(), 120, site, sqm, rated)


def _sqm(grid, site: Site) -> float:
    p = lightpollution.point(grid, site.lat, site.lon) if grid is not None else None
    return float(p["sqm"]) if p else sky.NATURAL_SQM


def cached_events(site: Site, sqm: float) -> list:
    """Events for the next four months, recomputed once a day (and per place)."""
    return _cached_events(site, live.utcnow().tz_convert(site.timezone).date().isoformat(), sqm)


def night_window(site: Site) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Tonight's dark window (or the one under way), without needing a weather forecast."""
    now = live.utcnow()
    w = events._dark_windows(site, now.tz_convert(site.timezone).date(), -18.0)
    w = w[w["dawn_utc"] > now]
    if w.empty:
        w = events._dark_windows(
            site, (now + pd.Timedelta(days=1)).tz_convert(site.timezone).date(), -18.0
        )
    row = w.iloc[0]
    return row["dusk_utc"], row["dawn_utc"]


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
    stack and the navigation wraps, so everything stays one tap away.

    The location menu is searchable: typing part of a name (or a ZIP code) narrows it to
    matching places, and picking one switches the whole app straight away."""
    query = st.query_params
    options, labels = place_menu()
    wanted = query.get("site", DEFAULT_PLACE).lower()
    wanted = OLD_SITES.get(wanted, wanted)
    if "lat" in query and "lon" in query:
        wanted = live.CUSTOM_ID
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
            index=options.index(wanted) if wanted in labels else options.index(DEFAULT_PLACE),
            key="site",
            format_func=labels.get,
            label_visibility="collapsed",
            placeholder="Type a city, town or ZIP code",
        )
    lat = lon = None
    name = ""
    if site_id == live.CUSTOM_ID:
        lat, lon, name = coordinates_panel(query)
    return page or "Tonight", site_id, lat, lon, name


def coordinates_panel(query) -> tuple[float, float, str]:
    """Latitude/longitude boxes for a spot with no town nearby (a trailhead, a dark site). The
    spot is named after the nearest place, or keeps the name given in a shared link."""
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
        c1, c2 = st.columns(2)
        lat = c1.number_input("Latitude", la0, la1, lat0, 0.01, key="lat", format="%.4f")
        lon = c2.number_input("Longitude", lo0, lo1, lon0, 0.01, key="lon", format="%.4f")
        st.caption(
            "For a spot with no town nearby. Type a value and press Enter (or use − / +). "
            "Anywhere in the Pacific-time West; forecasts use the version of the model built for "
            "new places."
        )
    given = clean_name(query.get("name", "")) if from_link else ""
    if given and (lat, lon) == (lat0, lon0):
        return lat, lon, given
    if (lat, lon) == (DEFAULT_CUSTOM["lat"], DEFAULT_CUSTOM["lon"]):
        return lat, lon, DEFAULT_CUSTOM["name"]
    gaz = gazetteer.load()
    return lat, lon, gaz.describe(lat, lon) if gaz is not None else "Custom location"


def sync_url(page: str, site_id: str, lat, lon, name: str) -> None:
    """Keep the address bar in step with what's shown, so the view can be shared or bookmarked."""
    params = {"page": next(s for s, p in SLUGS.items() if p == page)}
    if site_id == live.CUSTOM_ID and lat is not None:
        params |= {"lat": f"{lat:.4f}", "lon": f"{lon:.4f}"}
        if st.query_params.get("name"):  # a name given in a shared link travels with the spot
            params["name"] = name
    else:
        params["site"] = site_id.lower()
    if dict(st.query_params) != params:
        st.query_params.from_dict(params)


def main() -> None:
    st.set_page_config(
        page_title="SkyTrust · stargazing forecasts for California",
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
    except (ValueError, KeyError) as exc:  # out-of-bounds custom location, unknown place
        st.error(str(exc))
        return
    sync_url(page, site_id, lat, lon, name)
    needs_live = page == "Tonight"
    forecast, error = get_forecast(site_id, lat, lon, name) if needs_live else (None, None)
    summary = cached_forward_summary() if page == "Accuracy" else None
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
        {
            "sky": cached_sky,
            "chart": cached_chart,
            "events": cached_events,
            "night": night_window,
            "live": live_chart,
        },
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
