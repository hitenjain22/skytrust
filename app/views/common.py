"""Shared UI helpers: the page context, friendly names and times, and plain-English wording.

The wording functions turn forecast numbers into sentences a first-time stargazer can act on.
They only describe what the forecast already contains; they never change a probability.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from skytrust import sky
from skytrust.config import Settings, Site
from views.components import esc, pill
from views.theme import DAY, MOON, palette  # noqa: F401  (re-exported)

# The evaluated airports, by the place they're near (Track Record page, CLI links).
AIRPORTS = {
    "SAC": "Sacramento airport",
    "FAT": "Fresno airport",
    "AUN": "Auburn airport",
    "TRK": "Truckee airport",
    "BIH": "Bishop airport",
}

VERDICTS = {
    "Go": ("Clear skies tonight", "A good night to head out."),
    "Maybe": ("Worth a try", "It could go either way: have a backup plan and check again later."),
    "Skip": ("Probably cloudy", "Clouds are likely to get in the way. A good night to plan."),
    "No data": ("No forecast right now", "The weather models didn't return data for this night."),
}


@dataclass
class Context:
    settings: Settings
    site: Site
    sites: tuple[Site, ...]  # the places offered in the app (config/places.yaml)
    metrics: dict | None
    forecast: object | None  # live.LiveForecast
    forecast_error: str | None
    palette: dict
    forward_summary: dict | None = None
    forecast_for: object = None  # callable(site_id) -> (forecast, error), for multi-site pages
    now_utc: pd.Timestamp | None = None
    light_data: dict | None = None  # {"grid", "base", "sources", "model"}; see load_light_pollution
    services: dict | None = None  # cached computations from the app: sky, chart, events, night

    @property
    def light(self):
        """The latest light-pollution grid (lightpollution.Grid) or None."""
        return (self.light_data or {}).get("grid")

    def light_at(self, site: Site | None = None, glow: bool = True) -> dict | None:
        """Light-pollution report for a site (default: the selected one), with change since the
        2015 atlas and city glow when the data are available."""
        from skytrust import lightpollution, skyglow

        site = site or self.site
        if self.light is None:
            return None
        data = self.light_data or {}
        report = lightpollution.site_report(
            self.light, site.lat, site.lon, self.settings, data.get("base")
        )
        if glow and report is not None and data.get("sources") is not None:
            report["glow"] = skyglow.city_glow(data["sources"], site.lat, site.lon)
        return report

    def sqm_at(self, site: Site | None = None) -> float:
        """Moonless zenith sky brightness here (mag/arcsec²); a natural sky if unknown."""
        from skytrust import lightpollution

        site = site or self.site
        if self.light is None:
            return sky.NATURAL_SQM
        p = lightpollution.point(self.light, site.lat, site.lon)
        return float(p["sqm"]) if p else sky.NATURAL_SQM

    def darkness(self, site: Site | None = None) -> sky.Darkness:
        return sky.darkness(self.sqm_at(site))

    def service(self, name: str):
        return (self.services or {})[name]

    @property
    def site_label(self) -> str:
        return place_name(self.site)

    @property
    def nights(self) -> list:
        """Forecast nights that haven't ended yet. A forecast cached before dawn would otherwise
        keep showing last night as "tonight" until the cache refreshes."""
        return upcoming(self.forecast, self.now_utc)


def upcoming(forecast, now_utc: pd.Timestamp | None) -> list:
    if forecast is None:
        return []
    now = now_utc if now_utc is not None else pd.Timestamp.now(tz="UTC")
    return [n for n in forecast.nights if n.dawn_utc > now]


def place_name(site: Site) -> str:
    return AIRPORTS.get(site.id, site.name)


def place_detail(site: Site) -> str:
    """Region and elevation (from config/places.yaml), or coordinates for a custom spot."""
    if site.terrain_class and site.terrain_class != "custom":
        return f"{site.terrain_class} · {site.elevation_m:,.0f} m"
    return f"{site.lat:.3f}, {site.lon:.3f}"


def site_option_label(site: Site, darkness: sky.Darkness | None = None) -> str:
    """'Death Valley · very dark sky' in the location menu."""
    if darkness is None:
        return place_name(site)
    return f"{place_name(site)} · {darkness.title.lower()} sky"


# ---------- page helpers ----------


PLOT_CONFIG = {"displayModeBar": False, "responsive": True}


def show(fig) -> None:
    st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG, theme="streamlit")


def unavailable(ctx: Context) -> bool:
    """Show a friendly message (and stop) when there's no forecast at all."""
    if ctx.forecast is None or not ctx.nights:
        reason = f" ({ctx.forecast_error})" if ctx.forecast_error else ""
        st.warning(
            f"Live forecasts are unavailable right now and there's no saved copy yet.{reason} "
            "The Sky Guide, Events and Accuracy pages still work."
        )
        return True
    if ctx.forecast.warning:
        st.warning(ctx.forecast.warning)
    return False


# ---------- formatting ----------


def local_time(ts: pd.Timestamp, tz: str, fmt: str = "%-I:%M %p") -> str:
    return ts.tz_convert(tz).strftime(fmt)


def short_time(ts: pd.Timestamp, tz: str) -> str:
    """'9 PM' on the hour, '8:18 PM' otherwise."""
    t = ts.tz_convert(tz)
    return t.strftime("%-I %p") if t.minute == 0 else t.strftime("%-I:%M %p")


def duration(start: pd.Timestamp, end: pd.Timestamp) -> str:
    minutes = max(0, round((end - start).total_seconds() / 60))
    h, m = divmod(minutes, 60)
    if h and m:
        return f"{h} h {m} min"
    return f"{h} h" if h else f"{m} min"


def pct(x) -> str:
    return "–" if x is None or pd.isna(x) else f"{x:.0%}"


def with_ci(d: dict | None, metric: str, kind: str = "pct") -> str:
    if not d or d.get(metric) is None or pd.isna(d.get(metric)):
        return "–"
    fmt = (lambda v: f"{v:.0%}") if kind == "pct" else (lambda v: f"{v:.2f}")
    lo, hi = d.get(f"{metric}_lo"), d.get(f"{metric}_hi")
    if lo is None or pd.isna(lo):
        return fmt(d[metric])
    return f"{fmt(d[metric])} ({fmt(lo)}–{fmt(hi)})"


def day_label(night_date: dt.date, index: int, now_utc: pd.Timestamp | None, tz: str) -> str:
    """'Tonight', 'Tomorrow', or 'Fri'. Index 0 is always tonight (or the night in progress)."""
    if index == 0:
        return "Tonight"
    today = (now_utc or pd.Timestamp.now(tz="UTC")).tz_convert(tz).date()
    if night_date == today + dt.timedelta(days=1):
        return "Tomorrow"
    return f"{pd.Timestamp(night_date):%a}"


def lead_phrase(lead: int) -> str:
    return "tonight" if lead == 1 else f"{lead} days ahead"


def verdict_badge(verdict: str, pal: dict) -> str:
    return pill(verdict.upper(), pal[verdict])


# ---------- plain-English description of a night ----------


def moon_up_in_dark(night) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Moon-up stretches clipped to astronomical darkness."""
    out = []
    for a, b in getattr(night, "moon_up", []) or []:
        a, b = max(a, night.dusk_utc), min(b, night.dawn_utc)
        if b > a:
            out.append((a, b))
    return out


def _moon_minutes(night) -> tuple[float, float]:
    """(minutes the Moon is up during darkness, minutes of darkness)."""
    runs = moon_up_in_dark(night)
    dark = (night.dawn_utc - night.dusk_utc).total_seconds() / 60
    return sum((b - a).total_seconds() / 60 for a, b in runs), dark


def moon_sentence(night, tz: str) -> str:
    """When the Moon is up during darkness (a fact, no advice)."""
    illum = night.moon_illum or 0.0
    runs = moon_up_in_dark(night)
    if not runs:
        return "The Moon stays below the horizon all night."
    up_min, dark_min = _moon_minutes(night)
    first, last = runs[0][0], runs[-1][1]
    if up_min >= dark_min - 10:
        when = "up all night"
    elif first <= night.dusk_utc + pd.Timedelta(minutes=10):
        when = f"up until {short_time(last, tz)}"
    elif last >= night.dawn_utc - pd.Timedelta(minutes=10):
        when = f"rising at {short_time(first, tz)}"
    else:
        when = f"up from {short_time(first, tz)} to {short_time(last, tz)}"
    return f"The Moon ({illum:.0%} lit) is {when}."


def moon_advice(night) -> tuple[str, str] | None:
    """(kind, text) when the Moon changes what's worth shooting tonight, else None.
    kind is "bright" (a warning) or "dark" (good news for faint targets)."""
    illum = night.moon_illum or 0.0
    up_min, dark_min = _moon_minutes(night)
    if illum >= 0.5 and up_min >= 0.5 * dark_min:
        return (
            "bright",
            "Bright Moon for most of the dark hours: the Milky Way and faint galaxies will be "
            "washed out. The Moon itself, planets and bright stars are fine.",
        )
    if up_min == 0 or illum < 0.15:
        return "dark", "No Moon in the dark hours: the best kind of night for the Milky Way."
    return None


def night_summary(night, tz: str, threshold: float) -> str:
    """One or two sentences: when it's clear, and the Moon."""
    bw = night.best_window
    if bw:
        clear = (
            f"Clearest stretch: <b>{short_time(bw.start_utc, tz)} – "
            f"{short_time(bw.until_utc, tz)}</b> ({duration(bw.start_utc, bw.until_utc)} where "
            "the typical model shows clear sky)."
        )
    else:
        clear = (
            f"No dark hour looks clear (≤ {threshold:.0%} cloud) in the middle-of-the-road model "
            "forecast."
        )
    return f"{clear} {esc(moon_sentence(night, tz))}"


def agreement_text(night) -> tuple[str, str]:
    """(short label, explanation) for the model-spread indicator."""
    if night.agreement == "Models agree":
        return "Models agree", "The five weather models tell a similar story, so trust is higher."
    if night.agreement == "Models split":
        return "Models disagree", "The weather models disagree, so treat this one with caution."
    return "Few models", "Only one model has data for this night."


def footer() -> None:
    st.divider()
    st.caption(
        "Weather: [Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0). Observations: the "
        "[Iowa Environmental Mesonet](https://mesonet.agron.iastate.edu/). Satellite: NOAA "
        "GOES-18. Light pollution: Falchi et al. (2016), [World Atlas of Artificial Night Sky "
        "Brightness](https://doi.org/10.5880/GFZ.1.4.2016.001) (CC BY-NC 4.0), updated with NASA "
        "Black Marble night lights (CC0). Positions: JPL DE421 via Skyfield; stars from the ESA "
        "Hipparcos catalogue; constellation figures and Milky Way outline from "
        "[d3-celestial](https://github.com/ofrohn/d3-celestial) (BSD). Meteor showers: the "
        "[IMO](https://www.imo.net/) 2026 calendar. Places: [GeoNames](https://www.geonames.org/) "
        "(CC BY 4.0). SkyTrust is a student project by Hiten Jain, not an official forecast. "
        "[Code](https://github.com/hitenjain22/skytrust) · "
        "[Full results](https://github.com/hitenjain22/skytrust/blob/main/docs/RESULTS.md)"
    )


# ---------- light pollution in plain English ----------


def brightness_phrase(ratio: float) -> str:
    """'39% brighter than a natural sky' or '18x the natural sky brightness'."""
    if ratio < 0.01:
        return "as dark as a natural sky"
    if ratio < 1:
        return f"{ratio:.0%} brighter than a natural sky"
    return f"{1 + ratio:.0f}× the natural sky brightness"


def maps_link(lat: float, lon: float) -> str:
    return f"https://www.google.com/maps/search/?api=1&query={lat:.4f},{lon:.4f}"


def dark_clear_hours(night) -> float | None:
    """Expected hours tonight that are dark, clear and moonless: the sum over dark hours of the
    hourly model's P(clear), counting only hours with the Moon below the horizon."""
    p = getattr(night, "hourly_clear", None)
    if p is None or p.empty:
        return None
    up = moon_up_in_dark(night)
    down = [not any(a <= t < b for a, b in up) for t in p.index]
    return float((p.to_numpy() * down).sum())


def conditions(night, sqm: float | None, settings: Settings) -> list[tuple[str, bool | None, str]]:
    """The three things a good night of stargazing needs, each met or not: (name, met, detail)."""
    go_at = settings.raw["verdict"]["go"]
    dark_sqm = settings.raw["light_pollution"]["dark_sqm"]
    min_run = settings.min_run_hours
    p = night.p_usable
    clear = (
        "Clear",
        None if p is None else p >= go_at,
        "no forecast" if p is None else f"{p:.0%} chance of a clear night",
    )
    if sqm is not None:
        d = sky.darkness(sqm)
        dark = ("Dark sky", sqm >= dark_sqm, f"{d.title}: {sky.brightness_words(d.times_natural)}")
    else:
        dark = ("Dark sky", None, "no light-pollution data")
    moon = (
        "Moon down",
        night.moon_free_hours >= min_run,
        f"{night.moon_free_hours} moon-free dark hour{'s' if night.moon_free_hours != 1 else ''}",
    )
    return [clear, dark, moon]


def fingerprint(*paths: Path) -> str:
    """Modification time and size of each file, as a cache key. Streamlit Cloud reloads the code
    on a push without restarting the server, so anything cached per process must be keyed on
    the files it was read from, or a new page can meet old data (that caused a KeyError on
    Where to Go when the light-pollution settings were added)."""
    parts = []
    for p in paths:
        try:
            st_ = p.stat()
            parts.append(f"{p.name}:{st_.st_mtime_ns}:{st_.st_size}")
        except FileNotFoundError:
            parts.append(f"{p.name}:missing")
    return "|".join(parts)
