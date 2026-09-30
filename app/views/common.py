"""Shared UI helpers: the page context, friendly names and times, and plain-English wording.

The wording functions turn forecast numbers into sentences a first-time stargazer can act on.
They only describe what the forecast already contains; they never change a probability.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd
import streamlit as st

from skytrust.config import Settings, Site
from views.components import esc, pill
from views.theme import DAY, MOON, NIGHT, palette  # noqa: F401  (re-exported)

# The airport codes mean nothing to most people; these are the places they're near.
PLACES = {
    "SAC": "Sacramento",
    "FAT": "Fresno",
    "AUN": "Auburn",
    "TRK": "Truckee · Lake Tahoe",
    "BIH": "Bishop",
}

VERDICTS = {
    "Go": ("Great night to shoot", "Clear skies are likely. Pack the gear."),
    "Maybe": ("Worth a try", "It could go either way. Have a backup plan and check again later."),
    "Skip": ("Probably cloudy", "Clouds are likely to spoil it. A good night to plan instead."),
    "No data": ("No forecast right now", "The weather models didn't return data for this night."),
}


@dataclass
class Context:
    settings: Settings
    site: Site
    sites: tuple[Site, ...]
    metrics: dict | None
    forecast: object | None  # live.LiveForecast
    forecast_error: str | None
    palette: dict
    forward_summary: dict | None = None
    forecast_for: object = None  # callable(site_id) -> (forecast, error), for multi-site pages
    now_utc: pd.Timestamp | None = None

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
    return PLACES.get(site.id, site.name)


def place_detail(site: Site) -> str:
    """Terrain and elevation straight from config/sites.yaml."""
    if site.id in PLACES:
        return f"{site.terrain_class} · {site.elevation_m:,.0f} m"
    return f"{site.lat:.3f}, {site.lon:.3f}"


def site_option_label(site: Site) -> str:
    return f"{place_name(site)} ({site.id})" if site.id in PLACES else place_name(site)


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
            "Bright Moon for most of the dark hours: faint galaxies and nebulae will look washed "
            "out. The Moon, planets, star clusters and narrowband imaging are fine.",
        )
    if up_min == 0 or illum < 0.15:
        return "dark", "Moonless dark sky: ideal for faint galaxies, nebulae and the Milky Way."
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
        "Weather data by [Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0). "
        "ASOS observations courtesy of the "
        "[Iowa Environmental Mesonet](https://mesonet.agron.iastate.edu/), "
        "Iowa State University. Satellite: NOAA GOES-18. "
        "SkyTrust is a student portfolio project by Hiten Jain, not an official forecast. "
        "[Code](https://github.com/hitenjain22/skytrust) · "
        "[Full results](https://github.com/hitenjain22/skytrust/blob/main/docs/RESULTS.md)"
    )
