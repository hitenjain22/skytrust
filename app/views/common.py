"""Shared UI helpers: palettes (normal + red "night vision"), formatting, footer."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st

from skytrust.config import Settings, Site

DAY = {
    "bg": "#0b0f19",
    "paper": "#0b0f19",
    "text": "#e6e6e6",
    "muted": "#9aa4b2",
    "grid": "#253049",
    "accent": "#4ea1ff",
    "Go": "#2ecc71",
    "Maybe": "#f1c40f",
    "Skip": "#e74c3c",
    "No data": "#7f8c8d",
    "dark": "rgba(78,161,255,0.12)",
    "moon": "#f5f0c8",
    # Paul Tol's colour-blind-safe palettes (lighter variants for the dark background).
    "models": {
        "gfs": "#88CCEE",
        "hrrr": "#DDCC77",
        "ecmwf": "#44AA99",
        "gem": "#999933",
        "icon": "#CC6677",
        "median": "#ffffff",
    },
    "methods": {
        "blend": "#EE6677",
        "equal_weight": "#4ea1ff",
        "climatology": "#8C8C8C",
        "nbm_lr": "#EE3377",
        "equal_weight_cal": "#33BBEE",
    },
}
# Astronomers use dim red light to keep their eyes dark-adapted.
NIGHT = {
    "bg": "#000000",
    "paper": "#000000",
    "text": "#ff3b30",
    "muted": "#a3261f",
    "grid": "#3a0000",
    "accent": "#ff3b30",
    "Go": "#ff5a4f",
    "Maybe": "#c0392b",
    "Skip": "#7f1d1d",
    "No data": "#5a1a1a",
    "dark": "rgba(255,59,48,0.10)",
    "moon": "#ff8a80",
    "models": {
        k: c
        for k, c in zip(
            ["gfs", "hrrr", "ecmwf", "gem", "icon"],
            ["#ff6b60", "#e04b40", "#c0392b", "#a93226", "#8b1e1e"],
            strict=True,
        )
    }
    | {"median": "#ff3b30"},
    "methods": {
        "blend": "#ff3b30",
        "equal_weight": "#c0392b",
        "climatology": "#6b1a1a",
        "nbm_lr": "#a93226",
    },
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

    @property
    def site_label(self) -> str:
        return self.site.name if self.site.id.startswith("CUSTOM") else self.site.id


def palette(night_vision: bool) -> dict:
    return NIGHT if night_vision else DAY


def inject_night_vision_css() -> None:
    st.markdown(
        """
        <style>
        .stApp, [data-testid="stSidebar"], [data-testid="stHeader"] { background: #000 !important; }
        .stApp *, [data-testid="stSidebar"] * {
            color: #ff3b30 !important; border-color: #3a0000 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def local_time(ts: pd.Timestamp, tz: str, fmt: str = "%-I:%M %p") -> str:
    return ts.tz_convert(tz).strftime(fmt)


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


def verdict_badge(verdict: str, pal: dict) -> str:
    return (
        f"<span style='background:{pal[verdict]};color:#000;padding:4px 14px;border-radius:14px;"
        f"font-weight:700;font-size:1.1rem'>{verdict.upper()}</span>"
    )


def footer() -> None:
    st.divider()
    st.caption(
        "Weather data by [Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0). "
        "ASOS observations courtesy of the "
        "[Iowa Environmental Mesonet](https://mesonet.agron.iastate.edu/), "
        "Iowa State University. SkyTrust is a student portfolio project, not an official forecast."
    )
