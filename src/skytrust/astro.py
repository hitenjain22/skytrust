"""Astronomy: dark windows (astronomical dusk -> dawn), dark hours, and the Moon.

Uses Skyfield with a committed excerpt of JPL's DE421 ephemeris covering 2023-2030, so
nothing here touches the network (tests, CI, and the deployed app all work offline).
"""

from __future__ import annotations

import datetime as dt
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from skyfield import almanac
from skyfield.api import load, load_file, wgs84

from skytrust.config import REPO_ROOT, Site

log = logging.getLogger(__name__)

EPHEMERIS_PATH = REPO_ROOT / "data" / "ephemeris" / "de421_2023_2030.bsp"
SEARCH_STEP_DAYS = 1 / 24  # sample hourly; twilight crossings are ~>1 h apart, so none are missed


@lru_cache(maxsize=1)
def _ephemeris(path: Path = EPHEMERIS_PATH):
    return load_file(str(path))


@lru_cache(maxsize=1)
def _timescale():
    # builtin=True uses Skyfield's bundled leap-second/Delta-T tables: no download.
    return load.timescale(builtin=True)


def _topos(site: Site):
    return wgs84.latlon(site.lat, site.lon, elevation_m=site.elevation_m)


def _to_skyfield(times: pd.DatetimeIndex):
    if times.tz is None:
        raise ValueError("times must be timezone-aware (UTC)")
    return _timescale().from_datetimes(times.tz_convert("UTC").to_pydatetime())


def _to_utc_index(t) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(t.utc_datetime(), utc=True))


def _altitude_deg(site: Site, body_name: str, t) -> np.ndarray:
    """Geometric altitude (no atmospheric refraction), the convention for twilight angles."""
    eph = _ephemeris()
    observer = eph["earth"] + _topos(site)
    alt, _, _ = observer.at(t).observe(eph[body_name]).apparent().altaz()
    return np.atleast_1d(alt.degrees)


def dark_windows(
    site: Site, start: dt.date, end: dt.date, sun_altitude_deg: float = -18.0
) -> pd.DataFrame:
    """One row per night_date in [start, end] with `dusk_utc` and `dawn_utc`.

    A night is keyed by the local date of the evening it begins (SPEC 4.1). We search from
    local noon on `start` to local noon after `end` for moments the Sun crosses
    `sun_altitude_deg`: descending = dusk, ascending = dawn. A night with no darkness gets
    NaT for both and is logged; callers must exclude it (not treat it as cloudy).
    """
    ts = _timescale()
    tz = site.timezone
    t0 = pd.Timestamp(start).tz_localize(tz) + pd.Timedelta(hours=12)
    t1 = pd.Timestamp(end + dt.timedelta(days=1)).tz_localize(tz) + pd.Timedelta(hours=12)

    def is_dark(t) -> np.ndarray:
        return _altitude_deg(site, "sun", t) < sun_altitude_deg

    is_dark.step_days = SEARCH_STEP_DAYS
    times, became_dark = almanac.find_discrete(
        ts.from_datetime(t0.to_pydatetime()), ts.from_datetime(t1.to_pydatetime()), is_dark
    )
    crossings = _to_utc_index(times) if len(times) else pd.DatetimeIndex([], tz="UTC")

    nights = pd.date_range(start, end, freq="D").date
    result = pd.DataFrame(
        {"dusk_utc": pd.NaT, "dawn_utc": pd.NaT},
        index=pd.Index(nights, name="night_date"),
        dtype="datetime64[ns, UTC]",
    )
    pending_dusk = None
    for when, dark in zip(crossings, became_dark, strict=True):
        if dark:
            pending_dusk = when
        elif pending_dusk is not None:  # a dawn that closes an open dusk
            night = pending_dusk.tz_convert(tz).date()
            if night in result.index:
                result.loc[night] = [pending_dusk, when]
            pending_dusk = None
    for night in result.index[result["dusk_utc"].isna()]:
        log.warning("%s night %s has no astronomical darkness; excluded", site.id, night)
    return result


def dark_hours(dusk_utc: pd.Timestamp, dawn_utc: pd.Timestamp) -> pd.DatetimeIndex:
    """Top-of-hour UTC timestamps H with dusk <= H <= dawn (SPEC 4.3). Empty if no darkness.

    Built in UTC, where every hour exists exactly once, so DST changes can't create
    duplicated or missing hours.
    """
    if pd.isna(dusk_utc) or pd.isna(dawn_utc):
        return pd.DatetimeIndex([], tz="UTC")
    return pd.date_range(dusk_utc.ceil("h"), dawn_utc.floor("h"), freq="h")


def moon_illumination(times: pd.DatetimeIndex) -> np.ndarray:
    """Illuminated fraction of the Moon's disk (0 = new, 1 = full) at each time."""
    eph = _ephemeris()
    return np.atleast_1d(almanac.fraction_illuminated(eph, "moon", _to_skyfield(times)))


def moon_altitude(site: Site, times: pd.DatetimeIndex) -> np.ndarray:
    return _altitude_deg(site, "moon", _to_skyfield(times))


def moon_events(site: Site, start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> pd.DataFrame:
    """Moonrise/moonset times between two UTC instants (for the live view)."""
    ts = _timescale()
    eph = _ephemeris()
    f = almanac.risings_and_settings(eph, eph["moon"], _topos(site))
    times, is_rise = almanac.find_discrete(
        ts.from_datetime(start_utc.to_pydatetime()), ts.from_datetime(end_utc.to_pydatetime()), f
    )
    if not len(times):
        return pd.DataFrame({"time_utc": pd.DatetimeIndex([], tz="UTC"), "event": []})
    return pd.DataFrame(
        {"time_utc": _to_utc_index(times), "event": np.where(is_rise, "rise", "set")}
    )


def night_hours(windows: pd.DataFrame) -> pd.DataFrame:
    """Long table with one row per (night_date, dark hour), from `dark_windows` output.

    This is the backbone that labels and features are computed on: every hourly source is
    looked up at exactly these timestamps, so all of them describe the same dark window.
    Nights without darkness contribute no rows.
    """
    parts = [
        pd.DataFrame({"night_date": night, "hour": dark_hours(row.dusk_utc, row.dawn_utc)})
        for night, row in windows.iterrows()
    ]
    if not parts:
        return pd.DataFrame({"night_date": [], "hour": pd.DatetimeIndex([], tz="UTC")})
    return pd.concat(parts, ignore_index=True)


def night_table(
    site: Site,
    start: dt.date,
    end: dt.date,
    sun_altitude_deg: float = -18.0,
    moon_up_altitude_deg: float = 0.0,
    windows: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-night astronomy plus the long (night_date, hour) table it was built from.

    Per night: dusk, dawn, number of dark hours, mean Moon illumination over the dark hours,
    and how many dark hours have the Moon below `moon_up_altitude_deg` (moon-free hours).
    Pass `windows` (from dark_windows) to skip recomputing them.
    """
    if windows is None:
        windows = dark_windows(site, start, end, sun_altitude_deg)
    hours = night_hours(windows)
    if len(hours):
        idx = pd.DatetimeIndex(hours["hour"])
        hours = hours.assign(
            moon_illum=moon_illumination(idx),
            moon_down=moon_altitude(site, idx) < moon_up_altitude_deg,
        )
    else:
        hours = hours.assign(moon_illum=[], moon_down=[])
    per_night = hours.groupby("night_date").agg(
        dark_hours=("hour", "size"),
        moon_illum_mean=("moon_illum", "mean"),
        moon_free_dark_hours=("moon_down", "sum"),
    )
    out = windows.join(per_night)
    out["dark_hours"] = out["dark_hours"].fillna(0).astype(int)
    out["moon_free_dark_hours"] = out["moon_free_dark_hours"].fillna(0).astype(int)
    out.insert(0, "site", site.id)
    return out, hours[["night_date", "hour"]]
