"""Iowa Environmental Mesonet (IEM): station metadata, ASOS/METAR fetch, sky-cover parsing.

Ground-truth caveat: ASOS ceilometers only see clouds below 12,000 ft AGL, so "CLR"
means "no clouds below 12,000 ft". This source cannot see cirrus (SPEC 4.6a).
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from skytrust.data.cache import RAW_DIR, raw_path, read_cached, write_cached
from skytrust.data.http import BadResponseError, HttpClient

log = logging.getLogger(__name__)

SKY_COLS = ["skyc1", "skyc2", "skyc3", "skyc4"]
ASOS_DATA_COLS = SKY_COLS + ["skyl1", "skyl2", "skyl3", "skyl4", "metar"]


# ---------- station metadata ----------


def parse_network_geojson(payload: dict[str, Any], station_ids: list[str]) -> dict[str, dict]:
    """Pull name/lat/lon/elevation for the requested stations from IEM's network GeoJSON.

    Coordinates come from IEM, never hand-typed (SPEC 5). GeoJSON order is [lon, lat].
    """
    found: dict[str, dict] = {}
    for feat in payload.get("features", []):
        props = feat["properties"]
        sid = props.get("sid")
        if sid in station_ids:
            lon, lat = feat["geometry"]["coordinates"][:2]
            attrs = props.get("attributes") or {}
            found[sid] = {
                "id": sid,
                "name": props["sname"],
                "lat": float(lat),
                "lon": float(lon),
                "elevation_m": float(props["elevation"]),
                "timezone": props["tzname"],
                "archive_begin": props.get("archive_begin"),
                "online": props.get("online"),
                "is_awos": attrs.get("IS_AWOS") == "1",
            }
    return found


def fetch_network(client: HttpClient, url_template: str, network: str) -> dict[str, Any]:
    return client.get_json(url_template.format(network=network), params={})


# ---------- ASOS observations ----------


def fetch_asos_csv(
    client: HttpClient,
    url: str,
    station: str,
    start: dt.datetime,
    end: dt.datetime,
    report_types: list[int],
    refresh: bool = False,
) -> str:
    """Fetch one station's METAR sky data for [start, end) as CSV text, using the disk cache.

    IEM asks for small requests (it rejects >~1,000 station-years with HTTP 422), so
    callers request one station x one year at a time.
    """
    path = raw_path("asos", station, None, f"{start:%Y%m%d}", f"{end:%Y%m%d}", "csv")
    if not refresh and (cached := read_cached(path)) is not None:
        return cached
    params: list[tuple[str, Any]] = [("station", station)]
    params += [("data", c) for c in ASOS_DATA_COLS]
    params += [("report_type", r) for r in report_types]
    params += [
        ("sts", start.strftime("%Y-%m-%dT%H:%MZ")),
        ("ets", end.strftime("%Y-%m-%dT%H:%MZ")),
        ("tz", "UTC"),
        ("format", "onlycomma"),
        ("elev", "yes"),
    ]
    text = client.get(url, params).text
    parse_asos_csv(text)  # validate before caching so bad payloads never enter the cache
    write_cached(path, text)
    return text


def year_chunks(start: dt.date, end: dt.date) -> list[tuple[dt.datetime, dt.datetime]]:
    """Split [start, end] into calendar-year UTC windows (IEM etiquette: one station-year
    per request). The end bound is exclusive, so the last chunk ends the day after `end`."""
    chunks = []
    for year in range(start.year, end.year + 1):
        lo = max(dt.date(year, 1, 1), start)
        hi = min(dt.date(year + 1, 1, 1), end + dt.timedelta(days=1))
        chunks.append(
            (
                dt.datetime.combine(lo, dt.time(), tzinfo=dt.UTC),
                dt.datetime.combine(hi, dt.time(), tzinfo=dt.UTC),
            )
        )
    return chunks


def fetch_asos_range(
    client: HttpClient,
    url: str,
    station: str,
    first_night: dt.date,
    last_night: dt.date,
    report_types: list[int],
    refresh: bool = False,
) -> pd.DataFrame:
    """All reports needed to label nights first_night..last_night.

    The last night's dark hours run into the next UTC day, so data is fetched through
    last_night + 1 day.
    """
    frames = [
        parse_asos_csv(fetch_asos_csv(client, url, station, lo, hi, report_types, refresh))
        for lo, hi in year_chunks(first_night, last_night + dt.timedelta(days=1))
    ]
    return pd.concat(frames, ignore_index=True)


LABEL_COLUMNS = ["station", "valid", *SKY_COLS]


def load_asos(station: str, root: Path = RAW_DIR) -> pd.DataFrame:
    """Every cached report for a station, keeping only what labelling needs (time + the four
    sky-cover codes, stored as categories). 20 years of history with every column, including the
    raw METAR text, didn't fit in 8 GB of RAM. Overlapping chunks (the current year re-fetched with
    a later end date) are de-duplicated on (time, sky codes)."""
    files = sorted((root / "asos" / station / "na").glob("*.csv"))
    if not files:
        return parse_asos_csv(",".join(LABEL_COLUMNS) + "\n", columns=LABEL_COLUMNS)
    df = pd.concat(
        [parse_asos_csv(p.read_text(), columns=LABEL_COLUMNS) for p in files], ignore_index=True
    )
    df = df.drop_duplicates(subset=["valid", *SKY_COLS]).sort_values("valid").reset_index(drop=True)
    return df.astype({c: "category" for c in SKY_COLS})


def parse_asos_csv(text: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Parse IEM CSV into a DataFrame with a tz-aware UTC `valid` column.

    Raw METAR text has real-world quirks: a stray carriage return inside a remark (TRK, 2008)
    crashes pandas' C parser, and remarks use `"` as an inch mark (`NO SN 9"`). IEM's
    `onlycomma` format never quotes fields, so carriage returns become spaces and quoting is off.
    """
    try:
        df = pd.read_csv(
            io.StringIO(text.replace("\r", " ")),
            dtype=str,
            keep_default_na=False,
            quoting=csv.QUOTE_NONE,
            usecols=(lambda c: c in columns) if columns else None,
        )
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise BadResponseError(f"Malformed IEM CSV: {exc}") from exc
    missing = {"station", "valid", *SKY_COLS} - set(df.columns)
    if missing:
        raise BadResponseError(f"IEM CSV missing columns {sorted(missing)}")
    df["valid"] = pd.to_datetime(df["valid"], format="%Y-%m-%d %H:%M", utc=True)
    return df


def sky_code_to_fraction(code: str | None, mapping: dict[str, float]) -> float:
    """Map one METAR sky-cover code (CLR, FEW, ...) to a cloud fraction; unknown/missing -> NaN."""
    if code is None:
        return math.nan
    code = code.strip().upper()
    return mapping.get(code, math.nan)


def observation_cover(df: pd.DataFrame, mapping: dict[str, float]) -> pd.Series:
    """Cloud fraction per observation = max over its reported layers.

    METAR layer amounts are cumulative (each layer includes everything below it), so the
    highest layer's amount is the total coverage; taking the max is equivalent and safe.
    An observation with no parseable layer at all is NaN.
    """
    fractions = np.column_stack(
        [df[c].map(lambda v: sky_code_to_fraction(v, mapping)).to_numpy(float) for c in SKY_COLS]
    )
    all_nan = np.isnan(fractions).all(axis=1)
    cover = np.where(all_nan, np.nan, np.nanmax(np.where(all_nan[:, None], 0, fractions), axis=1))
    return pd.Series(cover, index=df.index, name="cover")


def assign_hour(valid: pd.Series) -> pd.Series:
    """Map each observation time t to the top-of-hour H whose window (H-30min, H+30min]
    contains it. Solving H-30 < t <= H+30 for H gives H = ceil_to_hour(t - 30min).
    Check: t = H+30 -> ceil(H) = H (included); t = H-30 -> ceil(H-60) = H-1 (excluded).
    """
    return (valid - pd.Timedelta(minutes=30)).dt.ceil("h")


HOUR_AGGREGATIONS = ("nearest", "max")


def hourly_cover(df: pd.DataFrame, mapping: dict[str, float], method: str = "nearest") -> pd.Series:
    """Hourly cloud fraction from all observations in each hour's window (H-30, H+30].

    - "nearest" (default, DECISIONS 2026-09-25): the valid report closest in time to H;
      ties go to the cloudier report. One reading per hour at every station, so labels are
      comparable across stations whose reporting frequency differs (AUN reports 3x/hour,
      SAC 1x) and stable if a station changes its schedule over time.
    - "max" (SPEC's original rule, kept for sensitivity analysis): max over all reports.
      More reports per hour make this stricter, which is why it is not the default.

    Hours whose reports all lack a parseable sky layer are NaN.
    """
    if method not in HOUR_AGGREGATIONS:
        raise ValueError(f"method must be one of {HOUR_AGGREGATIONS}, got {method!r}")
    hour = assign_hour(df["valid"])
    obs = pd.DataFrame(
        {
            "hour": hour,
            "cover": observation_cover(df, mapping),
            "dist": (df["valid"] - hour).abs(),
        }
    )
    all_hours = pd.Index(obs["hour"].unique()).sort_values()
    valid = obs.dropna(subset=["cover"])
    if method == "nearest":
        closest = valid.groupby("hour")["dist"].transform("min")
        valid = valid[valid["dist"] == closest]
    result = valid.groupby("hour")["cover"].max().reindex(all_hours)
    result.index.name = "hour"
    return result.rename("asos_cover")


def proxy_night_hours(
    start: dt.date, end: dt.date, tz: str, local_hours: list[int]
) -> pd.DatetimeIndex:
    """UTC timestamps of fixed local night hours, used only for Phase 0 station
    validation (real dark windows come from astro.py). Hours after midnight belong to
    the night that began the previous evening."""
    wall_times = []
    for day in pd.date_range(start, end, freq="D"):
        for h in local_hours:
            local_day = day + pd.Timedelta(days=1) if h < 12 else day
            wall_times.append(local_day + pd.Timedelta(hours=h))
    # Localize wall-clock times (not "midnight + N hours") so DST days are handled;
    # the non-existent 02:00 on spring-forward night shifts to 03:00 and is de-duplicated.
    local = pd.DatetimeIndex(wall_times).tz_localize(
        tz, nonexistent="shift_forward", ambiguous="NaT"
    )
    return local.dropna().tz_convert("UTC").unique()


def night_coverage(hourly: pd.Series, night_hours: pd.DatetimeIndex) -> float:
    """Share of the given night hours that have a valid (non-NaN) hourly sky cover."""
    if len(night_hours) == 0:
        return math.nan
    valid = hourly.reindex(night_hours).notna()
    return float(valid.mean())
