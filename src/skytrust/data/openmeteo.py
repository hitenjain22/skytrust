"""Open-Meteo clients: Previous Runs (backtest forecasts), ERA5 archive (label), live Forecast.

Findings that shape this module are in docs/DATA_NOTES.md:
- multi-model responses suffix every key with `_{model_id}`; single-model ones don't;
- values are percent (0-100) and are converted to fractions (0-1) here, once;
- ERA5 trails real time by ~6 days, and the trailing hours come back as null.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from skytrust.config import ModelSpec, Settings, Site
from skytrust.data.cache import RAW_DIR, raw_path, read_cached, write_cached
from skytrust.data.http import BadResponseError, HttpClient

log = logging.getLogger(__name__)

ERA5_VARS = ["cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high"]
LIVE_CLOUD_VARS = ERA5_VARS
LIVE_EXTRA_VARS = ["temperature_2m", "dew_point_2m", "wind_speed_10m", "wind_gusts_10m"]
# Real GFS data contains exactly -1 and 101 (0.008 % of values; see DATA_NOTES). Values that
# far out are clipped; anything further is treated as a corrupt payload.
ROUNDING_TOLERANCE_PCT = 1


def prevruns_var(lead: int) -> str:
    return f"cloud_cover_previous_day{lead}"


# ---------- chunking ----------


def month_chunks(start: dt.date, end: dt.date) -> list[tuple[dt.date, dt.date]]:
    """Split [start, end] (inclusive) into calendar-month pieces.

    Months keep each cached file small (~2 weighted API calls) and mean a daily update only
    re-downloads the current month, not the whole history.
    """
    chunks = []
    cur = start
    while cur <= end:
        next_month = (cur.replace(day=1) + dt.timedelta(days=32)).replace(day=1)
        chunk_end = min(next_month - dt.timedelta(days=1), end)
        chunks.append((cur, chunk_end))
        cur = next_month
    return chunks


# ---------- parsing ----------


def parse_hourly(
    payload: dict[str, Any], variables: list[str], model_id: str | None = None
) -> pd.DataFrame:
    """Turn an Open-Meteo `hourly` block into a DataFrame of fractions (0-1) indexed by
    tz-aware UTC time. Accepts both suffixed (multi-model) and plain (single-model) keys.
    A variable the API didn't return at all becomes an all-NaN column."""
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict) or not hourly.get("time"):
        raise BadResponseError("Open-Meteo payload has no hourly.time")
    if payload.get("timezone", "GMT") not in ("GMT", "UTC"):
        raise BadResponseError(f"Expected UTC timestamps, got timezone={payload.get('timezone')}")
    index = pd.DatetimeIndex(pd.to_datetime(hourly["time"], utc=True), name="time")
    columns = {}
    for var in variables:
        key = f"{var}_{model_id}" if model_id and f"{var}_{model_id}" in hourly else var
        values = hourly.get(key)
        if values is None:
            columns[var] = [None] * len(index)
            continue
        if len(values) != len(index):
            raise BadResponseError(f"{key}: {len(values)} values for {len(index)} timestamps")
        columns[var] = values
    df = pd.DataFrame(columns, index=index, dtype="float64")
    if cloud_cols := [c for c in df.columns if c.startswith("cloud_cover")]:
        values = df[cloud_cols].to_numpy()
        values = values[~np.isnan(values)]  # nulls are "missing", not out of range
        lo, hi = -ROUNDING_TOLERANCE_PCT, 100 + ROUNDING_TOLERANCE_PCT
        if ((values < lo) | (values > hi)).any():
            raise BadResponseError(f"cloud cover outside {lo}-{hi} %")
        n_clipped = int(((values < 0) | (values > 100)).sum())
        if n_clipped:
            log.debug("clipped %d rounding artifacts (-1/101 %%) to 0-100", n_clipped)
        df[cloud_cols] = df[cloud_cols].clip(0, 100) / 100.0
    return df


# ---------- cached chunk fetching ----------


def _last_day_populated(payload: dict[str, Any], key_prefix: str) -> bool:
    """True if the final day of a chunk has at least one non-null value for the given
    variable. Used to avoid caching chunks the provider hasn't finished publishing."""
    hourly = payload.get("hourly", {})
    key = next((k for k in hourly if k.startswith(key_prefix)), None)
    return key is not None and any(v is not None for v in hourly[key][-24:])


def _fetch_chunk(
    client: HttpClient,
    url: str,
    params: dict[str, Any],
    source: str,
    site: Site,
    model: str | None,
    start: dt.date,
    end: dt.date,
    completeness_key: str,
    refresh: bool,
    settled_after_days: int,
    today: dt.date | None,
) -> dict[str, Any]:
    path = raw_path(source, site.id, model, f"{start:%Y%m%d}", f"{end:%Y%m%d}", "json")
    if not refresh and (cached := read_cached(path)) is not None:
        return json.loads(cached)
    payload = client.get_json(
        url, {**params, "start_date": f"{start:%Y-%m-%d}", "end_date": f"{end:%Y-%m-%d}"}
    )
    # Validate before caching so a bad payload never enters the cache.
    parse_hourly(payload, [k for k in payload.get("hourly", {}) if k != "time"])
    # Cache if the last day has data, or if the chunk is old enough that any nulls are
    # permanent (e.g. months before a model's archive began), so they never re-download.
    today = today or dt.datetime.now(dt.UTC).date()
    settled = (today - end).days > settled_after_days
    if _last_day_populated(payload, completeness_key) or settled:
        write_cached(path, json.dumps(payload))
    else:
        log.warning("%s %s %s %s..%s: last day not published yet; not cached",
                    source, site.id, model, start, end)  # fmt: skip
    return payload


def location_params(site: Site) -> dict[str, Any]:
    # Station coordinates + elevation, so forecasts are downscaled to the station (SPEC 5).
    return {
        "latitude": site.lat,
        "longitude": site.lon,
        "elevation": site.elevation_m,
        "timezone": "UTC",
    }


def fetch_prevruns(
    client: HttpClient,
    settings: Settings,
    site: Site,
    model: ModelSpec,
    start: dt.date,
    end: dt.date,
    refresh: bool = False,
    today: dt.date | None = None,
) -> int:
    """Download (or read from cache) Previous Runs forecasts for one site x model, month by
    month. Only the leads the model actually has are requested. Returns chunks processed."""
    params = {
        **location_params(site),
        "models": model.id,
        "hourly": ",".join(prevruns_var(d) for d in model.leads),
    }
    chunks = month_chunks(start, end)
    for lo, hi in chunks:
        _fetch_chunk(
            client, settings.sources["prevruns_url"], params, "prevruns", site, model.id,
            lo, hi, prevruns_var(model.leads[0]), refresh,
            settings.sources["settled_after_days"], today,
        )  # fmt: skip
    return len(chunks)


def fetch_era5(
    client: HttpClient,
    settings: Settings,
    site: Site,
    start: dt.date,
    end: dt.date,
    refresh: bool = False,
    today: dt.date | None = None,
) -> int:
    params = {
        **location_params(site),
        "models": settings.sources["era5_model"],
        "hourly": ",".join(ERA5_VARS),
    }
    chunks = month_chunks(start, end)
    for lo, hi in chunks:
        _fetch_chunk(
            client, settings.sources["era5_url"], params, "era5", site, None,
            lo, hi, "cloud_cover", refresh,
            settings.sources["settled_after_days"], today,
        )  # fmt: skip
    return len(chunks)


def fetch_live(client: HttpClient, settings: Settings, site: Site) -> dict[str, Any]:
    """Live 8-day forecast for all models in one request (not disk-cached here; the live
    module keeps its own last-good copy)."""
    params = {
        **location_params(site),
        "models": ",".join(m.id for m in settings.models),
        "hourly": ",".join(LIVE_CLOUD_VARS + LIVE_EXTRA_VARS),
        "forecast_days": 8,
    }
    return client.get_json(settings.sources["forecast_url"], params)


# ---------- loading from the cache ----------


def _load_cached(
    source: str,
    site_id: str,
    model: str | None,
    variables: list[str],
    root: Path,
) -> pd.DataFrame:
    """Concatenate every cached chunk for a source/site/model into one hourly frame.

    If a month was fetched more than once as the end date advanced (e.g. 0901_0915 then
    0901_0922), the file with the later end wins for overlapping hours.
    """
    folder = root / source / site_id / (model or "na")
    files = sorted(folder.glob("*.json"), key=lambda p: (p.stem.split("_")[1], p.stem))
    frames = [parse_hourly(json.loads(p.read_text()), variables, model) for p in files]
    if not frames:
        return pd.DataFrame(columns=variables, index=pd.DatetimeIndex([], tz="UTC", name="time"))
    df = pd.concat(frames)
    return df[~df.index.duplicated(keep="last")].sort_index()


def load_prevruns(site_id: str, model: ModelSpec, root: Path = RAW_DIR) -> pd.DataFrame:
    """Hourly forecasts with columns `lead1`..`leadN` (fractions), for one site x model."""
    variables = [prevruns_var(d) for d in model.leads]
    df = _load_cached("prevruns", site_id, model.id, variables, root)
    return df.rename(columns={prevruns_var(d): f"lead{d}" for d in model.leads})


def load_era5(site_id: str, root: Path = RAW_DIR) -> pd.DataFrame:
    """Hourly ERA5 cloud cover (total + low/mid/high) as fractions for one site."""
    return _load_cached("era5", site_id, None, ERA5_VARS, root)
