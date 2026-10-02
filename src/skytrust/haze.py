"""Smoke and haze: how much tonight's air dims the stars.

A cloud-free night can still be a poor one: wildfire smoke, dust or thick haze dim every star
and wash out the Milky Way. The forecast for that is the aerosol optical depth (AOD) at 550 nm
from the Copernicus Atmosphere Monitoring Service (CAMS global, 0.4°, about 5 days ahead),
served by Open-Meteo's air-quality API (DATA_NOTES §17).

Extinction in the V band (magnitudes per airmass), following Schaefer's (1998, Sky & Telescope
95(5):57, program VISLIMIT.BAS) split into gases and particles:

- Rayleigh scattering by air, thinning with height: 0.1066 exp(-h / 8.2 km);
- ozone: 0.031;
- aerosols: 1.086 x AOD (an optical depth tau dims by 2.5 log10(e) tau = 1.086 tau mag).

Schaefer's standard aerosol term for a clear night is 0.1 mag in V, i.e. AOD ~0.092. The
limiting-magnitude formula is quoted for such a typical sky, so only the aerosol load *above*
that reference counts as haze. Water-vapour absorption (weak in V) and how smoke brightens city
glow are left out.
"""

from __future__ import annotations

import dataclasses
import logging
import math

import numpy as np
import pandas as pd

from skytrust.config import Settings
from skytrust.data.http import HttpClient

log = logging.getLogger(__name__)

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
RAYLEIGH_V, RAYLEIGH_H_KM = 0.1066, 8.2
OZONE_V = 0.031
MAG_PER_TAU = 2.5 * math.log10(math.e)  # 1.0857
AOD_REF = 0.1 / MAG_PER_TAU  # Schaefer's standard aerosol extinction, 0.1 mag in V
# Extra dimming overhead (magnitudes) at which the app says something about haze.
HAZE_LEVELS = [
    (1.0, "Thick smoke or haze", "only bright stars and planets will show"),
    (0.5, "Smoke or haze", "fewer stars, and the Milky Way is washed out"),
    (0.2, "Some haze", "the faintest stars are dimmed"),
]


def extinction(elevation_m: float, aod: float | None = None) -> float:
    """V-band extinction (mag per airmass) at this height for this aerosol load (the reference
    load when the forecast isn't available)."""
    h_km = max(float(elevation_m or 0.0), 0.0) / 1000.0
    a = AOD_REF if aod is None or not np.isfinite(aod) else max(float(aod), 0.0)
    return RAYLEIGH_V * math.exp(-h_km / RAYLEIGH_H_KM) + OZONE_V + MAG_PER_TAU * a


def dimming(aod: float | None) -> float:
    """How many magnitudes tonight's aerosols dim a star overhead, beyond a typical clear
    night (never negative: an unusually clean sky isn't promised as better than typical)."""
    if aod is None or not np.isfinite(aod):
        return 0.0
    return max(MAG_PER_TAU * (float(aod) - AOD_REF), 0.0)


def words(aod: float | None) -> tuple[str, str] | None:
    """('Smoke or haze', 'fewer stars, ...') when it's bad enough to matter, else None."""
    return words_for_dimming(dimming(aod))


def words_for_dimming(d: float) -> tuple[str, str] | None:
    for at, title, effect in HAZE_LEVELS:
        if d >= at:
            return title, effect
    return None


def fetch(lat: float, lon: float, settings: Settings, client: HttpClient | None = None
          ) -> pd.Series | None:  # fmt: skip
    """Hourly AOD forecast (UTC index) for the next 5 days, or None if the service doesn't
    answer quickly: haze is a refinement, so one short try only."""
    client = client or HttpClient(
        dataclasses.replace(settings.http, timeout_s=5.0, max_attempts=1, polite_delay_s=0.0)
    )
    params = {"latitude": round(lat, 3), "longitude": round(lon, 3),
              "hourly": "aerosol_optical_depth", "timezone": "UTC", "forecast_days": 5}  # fmt: skip
    try:
        payload = client.get_json(AIR_QUALITY_URL, params)
    except Exception as exc:  # any failure just means "no haze forecast"
        log.warning("air-quality forecast unavailable: %s", exc)
        return None
    return parse(payload)


def parse(payload: dict) -> pd.Series | None:
    hourly = payload.get("hourly") or {}
    times, values = hourly.get("time"), hourly.get("aerosol_optical_depth")
    if not times or values is None or len(times) != len(values):
        return None
    s = pd.Series(
        pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(),
        index=pd.DatetimeIndex(pd.to_datetime(times)).tz_localize("UTC"),
    ).dropna()
    return s if len(s) else None


def night_aod(series: pd.Series | None, dusk: pd.Timestamp, dawn: pd.Timestamp) -> float | None:
    """Median AOD over the night's dark hours (None when the forecast doesn't reach them:
    CAMS runs about 5 days ahead, so the last nights of the week have none)."""
    if series is None or series.empty:
        return None
    win = series[(series.index >= dusk.floor("h")) & (series.index <= dawn.ceil("h"))]
    if len(win) < max(2, int((dawn - dusk) / pd.Timedelta(hours=1)) // 2):
        return None
    return float(win.median())
