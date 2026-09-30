"""A small, deterministic, *fake* raw cache in the exact on-disk formats of the real sources
(IEM CSV, Open-Meteo JSON, GOES extract CSV), so the whole pipeline can run offline in tests.

The fake weather has structure a model can learn: each night has a cloudiness regime, the
"truth" follows it, and forecasts are truth plus noise that grows with lead time.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from skytrust.config import ModelSpec, Site

SYNTH_SITES = (
    Site("SAC", "Sacramento", 38.5069, -121.495, 8.0, "valley", "America/Los_Angeles"),
    Site("BIH", "Bishop", 37.3731, -118.3636, 1263.0, "desert", "America/Los_Angeles"),
)


def _code(fraction: float) -> str:
    """Inverse of the okta mapping: pick the METAR code whose midpoint is closest."""
    for code, upper in [("CLR", 0.05), ("FEW", 0.31), ("SCT", 0.6), ("BKN", 0.9)]:
        if fraction < upper:
            return code
    return "OVC"


def _openmeteo_payload(hours: pd.DatetimeIndex, columns: dict[str, np.ndarray]) -> dict:
    return {
        "timezone": "GMT",
        "hourly": {
            "time": [h.strftime("%Y-%m-%dT%H:%M") for h in hours],
            **{
                k: [None if np.isnan(v) else int(round(v)) for v in vals]
                for k, vals in columns.items()
            },
        },
    }


def write_synthetic_cache(
    root: Path,
    models: tuple[ModelSpec, ...],
    first_night: dt.date,
    last_night: dt.date,
    sites: tuple[Site, ...] = SYNTH_SITES,
    seed: int = 7,
    goes: bool = True,
) -> None:
    rng = np.random.default_rng(seed)
    # GOES gets its own stream so adding it didn't change any other synthetic value.
    goes_rng = np.random.default_rng([seed, 1])
    start = pd.Timestamp(first_night, tz="UTC")
    end = pd.Timestamp(last_night, tz="UTC") + pd.Timedelta(days=2)
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    tag = f"{first_night:%Y%m%d}_{last_night + dt.timedelta(days=1):%Y%m%d}"
    for site in sites:
        # Regime per UTC day (0 = clear, 1 = overcast), smooth within the day.
        days = hours.normalize().unique()
        regime = pd.Series(rng.beta(0.6, 0.6, size=len(days)), index=days)
        base = regime.reindex(hours.normalize()).to_numpy()
        truth = np.clip(base + rng.normal(0, 0.12, len(hours)), 0, 1)
        high = np.clip(truth * rng.uniform(0.2, 0.8, len(hours)), 0, 1)
        low = np.clip(truth - high, 0, 1)

        # ASOS: one routine report per hour at :53, seeing only the low part of the cloud.
        asos_times = hours - pd.Timedelta(minutes=7)
        rows = ["station,valid,elevation,skyc1,skyc2,skyc3,skyc4,skyl1,skyl2,skyl3,skyl4,metar"]
        for t, frac in zip(asos_times, low, strict=True):
            if rng.random() < 0.01:
                continue  # occasional missing report
            code = _code(frac)
            layers = f"{code},M,M,M,M,M,M,M"
            rows.append(
                f"{site.id},{t:%Y-%m-%d %H:%M},{site.elevation_m},{layers},K{site.id} {code}"
            )
        asos_path = root / "asos" / site.id / "na" / f"{tag}.csv"
        asos_path.parent.mkdir(parents=True, exist_ok=True)
        asos_path.write_text("\n".join(rows) + "\n")

        era5 = {
            "cloud_cover": truth * 100,
            "cloud_cover_low": low * 100,
            "cloud_cover_mid": np.zeros(len(hours)),
            "cloud_cover_high": high * 100,
        }
        era5_path = root / "era5" / site.id / "na" / f"{tag}.json"
        era5_path.parent.mkdir(parents=True, exist_ok=True)
        era5_path.write_text(json.dumps(_openmeteo_payload(hours, era5)))

        for m in models:
            cols = {}
            for d in m.leads:
                noisy = truth + rng.normal(0, 0.08 + 0.05 * d, len(hours))
                cols[f"cloud_cover_previous_day{d}"] = np.clip(noisy, 0, 1) * 100
            path = root / "prevruns" / site.id / m.id / f"{tag}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(_openmeteo_payload(hours, cols)))

        if goes:
            # GOES: 25-pixel cloud fraction of the full column (sees high cloud), a little noisy.
            sat = np.round(np.clip(truth + goes_rng.normal(0, 0.08, len(hours)), 0, 1) * 25) / 25
            frame = pd.DataFrame({"hour": hours, "goes_cover": sat, "n_good": 25, "key": "synth"})
            for month, part in frame.groupby(frame["hour"].dt.strftime("%Y-%m")):
                path = root / "goes" / site.id / f"{month}.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                part.assign(hour=part["hour"].map(pd.Timestamp.isoformat)).to_csv(path, index=False)
