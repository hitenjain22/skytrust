"""GOES-18 (GOES-West) satellite clear-sky mask: an independent, observation-based truth source.

ASOS ceilometers can't see above 12,000 ft and ERA5 is a model. The ABI Clear Sky Mask
(product ABI-L2-ACMC, CONUS, 2 km, every 5 minutes, free on AWS) is derived from satellite
radiances and sees every layer, including cirrus (caveat: at night only infrared channels are
available, so very thin cirrus can be missed).

For each dark hour we take the scan that starts closest to the top of the hour, convert each
site's latitude/longitude to the satellite's scan angles (GOES-R Product User Guide, vol. 3,
§4.2.8), and average the binary cloud mask over a small box of good-quality pixels around the
site. One ~3.5 MB file serves every site; only the per-site numbers are kept on disk.
"""

from __future__ import annotations

import io
import logging
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from skytrust.config import Site
from skytrust.data.cache import RAW_DIR
from skytrust.data.http import BadResponseError, HttpClient, SourceUnavailableError

log = logging.getLogger(__name__)

BUCKET_URL = "https://noaa-goes18.s3.amazonaws.com"
PRODUCT = "ABI-L2-ACMC"
HALF_WINDOW = 2  # pixels on each side -> 5 x 5 box of 2 km pixels (~10 km)
EXTRACT_DIR = RAW_DIR / "goes"
KEY_RE = re.compile(r"<Key>([^<]+)</Key>")


# ---------- geometry ----------


def scan_angles(lat_deg: float, lon_deg: float, proj: dict) -> tuple[float, float] | None:
    """Latitude/longitude -> ABI fixed-grid scan angles (x = E/W, y = N/S), in radians.
    Returns None if the point can't be seen from the satellite (PUG vol. 3, §4.2.8.1)."""
    r_eq, r_pol = proj["semi_major_axis"], proj["semi_minor_axis"]
    h = proj["perspective_point_height"] + r_eq
    lon0 = math.radians(proj["longitude_of_projection_origin"])
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    e2 = (r_eq**2 - r_pol**2) / r_eq**2
    lat_c = math.atan((r_pol**2 / r_eq**2) * math.tan(lat))
    r_c = r_pol / math.sqrt(1 - e2 * math.cos(lat_c) ** 2)
    sx = h - r_c * math.cos(lat_c) * math.cos(lon - lon0)
    sy = -r_c * math.cos(lat_c) * math.sin(lon - lon0)
    sz = r_c * math.sin(lat_c)
    if h * (h - sx) < sy**2 + (r_eq**2 / r_pol**2) * sz**2:
        return None
    x = math.asin(-sy / math.sqrt(sx**2 + sy**2 + sz**2))
    y = math.atan(sz / sx)
    return x, y


# ---------- reading one Clear Sky Mask file ----------


def _attr(ds, name: str, default=None):
    value = ds.attrs.get(name, default)
    return value.item() if hasattr(value, "item") else value


def extract_sites(
    file_bytes: bytes, sites: tuple[Site, ...], half_window: int = HALF_WINDOW
) -> dict:
    """{site_id: (cloud_fraction, n_good_pixels)} from one ACM NetCDF4 file (read with h5py).

    BCM is the binary cloud mask (0 clear, 1 cloudy); DQF 0 marks good-quality pixels. Pixels
    that are missing or not good are ignored; a site with no good pixel gets NaN.
    """
    import h5py

    out = {}
    with h5py.File(io.BytesIO(file_bytes), "r") as f:
        proj = {
            k: _attr(f["goes_imager_projection"], k)
            for k in (
                "semi_major_axis",
                "semi_minor_axis",
                "perspective_point_height",
                "longitude_of_projection_origin",
            )
        }
        x_ds, y_ds = f["x"], f["y"]
        x_scale, x_off = _attr(x_ds, "scale_factor", 1.0), _attr(x_ds, "add_offset", 0.0)
        y_scale, y_off = _attr(y_ds, "scale_factor", 1.0), _attr(y_ds, "add_offset", 0.0)
        bcm, dqf = f["BCM"], f["DQF"]
        n_rows, n_cols = bcm.shape
        for site in sites:
            angles = scan_angles(site.lat, site.lon, proj)
            if angles is None:
                out[site.id] = (math.nan, 0)
                continue
            col = int(round((angles[0] - x_off) / x_scale))
            row = int(round((angles[1] - y_off) / y_scale))
            r0, r1 = max(row - half_window, 0), min(row + half_window + 1, n_rows)
            c0, c1 = max(col - half_window, 0), min(col + half_window + 1, n_cols)
            if r0 >= r1 or c0 >= c1:
                out[site.id] = (math.nan, 0)
                continue
            mask = np.asarray(bcm[r0:r1, c0:c1], dtype=float)
            good = (np.asarray(dqf[r0:r1, c0:c1]) == 0) & np.isin(mask, (0.0, 1.0))
            out[site.id] = (float(mask[good].mean()) if good.any() else math.nan, int(good.sum()))
    return out


# ---------- finding and fetching the scan for an hour ----------


def hour_prefix(hour_utc: pd.Timestamp) -> str:
    return f"{PRODUCT}/{hour_utc:%Y}/{hour_utc.dayofyear:03d}/{hour_utc:%H}/"


def first_scan_key(client: HttpClient, hour_utc: pd.Timestamp) -> str | None:
    """The first scan starting in hour H (starts ~H:01, i.e. within ~2 min of the hour)."""
    resp = client.get(BUCKET_URL, {"list-type": 2, "prefix": hour_prefix(hour_utc), "max-keys": 2})
    keys = KEY_RE.findall(resp.text)
    return keys[0] if keys else None


def extract_path(site_id: str, month: str, root: Path = EXTRACT_DIR) -> Path:
    return root / site_id / f"{month}.csv"


def load_extracts(site_id: str, root: Path = EXTRACT_DIR) -> pd.DataFrame:
    files = sorted((root / site_id).glob("*.csv"))
    if not files:
        return pd.DataFrame(columns=["hour", "goes_cover", "n_good"]).astype({"goes_cover": float})
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    df["hour"] = pd.to_datetime(df["hour"], utc=True)
    return df.drop_duplicates("hour", keep="last").sort_values("hour").reset_index(drop=True)


def goes_hourly(site_id: str, root: Path = EXTRACT_DIR) -> pd.Series:
    """Hourly GOES cloud fraction for a site (index = top-of-hour UTC)."""
    df = load_extracts(site_id, root)
    index = pd.DatetimeIndex(pd.to_datetime(df["hour"], utc=True))  # tz-aware even when empty
    return pd.Series(df["goes_cover"].to_numpy(dtype=float), index=index, name="goes")


def _fetch_one(client: HttpClient, hour: pd.Timestamp, sites: tuple[Site, ...]):
    try:
        key = first_scan_key(client, hour)
        values = extract_sites(client.get(f"{BUCKET_URL}/{key}", {}).content, sites) if key else {}
        return hour, key, values, None
    except (SourceUnavailableError, BadResponseError, OSError) as exc:
        return hour, None, None, exc


def fetch_hours(
    client: HttpClient,
    hours: pd.DatetimeIndex,
    sites: tuple[Site, ...],
    root: Path = EXTRACT_DIR,
    workers: int = 1,
) -> int:
    """Download + extract every requested hour not already extracted for all sites.

    Results are appended to monthly per-site CSVs as they arrive, so an interrupted run resumes.
    `workers` > 1 downloads in parallel threads (AWS S3 handles that easily); writing stays in
    this thread, so the CSVs never interleave."""
    from concurrent.futures import ThreadPoolExecutor

    done = (
        set.intersection(*[set(load_extracts(s.id, root)["hour"]) for s in sites])
        if sites
        else set()
    )
    todo = [h for h in hours if h not in done]
    log.info("GOES: %d hours to fetch (%d already extracted)", len(todo), len(hours) - len(todo))
    fetched = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for i, (hour, key, values, error) in enumerate(
            pool.map(lambda h: _fetch_one(client, h, sites), todo)
        ):
            if error is not None:
                log.warning("GOES %s: %s", hour, error)
                continue
            for site in sites:
                cover, n_good = (values or {}).get(site.id, (math.nan, 0))
                path = extract_path(site.id, f"{hour:%Y-%m}", root)
                path.parent.mkdir(parents=True, exist_ok=True)
                new = not path.exists()
                with path.open("a") as fh:
                    if new:
                        fh.write("hour,goes_cover,n_good,key\n")
                    fh.write(f"{hour.isoformat()},{cover},{n_good},{key or ''}\n")
            fetched += 1
            if i and i % 250 == 0:
                log.info("GOES: %d / %d hours", i, len(todo))
    return fetched


def dark_hours_union(
    sites: tuple[Site, ...], first, last, sun_altitude_deg: float
) -> pd.DatetimeIndex:
    """Every top-of-hour that is astronomically dark at any site (one scan serves all sites)."""
    from skytrust import astro

    hours = set()
    for site in sites:
        windows = astro.dark_windows(site, first, last, sun_altitude_deg)
        hours |= set(astro.night_hours(windows)["hour"])
    return pd.DatetimeIndex(sorted(hours))
