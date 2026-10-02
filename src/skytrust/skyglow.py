"""Bringing the light-pollution atlas up to date, and mapping city glow, from NASA night lights.

The World Atlas (Falchi et al. 2016) is calibrated against thousands of sky-quality readings but
its lights are from 2014-2015. NASA's Black Marble annual night-lights composites (VNP46A4 /
VJ146A4, CC0; distributed as GeoTIFFs by lightpollutionmap.info) are available through 2025.
This module combines the two:

1. **Learn how light spreads.** Model the atlas's artificial zenith brightness at each place as
   a sum over every light source of emission x K(distance), with K a non-negative step function
   of distance (rings from 0 to 300 km). K is fitted to the 2016 atlas from 2015 lights by
   non-negative least squares on relative error, and checked by spatial cross-validation.
2. **Update to 2025.** artificial_2025 = atlas_2016 x model(2025 lights) / model(2015 lights).
   Used as a ratio, the atlas keeps what the simple model lacks (altitude, its own calibration)
   and only the *change* in lights comes from the model. The result is checked against David
   Lorenz's independent 2025 re-calculation of the atlas (used for validation only).
3. **City glow.** For one place, the same kernel splits the artificial glow overhead by the
   direction it comes from, and Walker's law (glow ~ distance^-2.5, Walker 1977, PASP 89:405)
   ranks the light domes on the horizon by direction, named after the nearest city (GeoNames,
   CC BY 4.0). Walker's law is used for relative dome strength only.

Everything is numpy; scipy (installed with scikit-learn) is used only at build time for NNLS.
The NASA GeoTIFFs are LZW-compressed with one row per strip, decoded by a small LZW reader.
"""

from __future__ import annotations

import json
import logging
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from skytrust.config import REPO_ROOT, Settings
from skytrust.data.cache import RAW_DIR
from skytrust.lightpollution import Grid, haversine_km

log = logging.getLogger(__name__)

LP_RAW = RAW_DIR / "light_pollution"
SOURCES_PATH = REPO_ROOT / "artifacts" / "light_sources_2025.npz"
CITIES_PATH = REPO_ROOT / "artifacts" / "cities.json"
MODEL_PATH = REPO_ROOT / "artifacts" / "skyglow.json"
RING_EDGES_KM = [0.0, 1.5, 3, 6, 12, 25, 50, 100, 200, 300]
# Dome strengths are reported relative to a fixed reference, so a faint dome looks faint: the
# Walker's-law dome of Sacramento's lights (within 20 km of downtown) seen from 50 km away.
WALKER_MIN_KM = 10.0
REF_LAT, REF_LON, REF_RADIUS_KM, REF_DISTANCE_KM = 38.5816, -121.4944, 20.0, 50.0
DOME_REFERENCE: dict[str, float | str] = {
    "name": "Sacramento", "lat": REF_LAT, "lon": REF_LON, "radius_km": REF_RADIUS_KM,
    "distance_km": REF_DISTANCE_KM,
}  # fmt: skip


def dome_reference(lat, lon, emission) -> float:
    d = haversine_km(REF_LAT, REF_LON, lat, lon)
    return float(emission[d <= REF_RADIUS_KM].sum() * REF_DISTANCE_KM**-2.5)


def city_radius_km(population) -> np.ndarray:
    """How far from its centre a place claims light: grows with the square root of population
    (built-up area scales roughly with population). 1,000 people ~2 km, 1.5 million ~17 km."""
    return np.clip(1.5 + 4.0 * np.sqrt(np.asarray(population, dtype=float) / 1e5), 2.0, 25.0)


COMPASS16 = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
             "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]  # fmt: skip


def viirs_path(year: int) -> Path:
    return LP_RAW / f"viirs_{year}_raw.tif"


# ---------- reading the NASA GeoTIFFs (LZW, one row per strip) ----------


def lzw_decode(data: bytes, max_out: int | None = None) -> bytes:
    """TIFF-flavoured LZW: MSB-first codes of 9-12 bits, Clear = 256, EOI = 257, and the code
    width grows one code early ("early change"). Stops once `max_out` bytes are produced, so a
    row can be decoded only as far as the columns that are needed."""
    limit = max_out if max_out is not None else 1 << 62
    out = bytearray()
    table = [bytes([i]) for i in range(256)] + [b"", b""]
    bitbuf = bitcnt = 0
    width = 9
    prev: bytes | None = None
    pos, n = 0, len(data)
    while len(out) < limit:
        while bitcnt < width:
            if pos >= n:
                return bytes(out)
            bitbuf = ((bitbuf << 8) | data[pos]) & 0xFFFFFFFF
            pos += 1
            bitcnt += 8
        bitcnt -= width
        code = (bitbuf >> bitcnt) & ((1 << width) - 1)
        if code == 256:
            table = table[:258]
            width = 9
            prev = None
            continue
        if code == 257:
            break
        if prev is None:
            entry = table[code]
        elif code < len(table):
            entry = table[code]
            table.append(prev + entry[:1])
        else:  # the code being defined right now (KwKwK case)
            entry = prev + prev[:1]
            table.append(entry)
        out += entry
        prev = entry
        size = len(table)
        if size >= 511 and width == 9:
            width = 10
        elif size >= 1023 and width == 10:
            width = 11
        elif size >= 2047 and width == 11:
            width = 12
    return bytes(out)


@dataclass
class StripTiff:
    width: int
    height: int
    compression: int
    offsets: np.ndarray
    counts: np.ndarray
    west: float
    north: float
    step: float
    nodata: float | None


def read_strip_info(path: Path) -> StripTiff:
    """Header of a little-endian float32 GeoTIFF stored one row per strip (LZW or none)."""
    with path.open("rb") as f:
        head = f.read(8)
        if head[:4] != b"II*\x00":
            raise ValueError(f"{path}: not a little-endian classic TIFF")
        f.seek(struct.unpack("<I", head[4:])[0])
        (n,) = struct.unpack("<H", f.read(2))
        tags = {}
        for _ in range(n):
            tag, typ, count, value = struct.unpack("<HHII", f.read(12))
            tags[tag] = (typ, count, value)

        def scalar(tag, default=None):
            if tag not in tags:
                return default
            typ, _, value = tags[tag]
            return value & 0xFFFF if typ == 3 else value

        def array(tag, dtype, width):
            _, count, offset = tags[tag]
            if count * width <= 4:
                return np.array([offset], dtype=np.int64)
            f.seek(offset)
            return np.frombuffer(f.read(width * count), dtype=dtype).astype(np.int64)

        def doubles(tag):
            _, count, offset = tags[tag]
            f.seek(offset)
            return struct.unpack(f"<{count}d", f.read(8 * count))

        if scalar(258) != 32 or scalar(339) != 3 or scalar(278) != 1:
            raise ValueError(f"{path}: expected float32 with one row per strip")
        if scalar(259) not in (1, 5) or scalar(317, 1) != 1:
            raise ValueError(f"{path}: only uncompressed or LZW without predictor is supported")
        nodata = None
        if 42113 in tags:
            _, count, offset = tags[42113]
            f.seek(offset)
            text = f.read(count).rstrip(b"\x00").decode() if count > 4 else ""
            nodata = float(text) if text else None
        scale, tie = doubles(33550), doubles(33922)
        return StripTiff(
            scalar(256), scalar(257), scalar(259), array(273, "<u4", 4), array(279, "<u4", 4),
            tie[3], tie[4], scale[0], nodata,
        )  # fmt: skip


def read_strip_window(path: Path, lat: tuple[float, float], lon: tuple[float, float]):
    """Values for a lat/lon box (nodata -> 0), plus (north, west, step) of the returned grid."""
    info = read_strip_info(path)
    r0 = max(0, math.floor((info.north - lat[1]) / info.step))
    r1 = min(info.height, math.ceil((info.north - lat[0]) / info.step))
    c0 = max(0, math.floor((lon[0] - info.west) / info.step))
    c1 = min(info.width, math.ceil((lon[1] - info.west) / info.step))
    out = np.zeros((r1 - r0, c1 - c0), dtype=np.float32)
    with path.open("rb") as f:
        for r in range(r0, r1):
            f.seek(int(info.offsets[r]))
            raw = f.read(int(info.counts[r]))
            row = lzw_decode(raw, c1 * 4) if info.compression == 5 else raw[: c1 * 4]
            out[r - r0] = np.frombuffer(row[: c1 * 4], dtype="<f4")[c0:c1]
    if info.nodata is not None:
        out[np.isclose(out, info.nodata)] = 0.0
    out[~np.isfinite(out) | (out < 0)] = 0.0
    return out, info.north - r0 * info.step, info.west + c0 * info.step, info.step


def emission(radiance: np.ndarray, north: float, step: float, factor: int = 2) -> np.ndarray:
    """Light emitted per coarse cell: mean radiance x cell area (km²), after summing
    `factor` x `factor` blocks. Units are arbitrary but consistent between years."""
    h, w = (radiance.shape[0] // factor) * factor, (radiance.shape[1] // factor) * factor
    r = radiance[:h, :w].reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))
    lat = north - (np.arange(r.shape[0]) + 0.5) * step * factor
    cell_km2 = (step * factor * 111.195) ** 2 * np.cos(np.radians(lat))
    return (r * cell_km2[:, None]).astype(np.float32)


# ---------- projection + convolution ----------


def laea(lat, lon, lat0: float, lon0: float):
    """Spherical Lambert azimuthal equal-area projection (km). Distances from points near the
    centre are preserved to a few percent over ~1000 km, enough for a ring kernel."""
    R = 6371.0
    p, lam = np.radians(lat), np.radians(np.asarray(lon) - lon0)
    p0 = math.radians(lat0)
    k = np.sqrt(2 / (1 + math.sin(p0) * np.sin(p) + math.cos(p0) * np.cos(p) * np.cos(lam)))
    x = R * k * np.cos(p) * np.sin(lam)
    y = R * k * (math.cos(p0) * np.sin(p) - math.sin(p0) * np.cos(p) * np.cos(lam))
    return x, y


@dataclass
class KmGrid:
    values: np.ndarray  # emission summed into cell_km squares
    x0: float  # x of column 0's left edge (km)
    y0: float  # y of row 0's bottom edge (km)
    cell_km: float

    def index(self, x, y):
        c = np.floor((np.asarray(x) - self.x0) / self.cell_km).astype(int)
        r = np.floor((np.asarray(y) - self.y0) / self.cell_km).astype(int)
        return np.clip(r, 0, self.values.shape[0] - 1), np.clip(c, 0, self.values.shape[1] - 1)


def to_km_grid(e: np.ndarray, north: float, west: float, step: float, center, cell_km=1.0):
    """Sum emission cells into a square-km grid in the LAEA projection (light is conserved)."""
    rows, cols = np.nonzero(e > 0)
    lat = north - (rows + 0.5) * step
    lon = west + (cols + 0.5) * step
    x, y = laea(lat, lon, *center)
    x0, y0 = float(x.min()) - cell_km, float(y.min()) - cell_km
    nx = int((x.max() - x0) / cell_km) + 2
    ny = int((y.max() - y0) / cell_km) + 2
    grid = np.zeros((ny, nx), dtype=np.float64)
    np.add.at(
        grid, (((y - y0) / cell_km).astype(int), ((x - x0) / cell_km).astype(int)), e[rows, cols]
    )
    return KmGrid(grid, x0, y0, cell_km)


def ring_masks(edges_km, cell_km: float) -> list[np.ndarray]:
    reach = int(math.ceil(edges_km[-1] / cell_km))
    ax = np.arange(-reach, reach + 1) * cell_km
    d = np.hypot(*np.meshgrid(ax, ax, indexing="ij"))
    return [
        ((d >= a) & (d < b)).astype(np.float64)
        for a, b in zip(edges_km[:-1], edges_km[1:], strict=True)
    ]


def convolve_rings(grid: KmGrid, edges_km) -> list[np.ndarray]:
    """For every cell, the emission summed over each distance ring (FFT convolution)."""
    masks = ring_masks(edges_km, grid.cell_km)
    k = masks[0].shape[0]
    ny, nx = grid.values.shape
    shape = (1 << int(math.ceil(math.log2(ny + k))), 1 << int(math.ceil(math.log2(nx + k))))
    f_grid = np.fft.rfft2(grid.values, shape)
    half = k // 2
    out = []
    for m in masks:
        conv = np.fft.irfft2(f_grid * np.fft.rfft2(m, shape), shape)
        out.append(np.maximum(conv[half : half + ny, half : half + nx], 0.0))
    return out


# ---------- the light-spread kernel ----------


def fit_kernel(features: np.ndarray, target: np.ndarray, natural: float) -> np.ndarray:
    """Non-negative weights w with target ≈ features @ w, minimising *relative* error of the
    total sky brightness (rows divided by target + natural), which is what magnitudes measure."""
    from scipy.optimize import nnls

    scale = features.std(axis=0)
    scale[scale == 0] = 1.0
    rows = 1.0 / (target + natural)
    w, _ = nnls((features / scale) * rows[:, None], target * rows)
    return w / scale


def mag_error(pred_ucd, true_ucd, natural: float) -> np.ndarray:
    """Error in mag/arcsec² of the total sky brightness (positive = predicted darker)."""
    return 2.5 * np.log10((np.asarray(true_ucd) + natural) / (np.asarray(pred_ucd) + natural))


def spatial_cv(features, target, lon, natural: float, folds: int = 5) -> dict:
    """Fit on all but one longitude band, predict that band: honest out-of-area error."""
    edges = np.quantile(lon, np.linspace(0, 1, folds + 1))
    errs = np.empty_like(target)
    for i in range(folds):
        test = (lon >= edges[i]) & (
            (lon < edges[i + 1]) if i < folds - 1 else (lon <= edges[i + 1])
        )
        w = fit_kernel(features[~test], target[~test], natural)
        errs[test] = mag_error(features[test] @ w, target[test], natural)
    return {
        "folds": folds,
        "rmse_mag": float(np.sqrt(np.mean(errs**2))),
        "median_abs_mag": float(np.median(np.abs(errs))),
        "bias_mag": float(np.mean(errs)),
        "share_within_0_15": float(np.mean(np.abs(errs) <= 0.15)),
        "share_within_0_30": float(np.mean(np.abs(errs) <= 0.30)),
    }


# ---------- the independent 2025 atlas (validation only) ----------

# David Lorenz, World Atlas of the Artificial Night Sky Brightness 2025 (djlorenz.github.io):
# zone colours and light-pollution-index (artificial / natural) bounds, from its colour bar.
LORENZ_ZONES = [
    ((0, 0, 0), 0.0), ((34, 34, 34), 0.01), ((66, 66, 66), 0.06), ((20, 47, 114), 0.11),
    ((33, 84, 216), 0.19), ((15, 87, 20), 0.33), ((31, 161, 42), 0.58), ((110, 100, 30), 1.0),
    ((184, 166, 37), 1.73), ((191, 100, 30), 3.0), ((253, 150, 80), 5.2), ((251, 90, 73), 9.0),
    ((251, 153, 138), 15.59), ((160, 160, 160), 27.0), ((242, 242, 242), 46.77),
]  # fmt: skip
LORENZ_NA = {"north": 75.0, "west": -180.0, "step": 1 / 120}  # NorthAmerica2025.png: 7-75N, 180-51W


def zone_of_ratio(r) -> np.ndarray:
    """Zone index 0-14 of an artificial/natural ratio, on Lorenz's scale."""
    bounds = np.array([b for _, b in LORENZ_ZONES[1:]])
    return np.searchsorted(bounds, np.asarray(r), side="right")


def lorenz_zones(png: Path, north: float, west: float, step: float, shape) -> np.ndarray:
    """Zone index on our grid, decoded from the 2025 atlas image by nearest palette colour."""
    from PIL import Image  # Pillow ships with matplotlib; build-time validation only

    Image.MAX_IMAGE_PIXELS = None
    r0 = round((LORENZ_NA["north"] - north) / LORENZ_NA["step"])
    c0 = round((west - LORENZ_NA["west"]) / LORENZ_NA["step"])
    with Image.open(png) as im:
        rgb = np.asarray(
            im.convert("RGB").crop((c0, r0, c0 + shape[1], r0 + shape[0])), dtype=np.int32
        )
    palette = np.array([c for c, _ in LORENZ_ZONES], dtype=np.int32)
    d = ((rgb[..., None, :] - palette[None, None]) ** 2).sum(-1)
    return d.argmin(-1)


# ---------- build ----------


def build(settings: Settings, atlas: Grid, base_year: int = 2015, year: int = 2025,
          margin_deg: float = 3.0, sample_every: int = 3) -> tuple[Grid, dict]:  # fmt: skip
    """The atlas updated to `year`, plus the model card (kernel, validation)."""
    cfg = settings.raw["light_pollution"]
    natural = atlas.natural_ucd
    lat = (atlas.south - margin_deg, atlas.north + margin_deg)
    lon = (atlas.west - margin_deg, atlas.east + margin_deg)
    center = ((atlas.south + atlas.north) / 2, (atlas.west + atlas.east) / 2)
    grids, sources = {}, None
    for y in (base_year, year):
        log.info("reading VIIRS %d", y)
        rad, north, west, step = read_strip_window(viirs_path(y), lat, lon)
        e = emission(rad, north, step)
        grids[y] = to_km_grid(e, north, west, step * 2, center)
        if y == year:
            sources = (e, north, west, step * 2)
    # both years on the same km grid (union extent) for comparable features
    feats = {y: convolve_rings(grids[y], RING_EDGES_KM) for y in grids}

    # targets: atlas cells (every `sample_every`th), at their km-grid position
    rr, cc = np.mgrid[0 : atlas.ucd.shape[0] : sample_every, 0 : atlas.ucd.shape[1] : sample_every]
    tlat = atlas.north - (rr.ravel() + 0.5) * atlas.step
    tlon = atlas.west + (cc.ravel() + 0.5) * atlas.step
    target = atlas.ucd[rr.ravel(), cc.ravel()].astype(np.float64)
    ok = np.isfinite(target)
    tx, ty = laea(tlat, tlon, *center)

    def at(y, x, yy):
        r, c = grids[y].index(x, yy)
        return np.stack([f[r, c] for f in feats[y]], axis=1)

    X15 = at(base_year, tx, ty)
    w = fit_kernel(X15[ok], target[ok], natural)
    in_sample = mag_error(X15[ok] @ w, target[ok], natural)
    cv = spatial_cv(X15[ok], target[ok], tlon[ok], natural)
    log.info("kernel fitted: in-sample RMSE %.3f mag, spatial CV RMSE %.3f mag",
             float(np.sqrt(np.mean(in_sample**2))), cv["rmse_mag"])  # fmt: skip

    # every atlas cell: modelled glow in both years -> ratio -> updated atlas
    ar, ac = np.mgrid[0 : atlas.ucd.shape[0], 0 : atlas.ucd.shape[1]]
    alat = atlas.north - (ar.ravel() + 0.5) * atlas.step
    alon = atlas.west + (ac.ravel() + 0.5) * atlas.step
    ax, ay = laea(alat, alon, *center)
    m_base = np.concatenate([at(base_year, ax[i : i + 400_000], ay[i : i + 400_000]) @ w
                             for i in range(0, len(ax), 400_000)])  # fmt: skip
    m_year = np.concatenate([at(year, ax[i : i + 400_000], ay[i : i + 400_000]) @ w
                             for i in range(0, len(ax), 400_000)])  # fmt: skip
    floor = 0.03 * natural  # below ~3% of natural the model's ratio is noise, and irrelevant
    ratio = np.clip((m_year + floor) / (m_base + floor), 0.1, 10.0).reshape(atlas.ucd.shape)
    updated = Grid((atlas.ucd * ratio).astype(np.float32), atlas.north, atlas.west, atlas.step,
                   atlas.natural_ucd, atlas.natural_sqm)  # fmt: skip

    card = {
        "base_year": base_year,
        "year": year,
        "ring_edges_km": RING_EDGES_KM,
        "kernel": w.tolist(),
        "in_sample": {
            "rmse_mag": float(np.sqrt(np.mean(in_sample**2))),
            "median_abs_mag": float(np.median(np.abs(in_sample))),
        },  # fmt: skip
        "spatial_cv": cv,
        "atlas_sigma_mag": cfg["sigma_sqm"],
        "ratio": {
            "median_lit": float(np.median(ratio[atlas.ucd > natural])),
            "share_brighter": float(np.mean(ratio[atlas.ucd > 0.1 * natural] > 1.0)),
            "share_clipped": float(np.mean((ratio <= 0.1) | (ratio >= 10.0))),
        },
        "n_targets": int(ok.sum()),
    }
    assert sources is not None  # set when reading `year` above
    e, north, west, step = sources
    rows, cols = np.nonzero(e > 0)
    card["dome_reference"] = DOME_REFERENCE | {
        "walker": dome_reference(north - (rows + 0.5) * step, west + (cols + 0.5) * step,
                                 e[rows, cols]),
    }  # fmt: skip
    return updated, card | {"_sources": sources}


def zone_center_ratio() -> np.ndarray:
    """Representative artificial/natural ratio of each zone: the geometric middle of its bounds
    (zone 0 and the open top zone use the half / double of their single bound)."""
    b = [x for _, x in LORENZ_ZONES]
    mids = [b[1] / 2] + [math.sqrt(b[i] * b[i + 1]) for i in range(1, len(b) - 1)] + [b[-1] * 2]
    return np.array(mids)


def validate_against_lorenz(atlas: Grid, updated: Grid, png: Path) -> dict | None:
    """Agreement with the independent 2025 atlas, in sky-quality magnitudes (what an observer
    sees; zone steps are tiny in dark places and large in cities, so zones alone mislead)."""
    if not png.exists():
        return None
    theirs = lorenz_zones(png, atlas.north, atlas.west, atlas.step, atlas.ucd.shape)
    their_sqm = atlas.sqm(zone_center_ratio()[theirs] * atlas.natural_ucd)
    lit = theirs >= 7  # Lorenz: artificial light at least equal to natural
    out: dict = {"source": "Lorenz, World Atlas of the Artificial Night Sky Brightness 2025"}
    for name, g in [("atlas_2016", atlas), ("updated", updated)]:
        ours = zone_of_ratio(g.ucd / g.natural_ucd)
        diff = g.sqm(g.ucd) - their_sqm  # positive = ours darker
        out[name] = {
            "within_one_zone": float(np.mean(np.abs(ours - theirs) <= 1)),
            "median_abs_mag": float(np.median(np.abs(diff))),
            "share_within_0_3_mag": float(np.mean(np.abs(diff) <= 0.3)),
            "lit_median_abs_mag": float(np.median(np.abs(diff[lit]))),
            "lit_bias_mag": float(np.mean(diff[lit])),
            "lit_within_one_zone": float(np.mean(np.abs(ours[lit] - theirs[lit]) <= 1)),
        }
    out["lit_share_of_cells"] = float(lit.mean())
    return out


def save_sources(sources, path: Path = SOURCES_PATH, factor: int = 2, min_emission: float = 0.5):
    """2025 light sources for city-glow queries: (lat, lon, emission) of lit cells, coarsened
    to 1 arc-minute (~1.5 km) to keep the file small."""
    e, north, west, step = sources
    h, w = (e.shape[0] // factor) * factor, (e.shape[1] // factor) * factor
    c = e[:h, :w].reshape(h // factor, factor, w // factor, factor).sum(axis=(1, 3))
    rows, cols = np.nonzero(c >= min_emission)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        lat=(north - (rows + 0.5) * step * factor).astype(np.float32),
        lon=(west + (cols + 0.5) * step * factor).astype(np.float32),
        emission=c[rows, cols].astype(np.float32),
    )
    return path


def save_cities(bounds: dict, path: Path = CITIES_PATH, src: Path = LP_RAW / "cities1000.txt",
                margin: float = 3.0) -> Path:  # fmt: skip
    """GeoNames places with 1,000+ people around the mapped region: name, lat, lon, population.
    Small towns matter: a town of 4,000 next to a dark site is its biggest light dome."""
    keep = []
    with src.open(encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            la, lo, pop = float(p[4]), float(p[5]), int(p[14] or 0)
            if (bounds["lat"][0] - margin <= la <= bounds["lat"][1] + margin
                    and bounds["lon"][0] - margin <= lo <= bounds["lon"][1] + margin):  # fmt: skip
                keep.append({"name": p[1], "lat": la, "lon": lo, "pop": pop, "country": p[8]})
    path.write_text(json.dumps(keep, ensure_ascii=False))
    return path


def save_model(card: dict, path: Path = MODEL_PATH) -> Path:
    path.write_text(json.dumps({k: v for k, v in card.items() if not k.startswith("_")}, indent=1))
    return path


# ---------- city glow at one place (app time, numpy only) ----------


@dataclass
class Sources:
    lat: np.ndarray
    lon: np.ndarray
    emission: np.ndarray
    cities: list[dict]
    kernel: np.ndarray
    edges: np.ndarray
    dome_ref: float


def load_sources(sources: Path = SOURCES_PATH, cities: Path = CITIES_PATH,
                 model: Path = MODEL_PATH) -> Sources | None:  # fmt: skip
    if not (sources.exists() and cities.exists() and model.exists()):
        return None
    card = json.loads(model.read_text())
    with np.load(sources) as z:
        return Sources(
            z["lat"].astype(np.float64), z["lon"].astype(np.float64),
            z["emission"].astype(np.float64), json.loads(cities.read_text()),
            np.array(card["kernel"]), np.array(card["ring_edges_km"]),
            float(card["dome_reference"]["walker"]),
        )  # fmt: skip


def bearings(lat, lon, lat2, lon2) -> np.ndarray:
    p1, p2 = math.radians(lat), np.radians(lat2)
    dl = np.radians(np.asarray(lon2) - lon)
    x = np.sin(dl) * np.cos(p2)
    y = math.cos(p1) * np.sin(p2) - math.sin(p1) * np.cos(p2) * np.cos(dl)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def zenith_shares(src: Sources, lat: float, lon: float, radius_km: float = 300.0):
    """Each light source within `radius_km`: distance (km), bearing (deg) and its share of the
    artificial glow overhead under the fitted kernel (emission x K(distance))."""
    near = (np.abs(src.lat - lat) < radius_km / 111) & (
        np.abs(src.lon - lon) < radius_km / (111 * max(math.cos(math.radians(lat)), 0.2))
    )
    la, lo, e = src.lat[near], src.lon[near], src.emission[near]
    d = haversine_km(lat, lon, la, lo)
    keep = d <= radius_km
    la, lo, e, d = la[keep], lo[keep], e[keep], d[keep]
    ring = np.clip(np.searchsorted(src.edges, d, side="right") - 1, 0, len(src.kernel) - 1)
    return d, bearings(lat, lon, la, lo), e * src.kernel[ring]


def darkest_window(dome: np.ndarray, size: int = 4) -> int:
    """Start sector of the `size` adjacent sectors with the least dome light. Ties (common when
    most directions have no lights at all) go to the window facing farthest from the strongest
    dome, so the advice points away from the glow rather than just beside it."""
    n = len(dome)
    sums = np.convolve(np.r_[dome, dome[: size - 1]], np.ones(size), "valid")[:n]
    centre = (np.arange(n) + (size - 1) / 2) % n
    away = np.abs(((centre - int(np.argmax(dome))) + n / 2) % n - n / 2)  # 0 .. n/2 sectors
    tol = 1e-9 * max(float(dome.sum()), 1e-12)
    best = np.flatnonzero(sums <= sums.min() + tol)
    return int(best[np.argmax(away[best])])


def city_glow(src: Sources, lat: float, lon: float, radius_km: float = 300.0,
              sectors: int = 16) -> dict:  # fmt: skip
    """Where tonight's artificial glow comes from, seen from (lat, lon).

    - `overhead`: the fitted kernel's split of the zenith glow by source direction (shares).
    - `sectors[i]["dome"]`: Walker's-law (distance^-2.5) strength of the light dome in each
      direction, relative to DOME_REFERENCE (Sacramento seen from 50 km = 1.0). Sources within
      2 km are excluded: that's local lighting, not a dome on the horizon.
    - `cities`: the biggest named contributors (>1% of the dome light), with direction,
      distance, share and strength; light not inside any town is "scattered".
    """
    near = (np.abs(src.lat - lat) < radius_km / 111) & (
        np.abs(src.lon - lon) < radius_km / (111 * max(math.cos(math.radians(lat)), 0.2))
    )
    la, lo, e = src.lat[near], src.lon[near], src.emission[near]
    d = haversine_km(lat, lon, la, lo)
    keep = d <= radius_km
    la, lo, e, d = la[keep], lo[keep], e[keep], d[keep]
    b = bearings(lat, lon, la, lo)
    ring = np.clip(np.searchsorted(src.edges, d, side="right") - 1, 0, len(src.kernel) - 1)
    zen = e * src.kernel[ring]
    # Walker's law was measured for cities ~10 km and farther; closer sources count as at 10 km
    # (d^-2.5 would explode), and lights within 2 km are local lighting, not a dome.
    walker = np.where(d >= 2.0, e * np.maximum(d, WALKER_MIN_KM) ** -2.5, 0.0)
    width = 360 / sectors
    sec = ((b + width / 2) // width).astype(int) % sectors
    zen_s = np.bincount(sec, zen, sectors)
    dome_s = np.bincount(sec, walker, sectors)
    names = COMPASS16 if sectors == 16 else [f"{i * width:.0f}°" for i in range(sectors)]

    # attribute each source to the nearest place whose built-up radius covers it
    cities = [c for c in src.cities
              if haversine_km(lat, lon, c["lat"], c["lon"]) <= radius_km + 25]  # fmt: skip
    top = []
    total = walker.sum()
    if cities and len(e) and total > 0:
        clat = np.array([c["lat"] for c in cities])
        clon = np.array([c["lon"] for c in cities])
        reach = city_radius_km([c["pop"] for c in cities])
        # each source belongs to the nearest place whose radius covers it; checking only the
        # sources inside each place's bounding box keeps this ~O(sources) instead of a full
        # sources x places distance matrix
        best = np.full(len(e), np.inf)
        nearest = np.zeros(len(e), dtype=int)
        coslat = max(math.cos(math.radians(lat)), 0.2)
        for j in range(len(cities)):
            box = (np.abs(la - clat[j]) <= reach[j] / 111) & (
                np.abs(lo - clon[j]) <= reach[j] / (111 * coslat)
            )
            if not box.any():
                continue
            idx = np.flatnonzero(box)
            dj = haversine_km(clat[j], clon[j], la[idx], lo[idx])
            closer = (dj <= reach[j]) & (dj < best[idx])
            best[idx[closer]] = dj[closer]
            nearest[idx[closer]] = j
        owned = np.isfinite(best)
        weight = np.bincount(nearest[owned], walker[owned], len(cities))
        for i in np.argsort(weight)[::-1][:6]:
            if weight[i] <= 0.01 * total:
                break
            c = cities[i]
            dist = float(haversine_km(lat, lon, c["lat"], c["lon"]))
            bc = float(bearings(lat, lon, np.array([c["lat"]]), np.array([c["lon"]]))[0])
            top.append({
                "name": c["name"], "population": c["pop"], "distance_km": dist,
                "direction": names[int(((bc + width / 2) // width) % sectors)], "bearing": bc,
                "share": float(weight[i] / total),
                "strength": float(weight[i] / src.dome_ref),
            })  # fmt: skip
        scattered = float(walker[~owned].sum() / total)
    else:
        scattered = 0.0
    darkest = darkest_window(dome_s)
    return {
        "sectors": [
            {
                "direction": names[i],
                "bearing": i * width,
                "overhead_ucd": float(zen_s[i]),
                "overhead_share": float(zen_s[i] / (zen_s.sum() or 1.0)),
                "dome": float(dome_s[i] / src.dome_ref),
            }
            for i in range(sectors)
        ],  # fmt: skip
        "overhead_ucd": float(zen_s.sum()),
        "cities": top,
        "scattered_share": scattered,
        "dome_total": float(dome_s.sum() / src.dome_ref),
        # the darkest quarter of the horizon (4 adjacent sectors with the weakest domes)
        "darkest_quarter": [names[(darkest + k) % sectors] for k in range(4)],
    }
