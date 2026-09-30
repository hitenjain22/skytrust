"""Light pollution from the World Atlas of Artificial Night Sky Brightness.

Source: Falchi et al. (2016), "The new world atlas of artificial night sky brightness",
Science Advances 2:e1600377; data doi:10.5880/GFZ.1.4.2016.001 (CC BY-NC 4.0). The atlas models
the *artificial* zenith sky brightness (mcd/m²) on a 30-arcsecond grid (about 0.9 km x 0.7 km
here) from VIIRS satellite radiances and a light-propagation model, calibrated against sky
quality meter (SQM) measurements with a standard deviation of 0.15 mag/arcsec².

Everything below uses the paper's own numbers:

- natural sky background 174 µcd/m² = 22.0 mag/arcsec², so total sky brightness in
  mag/arcsec² is 22.0 - 2.5·log10((artificial + 174) / 174);
- light-pollution levels are the ratio artificial / natural, doubling per level (the colour
  scale of the atlas);
- Bortle classes come from the SQM ranges tabulated in the Bortle scale article (Bortle 2001;
  the visual scale has no exact SQM definition, so these ranges are approximate).

The global file is 2.9 GB; only the tiles covering `light_pollution.bounds` are read (it is an
uncompressed, tiled GeoTIFF, so plain numpy suffices) and saved to a small compressed artifact.
"""

from __future__ import annotations

import logging
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from skytrust.config import REPO_ROOT, Settings
from skytrust.data.cache import RAW_DIR

log = logging.getLogger(__name__)

ATLAS_TIF = RAW_DIR / "light_pollution" / "World_Atlas_2015.tif"
GRID_PATH = REPO_ROOT / "artifacts" / "light_pollution.npz"
OVERLAY_PATH = REPO_ROOT / "artifacts" / "light_pollution.png"
EARTH_KM = 6371.0

# Atlas colour levels: ratio of artificial to natural brightness (Falchi et al. 2016, Fig. 1).
# Descriptions paraphrase the paper where it describes a level; the rest are plain labels.
LEVELS = [
    (0.01, "Pristine sky", "Less than 1% brighter than natural"),
    (0.02, "Near pristine", "1–2% brighter than natural"),
    (0.04, "Very slightly degraded", "2–4% brighter than natural"),
    (0.08, "Slightly degraded", "4–8% brighter than natural"),
    (0.16, "Degraded near the horizon", "8–16% brighter: astronomically polluted"),
    (0.32, "Moderately polluted", "16–32% brighter than natural"),
    (0.64, "Polluted", "32–64% brighter than natural"),
    (1.28, "Heavily polluted", "Up to about double the natural brightness"),
    (2.56, "Milky Way fading", "2–3.5x natural: the Milky Way loses detail"),
    (5.12, "Milky Way hidden", "3.5–6x natural: the summer Milky Way is masked"),
    (10.24, "Very bright sky", "6–11x natural: near the limit of night (scotopic) vision"),
    (20.48, "City sky", "11–21x natural"),
    (40.96, "Bright city sky", "21–41x natural"),
    (math.inf, "No dark adaptation", "More than 41x natural: the eye can't dark-adapt"),
]

# Bortle class by total sky brightness (mag/arcsec²), brightest first; lower bound of each class.
BORTLE = [
    (21.76, "1", "Excellent dark-sky site", "Milky Way casts shadows; zodiacal light obvious"),
    (21.60, "2", "Truly dark site", "Milky Way highly structured; many Messier objects by eye"),
    (21.30, "3", "Rural sky", "Milky Way complex; some glow near the horizon"),
    (20.80, "4", "Rural/suburban transition", "Milky Way visible but lacks detail"),
    (20.30, "4.5", "Semi-suburban", "Milky Way only vaguely visible"),
    (19.25, "5", "Suburban sky", "Milky Way washed out overhead"),
    (18.50, "6", "Bright suburban sky", "Milky Way only near the zenith"),
    (18.00, "7", "Suburban/urban transition", "Milky Way nearly invisible; sky light gray"),
    (-math.inf, "8–9", "City sky", "Only the Moon, planets and bright stars and clusters"),
]


# ---------- conversions ----------


def sqm(artificial_ucd, natural_ucd: float = 174.0, natural_sqm: float = 22.0):
    """Total zenith sky brightness in mag/arcsec² (higher = darker)."""
    a = np.asarray(artificial_ucd, dtype=float)
    return natural_sqm - 2.5 * np.log10((np.maximum(a, 0) + natural_ucd) / natural_ucd)


def ratio(artificial_ucd, natural_ucd: float = 174.0):
    """Artificial brightness as a fraction of the natural sky (0.5 = 50% brighter)."""
    return np.asarray(artificial_ucd, dtype=float) / natural_ucd


def level(r: float) -> tuple[int, str, str]:
    """(index 0-13, name, description) of the atlas colour level for a ratio."""
    for i, (upper, name, text) in enumerate(LEVELS):
        if r < upper:
            return i, name, text
    raise AssertionError("unreachable")


def bortle(m: float) -> tuple[str, str, str]:
    """(class, title, what you see) for a total sky brightness in mag/arcsec²."""
    for lower, cls, title, text in BORTLE:
        if m >= lower:
            return cls, title, text
    raise AssertionError("unreachable")


def bortle_number(m) -> np.ndarray:
    """Numeric Bortle class for arrays (4.5 stays 4.5; 8-9 counts as 8.5)."""
    m = np.asarray(m, dtype=float)
    out = np.full(m.shape, 8.5)
    for lower, cls, *_ in reversed(BORTLE[:-1]):
        out = np.where(m >= lower, float(cls), out)
    return out


# ---------- reading the atlas GeoTIFF ----------


@dataclass
class TiffInfo:
    width: int
    height: int
    tile: int
    tile_offsets: np.ndarray
    west: float
    north: float
    step: float


def read_tiff_info(path: Path) -> TiffInfo:
    """Minimal parser for the atlas's layout: little-endian, uncompressed, 128 x 128 float32
    tiles, with GeoTIFF pixel-scale and tie-point tags. Anything else is rejected, not guessed."""
    with path.open("rb") as f:
        head = f.read(8)
        if head[:4] != b"II*\x00":
            raise ValueError(f"{path}: not a little-endian classic TIFF")
        (ifd,) = struct.unpack("<I", head[4:])
        f.seek(ifd)
        (n,) = struct.unpack("<H", f.read(2))
        tags = {}
        for _ in range(n):
            tag, typ, count, value = struct.unpack("<HHII", f.read(12))
            tags[tag] = (typ, count, value)

        def scalar(tag):
            typ, _, value = tags[tag]
            return value & 0xFFFF if typ == 3 else value

        def doubles(tag):
            _, count, offset = tags[tag]
            f.seek(offset)
            return struct.unpack(f"<{count}d", f.read(8 * count))

        if scalar(259) != 1 or scalar(258) != 32 or scalar(339) != 3 or scalar(322) != scalar(323):
            raise ValueError(f"{path}: expected uncompressed square-tiled float32")
        _, n_tiles, offsets_at = tags[324]
        if n_tiles == 1:  # a single offset is stored inline in the tag itself
            offsets = np.array([offsets_at], dtype=np.int64)
        else:
            f.seek(offsets_at)
            offsets = np.frombuffer(f.read(4 * n_tiles), dtype="<u4").astype(np.int64)
        scale, tie = doubles(33550), doubles(33922)
    return TiffInfo(scalar(256), scalar(257), scalar(322), offsets, tie[3], tie[4], scale[0])


def read_window(path: Path, lat: tuple[float, float], lon: tuple[float, float]):
    """Atlas values (mcd/m²) for a lat/lon box, plus (north, west, step) of the returned grid."""
    info = read_tiff_info(path)
    r0 = max(0, math.floor((info.north - lat[1]) / info.step))
    r1 = min(info.height, math.ceil((info.north - lat[0]) / info.step))
    c0 = max(0, math.floor((lon[0] - info.west) / info.step))
    c1 = min(info.width, math.ceil((lon[1] - info.west) / info.step))
    t = info.tile
    tiles_across = math.ceil(info.width / t)
    out = np.empty((r1 - r0, c1 - c0), dtype=np.float32)
    with path.open("rb") as f:  # tile offsets need not be 4-byte aligned, so read bytes per tile
        for tr in range(r0 // t, (r1 - 1) // t + 1):
            for tc in range(c0 // t, (c1 - 1) // t + 1):
                f.seek(int(info.tile_offsets[tr * tiles_across + tc]))
                tile = np.frombuffer(f.read(4 * t * t), dtype="<f4").reshape(t, t)
                rs, cs = tr * t, tc * t
                a0, a1 = max(r0, rs), min(r1, rs + t)
                b0, b1 = max(c0, cs), min(c1, cs + t)
                block = tile[a0 - rs : a1 - rs, b0 - cs : b1 - cs]
                out[a0 - r0 : a1 - r0, b0 - c0 : b1 - c0] = block
    north = info.north - r0 * info.step
    west = info.west + c0 * info.step
    return out, north, west, info.step


# ---------- the regional grid ----------


@dataclass
class Grid:
    """Artificial zenith brightness (µcd/m²) on a regular lat/lon grid (row 0 = north edge)."""

    ucd: np.ndarray
    north: float
    west: float
    step: float
    natural_ucd: float = 174.0
    natural_sqm: float = 22.0

    @property
    def south(self) -> float:
        return self.north - self.ucd.shape[0] * self.step

    @property
    def east(self) -> float:
        return self.west + self.ucd.shape[1] * self.step

    def contains(self, lat: float, lon: float) -> bool:
        return self.south <= lat < self.north and self.west <= lon < self.east

    def index(self, lat: float, lon: float) -> tuple[int, int]:
        r = int((self.north - lat) / self.step)
        c = int((lon - self.west) / self.step)
        return min(max(r, 0), self.ucd.shape[0] - 1), min(max(c, 0), self.ucd.shape[1] - 1)

    def cell_center(self, r: int, c: int) -> tuple[float, float]:
        return self.north - (r + 0.5) * self.step, self.west + (c + 0.5) * self.step

    def sqm(self, ucd):
        return sqm(ucd, self.natural_ucd, self.natural_sqm)

    def around(self, lat: float, lon: float, radius_km: float):
        """(values, lats, lons, distances in km) of every cell within radius_km."""
        dlat = radius_km / 111.2
        dlon = radius_km / (111.2 * math.cos(math.radians(lat)))
        r0, c0 = self.index(lat + dlat, lon - dlon)
        r1, c1 = self.index(lat - dlat, lon + dlon)
        rows = np.arange(r0, r1 + 1)
        cols = np.arange(c0, c1 + 1)
        lats = self.north - (rows + 0.5) * self.step
        lons = self.west + (cols + 0.5) * self.step
        glat, glon = np.meshgrid(lats, lons, indexing="ij")
        dist = haversine_km(lat, lon, glat, glon)
        inside = dist <= radius_km
        vals = self.ucd[r0 : r1 + 1, c0 : c1 + 1]
        return vals[inside], glat[inside], glon[inside], dist[inside]


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(a))


def bearing(lat1, lon1, lat2, lon2) -> str:
    """Compass direction from point 1 to point 2 (8 points)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    deg = (math.degrees(math.atan2(x, y)) + 360) % 360
    return ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][int((deg + 22.5) // 45) % 8]


def build(settings: Settings, tif: Path = ATLAS_TIF) -> Grid:
    cfg = settings.raw["light_pollution"]
    mcd, north, west, step = read_window(
        tif, tuple(cfg["bounds"]["lat"]), tuple(cfg["bounds"]["lon"])
    )
    ucd = np.where(np.isfinite(mcd) & (mcd >= 0), mcd * 1000.0, np.nan).astype(np.float32)
    return Grid(ucd, north, west, step, cfg["natural_ucd"], cfg["natural_sqm"])


def save(grid: Grid, path: Path = GRID_PATH) -> Path:
    """Stored as total sky brightness in thousandths of a mag/arcsec² (uint16): exact to 0.0005
    mag, far below the atlas's 0.15 mag error, and a fraction of the float32 size."""
    path.parent.mkdir(parents=True, exist_ok=True)
    milli = np.round(np.nan_to_num(grid.sqm(grid.ucd), nan=0.0) * 1000).astype(np.uint16)
    np.savez_compressed(
        path,
        sqm_milli=milli,
        meta=np.array([grid.north, grid.west, grid.step, grid.natural_ucd, grid.natural_sqm]),
    )
    return path


def load(path: Path = GRID_PATH) -> Grid | None:
    if not path.exists():
        return None
    with np.load(path) as z:
        north, west, step, nat_ucd, nat_sqm = z["meta"].tolist()
        m = z["sqm_milli"].astype(np.float64) / 1000
    # invert sqm(): artificial = natural * (10^((natural_sqm - m) / 2.5) - 1); 0 marks no data
    ucd = np.where(m > 0, nat_ucd * (10 ** ((nat_sqm - m) / 2.5) - 1), np.nan)
    return Grid(np.maximum(ucd, 0).astype(np.float32), north, west, step, nat_ucd, nat_sqm)


# ---------- what the app shows ----------


def point(grid: Grid, lat: float, lon: float, sigma: float = 0.15) -> dict | None:
    """Everything about the sky brightness at one place."""
    if not grid.contains(lat, lon):
        return None
    r, c = grid.index(lat, lon)
    a = float(grid.ucd[r, c])
    if not math.isfinite(a):
        return None
    m = float(grid.sqm(a))
    rt = a / grid.natural_ucd
    lvl, lvl_name, lvl_text = level(rt)
    b_cls, b_title, b_text = bortle(m)
    return {
        "artificial_ucd": a,
        "ratio": rt,
        "sqm": m,
        "sqm_lo": m - sigma,
        "sqm_hi": m + sigma,
        "level": lvl,
        "level_name": lvl_name,
        "level_text": lvl_text,
        "bortle": b_cls,
        "bortle_title": b_title,
        "bortle_text": b_text,
    }


def _describe(grid: Grid, lat, lon, lats, lons, dist, i: int) -> dict:
    out = point(grid, float(lats[i]), float(lons[i])) or {}
    return out | {
        "lat": float(lats[i]),
        "lon": float(lons[i]),
        "distance_km": float(dist[i]),
        "direction": bearing(lat, lon, float(lats[i]), float(lons[i])),
    }


def darkest_within(
    grid: Grid, lat: float, lon: float, radius_km: float, tolerance: float = 0.15
) -> dict | None:
    """The nearest spot within radius_km whose sky is as dark as the darkest one there, up to
    `tolerance` mag/arcsec² (the model's own error: smaller differences aren't meaningful, so a
    close spot beats a marginally darker far one). Straight-line distance; roads not considered."""
    vals, lats, lons, dist = grid.around(lat, lon, radius_km)
    ok = np.isfinite(vals)
    if not ok.any():
        return None
    vals, lats, lons, dist = vals[ok], lats[ok], lons[ok], dist[ok]
    m = grid.sqm(vals)
    candidates = np.flatnonzero(m >= m.max() - tolerance)
    i = int(candidates[np.argmin(dist[candidates])])
    return _describe(grid, lat, lon, lats, lons, dist, i)


def nearest_dark(
    grid: Grid, lat: float, lon: float, max_bortle: float = 3, radius_km: float = 200
) -> dict | None:
    """The nearest spot with a Bortle class of max_bortle or darker, within radius_km."""
    vals, lats, lons, dist = grid.around(lat, lon, radius_km)
    ok = np.isfinite(vals)
    good = np.flatnonzero(ok & (bortle_number(grid.sqm(np.where(ok, vals, 0))) <= max_bortle))
    if not len(good):
        return None
    i = int(good[np.argmin(dist[good])])
    return _describe(grid, lat, lon, lats, lons, dist, i)


def area_shares(grid: Grid, lat: float, lon: float, radius_km: float) -> dict[str, float]:
    """Share of the area within radius_km in each Bortle band (grid cells shrink with latitude
    equally within a small disc, so cell counts are area shares to good approximation)."""
    vals, *_ = grid.around(lat, lon, radius_km)
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return {}
    b = bortle_number(grid.sqm(vals))
    bands = {"1–3": b <= 3, "4–4.5": (b > 3) & (b <= 4.5), "5–6": (b > 4.5) & (b <= 6),
             "7–9": b > 6}  # fmt: skip
    return {k: float(v.mean()) for k, v in bands.items()}


def site_report(grid: Grid, lat: float, lon: float, settings: Settings) -> dict | None:
    cfg = settings.raw["light_pollution"]
    here = point(grid, lat, lon, cfg["sigma_sqm"])
    if here is None:
        return None
    return {
        "here": here,
        "darkest": {
            r: darkest_within(grid, lat, lon, r, cfg["sigma_sqm"]) for r in cfg["radii_km"]
        },
        "nearest_dark": nearest_dark(grid, lat, lon, 3, max(cfg["radii_km"]) * 2),
        "shares": area_shares(grid, lat, lon, cfg["radii_km"][1]),
        "shares_radius_km": cfg["radii_km"][1],
    }


# ---------- map overlay ----------

# Warm glow on transparent: pristine sky shows the base map, cities glow cream-white. One colour
# per atlas level (index 0-13); alpha rises with brightness.
# Levels below 4% stay fully transparent: a faint tint there would darken the base map and
# show a seam at the region's edge.
OVERLAY_RGBA = [
    (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (110, 90, 64, 55), (130, 104, 70, 90),
    (150, 118, 72, 120), (175, 135, 76, 145), (196, 150, 80, 165), (214, 166, 88, 185),
    (228, 184, 104, 200), (238, 202, 128, 215), (244, 218, 158, 228), (248, 232, 190, 238),
    (252, 244, 222, 245),
]  # fmt: skip


def overlay_rgba(grid: Grid, every: int = 1) -> np.ndarray:
    """RGBA image (rows north to south) of the atlas levels, downsampled by `every`."""
    r = ratio(grid.ucd[::every, ::every], grid.natural_ucd)
    edges = np.array([u for u, *_ in LEVELS[:-1]])
    idx = np.searchsorted(edges, np.nan_to_num(r, nan=0.0), side="right")
    return np.array(OVERLAY_RGBA, dtype=np.uint8)[idx]


def save_overlay(grid: Grid, path: Path = OVERLAY_PATH) -> Path:
    import matplotlib.image as mpimg

    path.parent.mkdir(parents=True, exist_ok=True)
    mpimg.imsave(path, overlay_rgba(grid))
    return path
