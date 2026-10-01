"""Build the small sky catalogues the app draws and reasons with (run once, offline).

Sources (all in data/raw/sky/, see docs/DATA_NOTES.md §13):
- Stars: the Hipparcos main catalogue (ESA 1997, CDS I/239): position, V magnitude, B-V colour.
  Proper names from Skyfield's IAU-based `named_star_dict`.
- Constellation stick figures, label positions and the Milky Way outline: d3-celestial
  (Olaf Frohn, BSD-3-Clause).
- Deep-sky showpieces: d3-celestial's Messier and bright-DSO lists, each checked against the
  constellation it must fall in (that check caught M4 filed at M42's right ascension in the
  bright-DSO list, so Messier objects come from the Messier list).

Constellation membership uses Skyfield's bundled IAU boundaries (Roman 1987), so it is an
independent check on every coordinate taken from the third-party files.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np

from skytrust.config import REPO_ROOT

log = logging.getLogger(__name__)

RAW = REPO_ROOT / "data" / "raw" / "sky"
SKY_DIR = REPO_ROOT / "artifacts" / "sky"
STARS_PATH = SKY_DIR / "stars.json"
CONSTELLATIONS_PATH = SKY_DIR / "constellations.json"
MILKY_WAY_PATH = SKY_DIR / "milkyway.json"
DSO_PATH = SKY_DIR / "dso.json"

STAR_MAG_LIMIT = 6.5  # what the darkest sky in the app shows to a typical observer (~6.6)

# Naked-eye and binocular showpieces visible from California, with the constellation each must
# fall in. Source keys: Messier list ("M..") or the bright-DSO list (others).
SHOWPIECES = [
    ("M45", "Pleiades", "cluster", "Tau", "A tight little dipper of blue stars"),
    ("C 41", "Hyades", "cluster", "Tau", "The V-shaped face of Taurus around Aldebaran"),
    ("Cr 39", "Alpha Persei Cluster", "cluster", "Per", "A loose spray of stars around Mirfak"),
    ("Cr 256", "Coma Star Cluster", "cluster", "Com", "A faint scatter of stars below Leo's tail"),
    ("M44", "Beehive Cluster", "cluster", "Cnc", "A misty patch between Gemini and Leo"),
    ("M7", "Ptolemy's Cluster", "cluster", "Sco", "Above the tail of Scorpius"),
    ("M6", "Butterfly Cluster", "cluster", "Sco", "Just above Ptolemy's Cluster"),
    ("NGC 869", "Double Cluster", "cluster", "Per", "Two star clusters side by side"),
    ("M31", "Andromeda Galaxy", "galaxy", "And", "The farthest thing you can see by eye"),
    ("M33", "Triangulum Galaxy", "galaxy", "Tri", "Only from very dark skies"),
    ("M42", "Orion Nebula", "nebula", "Ori", "The fuzzy 'star' in Orion's sword"),
    ("M8", "Lagoon Nebula", "nebula", "Sgr", "A glow above the spout of the Teapot"),
    ("M13", "Hercules Cluster", "cluster", "Her", "A ball of ~300,000 stars, a smudge by eye"),
    ("M22", "M22 globular cluster", "cluster", "Sgr", "Above the lid of the Teapot"),
    ("M41", "M41 cluster", "cluster", "CMa", "Just below Sirius"),
    ("M35", "M35 cluster", "cluster", "Gem", "At the feet of Gemini"),
    ("NGC 5139", "Omega Centauri", "cluster", "Cen", "Low in the south; best from SoCal"),
]


def parse_hipparcos(path: Path = RAW / "hip_main.dat", limit: float = STAR_MAG_LIMIT) -> dict:
    """HIP number, RA/Dec (deg, ICRS, epoch J1991.25), V and B-V for stars brighter than
    `limit`. Proper motion is ignored: even the fastest naked-eye star moves < 0.1° in 35 years."""
    hip, ra, dec, mag, bv = [], [], [], [], []
    with path.open() as f:
        for line in f:
            p = line.split("|")
            vmag = p[5].strip()
            if not vmag or float(vmag) > limit:
                continue
            if p[8].strip():
                r, d = float(p[8]), float(p[9])
            else:  # a few entries have no astrometric solution: use the sexagesimal columns
                h, m, s = (float(x) for x in p[3].split())
                sign = -1 if p[4].strip().startswith("-") else 1
                dd, dm, ds = (abs(float(x)) for x in p[4].split())
                r, d = 15 * (h + m / 60 + s / 3600), sign * (dd + dm / 60 + ds / 3600)
            hip.append(int(p[1]))
            ra.append(round(r, 4))
            dec.append(round(d, 4))
            mag.append(float(vmag))
            bv.append(float(p[37]) if p[37].strip() else None)
    return {"hip": hip, "ra": ra, "dec": dec, "mag": mag, "bv": bv}


@lru_cache(maxsize=1)
def _constellation_map():
    from skyfield.api import load_constellation_map

    return load_constellation_map()


def constellation_of(ra_deg, dec_deg) -> list[str]:
    """IAU constellation abbreviation for J2000 positions (Skyfield's bundled boundaries)."""
    from skyfield.api import position_of_radec

    lookup = _constellation_map()
    pos = position_of_radec(np.asarray(ra_deg, float) / 15.0, np.asarray(dec_deg, float))
    return [str(c) for c in np.atleast_1d(lookup(pos))]


def build_stars(path: Path = STARS_PATH) -> dict:
    from skyfield.named_stars import named_star_dict

    stars = parse_hipparcos()
    stars["con"] = constellation_of(stars["ra"], stars["dec"])
    by_hip = {h: name for name, h in named_star_dict.items()}
    stars["names"] = {str(h): by_hip[h] for h in stars["hip"] if h in by_hip}
    _write(path, stars)
    log.info(
        "%d stars to V %.1f (%d named)", len(stars["hip"]), STAR_MAG_LIMIT, len(stars["names"])
    )
    return stars


def _ra360(x: float) -> float:
    return round(x % 360.0, 3)


def build_constellations(path: Path = CONSTELLATIONS_PATH) -> dict:
    """{abbr: {name, label [ra, dec], lines [[[ra, dec], ...], ...]}} with RA in 0-360°."""
    from skyfield.api import load_constellation_names

    names = dict(load_constellation_names())
    lines = json.loads((RAW / "constellations.lines.json").read_text())
    labels = json.loads((RAW / "constellations.json").read_text())
    out: dict[str, dict] = {}
    for f in labels["features"]:
        abbr = f["id"]
        lon, lat = f["geometry"]["coordinates"]
        out[abbr] = {
            "name": names.get(abbr, f["properties"]["name"]),
            "label": [_ra360(lon), round(lat, 3)],
            "lines": [],
        }
    for f in lines["features"]:
        out.setdefault(f["id"], {"name": names.get(f["id"], f["id"]), "label": None, "lines": []})
        out[f["id"]]["lines"] = [
            [[_ra360(lon), round(lat, 3)] for lon, lat in seg]
            for seg in f["geometry"]["coordinates"]
        ]
    _write(path, out)
    return out


def build_milky_way(path: Path = MILKY_WAY_PATH, max_points: int = 600) -> dict:
    """The five brightness levels of the Milky Way outline (faintest first), each a list of
    rings [[ra, dec], ...]. Long rings are thinned to `max_points` (the chart is ~700 px)."""
    raw = json.loads((RAW / "mw.json").read_text())
    levels = []
    for f in sorted(raw["features"], key=lambda f: f["id"]):
        rings = []
        for poly in f["geometry"]["coordinates"]:
            for ring in poly:
                step = max(1, len(ring) // max_points)
                rings.append([[_ra360(lon), round(lat, 2)] for lon, lat in ring[::step]])
        levels.append(rings)
    out = {"levels": levels}
    _write(path, out)
    return out


# Clusters wider than this are seen star by star (Pleiades, Hyades); smaller ones as a glow,
# like M44, which the Bortle scale treats like M31.
LOOSE_CLUSTER_ARCMIN = 100


def _size_arcmin(dim) -> float:
    """First number of d3-celestial's 'dim' field ('189x61' -> 189)."""
    try:
        return float(str(dim).lower().split("x")[0])
    except ValueError:
        return float("nan")


def group_magnitude(ra: float, dec: float, radius_deg: float, stars: dict, nth: int = 3) -> float:
    """Magnitude of the nth-brightest catalogue star within `radius_deg`: a wide cluster shows as
    a group once a few of its stars are visible, however bright its combined light."""
    from skytrust.sky import separation_deg

    sep = separation_deg(ra, dec, np.array(stars["ra"]), np.array(stars["dec"]))
    mags = np.sort(np.array(stars["mag"])[sep <= radius_deg])
    return float(mags[min(nth, len(mags)) - 1]) if len(mags) else float("nan")


def build_dsos(path: Path = DSO_PATH, stars: dict | None = None) -> list[dict]:
    """Showpieces with position, magnitude and how to judge their visibility: `vis_mag` is the
    magnitude to compare with the limiting magnitude, `extended` whether the extended-object
    margin applies (a diffuse glow) or not (a wide cluster seen as separate stars)."""
    stars = stars or json.loads(STARS_PATH.read_text())
    messier = {f["id"]: f for f in json.loads((RAW / "messier.json").read_text())["features"]}
    bright = {f["id"]: f for f in json.loads((RAW / "dsos.bright.json").read_text())["features"]}
    out = []
    for key, name, kind, con, note in SHOWPIECES:
        f = messier.get(key) or bright.get(key)
        if f is None:
            raise KeyError(f"{key} not found in the d3-celestial lists")
        lon, lat = f["geometry"]["coordinates"]
        ra, dec = _ra360(lon), round(lat, 3)
        found = constellation_of([ra], [dec])[0]
        if found != con:
            raise ValueError(f"{key} ({name}) lies in {found}, expected {con}: bad coordinates")
        mag, size = float(f["properties"]["mag"]), _size_arcmin(f["properties"].get("dim"))
        loose = kind == "cluster" and size >= LOOSE_CLUSTER_ARCMIN
        vis_mag = group_magnitude(ra, dec, size / 120, stars) if loose else mag
        out.append(
            {
                "id": key,
                "name": name,
                "kind": kind,
                "con": con,
                "ra": ra,
                "dec": dec,
                "mag": mag,
                "size_arcmin": size,
                "vis_mag": round(vis_mag, 2),
                "extended": not loose,
                "note": note,
            }
        )
    _write(path, out)
    return out


def build_all() -> None:
    SKY_DIR.mkdir(parents=True, exist_ok=True)
    stars = build_stars()
    build_constellations()
    build_milky_way()
    build_dsos(stars=stars)


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, separators=(",", ":"), ensure_ascii=False))
