"""What's up tonight, for a person standing at a place: where the stars, planets, Moon and Milky
Way are, how dark the sky is in plain words, and which of those things a person can actually see
through the local light pollution and moonlight.

Everything here is computed, not looked up: positions from JPL's DE421 ephemeris and the
Hipparcos catalogue via Skyfield; visibility from published sky-brightness models:

- Limiting magnitude (faintest star a typical observer sees at the zenith) from sky brightness:
  NELM = 7.93 - 5 log10(10^(4.316 - B/5) + 1)  (Schaefer 1990, PASP 102:212, as converted by
  Unihedron's NELM calculator; B in mag/arcsec²).
- Moonlight: Krisciunas & Schaefer (1991, PASP 103:1033), equations as reproduced by Yao et al.
  (2013, RAA 13:1255): Rayleigh + Mie scattering of moonlight, the van Rhijn brightening of the
  dark sky towards the horizon, and atmospheric extinction.
- Units: B[nL] = 34.08 exp(20.7233 - 0.92104 V) (Garstang 1989).
- Plain-language darkness tiers use the Bortle scale's SQM ranges and descriptions.

Positions are geometric (no refraction), like the twilight definitions in `astro`. Within one
night, stars' apparent equatorial positions are fixed to well under 0.01°, so they are computed
once and turned into altitude/azimuth with the local sidereal time; planets and the Moon are
computed at every time.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from skytrust import astro
from skytrust.config import Site
from skytrust.skycatalog import CONSTELLATIONS_PATH, DSO_PATH, MILKY_WAY_PATH, STARS_PATH

NATURAL_SQM = 22.0  # mag/arcsec², the natural sky in Falchi et al. (2016)
# V-band extinction per airmass. K&S measured 0.172 on Mauna Kea; Yao et al. fitted 0.23 at
# Xinglong (900 m). 0.2 sits between them; the model is not sensitive to it at these precisions.
EXTINCTION_K = 0.20
# Extended objects (clusters, galaxies, nebulae) are harder to see than a star of the same total
# brightness. Rule of thumb: visible if magnitude <= NELM - 0.5. Chosen so the Bortle scale's
# descriptions come out right: M31's core barely visible at class 7-8 (SQM ~18.0, NELM ~4.0)
# and M33 a difficult averted-vision object at class 4 (SQM ~21, NELM ~6.2).
EXTENDED_MARGIN = 0.5
MILKY_WAY_MIN_SQM = 18.5  # Bortle class 6: "only visible near the zenith"; below that, invisible

COMPASS = [
    "N",
    "NNE",
    "NE",
    "ENE",
    "E",
    "ESE",
    "SE",
    "SSE",
    "S",
    "SSW",
    "SW",
    "WSW",
    "W",
    "WNW",
    "NW",
    "NNW",
]
COMPASS_WORDS = {"N": "north", "E": "east", "S": "south", "W": "west"}

PLANETS = {
    "Mercury": "mercury",
    "Venus": "venus",
    "Mars": "mars",
    "Jupiter": "jupiter barycenter",
    "Saturn": "saturn barycenter",
    "Uranus": "uranus barycenter",
}

# Patterns a beginner can learn first: (name, constellation, how to recognise it). Editorial
# choice; the positions come from the catalogue.
PATTERNS = [
    ("Big Dipper", "UMa", "seven stars in a ladle shape; its end points to Polaris"),
    ("Cassiopeia", "Cas", "a W (or M) of five stars"),
    ("Orion", "Ori", "three stars in a row for the belt, Betelgeuse and Rigel at the corners"),
    ("Scorpius", "Sco", "a curving tail and the red star Antares"),
    ("Sagittarius (the Teapot)", "Sgr", "a teapot shape, with the Milky Way rising like steam"),
    ("Cygnus (the Northern Cross)", "Cyg", "a long cross lying along the Milky Way"),
    ("Pegasus (the Great Square)", "Peg", "four stars in a big square"),
    ("Leo", "Leo", "a backwards question mark with Regulus at the bottom"),
    ("Taurus", "Tau", "a V of stars with orange Aldebaran, and the Pleiades nearby"),
    ("Gemini", "Gem", "two parallel lines of stars ending in Castor and Pollux"),
    ("Lyra", "Lyr", "a small parallelogram hanging from brilliant Vega"),
    ("Perseus", "Per", "an arc of stars between Cassiopeia and the Pleiades"),
    ("Auriga", "Aur", "a pentagon of stars led by yellow Capella"),
    ("Boötes", "Boo", "a kite shape with orange Arcturus at its tail"),
    ("Andromeda", "And", "two lines of stars leading from the Great Square to the galaxy"),
    ("Canis Major", "CMa", "the big dog, led by Sirius, the brightest star"),
]

# Star groupings that span constellations (asterisms), by their member stars' proper names.
ASTERISMS = {
    "Summer Triangle": ("Vega", "Deneb", "Altair"),
    "Winter Triangle": ("Betelgeuse", "Sirius", "Procyon"),
}


# ---------- darkness in plain words ----------


@dataclass(frozen=True)
class Darkness:
    key: str  # very-dark | dark | suburban | bright | city
    title: str  # "Very dark"
    verdict: str  # one line a beginner can act on
    milky_way: str  # what the Milky Way looks like
    times_natural: float  # total sky brightness / natural sky
    nelm: float  # faintest star a typical observer sees overhead
    sqm: float


# Lower SQM bound of each tier = Bortle class boundaries (classes 1-3, 4-4.5, 5, 6-7, 8-9).
TIERS = [
    (
        21.30,
        "very-dark",
        "Very dark",
        "Excellent for stargazing",
        "The Milky Way is bright and full of detail",
    ),
    (
        20.30,
        "dark",
        "Dark",
        "Good for stargazing",
        "The Milky Way is visible but faint and without detail",
    ),
    (
        19.25,
        "suburban",
        "Suburban",
        "Fine for constellations, planets and bright clusters",
        "The Milky Way is washed out",
    ),
    (
        18.00,
        "bright",
        "Bright",
        "Planets, the Moon and bright stars",
        "The Milky Way is hard to see, at most a faint band overhead",
    ),
    (
        -math.inf,
        "city",
        "City",
        "The Moon, planets and only the brightest stars",
        "The Milky Way is hidden by the city glow",
    ),
]


def times_natural(sqm) -> np.ndarray | float:
    """How many times brighter than a natural sky (1.0 = natural)."""
    return 10 ** (0.4 * (NATURAL_SQM - np.asarray(sqm, dtype=float)))


def limiting_magnitude(sqm):
    """Faintest star a typical observer can see at the zenith under sky brightness `sqm`."""
    b = np.asarray(sqm, dtype=float)
    return 7.93 - 5 * np.log10(10 ** (4.316 - b / 5) + 1)


def darkness(sqm: float) -> Darkness:
    for lower, key, title, verdict, mw in TIERS:
        if sqm >= lower:
            return Darkness(
                key,
                title,
                verdict,
                mw,
                float(times_natural(sqm)),
                float(limiting_magnitude(sqm)),
                float(sqm),
            )
    raise AssertionError("unreachable")


def brightness_words(times: float) -> str:
    """'About as dark as nature' / '1.6× brighter than a natural sky' / '76× brighter ...'."""
    if times < 1.05:
        return "About as dark as a natural sky"
    if times < 10:
        return f"{times:.1f}× brighter than a natural sky"
    return f"{times:.0f}× brighter than a natural sky"


# ---------- sky brightness with the Moon (Krisciunas & Schaefer 1991) ----------


def mag_to_nl(v):
    return 34.08 * np.exp(20.7233 - 0.92104 * np.asarray(v, dtype=float))


def nl_to_mag(b):
    return (20.7233 - np.log(np.asarray(b, dtype=float) / 34.08)) / 0.92104


def airmass(zenith_deg):
    """Airmass in K&S's formula; finite down to the horizon. Used only inside the K&S sky-
    brightness model, which was calibrated with it (it reaches 5 at the horizon, where the real
    air path is ~38 times the zenith one, so it must not be used for a star's own extinction)."""
    z = np.radians(np.clip(np.asarray(zenith_deg, dtype=float), 0, 90))
    return (1 - 0.96 * np.sin(z) ** 2) ** -0.5


def extinction_airmass(zenith_deg):
    """Relative air path towards a star (Kasten & Young 1989, Applied Optics 28:4735):
    X = 1 / (cos z + 0.50572 (96.07995 - z)^-1.6364), z in degrees. 1 at the zenith, ~2 at 60°,
    ~10.3 at 5° altitude and ~38 at the horizon (plane-parallel sec z diverges there)."""
    z = np.clip(np.asarray(zenith_deg, dtype=float), 0, 90)
    return 1.0 / (np.cos(np.radians(z)) + 0.50572 * (96.07995 - z) ** -1.6364)


def moon_sky_nl(sep_deg, zenith_deg, moon_zenith_deg, phase_angle_deg, k=EXTINCTION_K):
    """Scattered moonlight (nanolamberts) at a sky position `sep_deg` from the Moon."""
    rho = np.clip(np.asarray(sep_deg, dtype=float), 0.5, 180)
    alpha = np.abs(np.asarray(phase_angle_deg, dtype=float))
    i_star = 10 ** (-0.4 * (3.84 + 0.026 * alpha + 4e-9 * alpha**4))
    rayleigh = 10**5.36 * (1.06 + np.cos(np.radians(rho)) ** 2)
    mie = np.where(rho >= 10, 10 ** (6.15 - rho / 40), 6.2e7 * rho**-2.0)
    x, xm = airmass(zenith_deg), airmass(moon_zenith_deg)
    return (rayleigh + mie) * i_star * 10 ** (-0.4 * k * xm) * (1 - 10 ** (-0.4 * k * x))


def sky_brightness(alt_deg, zenith_sqm, moon=None, k=EXTINCTION_K):
    """Sky brightness (mag/arcsec²) at altitude(s) `alt_deg`. `zenith_sqm` is the moonless
    zenith brightness here (natural + light pollution). `moon` = dict(alt, sep, phase_angle)
    with `sep` the angular distance of each sky position from the Moon (same shape as alt)."""
    z = 90 - np.asarray(alt_deg, dtype=float)
    x = airmass(z)
    b = mag_to_nl(zenith_sqm) * 10 ** (-0.4 * k * (x - 1)) * x
    if moon is not None and moon["alt"] > 0:
        b = b + moon_sky_nl(moon["sep"], z, 90 - moon["alt"], moon["phase_angle"], k)
    return nl_to_mag(b)


def faintest_visible(alt_deg, zenith_sqm, moon=None, k=EXTINCTION_K):
    """Catalogue magnitude of the faintest star visible at each altitude: the limiting magnitude
    for the local sky brightness, minus the extra extinction an object there suffers compared
    with the zenith (NELM is quoted for the zenith, so the zenith's own extinction is already in
    it)."""
    alt = np.asarray(alt_deg, dtype=float)
    nelm = limiting_magnitude(sky_brightness(alt, zenith_sqm, moon, k))
    return nelm - k * (extinction_airmass(90 - alt) - 1)


# ---------- visibility tables for the browser's live chart ----------
# The Sky Guide's chart is redrawn in the browser while the time slider moves. The visibility
# physics stays here, in one place: for each Moon state the faintest visible magnitude is
# tabulated on an (altitude x distance-from-the-Moon) grid and the browser interpolates
# bilinearly. Distance from the Moon is spaced tightly near the Moon, where the glare changes
# fastest. `interpolate_limit` is the same lookup in Python, used to bound the error in tests.
# Altitude steps are fine near the horizon, where the air path (and so the extinction) changes
# fastest: from 38 airmasses at 0° to 10 at 5°.
LIMIT_ALTS = np.array(
    [0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 10, 12, 14, 17, 20, 24, 28, 33, 40, 48, 56, 65,
     75, 90.0]
)  # fmt: skip
# K&S's glare term changes formula at 10° from the Moon (a 28% step in the published equations),
# so the grid has a node on each side of it rather than interpolating across the jump.
LIMIT_SEPS = np.array(
    [0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8.5, 9.999, 10, 11.5, 13, 15, 17, 20, 24, 28, 35, 45, 55, 70,
     85, 100, 120, 140, 160, 180.0]
)  # fmt: skip


def limit_table(zenith_sqm: float, moon_alt: float, phase_angle: float) -> np.ndarray:
    """Faintest visible magnitude on the LIMIT_ALTS x LIMIT_SEPS grid with the Moon at
    `moon_alt`; without the Moon (moon_alt <= 0) every column is the same."""
    aa, ss = np.meshgrid(LIMIT_ALTS, LIMIT_SEPS, indexing="ij")
    moon = {"alt": float(moon_alt), "sep": ss, "phase_angle": float(phase_angle)}
    return np.asarray(faintest_visible(aa, zenith_sqm, moon), dtype=float)


def interpolate_limit(table: np.ndarray, alt, sep) -> np.ndarray:
    """Bilinear lookup in a limit_table (what the browser does)."""
    alt = np.clip(np.asarray(alt, dtype=float), LIMIT_ALTS[0], LIMIT_ALTS[-1])
    sep = np.clip(np.asarray(sep, dtype=float), LIMIT_SEPS[0], LIMIT_SEPS[-1])
    i = np.clip(np.searchsorted(LIMIT_ALTS, alt, side="right") - 1, 0, len(LIMIT_ALTS) - 2)
    j = np.clip(np.searchsorted(LIMIT_SEPS, sep, side="right") - 1, 0, len(LIMIT_SEPS) - 2)
    fa = (alt - LIMIT_ALTS[i]) / (LIMIT_ALTS[i + 1] - LIMIT_ALTS[i])
    fs = (sep - LIMIT_SEPS[j]) / (LIMIT_SEPS[j + 1] - LIMIT_SEPS[j])
    top = table[i, j] * (1 - fs) + table[i, j + 1] * fs
    bottom = table[i + 1, j] * (1 - fs) + table[i + 1, j + 1] * fs
    return top * (1 - fa) + bottom * fa


# ---------- catalogue ----------


@dataclass(frozen=True)
class Catalog:
    stars: pd.DataFrame  # hip, ra, dec, mag, bv, con, name
    constellations: dict  # abbr -> {name, label, lines}
    milky_way: list  # levels -> rings -> [[ra, dec], ...]
    dsos: pd.DataFrame  # id, name, kind, con, ra, dec, mag, note


@lru_cache(maxsize=1)
def load_catalog(stars_path: Path = STARS_PATH) -> Catalog:
    s = json.loads(stars_path.read_text())
    names = s.pop("names")
    stars = pd.DataFrame(s)
    stars["name"] = stars["hip"].astype(str).map(names)
    return Catalog(
        stars,
        json.loads(CONSTELLATIONS_PATH.read_text()),
        json.loads(MILKY_WAY_PATH.read_text())["levels"],
        pd.DataFrame(json.loads(DSO_PATH.read_text())),
    )


# ---------- coordinates ----------


def _unit(ra_deg, dec_deg) -> np.ndarray:
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], axis=-1)


def separation_deg(ra1, dec1, ra2, dec2):
    """Great-circle angle between two directions (degrees); works for alt/az pairs too."""
    a, b = _unit(ra1, dec1), _unit(ra2, dec2)
    return np.degrees(np.arccos(np.clip((a * b).sum(axis=-1), -1, 1)))


def compass(az_deg) -> str:
    return COMPASS[int(((float(az_deg) % 360) + 11.25) // 22.5) % 16]


def compass_words(point: str) -> str:
    """'SSW' -> 'south-southwest'."""
    if len(point) == 1:
        return COMPASS_WORDS[point]
    if len(point) == 2:
        return COMPASS_WORDS[point[0]] + COMPASS_WORDS[point[1]]
    return COMPASS_WORDS[point[0]] + "-" + compass_words(point[1:])


def height_words(alt_deg: float) -> str:
    """Plain words for altitude. A fist at arm's length is about 10°."""
    a = float(alt_deg)
    if a >= 70:
        return "nearly overhead"
    if a >= 45:
        return "high"
    if a >= 20:
        return "halfway up"
    if a >= 5:
        return "low"
    return "on the horizon"


def where_words(alt: float, az: float) -> str:
    """'low in the SE' / 'nearly overhead'."""
    h = height_words(alt)
    return h if h == "nearly overhead" else f"{h} in the {compass(az)}"


_APPARENT: dict[tuple[str, pd.Timestamp], tuple[np.ndarray, np.ndarray]] = {}


class Observer:
    """Altitude/azimuth of catalogue objects and solar-system bodies for a site."""

    def __init__(self, site: Site):

        self.site = site
        self.eph = astro._ephemeris()
        self.ts = astro._timescale()
        self.topos = astro._topos(site)
        self.observer = self.eph["earth"] + self.topos
        self.lat = math.radians(site.lat)

    def times(self, utc: pd.DatetimeIndex):
        return astro._to_skyfield(pd.DatetimeIndex(utc))

    def apparent_radec(self, ra_deg, dec_deg, when: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
        """Apparent RA/Dec of date (degrees) at `when`, for J2000 catalogue positions."""
        from skyfield.api import Star

        t = astro._to_skyfield(pd.DatetimeIndex([pd.Timestamp(when).floor("s")]))[0]
        star = Star(
            ra_hours=np.asarray(ra_deg, float) / 15.0, dec_degrees=np.asarray(dec_deg, float)
        )
        ra, dec, _ = self.eph["earth"].at(t).observe(star).apparent().radec(epoch="date")
        return np.atleast_1d(ra._degrees), np.atleast_1d(dec.degrees)

    def lst_deg(self, utc: pd.DatetimeIndex) -> np.ndarray:
        t = self.times(utc)
        return (np.atleast_1d(t.gast) * 15.0 + self.site.lon) % 360

    def altaz_from_radec(self, ra_date, dec_date, utc: pd.DatetimeIndex):
        """(alt, az) in degrees, shape (times, objects), from apparent RA/Dec of date."""
        ha = np.radians(self.lst_deg(utc)[:, None] - np.asarray(ra_date)[None, :])
        dec = np.radians(np.asarray(dec_date))[None, :]
        sin_alt = np.sin(self.lat) * np.sin(dec) + np.cos(self.lat) * np.cos(dec) * np.cos(ha)
        alt = np.arcsin(np.clip(sin_alt, -1, 1))
        y = -np.cos(dec) * np.sin(ha)
        x = np.sin(dec) * np.cos(self.lat) - np.cos(dec) * np.sin(self.lat) * np.cos(ha)
        return np.degrees(alt), np.degrees(np.arctan2(y, x)) % 360

    def altaz(self, ra_deg, dec_deg, utc: pd.DatetimeIndex, key: str | None = None):
        """(alt, az) for J2000 catalogue positions at each time (shape times × objects).

        `key` names a fixed set of positions (e.g. "stars") so their apparent coordinates can be
        cached: precession, nutation and aberration don't depend on where the observer stands
        and change by < 0.001° in two hours, so one computation serves every place."""
        utc = pd.DatetimeIndex(utc)
        mid = utc[0] + (utc[-1] - utc[0]) / 2
        if key is None:
            ra, dec = self.apparent_radec(ra_deg, dec_deg, mid)
        else:
            slot = mid.floor("2h")
            cached = _APPARENT.get((key, slot))
            if cached is None or len(cached[0]) != len(np.atleast_1d(ra_deg)):
                cached = self.apparent_radec(ra_deg, dec_deg, slot)
                if len(_APPARENT) > 64:
                    _APPARENT.clear()
                _APPARENT[(key, slot)] = cached
            ra, dec = cached
        return self.altaz_from_radec(ra, dec, utc)

    def body(self, name: str, utc: pd.DatetimeIndex) -> dict:
        """Planet / Moon / Sun at each time: alt, az, RA/Dec (J2000) and, for planets, magnitude."""
        from skyfield.magnitudelib import planetary_magnitude

        t = self.times(utc)
        target = self.eph[name]
        astrometric = self.observer.at(t).observe(target)
        app = astrometric.apparent()
        alt, az, _ = app.altaz()
        ra, dec, _ = astrometric.radec()  # J2000, what constellation boundaries are drawn in
        out = {
            "alt": np.atleast_1d(alt.degrees),
            "az": np.atleast_1d(az.degrees),
            "ra": np.atleast_1d(ra._degrees),
            "dec": np.atleast_1d(dec.degrees),
        }
        if name not in ("moon", "sun"):
            out["mag"] = np.atleast_1d(planetary_magnitude(self.eph["earth"].at(t).observe(target)))
        return out

    def radec_of_date(self, name: str, utc: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
        """Topocentric apparent RA/Dec of date (degrees) of a solar-system body: with the local
        sidereal time these give exactly Skyfield's altitude/azimuth (no refraction), so the
        browser can place the Moon and planets at any moment between the computed ones."""
        t = self.times(utc)
        app = self.observer.at(t).observe(self.eph[name]).apparent()
        ra, dec, _ = app.radec(epoch="date")
        return np.atleast_1d(ra._degrees), np.atleast_1d(dec.degrees)

    def moon(self, utc: pd.DatetimeIndex) -> dict:
        """The Moon plus its illuminated fraction and phase angle (Sun-Moon-Earth angle)."""
        from skyfield import almanac

        out = self.body("moon", utc)
        t = self.times(utc)
        e = self.eph["earth"].at(t)
        sun, moon = e.observe(self.eph["sun"]), e.observe(self.eph["moon"])
        elong = np.atleast_1d(sun.separation_from(moon).degrees)
        out["phase_angle"] = 180.0 - elong  # good to ~0.1° for the Moon
        out["illum"] = np.atleast_1d(almanac.fraction_illuminated(self.eph, "moon", t))
        out["phase_deg"] = np.atleast_1d(
            almanac.moon_phase(self.eph, t).degrees
        )  # 0 new, 90 first quarter, 180 full
        return out


def constellation_at(ra_deg, dec_deg) -> list[str]:
    """IAU constellation abbreviation for J2000 positions."""
    from skytrust.skycatalog import constellation_of

    return constellation_of(ra_deg, dec_deg)


def constellation_name(abbr: str, catalog: Catalog | None = None) -> str:
    cat = catalog or load_catalog()
    return cat.constellations.get(abbr, {}).get("name", abbr)


# ---------- the Milky Way ----------


def galactic_to_equatorial(l_deg, b_deg) -> tuple[np.ndarray, np.ndarray]:
    """Galactic (l, b) to J2000 RA/Dec (degrees), using Skyfield's IAU galactic frame."""
    from skyfield.framelib import galactic_frame

    ts = astro._timescale()
    r = galactic_frame.rotation_at(ts.J2000)  # ICRS -> galactic
    g = _unit(np.asarray(l_deg, float), np.asarray(b_deg, float))
    eq = g @ r  # galactic -> ICRS (r is orthogonal)
    ra = np.degrees(np.arctan2(eq[..., 1], eq[..., 0])) % 360
    dec = np.degrees(np.arcsin(np.clip(eq[..., 2], -1, 1)))
    return ra, dec


GALACTIC_L = np.arange(0, 360, 5.0)
CORE_HALF_WIDTH = 10  # galactic longitudes within ±10° of the centre: the Sagittarius star clouds


def milky_way_band(obs: Observer, utc: pd.DatetimeIndex) -> dict:
    """Altitude/azimuth of the galactic equator (every 5° of longitude) at each time."""
    ra, dec = galactic_to_equatorial(GALACTIC_L, np.zeros_like(GALACTIC_L))
    alt, az = obs.altaz(ra, dec, utc, key="galactic-equator")
    return {"l": GALACTIC_L, "alt": alt, "az": az, "ra": ra, "dec": dec}


def describe_band(
    alt: np.ndarray, az: np.ndarray, l_deg: np.ndarray, cons: list[str], catalog: Catalog
) -> dict | None:
    """Plain-language description of the Milky Way's arc at one moment."""
    up = alt > 0
    if not up.any():
        return None
    top = int(np.argmax(alt))
    # horizon crossings, ordered from the centre of the galaxy outward along the band
    ends = []
    n = len(alt)
    for i in range(n):
        j = (i + 1) % n
        if up[i] != up[j]:
            f = alt[i] / (alt[i] - alt[j])
            a0, a1 = az[i], az[j]
            d = ((a1 - a0 + 540) % 360) - 180
            ends.append(compass((a0 + f * d) % 360))
    core = np.abs(((l_deg + 180) % 360) - 180) <= CORE_HALF_WIDTH
    core_alt = float(alt[core].max())
    core_az = float(az[core][int(np.argmax(alt[core]))])
    along = []
    for i in np.argsort(-alt):
        c = cons[i]
        if alt[i] > 15 and c not in along:
            along.append(c)
    return {
        "top_alt": float(alt[top]),
        "top_az": float(az[top]),
        "ends": ends[:2],
        "core_alt": core_alt,
        "core_az": core_az,
        "through": [constellation_name(c, catalog) for c in along[:4]],
    }


# ---------- tonight, summarised ----------


@dataclass
class SkyItem:
    """Something to look at tonight: where and when it is best, and whether it shows from here.
    `track_alt` / `track_az` hold its position at every time of the night's grid."""

    name: str
    kind: str  # planet | moon | star | pattern | cluster | galaxy | nebula
    mag: float | None  # brightness shown to people (combined magnitude)
    vis_mag: float  # magnitude compared with the limiting magnitude
    extended: bool  # a diffuse glow (needs EXTENDED_MARGIN more darkness)
    track_alt: np.ndarray = field(repr=False)
    track_az: np.ndarray = field(repr=False)
    best: int = 0  # index of the best (highest) moment
    visible: bool = False  # visible to the eye from here at its best moment
    note: str = ""
    con: str = ""
    up_from: pd.Timestamp | None = None
    up_until: pd.Timestamp | None = None
    best_utc: pd.Timestamp | None = None

    @property
    def alt(self) -> float:
        return float(self.track_alt[self.best])

    @property
    def az(self) -> float:
        return float(self.track_az[self.best])

    @property
    def where(self) -> str:
        return where_words(self.alt, self.az)


def _times(dusk: pd.Timestamp, dawn: pd.Timestamp, step_min: int = 20) -> pd.DatetimeIndex:
    n = max(2, int((dawn - dusk) / pd.Timedelta(minutes=step_min)) + 1)
    return pd.date_range(dusk, dawn, periods=n)


def moon_at(moon: dict, i: int, alt, az) -> dict:
    """The Moon's state at time index i, with the separation of each (alt, az) from it."""
    return {
        "alt": float(moon["alt"][i]),
        "phase_angle": float(moon["phase_angle"][i]),
        "sep": separation_deg(az, alt, moon["az"][i], moon["alt"][i]),
    }


def is_visible(
    alt: float, az: float, vis_mag: float, extended: bool, zenith_sqm: float, moon: dict, i: int
) -> bool:
    """Can a typical observer see it with the naked eye here, at time index i?"""
    if alt <= 0:
        return False
    limit = float(faintest_visible(alt, zenith_sqm, moon_at(moon, i, alt, az)))
    return vis_mag <= limit - (EXTENDED_MARGIN if extended else 0.0)


def _finish(
    item: SkyItem, times: pd.DatetimeIndex, min_alt: float, zenith_sqm: float, moon: dict
) -> SkyItem | None:
    """Best moment = highest point; drop it if it never gets `min_alt` high."""
    item.best = int(np.argmax(item.track_alt))
    if item.alt < min_alt:
        return None
    above = np.flatnonzero(item.track_alt >= min_alt)
    item.up_from, item.up_until = times[above[0]], times[above[-1]]
    item.best_utc = times[item.best]
    item.visible = is_visible(
        item.alt, item.az, item.vis_mag, item.extended, zenith_sqm, moon, item.best
    )
    return item


def tonight(
    site: Site,
    dusk: pd.Timestamp,
    dawn: pd.Timestamp,
    zenith_sqm: float,
    catalog: Catalog | None = None,
) -> dict:
    """Everything the sky guide needs for one night of darkness (dusk -> dawn, UTC)."""
    cat = catalog or load_catalog()
    obs = Observer(site)
    times = _times(dusk, dawn)
    moon = obs.moon(times)
    found: list[SkyItem | None] = []

    for label, key in PLANETS.items():
        b = obs.body(key, times)
        i = int(np.argmax(b["alt"]))
        mag = float(b["mag"][i])
        con = constellation_at([b["ra"][i]], [b["dec"][i]])[0]
        found.append(
            _finish(
                SkyItem(label, "planet", mag, mag, False, b["alt"], b["az"], con=con),
                times,
                5.0,
                zenith_sqm,
                moon,
            )
        )

    moon_item = _finish(
        SkyItem("Moon", "moon", -12.0, -12.0, False, moon["alt"], moon["az"]),
        times,
        0.0,
        zenith_sqm,
        moon,
    )
    if moon_item:
        moon_item.note = f"{moon['illum'][moon_item.best]:.0%} lit"
        found.append(moon_item)

    stars = cat.stars[cat.stars["name"].notna() & (cat.stars["mag"] <= 1.6)]
    alt, az = obs.altaz(stars["ra"].to_numpy(), stars["dec"].to_numpy(), times, key="bright")
    for k, (_, s) in enumerate(stars.iterrows()):
        m = float(s["mag"])
        found.append(
            _finish(
                SkyItem(s["name"], "star", m, m, False, alt[:, k], az[:, k], con=s["con"]),
                times,
                10.0,
                zenith_sqm,
                moon,
            )
        )

    d = cat.dsos
    alt, az = obs.altaz(d["ra"].to_numpy(), d["dec"].to_numpy(), times, key="dsos")
    for k, (_, o) in enumerate(d.iterrows()):
        item = SkyItem(
            o["name"],
            o["kind"],
            float(o["mag"]),
            float(o["vis_mag"]),
            bool(o["extended"]),
            alt[:, k],
            az[:, k],
            note=o["note"],
            con=o["con"],
        )
        found.append(_finish(item, times, 15.0 if o["dec"] > -40 else 3.0, zenith_sqm, moon))

    patterns = pattern_positions(obs, cat, times)
    for name, con, how in PATTERNS:
        p = patterns[con]
        item = SkyItem(
            name, "pattern", p["mag"], p["faint_mag"], False, p["alt"], p["az"], note=how, con=con
        )
        found.append(_finish(item, times, 20.0, zenith_sqm, moon))
    for name, members in ASTERISMS.items():
        s = cat.stars[cat.stars["name"].isin(members)]
        a, z = obs.altaz(s["ra"].to_numpy(), s["dec"].to_numpy(), times)
        # the pattern's height = its lowest star (all three must be up); azimuth = middle star
        mid = np.argsort(a, axis=1)[:, 1]
        item = SkyItem(
            name,
            "pattern",
            float(s["mag"].min()),
            float(s["mag"].max()),
            False,
            a.min(axis=1),
            z[np.arange(len(times)), mid],
            note=" + ".join(members),
        )
        found.append(_finish(item, times, 15.0, zenith_sqm, moon))

    items = [f for f in found if f is not None]
    band = milky_way_band(obs, times)
    band_cons = constellation_at(band["ra"], band["dec"])
    return {
        "times": times,
        "items": items,
        "moon": moon,
        "milky_way": milky_way_tonight(band, band_cons, moon, times, zenith_sqm, cat),
        "opposite_moon": opposite_moon(items, moon, times, zenith_sqm),
        "darkness": darkness(zenith_sqm),
        "stars_visible": stars_visible_count(obs, cat, times, moon, zenith_sqm),
    }


def pattern_positions(obs: Observer, cat: Catalog, times: pd.DatetimeIndex) -> dict:
    """For each pattern's constellation: the alt/az of the centre of its stick figure, the
    magnitude of its brightest star, and of its fifth-brightest stick-figure star (the shape
    is recognisable once about five of its stars show)."""
    out = {}
    stars = cat.stars
    sra, sdec = stars["ra"].to_numpy(), stars["dec"].to_numpy()
    for _, con, _ in PATTERNS:
        pts = np.array([p for seg in cat.constellations[con]["lines"] for p in seg])
        centre = _unit(pts[:, 0], pts[:, 1]).mean(axis=0)
        ra = math.degrees(math.atan2(centre[1], centre[0])) % 360
        dec = math.degrees(math.asin(centre[2] / np.linalg.norm(centre)))
        alt, az = obs.altaz([ra], [dec], times)
        sep = separation_deg(pts[:, None, 0], pts[:, None, 1], sra[None, :], sdec[None, :])
        idx = np.unique(np.argmin(sep, axis=1)[sep.min(axis=1) < 0.15])
        mags = np.sort(stars["mag"].to_numpy()[idx]) if len(idx) else np.array([3.0])
        out[con] = {
            "alt": alt[:, 0],
            "az": az[:, 0],
            "mag": float(mags[0]),
            "faint_mag": float(mags[min(len(mags), 5) - 1]),
        }
    return out


def milky_way_tonight(
    band: dict,
    band_cons: list[str],
    moon: dict,
    times: pd.DatetimeIndex,
    zenith_sqm: float,
    cat: Catalog,
) -> dict:
    """When and where the Milky Way is best tonight, and whether it shows from here.

    Best time: the moment the inner half of the band (the bright side, within 90° of the
    galactic centre) stands highest, preferring moments with the Moon down."""
    inner = np.abs(((band["l"] + 180) % 360) - 180) <= 90
    score = np.clip(band["alt"][:, inner], 0, None).sum(axis=1)
    moon_down = moon["alt"] <= 0
    i = int(np.argmax(np.where(moon_down, score, score * 0.25)))
    zen = float(sky_brightness(90.0, zenith_sqm, moon_at(moon, i, 90.0, 0.0)))
    desc = describe_band(band["alt"][i], band["az"][i], band["l"], band_cons, cat)
    centre = int(np.flatnonzero(band["l"] == 0)[0])
    centre_alt = band["alt"][:, centre]
    centre_up = np.flatnonzero(centre_alt > 10)
    return {
        "best_utc": times[i],
        "moon_down": bool(moon_down[i]),
        "zenith_sqm": zen,
        "visible": zen >= MILKY_WAY_MIN_SQM and desc is not None and desc["top_alt"] > 20,
        "looks": darkness(zen).milky_way,
        "band": desc,
        # the galactic centre (in Sagittarius, the brightest part) above 10° during darkness
        "centre_until": times[centre_up[-1]] if len(centre_up) else None,
        "centre_from": times[centre_up[0]] if len(centre_up) else None,
        "centre_max_alt": float(centre_alt.max()),
        "centre_az": float(band["az"][int(np.argmax(centre_alt)), centre]),
    }


def opposite_moon(
    items: list[SkyItem], moon: dict, times: pd.DatetimeIndex, zenith_sqm: float
) -> dict | None:
    """Where the sky is darkest while the Moon is up, and what to look at there.

    Evaluated in the middle of the Moon's hours above the horizon during darkness. Scattered
    moonlight is weakest about 90° from the Moon (where Rayleigh scattering is least, K&S 1991)
    and the sky brightens towards the horizon, so the darkest patch is usually well up on the
    side facing away from the Moon."""
    up = np.flatnonzero(moon["alt"] > 5)
    if not len(up):
        return None
    i = int(up[len(up) // 2])
    alts, azs = np.arange(20, 90, 5.0), np.arange(0, 360, 10.0)
    aa, zz = np.meshgrid(alts, azs, indexing="ij")
    b = sky_brightness(aa, zenith_sqm, moon_at(moon, i, aa, zz))
    r, c = np.unravel_index(int(np.argmax(b)), b.shape)
    picks = []
    for it in items:
        if it.kind == "moon":
            continue
        a, z = float(it.track_alt[i]), float(it.track_az[i])
        sep = float(separation_deg(z, a, moon["az"][i], moon["alt"][i]))
        if a < 20 or sep < 60:
            continue
        if not is_visible(a, z, it.vis_mag, it.extended, zenith_sqm, moon, i):
            continue
        picks.append((float(sky_brightness(a, zenith_sqm, moon_at(moon, i, a, z))), it, a, z))
    order = {"planet": 0, "pattern": 1, "star": 2}
    picks.sort(key=lambda p: (order.get(p[1].kind, 3), -p[0]))
    return {
        "utc": times[i],
        "moon_alt": float(moon["alt"][i]),
        "moon_az": float(moon["az"][i]),
        "illum": float(moon["illum"][i]),
        "face": compass((float(moon["az"][i]) + 180) % 360),
        "darkest_alt": float(aa[r, c]),
        "darkest_az": float(zz[r, c]),
        "darkest_sqm": float(b[r, c]),
        "near_moon_sqm": float(b.min()),
        "objects": [(it, a, z) for _, it, a, z in picks[:6]],
    }


def stars_visible_count(
    obs: Observer, cat: Catalog, times: pd.DatetimeIndex, moon: dict, zenith_sqm: float
) -> dict:
    """How many stars a typical observer can see at the darkest moment of the night (the Moon
    lowest), from here and from a natural sky at the same moment."""
    i = int(np.argmin(moon["alt"]))
    stars = cat.stars
    alt, az = obs.altaz(
        stars["ra"].to_numpy(), stars["dec"].to_numpy(), times[i : i + 1], key="stars"
    )
    alt, az = alt[0], az[0]
    up = alt > 0
    mags = cat.stars["mag"].to_numpy()[up]
    m = moon_at(moon, i, alt[up], az[up])
    return {
        "utc": times[i],
        "here": int((mags <= faintest_visible(alt[up], zenith_sqm, m)).sum()),
        "natural": int((mags <= faintest_visible(alt[up], NATURAL_SQM, m)).sum()),
        "moon_up": bool(moon["alt"][i] > 0),
    }
