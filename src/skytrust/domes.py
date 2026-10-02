"""City glow across the whole sky: how bright the light pollution is in every direction, not
just overhead.

The light-pollution atlas gives the artificial glow at the zenith only. Towards a city the sky
is far brighter, most of all near the horizon, which is where a Santa Barbara or Pasadena
observer loses stars in the dome of Los Angeles. This module spreads the zenith value over the
sky in three steps:

1. **Shape of one light dome.** For a source of light on the ground at distance D, the
   brightness at each altitude and azimuth relative to the zenith, S(D, alt, daz), from a
   single-scattering calculation with the classic Garstang (1986, PASP 98:364) atmosphere as
   restated by Cinzano, Falchi, Elvidge & Baugh (2000, MNRAS 318:641, sec. 4):
   - molecules: density ~ exp(-c h), c = 0.104 /km; sea-level scattering coefficient
     N_m0 sigma_m = 2.55e19 cm^-3 x 4.6e-27 cm^2 (V band; the paper's text lists the V and B
     cross-sections the other way round, but only 4.6e-27 reproduces its own stated V-band
     extinction for K = 1: 0.33 mag, tau = 0.3);
   - aerosols: K = 1 (the clean atmosphere of the atlas maps), sea-level coefficient
     11.11 K N_m0 sigma_m, scale a = 0.657 + 0.059 K /km;
   - phase functions: Rayleigh 3(1 + cos^2 w)/16 pi; aerosols McClatchey et al. as fitted by
     Garstang (1991), eq. 14 (both integrate to 1 over the sphere; checked in the tests);
   - emission: Garstang's city function I(psi) = [2 a1 cos psi + 0.554 a2 psi^4] / 2 pi with
     a1 = 0.46, a2 = 0.54 (G = F = 0.15), zero below the horizontal;
   - a spherical Earth (R = 6371 km); extinction on both legs of the light path.
   Double scattering, terrain and the observer's height are left out (Garstang's double
   scattering is a ~10% correction at tau = 0.3, and only ratios to the zenith are used).
2. **Which sources light this sky.** The fitted ring kernel of `skyglow` already splits the
   zenith glow among the NASA night-lights sources (weight = emission x K(distance)).
3. **Put together.** artificial(alt, az) = atlas zenith x sum_i w_i S(D_i, alt, az - az_i) /
   sum_i w_i. The atlas keeps the calibration at the zenith; the physics only supplies how
   that glow is spread over the sky.

`build_shape` runs once (`python -m skytrust build-domes`, ~1 min) and writes a small table;
the app only interpolates in it.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

from skytrust.config import REPO_ROOT

log = logging.getLogger(__name__)

SHAPE_PATH = REPO_ROOT / "artifacts" / "sky" / "dome_shape.npz"
EARTH_R_KM = 6371.0
C_MOL = 0.104  # /km, molecular inverse scale height (Garstang 1986)
BETA_MOL = 2.55e19 * 4.6e-27 * 1e5  # /km, sea-level Rayleigh scattering coefficient (V)
K_AEROSOL = 1.0
A_AER = 0.657 + 0.059 * K_AEROSOL  # /km
BETA_AER = 11.11 * K_AEROSOL * BETA_MOL  # /km
EMISSION_A1, EMISSION_A2 = 0.46, 0.54

# Table nodes. Distances follow the kernel's rings out to 300 km; altitudes are dense near the
# horizon, where domes change fastest; azimuth offsets from the source cover 0-180 (symmetric).
SHAPE_D_KM = np.array([0.5, 1, 2, 3, 4.5, 6, 9, 12, 18, 25, 35, 50, 70, 100, 140, 200, 250, 300.0])
SHAPE_ALTS = np.array([0, 1, 2, 3, 5, 7, 10, 14, 20, 28, 40, 55, 70, 90.0])
SHAPE_DAZ = np.arange(0, 181, 5.0)
# The site grid the app uses (the browser interpolates in it): every 10 degrees of azimuth.
SITE_AZ = np.arange(0, 360, 10.0)


def extinction_per_km(h_km):
    """Scattering (= extinction; absorption neglected, as in Garstang) per km at height h."""
    h = np.maximum(h_km, 0.0)
    return BETA_MOL * np.exp(-C_MOL * h) + BETA_AER * np.exp(-A_AER * h)


def aerosol_phase(w_deg):
    """Garstang's (1991) fit to McClatchey's haze phase function; w = scattering angle."""
    w = np.asarray(w_deg, dtype=float)
    return np.where(
        w <= 10,
        7.5 * np.exp(-0.1249 * w**2 / (1 + 0.04996 * w**2)),
        np.where(
            w <= 124,
            1.88 * np.exp(-0.07226 * w + 0.0002406 * w**2),
            0.025 + 0.015 * np.sin(np.radians(2.25 * w - 369.0)),
        ),
    )


def rayleigh_phase(w_deg):
    return 3 * (1 + np.cos(np.radians(w_deg)) ** 2) / (16 * math.pi)


def emission(psi_rad):
    """Garstang's normalised upward emission per unit solid angle (zero below horizontal)."""
    psi = np.asarray(psi_rad, dtype=float)
    up = (2 * EMISSION_A1 * np.cos(psi) + 0.554 * EMISSION_A2 * psi**4) / (2 * math.pi)
    return np.where(psi <= math.pi / 2, up, 0.0)


def _height(p):
    """Height above a spherical Earth of points p (..., 3), observer at the origin."""
    return np.sqrt(p[..., 0] ** 2 + p[..., 1] ** 2 + (EARTH_R_KM + p[..., 2]) ** 2) - EARTH_R_KM


def _los_grid(n: int = 220, near: float = 0.02, far: float = 700.0) -> np.ndarray:
    """Distances along the line of sight: log-spaced, fine near the observer."""
    return np.r_[0.0, np.geomspace(near, far, n)]


def brightness(d_km: float, alt_deg, daz_deg, n_sub: int = 24) -> np.ndarray:
    """Single-scattered brightness (arbitrary units, per unit upward flux) at sky positions
    (alt, daz) for a source on the ground d_km away at azimuth offset 0."""
    alt = np.radians(np.atleast_1d(np.asarray(alt_deg, dtype=float)))
    daz = np.radians(np.atleast_1d(np.asarray(daz_deg, dtype=float)))
    # observer at the origin, z up, x towards the source (azimuth offset 0)
    v = np.stack([np.cos(alt) * np.cos(daz), np.cos(alt) * np.sin(daz), np.sin(alt)], -1)
    g = d_km / EARTH_R_KM
    src = np.array([EARTH_R_KM * math.sin(g), 0.0, EARTH_R_KM * (math.cos(g) - 1)])
    up_src = np.array([math.sin(g), 0.0, math.cos(g)])  # local vertical at the source
    u = _los_grid()
    p = u[None, :, None] * v[:, None, :]  # (dirs, steps, 3)
    h = _height(p)
    beta_p = extinction_per_km(h)
    # transmission observer <- P (cumulative trapezoid along the line of sight)
    seg = 0.5 * (beta_p[:, 1:] + beta_p[:, :-1]) * np.diff(u)[None, :]
    t_po = np.exp(-np.concatenate([np.zeros((len(v), 1)), np.cumsum(seg, axis=1)], axis=1))
    # source -> P: distance, emission angle, scattering angle, transmission
    r = p - src
    s = np.linalg.norm(r, axis=-1)
    s = np.maximum(s, 1e-3)
    cos_psi = np.clip((r @ up_src) / s, -1, 1)
    psi = np.arccos(cos_psi)
    cos_w = np.clip(-(r * v[:, None, :]).sum(-1) / s, -1, 1)  # incoming vs outgoing to observer
    w = np.degrees(np.arccos(cos_w))
    frac = (np.arange(n_sub) + 0.5) / n_sub  # midpoint rule along source -> P
    pts = src + frac[None, None, :, None] * r[:, :, None, :]
    tau_sp = extinction_per_km(_height(pts)).mean(axis=-1) * s
    beta_m = BETA_MOL * np.exp(-C_MOL * np.maximum(h, 0))
    beta_a = BETA_AER * np.exp(-A_AER * np.maximum(h, 0))
    scatter = beta_m * rayleigh_phase(w) + beta_a * aerosol_phase(w)
    f = scatter * emission(psi) / s**2 * np.exp(-tau_sp) * t_po
    f = np.where(h < 40, f, 0.0)
    return np.trapezoid(f, u, axis=1)


def build_shape(path: Path = SHAPE_PATH) -> Path:
    """S(D, alt, daz) = brightness / brightness at the zenith, for every table node."""
    aa, zz = np.meshgrid(SHAPE_ALTS, SHAPE_DAZ, indexing="ij")
    shape = np.empty((len(SHAPE_D_KM), len(SHAPE_ALTS), len(SHAPE_DAZ)))
    for k, d in enumerate(SHAPE_D_KM):
        b = brightness(float(d), aa.ravel(), zz.ravel()).reshape(aa.shape)
        zen = float(brightness(float(d), [90.0], [0.0])[0])
        shape[k] = b / zen
        # a horizontal line of sight skims point sources at ground level (1/s^2 blows up, or
        # the light is blocked entirely); real lights are spread out and hidden by terrain, so
        # the horizon takes the 1-degree values (stars that low are lost to extinction anyway)
        shape[k, 0] = shape[k, 1]
        log.info("dome shape D=%5.1f km: horizon towards it = %.1f x zenith", d, shape[k, 0, 0])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, shape=shape.astype(np.float32), d_km=SHAPE_D_KM,
                        alts=SHAPE_ALTS, daz=SHAPE_DAZ)  # fmt: skip
    return path


_SHAPE: dict[str, np.ndarray] = {}


def load_shape(path: Path = SHAPE_PATH) -> np.ndarray | None:
    if "shape" not in _SHAPE:
        if not path.exists():
            return None
        with np.load(path) as z:
            _SHAPE["shape"] = z["shape"].astype(np.float64)
    return _SHAPE["shape"]


def sky_shape(d_km, az_src, weight, alts=None, az=SITE_AZ, shape=None) -> np.ndarray:
    """How the artificial glow overhead spreads over the sky at one place, from its light
    sources (distance, azimuth, share of the zenith glow): the weighted mean of the sources'
    dome shapes. 1 at the zenith; shape (len(alts), len(az))."""
    alts = SHAPE_ALTS if alts is None else np.asarray(alts, dtype=float)
    shape = load_shape() if shape is None else shape
    w = np.asarray(weight, dtype=float)
    if shape is None or w.sum() <= 0:
        return np.ones((len(alts), len(az)))
    # spread each source's weight over the two nearest distance nodes (in log distance) and
    # 5-degree azimuth bins, then add up the tabulated shapes: the same as interpolating each
    # source's shape, at a tiny fraction of the cost
    logd = np.log(np.clip(d_km, SHAPE_D_KM[0], SHAPE_D_KM[-1]))
    nodes = np.log(SHAPE_D_KM)
    i = np.clip(np.searchsorted(nodes, logd, side="right") - 1, 0, len(nodes) - 2)
    f = (logd - nodes[i]) / (nodes[i + 1] - nodes[i])
    nb = 72
    b = np.rint(np.asarray(az_src, dtype=float) % 360 / 5).astype(int) % nb
    hist = np.zeros((len(nodes), nb))
    np.add.at(hist, (i, b), w * (1 - f))
    np.add.at(hist, (i + 1, b), w * f)
    # shape at the table's altitudes for every (distance node, offset of 0..355 in 5° steps)
    off = np.arange(nb) * 5.0
    daz = np.minimum(off, 360 - off)  # 0..180
    j = np.clip((daz / 5).astype(int), 0, len(SHAPE_DAZ) - 1)
    s_full = shape[:, :, j]  # (D, alts, 72 offsets)
    # sky azimuth a (5° bins) gets source bin b at offset (a - b)
    out = np.zeros((len(SHAPE_ALTS), nb))
    for k in range(len(nodes)):
        if not hist[k].any():
            continue
        for bb in np.flatnonzero(hist[k]):
            out += hist[k, bb] * np.roll(s_full[k], bb, axis=1)
    out /= w.sum()
    # resample to the requested altitudes and azimuths
    a_idx = (np.asarray(az, dtype=float) % 360) / 5
    a0 = np.floor(a_idx).astype(int) % nb
    fa = a_idx - np.floor(a_idx)
    by_az = out[:, a0] * (1 - fa) + out[:, (a0 + 1) % nb] * fa
    res = np.empty((len(alts), len(az)))
    for c in range(len(az)):
        res[:, c] = np.interp(alts, SHAPE_ALTS, by_az[:, c])
    return res
