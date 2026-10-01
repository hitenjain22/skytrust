"""The Sky Guide's live chart: the sky redraws while the time slider is dragged, not only when it's
let go.

Python computes everything that is astronomy or physics, once per place and night:
- apparent RA/Dec of date of every star, Milky Way outline and constellation figure (fixed to
  < 0.001° over a night), and the local sidereal time at dusk;
- RA/Dec of date of the Moon and planets at every step of the night's 20-minute grid (seen from
  the place, so with the local sidereal time they give Skyfield's altitude/azimuth);
- the faintest visible magnitude on an altitude x distance-from-the-Moon grid for every step
  (sky.limit_table: light pollution, moonlight, extinction);
- each object's track for the "Up at ..." list.

The browser (skylive.js) only interpolates in time, turns RA/Dec into altitude/azimuth with the
sidereal time (the same formula as sky.Observer.altaz_from_radec), projects and draws. Numbers
travel as base64 typed arrays to keep the page light (~250 KB for a whole night).
"""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from skytrust import sky
from skytrust.config import Site
from views import lookup
from views.skychart import BV_TINTS

HERE = Path(__file__).resolve().parent
JS_PATH, CSS_PATH = HERE / "skylive.js", HERE / "skylive.css"
SIDEREAL_DEG_PER_MS = 360.98564736629 / 86_400_000  # Earth's rotation relative to the stars
MW_POINTS_PER_RING = 240  # as in skychart._milky_way: plenty at chart size
STEP_MINUTES = 5  # slider resolution


def _b64(values, dtype: str) -> str:
    return base64.b64encode(np.ascontiguousarray(values, dtype=dtype).tobytes()).decode("ascii")


def _ms(ts: pd.Timestamp) -> int:
    return int(pd.Timestamp(ts).value // 1_000_000)


def tint_index(bv) -> np.ndarray:
    """Index into BV_TINTS for each B-V colour (unknown -> the neutral white)."""
    bv = np.asarray(bv, dtype=float)
    limits = np.array([lim for lim, _ in BV_TINTS])
    idx = np.searchsorted(limits, bv, side="right")
    return np.where(np.isnan(bv), 2, np.minimum(idx, len(limits) - 1)).astype(np.uint8)


def star_block(obs: sky.Observer, cat: sky.Catalog, mid: pd.Timestamp) -> dict:
    s = cat.stars
    ra, dec = obs.apparent_radec(s["ra"].to_numpy(), s["dec"].to_numpy(), mid)
    names = {int(i): str(n) for i, n in enumerate(s["name"]) if isinstance(n, str)}
    return {
        "ra": _b64(np.round(ra * 100) % 36000, "<u2"),
        "dec": _b64(np.round(dec * 100), "<i2"),
        "mag": _b64(np.round(s["mag"].to_numpy() * 100), "<i2"),
        "tint": _b64(tint_index(s["bv"].to_numpy()), "u1"),
        "names": names,
    }


def _rings_block(obs: sky.Observer, rings: list[np.ndarray], mid: pd.Timestamp) -> dict:
    """Polylines of J2000 points -> one interleaved array of apparent (RA, Dec) in tenths of a
    degree, plus each polyline's length."""
    flat = np.concatenate(rings)
    ra, dec = obs.apparent_radec(flat[:, 0], flat[:, 1], mid)
    pts = np.empty(2 * len(flat))
    pts[0::2], pts[1::2] = np.round(ra * 10) % 3600, np.round(dec * 10)
    return {"pts": _b64(pts, "<i2"), "len": [int(len(r)) for r in rings]}


def milky_way_block(obs: sky.Observer, cat: sky.Catalog, mid: pd.Timestamp) -> list[dict]:
    levels = []
    for rings in cat.milky_way:
        kept = []
        for r in rings:
            r = np.asarray(r, dtype=float)
            if len(r) > 2:
                kept.append(r[:: max(1, len(r) // MW_POINTS_PER_RING)])
        levels.append(_rings_block(obs, kept, mid) if kept else {"pts": "", "len": []})
    return levels


def figures_block(obs: sky.Observer, cat: sky.Catalog, mid: pd.Timestamp) -> dict:
    segs = [np.asarray(seg, dtype=float) for c in cat.constellations.values() for seg in c["lines"]]
    lines = _rings_block(obs, segs, mid)
    major = [
        (c["name"].upper(), c["label"])
        for c in cat.constellations.values()
        if c.get("label") and int(c.get("rank", 3)) == 1
    ]
    pos = np.array([p for _, p in major], dtype=float)
    ra, dec = obs.apparent_radec(pos[:, 0], pos[:, 1], mid)
    labels = [
        [n, round(float(a), 2), round(float(d), 2)]
        for (n, _), a, d in zip(major, ra, dec, strict=True)
    ]
    return {"lines": lines, "labels": labels}


def _round(values, nd: int = 3) -> list[float]:
    return [round(float(v), nd) for v in np.asarray(values, dtype=float)]


def bodies_block(obs: sky.Observer, times: pd.DatetimeIndex, moon: dict) -> dict:
    mra, mdec = obs.radec_of_date("moon", times)
    planets = []
    for label, key in sky.PLANETS.items():
        ra, dec = obs.radec_of_date(key, times)
        mag = obs.body(key, times)["mag"]
        planets.append({"name": label, "ra": _round(ra, 4), "dec": _round(dec, 4),
                        "mag": _round(mag, 2)})  # fmt: skip
    return {
        "moon": {
            "ra": _round(mra, 4),
            "dec": _round(mdec, 4),
            "phase": _round(moon["phase_deg"], 2),
            "illum": _round(moon["illum"], 3),
        },
        "planets": planets,
    }


def tables_block(times: pd.DatetimeIndex, moon: dict, zenith_sqm: float) -> dict:
    """Visibility tables per step for "your sky" (light pollution + the Moon), and the moonless
    curves for "your sky" and "a perfectly dark sky" (altitude only)."""
    tables = []
    for i in range(len(times)):
        if float(moon["alt"][i]) <= 0:
            tables.append(None)  # no moonlight: the moonless curve applies
            continue
        t = sky.limit_table(zenith_sqm, float(moon["alt"][i]), float(moon["phase_angle"][i]))
        tables.append(_b64(np.round(t * 100), "<i2"))
    here = sky.limit_table(zenith_sqm, -1.0, 0.0)[:, 0]
    dark = sky.limit_table(sky.NATURAL_SQM, -1.0, 0.0)[:, 0]
    zen = [
        float(sky.sky_brightness(90.0, zenith_sqm, sky.moon_at(moon, i, 90.0, 0.0)))
        for i in range(len(times))
    ]
    return {
        "alts": _round(sky.LIMIT_ALTS, 3),
        "seps": _round(sky.LIMIT_SEPS, 3),
        "tables": tables,
        "curveHere": _round(here, 3),
        "curveDark": _round(dark, 3),
        "zen": _round(zen, 3),
    }


def item_sub(it: sky.SkyItem) -> str:
    """The grey line under an object's name in the "Up at ..." list."""
    if it.kind == "moon":
        return it.note
    if it.kind in ("pattern", "cluster", "galaxy", "nebula") and it.note:
        return lookup.cap(it.note)
    return lookup.KIND_LABELS.get(it.kind, "")


def items_block(guide: dict) -> list[dict]:
    return [
        {
            "n": it.name,
            "k": it.kind,
            "m": None if it.mag is None else round(float(it.mag), 2),
            "v": round(float(it.vis_mag), 2),
            "x": bool(it.extended),
            "s": item_sub(it),
            "alt": _round(it.track_alt, 1),
            "az": _round(it.track_az, 1),
        }
        for it in guide["items"]
    ]


def payload(site: Site, guide: dict, zenith_sqm: float, start: pd.Timestamp) -> dict:
    """Everything the browser needs to draw the night from dusk to dawn."""
    times = pd.DatetimeIndex(guide["times"])
    dusk, dawn = times[0], times[-1]
    mid = times[len(times) // 2]
    obs = sky.Observer(site)
    cat = sky.load_catalog()
    lst0 = float(obs.lst_deg(pd.DatetimeIndex([dusk]))[0])
    return {
        "id": f"{site.id}|{dusk.isoformat()}|{zenith_sqm:.3f}",
        "tz": site.timezone,
        "lat": site.lat,
        "t0": _ms(dusk),
        "t1": _ms(dawn),
        "start": _ms(start),
        "stepMinutes": STEP_MINUTES,
        "lst0": lst0,
        "lstRate": SIDEREAL_DEG_PER_MS,
        "steps": [_ms(t) for t in times],
        "stars": star_block(obs, cat, mid),
        "milkyWay": milky_way_block(obs, cat, mid),
        "figures": figures_block(obs, cat, mid),
        "tints": [c for _, c in BV_TINTS],
        "items": items_block(guide),
        **bodies_block(obs, times, guide["moon"]),
        **tables_block(times, guide["moon"], zenith_sqm),
        "place": site.name,
        "minMilkyWaySqm": sky.MILKY_WAY_MIN_SQM,
        "extendedMargin": sky.EXTENDED_MARGIN,
    }


COMPONENT_NAME = "skytrust_sky_live"
_REGISTERED: dict = {}


def component():
    """The component, registered again only when skylive.js or skylive.css changed on disk.
    Streamlit Cloud pulls new code into the running process; the app's module reload guard
    watches .py files only, so without this the browser would keep getting the old script."""
    from streamlit.components.v2.get_bidi_component_manager import get_bidi_component_manager

    stamp = tuple((p.stat().st_mtime_ns, p.stat().st_size) for p in (JS_PATH, CSS_PATH))
    # the registry belongs to the running Streamlit instance (tests start a fresh one)
    missing = get_bidi_component_manager().get(COMPONENT_NAME) is None
    if missing or _REGISTERED.get("stamp") != stamp:
        _REGISTERED["renderer"] = st.components.v2.component(
            COMPONENT_NAME,
            html='<div class="sk-live-root"></div>',
            css=CSS_PATH.read_text(),
            js=JS_PATH.read_text(),
            isolate_styles=False,
        )
        _REGISTERED["stamp"] = stamp
    return _REGISTERED["renderer"]


def render(data: dict, key: str = "sky_live") -> None:
    component()(data=data, key=key)
