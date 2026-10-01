"""The sky over a place at a moment, drawn as an SVG planisphere.

Looking up, with north at the top and east on the left (as on a star chart held overhead): the
centre is the zenith, the rim is the horizon, and distance from the centre is proportional to
the angle from the zenith. Only the stars a typical observer could see through the local light
pollution and moonlight are drawn ("your sky"), or every star a natural sky shows ("dark sky").
Star colours come from each star's B-V colour index.
"""

from __future__ import annotations

import html
import math

import numpy as np
import pandas as pd

from skytrust import sky
from skytrust.config import Site

SIZE = 1000  # viewBox units
R = 440  # horizon radius
CX = CY = SIZE / 2

# B-V colour index -> a star tint (blue-white hot stars to orange cool ones). A display choice.
BV_TINTS = [
    (-0.1, "#A9C1FF"),
    (0.15, "#D2DEFF"),
    (0.45, "#F2F4FF"),
    (0.7, "#FFF2DC"),
    (1.0, "#FFE1B8"),
    (1.4, "#FFCB90"),
    (9.0, "#FFB37E"),
]


def star_tint(bv) -> str:
    if bv is None or (isinstance(bv, float) and math.isnan(bv)):
        return "#F2F4FF"
    return next(c for limit, c in BV_TINTS if bv < limit)


def project(alt, az) -> tuple[np.ndarray, np.ndarray]:
    """Zenith at the centre, horizon on the rim, north up, east left."""
    r = (90 - np.asarray(alt, dtype=float)) / 90 * R
    a = np.radians(np.asarray(az, dtype=float))
    return CX - r * np.sin(a), CY - r * np.cos(a)


def _f(x: float) -> str:
    return f"{x:.1f}"


def _milky_way(
    obs: sky.Observer, cat: sky.Catalog, t: pd.DatetimeIndex, strength: float, uid: str
) -> str:
    if strength <= 0.02:
        return ""
    paths = []
    for level, rings in enumerate(cat.milky_way):
        pts = [np.asarray(r) for r in rings if len(r) > 2]
        if not pts:
            continue
        flat = np.concatenate(pts)
        alt, az = obs.altaz(flat[:, 0], flat[:, 1], t, key=f"milky-way-{level}")
        x, y = project(alt[0], az[0])
        d, k = [], 0
        for ring in pts:
            n = len(ring)
            xs, ys, al = x[k : k + n], y[k : k + n], alt[0][k : k + n]
            k += n
            if al.max() < -10:
                continue
            step = max(1, n // 240)  # ~240 points per ring is plenty at chart size
            d.append(
                "M"
                + "L".join(f"{a:.0f} {b:.0f}" for a, b in zip(xs[::step], ys[::step], strict=True))
                + "Z"
            )
        if d:
            opacity = (0.05 + 0.02 * level) * strength
            paths.append(
                f'<path d="{" ".join(d)}" fill="#C9D4F2" fill-opacity="{opacity:.3f}" '
                'fill-rule="evenodd"/>'
            )
    return f'<g filter="url(#mwblur-{uid})">{"".join(paths)}</g>'


def _lines(obs: sky.Observer, cat: sky.Catalog, t: pd.DatetimeIndex, color: str, op: float) -> str:
    segs = [(abbr, np.asarray(seg)) for abbr, c in cat.constellations.items() for seg in c["lines"]]
    if not segs:
        return ""
    flat = np.concatenate([s for _, s in segs])
    alt, az = obs.altaz(flat[:, 0], flat[:, 1], t, key="figures")
    x, y = project(alt[0], az[0])
    out, k = [], 0
    for _, seg in segs:
        n = len(seg)
        al = alt[0][k : k + n]
        if al.min() > -15:
            pts = " ".join(
                f"{a:.0f},{b:.0f}" for a, b in zip(x[k : k + n], y[k : k + n], strict=True)
            )
            out.append(f'<polyline points="{pts}"/>')
        k += n
    return (
        f'<g fill="none" stroke="{color}" stroke-opacity="{op}" stroke-width="1.6" '
        f'stroke-linejoin="round">{"".join(out)}</g>'
    )


def _labels(obs: sky.Observer, cat: sky.Catalog, t: pd.DatetimeIndex, color: str) -> str:
    items = [
        (c["name"], c["label"])
        for c in cat.constellations.values()
        if c.get("label") and int(c.get("rank", 3)) == 1
    ]  # the 22 major ones
    pos = np.array([p for _, p in items])
    alt, az = obs.altaz(pos[:, 0], pos[:, 1], t, key="labels")
    x, y = project(alt[0], az[0])
    out = []
    for (name, _), a, xi, yi in zip(items, alt[0], x, y, strict=True):
        if a > 12:
            out.append(f'<text x="{_f(xi)}" y="{_f(yi)}">{html.escape(name.upper())}</text>')
    return (
        f'<g fill="{color}" fill-opacity=".42" font-family="Geist Mono, monospace" '
        f'font-size="15" letter-spacing="2.4" text-anchor="middle">{"".join(out)}</g>'
    )


def _moon_glyph(x: float, y: float, r: float, phase_deg: float, uid: str = "m") -> str:
    """The Moon with its lit part (phase 0 new, 180 full); waxing is lit on the right as seen
    from the northern hemisphere. Kept simple: a disc plus a terminator ellipse."""
    lit = (1 - math.cos(math.radians(phase_deg))) / 2
    waxing = phase_deg < 180
    sx = 1 - 2 * lit  # terminator ellipse x-scale: +1 new ... -1 full
    rx = abs(sx) * r
    # dark disc, lit half, then the terminator ellipse in lit or dark colour
    side = 1 if waxing else -1
    half = (
        f"M {_f(x)} {_f(y - r)} A {_f(r)} {_f(r)} 0 0 {1 if side > 0 else 0} {_f(x)} {_f(y + r)} Z"
    )
    ell_fill = "#2A3248" if sx > 0 else "#F3EEDF"
    return (
        f'<g><circle cx="{_f(x)}" cy="{_f(y)}" r="{_f(r * 2.6)}" fill="url(#moonglow-{uid})"/>'
        f'<circle cx="{_f(x)}" cy="{_f(y)}" r="{_f(r)}" fill="#2A3248"/>'
        f'<path d="{half}" fill="#F3EEDF"/>'
        f'<ellipse cx="{_f(x)}" cy="{_f(y)}" rx="{_f(rx)}" ry="{_f(r)}" fill="{ell_fill}"/></g>'
    )


def _stars_svg(x, y, mags, bvs, limits, names, scale: float) -> list[str]:
    """Named bright stars as circles with a tooltip (and a gentle twinkle); the thousands of
    others as round-capped zero-length strokes batched by colour, size and opacity, which keeps
    a full dark-sky chart small enough for a phone (~60 KB instead of ~300 KB)."""
    out, batches = [], {}
    for xi, yi, mg, bv, lm, name in zip(x, y, mags, bvs, limits, names, strict=True):
        r = float(np.clip(3.4 - 0.52 * mg, 0.7, 5.2)) * scale
        o = float(np.clip(0.45 + 0.55 * (lm - mg) / 2.0, 0.42, 1.0))
        if isinstance(name, str) or mg < 2.0:
            cls = ' class="sk-tw"' if mg < 2.2 else ""
            tip = (
                f"<title>{html.escape(str(name))} (mag {mg:.1f})</title>"
                if isinstance(name, str)
                else ""
            )
            out.append(
                f'<circle{cls} cx="{_f(xi)}" cy="{_f(yi)}" r="{r:.2f}" fill="{star_tint(bv)}" '
                f'style="--o:{o:.2f}" fill-opacity="{o:.2f}">{tip}</circle>'
            )
            continue
        key = (star_tint(bv), round(r * 4) / 4, round(o, 1))
        batches.setdefault(key, []).append(f"M{xi:.0f} {yi:.0f}h0")
    for (tint, r, o), pts in batches.items():
        out.insert(
            0,
            f'<path d="{"".join(pts)}" stroke="{tint}" stroke-width="{2 * r:.2f}" '
            f'stroke-opacity="{o}" stroke-linecap="round" fill="none"/>',
        )
    return out


def sky_svg(
    site: Site,
    utc: pd.Timestamp,
    zenith_sqm: float,
    *,
    mode: str = "here",
    labels: bool = True,
    lines: bool = True,
    compact: bool = False,
    title: str = "",
) -> tuple[str, dict]:
    """SVG markup and a small summary (stars drawn, Moon up, planets shown) for one moment."""
    cat = sky.load_catalog()
    obs = sky.Observer(site)
    t = pd.DatetimeIndex([pd.Timestamp(utc)])
    # SVG ids are global to the page: several charts on one page must not share them
    uid = "c" + format(abs(hash((site.id, str(utc), mode, compact))) % 16**8, "08x")
    moon = obs.moon(t)
    moon_up = float(moon["alt"][0]) > 0
    sqm_here = zenith_sqm if mode == "here" else sky.NATURAL_SQM

    s = cat.stars
    alt, az = obs.altaz(s["ra"].to_numpy(), s["dec"].to_numpy(), t, key="stars")
    alt, az = alt[0], az[0]
    up = alt > 0
    m_state = sky.moon_at(moon, 0, alt[up], az[up]) if mode == "here" else None
    limit = sky.faintest_visible(alt[up], sqm_here, m_state)
    mags = s["mag"].to_numpy()[up]
    show = mags <= limit
    x, y = project(alt[up][show], az[up][show])
    mags_s, bv_s = mags[show], s["bv"].to_numpy()[up][show]
    names_s = s["name"].to_numpy()[up][show]
    lim_s = limit[show]

    scale = 0.8 if compact else 1.0
    stars = _stars_svg(x, y, mags_s, bv_s, lim_s, names_s, scale)

    # the Milky Way shows only where the sky is dark enough (Bortle 6 and darker overhead)
    zen = float(
        sky.sky_brightness(
            90.0, sqm_here, sky.moon_at(moon, 0, 90.0, 0.0) if mode == "here" else None
        )
    )
    mw_strength = float(
        np.clip((zen - sky.MILKY_WAY_MIN_SQM) / (21.6 - sky.MILKY_WAY_MIN_SQM), 0, 1)
    )

    planets_svg, shown = [], []
    for label, key in sky.PLANETS.items():
        b = obs.body(key, t)
        a, z, mg = float(b["alt"][0]), float(b["az"][0]), float(b["mag"][0])
        if a <= 0:
            continue
        lim = float(
            sky.faintest_visible(
                a, sqm_here, sky.moon_at(moon, 0, a, z) if mode == "here" else None
            )
        )
        if mg > lim:
            continue
        px, py = project(a, z)
        r = float(np.clip(4.6 - 0.55 * mg, 3.2, 7.5)) * scale
        shown.append(label)
        planets_svg.append(
            f'<g><title>{label} (mag {mg:.1f})</title><circle cx="{_f(float(px))}" cy="{_f(float(py))}" '
            f'r="{r:.1f}" fill="#F0C987"/><circle cx="{_f(float(px))}" cy="{_f(float(py))}" r="{r + 5:.1f}" '
            'fill="none" stroke="#F0C987" stroke-opacity=".35"/>'
            + (
                f'<text x="{_f(float(px) + r + 9)}" y="{_f(float(py) + 5)}" fill="#F0C987" '
                f'font-size="{22 if not compact else 26}" font-family="Geist, sans-serif">{label}</text>'
                if labels or compact
                else ""
            )
            + "</g>"
        )

    moon_svg = ""
    if moon_up:
        mx, my = project(float(moon["alt"][0]), float(moon["az"][0]))
        moon_svg = _moon_glyph(
            float(mx), float(my), 13 * scale + 3, float(moon["phase_deg"][0]), uid
        )
        if labels or compact:
            moon_svg += (
                f'<text x="{_f(float(mx) + 22)}" y="{_f(float(my) + 6)}" fill="#E8E2D0" '
                f'font-size="{22 if not compact else 26}" font-family="Geist, sans-serif">Moon</text>'
            )

    ink = "#C9D2EA"
    rings = "".join(
        f'<circle cx="{CX}" cy="{CY}" r="{R * (90 - a) / 90:.1f}" fill="none" stroke="{ink}" '
        f'stroke-opacity=".08" stroke-dasharray="3 7"/>'
        for a in (30, 60)
    )
    cardinals = ""
    for txt, a in (("N", 0), ("E", 90), ("S", 180), ("W", 270)):
        r_lab = R + 30
        cx = CX - r_lab * math.sin(math.radians(a))
        cy = CY - r_lab * math.cos(math.radians(a)) + 9
        cardinals += (
            f'<text x="{_f(cx)}" y="{_f(cy)}" text-anchor="middle" fill="currentColor" '
            f'fill-opacity=".7" font-size="26" font-family="Geist, sans-serif" '
            f'font-weight="500">{txt}</text>'
        )
    ticks = "".join(
        f'<line x1="{_f(CX - R * math.sin(math.radians(a)))}" y1="{_f(CY - R * math.cos(math.radians(a)))}" '
        f'x2="{_f(CX - (R + 9) * math.sin(math.radians(a)))}" y2="{_f(CY - (R + 9) * math.cos(math.radians(a)))}" '
        'stroke="currentColor" stroke-opacity=".35"/>'
        for a in range(0, 360, 15)
    )
    sky_fill = _sky_fill(zen, moon_up)
    body = (
        f'<circle cx="{CX}" cy="{CY}" r="{R}" fill="url(#skyfill-{uid})"/>'
        f'<g clip-path="url(#horizon-{uid})">'
        + _milky_way(obs, cat, t, mw_strength, uid)
        + rings
        + (_lines(obs, cat, t, ink, 0.22) if lines else "")
        + (_labels(obs, cat, t, ink) if labels and not compact else "")
        + "".join(stars)
        + "".join(planets_svg)
        + moon_svg
        + "</g>"
        + f'<circle cx="{CX}" cy="{CY}" r="{R}" fill="none" stroke="{ink}" stroke-opacity=".28" stroke-width="1.5"/>'
        + ticks
        + cardinals
    )
    defs = (
        "<defs>"
        f'<clipPath id="horizon-{uid}"><circle cx="{CX}" cy="{CY}" r="{R}"/></clipPath>'
        f'<radialGradient id="skyfill-{uid}" cx="50%" cy="50%" r="50%">{sky_fill}</radialGradient>'
        f'<radialGradient id="moonglow-{uid}"><stop offset="0" stop-color="#F3EEDF" stop-opacity=".35"/>'
        '<stop offset="1" stop-color="#F3EEDF" stop-opacity="0"/></radialGradient>'
        f'<filter id="mwblur-{uid}" x="-5%" y="-5%" width="110%" height="110%">'
        '<feGaussianBlur stdDeviation="6"/></filter>'
        "</defs>"
    )
    label = html.escape(title or f"The sky over {site.name}")
    svg = (
        f'<svg viewBox="0 0 {SIZE} {SIZE}" xmlns="http://www.w3.org/2000/svg" role="img" '
        f'aria-label="{label}">{defs}{body}</svg>'
    )
    return svg, {
        "stars": int(show.sum()),
        "moon_up": moon_up,
        "planets": shown,
        "milky_way": mw_strength > 0.15,
        "zenith_sqm": zen,
    }


def _sky_fill(zenith_sqm: float, moon_up: bool) -> str:
    """Background of the dome: deep blue under a dark sky, washed-out grey-blue under a bright
    one, so light pollution and moonlight are visible at a glance."""
    t = float(np.clip((zenith_sqm - 17.0) / (22.0 - 17.0), 0, 1))  # 0 = city glow, 1 = natural

    def mix(a: tuple, b: tuple) -> str:
        return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b, strict=True))

    centre = mix((0x3A, 0x45, 0x5E), (0x10, 0x18, 0x2E))
    edge = mix((0x5A, 0x58, 0x5E), (0x16, 0x20, 0x3A))
    return (
        f'<stop offset="0" stop-color="{centre}"/><stop offset=".78" stop-color="{centre}"/>'
        f'<stop offset="1" stop-color="{edge}"/>'
    )
