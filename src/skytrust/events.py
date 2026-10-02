"""Upcoming sky events for California: meteor showers, the Moon's phases, planets at their best,
close pairings of the Moon and planets, eclipses and the seasons.

Meteor showers come from the IMO's working list (config/meteor_showers.yaml); everything else is
computed from JPL's DE421 ephemeris with Skyfield, so dates and times are reproducible and were
checked against published values (tests/test_events.py): the IMO 2026 lunar-phase table and
shower peak times, Saturn's 2026 opposition, and the 2026 lunar eclipses.

For each event the module also works out, for a given place, when and where to look; and for
meteor showers, how many meteors an observer can expect at each place, from the IMO's standard
rate formula:  HR = ZHR · sin(radiant altitude) / r^(6.5 - limiting magnitude).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from skytrust import astro, sky
from skytrust.config import CONFIG_DIR, Site

SHOWERS_PATH = CONFIG_DIR / "meteor_showers.yaml"
SOLAR_DEG_PER_DAY = 360 / 365.2422

# Close pairings worth pointing out: the Moon with bright planets and ecliptic stars, and
# planet pairs. Thresholds keep it to what looks striking to the eye.
MOON_TARGETS = {
    "Venus": ("venus", 4.0),
    "Mars": ("mars", 3.0),
    "Jupiter": ("jupiter barycenter", 4.0),
    "Saturn": ("saturn barycenter", 3.0),
}
MOON_STARS = ["Aldebaran", "Regulus", "Spica", "Antares"]
MOON_STAR_MAX_SEP = 2.5
PLANET_PAIR_MAX_SEP = 3.0
BRIGHT_PLANETS = {
    "Mercury": "mercury",
    "Venus": "venus",
    "Mars": "mars",
    "Jupiter": "jupiter barycenter",
    "Saturn": "saturn barycenter",
}
OUTER_PLANETS = {
    "Mars": "mars",
    "Jupiter": "jupiter barycenter",
    "Saturn": "saturn barycenter",
    "Uranus": "uranus barycenter",
    "Neptune": "neptune barycenter",
}

# Meteor outlook from the expected hourly rate at the best place (dark site, Moon included).
RATE_OUTLOOK = [(30, "Great"), (12, "Good"), (5, "Fair"), (0, "Weak")]


@dataclass(frozen=True)
class Shower:
    code: str
    name: str
    active: tuple[str, str]  # "MM-DD"
    peak_sol: float
    ra: float
    dec: float
    speed: float
    r: float
    zhr: float
    notes: str = ""


@dataclass
class Event:
    kind: str  # meteor | moon | planet | pairing | eclipse | season
    title: str
    utc: pd.Timestamp  # the moment (peak, exact phase, closest approach, ...)
    summary: str  # one plain sentence
    look: str = ""  # when and where to look, for the chosen place
    outlook: str = ""  # Great / Good / Fair / Weak, or "" when not rated
    best_start: pd.Timestamp | None = None  # best viewing, for the chosen place
    best_end: pd.Timestamp | None = None
    active: tuple[pd.Timestamp, pd.Timestamp] | None = None
    places: list[dict] = field(default_factory=list)  # meteor showers: per-place outlook
    details: dict = field(default_factory=dict)
    source: str = ""


# ---------- time helpers ----------


def _ts():
    return astro._timescale()


def _sf(times) -> Any:
    """Skyfield Time array for UTC timestamps (whole seconds)."""
    return astro._to_skyfield(pd.DatetimeIndex(times).floor("s"))


def _utc(t) -> pd.Timestamp:
    return pd.Timestamp(t.utc_datetime()).tz_convert("UTC").floor("s")


def solar_longitude(times: pd.DatetimeIndex) -> np.ndarray:
    """Apparent geocentric longitude of the Sun on the ecliptic of J2000 (the IMO's λ⊙)."""
    from skyfield.framelib import ecliptic_J2000_frame

    eph = astro._ephemeris()
    sun = eph["earth"].at(_sf(times)).observe(eph["sun"]).apparent()
    _, lon, _ = sun.frame_latlon(ecliptic_J2000_frame)
    return np.atleast_1d(lon.degrees)


def time_of_solar_longitude(target: float, near: pd.Timestamp) -> pd.Timestamp:
    """When the Sun reaches ecliptic longitude `target` (J2000), nearest to `near`."""
    t = pd.Timestamp(near)
    for _ in range(6):  # Newton steps; the Sun moves ~0.986°/day, converges to seconds
        lam = float(solar_longitude(pd.DatetimeIndex([t]))[0])
        diff = ((target - lam + 180) % 360) - 180
        t = t + pd.Timedelta(days=diff / SOLAR_DEG_PER_DAY)
        if abs(diff) < 1e-6:
            break
    return t.floor("s")


# ---------- meteor showers ----------


@lru_cache(maxsize=4)
def load_showers(path: Path = SHOWERS_PATH) -> tuple[Shower, ...]:
    raw = yaml.safe_load(path.read_text())["showers"]
    return tuple(
        Shower(
            s["code"],
            s["name"],
            tuple(s["active"]),
            float(s["peak_sol"]),
            float(s["ra"]),
            float(s["dec"]),
            float(s["speed"]),
            float(s["r"]),
            float(s["zhr"]),
            " ".join(str(s.get("notes", "")).split()),
        )
        for s in raw
    )


def expected_rate(zhr: float, r: float, radiant_alt_deg, limiting_mag) -> np.ndarray:
    """Meteors per hour a single observer can expect (IMO formula, zenith exponent 1). The
    limiting magnitude is capped at the ZHR's reference 6.5, so the rate never exceeds the ZHR:
    the ZHR already describes an ideal observer under a perfect sky."""
    alt = np.asarray(radiant_alt_deg, dtype=float)
    lm = np.minimum(np.asarray(limiting_mag, dtype=float), 6.5)
    rate = zhr * np.sin(np.radians(np.clip(alt, 0, 90))) / r ** (6.5 - lm)
    return np.where(alt > 0, rate, 0.0)


def active_window(shower: Shower, peak: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Activity start/end around this year's peak (dates from the IMO list; the year is chosen
    so the window contains the peak, which handles the Quadrantids' new-year wrap)."""

    def on(md: str, year: int) -> pd.Timestamp:
        m, d = (int(x) for x in md.split("-"))
        return pd.Timestamp(year=year, month=m, day=d, tz="UTC")

    start, end = on(shower.active[0], peak.year), on(shower.active[1], peak.year)
    if start > peak:
        start = on(shower.active[0], peak.year - 1)
    if end < peak:
        end = on(shower.active[1], peak.year + 1)
    return start, end + pd.Timedelta(days=1)


def _approx_peak(shower: Shower, year: int) -> pd.Timestamp:
    """Rough date the Sun reaches the shower's solar longitude in `year` (within ~2 days): the
    March equinox is λ⊙ = 0 near March 20."""
    return pd.Timestamp(year=year, month=3, day=20, tz="UTC") + pd.Timedelta(
        days=shower.peak_sol / SOLAR_DEG_PER_DAY
    )


def shower_peaks(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[Shower, pd.Timestamp]]:
    """Every shower whose activity period overlaps [start, end], with its peak time. Only years
    whose rough peak date can matter are solved exactly (activity lasts at most ~2 months)."""
    out = []
    for s in load_showers():
        for year in range(start.year - 1, end.year + 2):
            guess = _approx_peak(s, year)
            if not (start - pd.Timedelta(days=75) <= guess <= end + pd.Timedelta(days=75)):
                continue
            peak = time_of_solar_longitude(s.peak_sol, guess)
            a, b = active_window(s, peak)
            if b >= start and a <= end:
                out.append((s, peak))
    return sorted(out, key=lambda x: x[1])


def next_peaks(now: pd.Timestamp, days: int = 60) -> list[tuple[Shower, pd.Timestamp]]:
    """Meteor-shower peaks still to come in the next `days` days (cheap: no per-place work)."""
    end = now + pd.Timedelta(days=days)
    return [(s, p) for s, p in shower_peaks(now, end) if now - pd.Timedelta(hours=12) <= p <= end]


@lru_cache(maxsize=1024)
def _dark_windows(site: Site, day: dt.date, sun_alt: float) -> pd.DataFrame:
    """Dusk and dawn for the nights starting `day - 1` and `day` (local dates), from the Sun's
    geometric altitude on a 10-minute grid, interpolated to the crossing. Same definition as
    `astro.dark_windows`, much faster for the many places × showers here (agrees to < 1 min)."""
    tz = site.timezone
    t0 = pd.Timestamp(day - dt.timedelta(days=1)).tz_localize(tz) + pd.Timedelta(hours=12)
    times = pd.date_range(t0, t0 + pd.Timedelta(hours=48), freq="10min").tz_convert("UTC")
    alt = sky.Observer(site).body("sun", times)["alt"] - sun_alt
    rows = []
    dusk = None
    for i in range(len(alt) - 1):
        if (alt[i] > 0) == (alt[i + 1] > 0):
            continue
        f = alt[i] / (alt[i] - alt[i + 1])
        when = times[i] + (times[i + 1] - times[i]) * f
        if alt[i] > 0:
            dusk = when
        elif dusk is not None:
            rows.append(
                {"night_date": dusk.tz_convert(tz).date(), "dusk_utc": dusk, "dawn_utc": when}
            )
            dusk = None
    return pd.DataFrame(rows).set_index("night_date") if rows else pd.DataFrame()


def viewing_night(peak: pd.Timestamp, site: Site, sun_alt: float = -18.0) -> pd.Series:
    """The night of darkness to watch a peak: the one containing it, else the nearer one."""
    w = _dark_windows(site, peak.tz_convert(site.timezone).date(), sun_alt)
    mids = w["dusk_utc"] + (w["dawn_utc"] - w["dusk_utc"]) / 2
    inside = w[(w["dusk_utc"] <= peak) & (w["dawn_utc"] >= peak)]
    if len(inside):
        return inside.iloc[0]
    return w.iloc[int(np.argmin(np.abs((mids - peak).dt.total_seconds())))]


@lru_cache(maxsize=1024)
def shower_at(shower: Shower, peak: pd.Timestamp, site: Site, zenith_sqm: float) -> dict:
    """The peak night at one place: hour-by-hour radiant height and expected rate (Moon and
    light pollution included), and the best stretch. Cached: it doesn't depend on which place
    the user picked, so every place's outlook is computed once per process."""
    night = viewing_night(peak, site)
    times = pd.date_range(night["dusk_utc"], night["dawn_utc"], freq="15min")
    obs = sky.Observer(site)
    ralt, raz = obs.altaz([shower.ra], [shower.dec], times)
    ralt, raz = ralt[:, 0], raz[:, 0]
    moon = obs.moon(times)
    lm = sky.limiting_magnitude(zenith_with_moon(moon, zenith_sqm))
    rate = expected_rate(shower.zhr, shower.r, ralt, lm)
    i = int(np.argmax(rate))
    good = np.flatnonzero(rate >= 0.6 * rate[i]) if rate[i] > 0 else np.array([i])
    con = sky.constellation_at([shower.ra], [shower.dec])[0]
    return {
        "night": night.name,
        "times": times,
        "rate": rate,
        "radiant_alt": ralt,
        "best_utc": times[i],
        "best_rate": float(rate[i]),
        "best_start": times[good[0]],
        "best_end": times[good[-1]],
        "radiant_where": sky.where_words(float(ralt[i]), float(raz[i])),
        "radiant_con": sky.constellation_name(con),
        "moon_illum": float(moon["illum"][i]),
        "moon_up_at_best": bool(moon["alt"][i] > 0),
        "moon_up_hours": float((moon["alt"] > 0).mean() * len(times) / 4),
        "dark_rate": float(
            expected_rate(shower.zhr, shower.r, ralt[i], sky.limiting_magnitude(sky.NATURAL_SQM))
        ),
    }


def zenith_with_moon(moon: dict, zenith_sqm: float) -> np.ndarray:
    """Zenith sky brightness (mag/arcsec²) at each time, with scattered moonlight added when the
    Moon is up (the separation of the zenith from the Moon is 90° minus its altitude)."""
    z = np.zeros_like(moon["alt"])
    base = sky.mag_to_nl(zenith_sqm) * np.ones_like(z)
    up = moon["alt"] > 0
    lit = sky.moon_sky_nl(90 - moon["alt"], z, 90 - moon["alt"], moon["phase_angle"])
    return sky.nl_to_mag(base + np.where(up, lit, 0.0))


def rate_words(rate: float) -> str:
    if rate < 1:
        return "fewer than 1 an hour"
    return f"about {rate:.0f} an hour"


def outlook_for(rate: float) -> str:
    return next(label for limit, label in RATE_OUTLOOK if rate >= limit)


def meteor_events(
    start: pd.Timestamp,
    end: pd.Timestamp,
    site: Site,
    zenith_sqm: float,
    places: list[tuple[Site, float]],
) -> list[Event]:
    """Showers active in [start, end], seen from `site`, with the outlook at every place."""
    events = []
    for s, peak in shower_peaks(start, end):
        here = shower_at(s, peak, site, zenith_sqm)
        ranked = []
        for p, p_sqm in places:
            v = shower_at(s, peak, p, p_sqm)
            ranked.append(
                {
                    "site": p,
                    "rate": v["best_rate"],
                    "best_utc": v["best_utc"],
                    "sqm": p_sqm,
                    "night": v["night"],
                }
            )
        ranked.sort(key=lambda x: -x["rate"])
        top = ranked[0]["rate"] if ranked else here["best_rate"]
        moon = f"Moon {here['moon_illum']:.0%} lit" + (
            ", up at the best time" if here["moon_up_at_best"] else ", down at the best time"
        )
        events.append(
            Event(
                kind="meteor",
                title=f"{s.name} peak",
                utc=peak,
                summary=(
                    f"{rate_words(here['best_rate']).capitalize()} from here (up to "
                    f"{s.zhr:.0f} in a perfect sky)."
                ),
                look=(
                    f"{moon}. The meteors seem to come from {here['radiant_where']} "
                    f"(in {here['radiant_con']}) but can appear anywhere."
                ),
                outlook=outlook_for(top),
                best_start=here["best_start"],
                best_end=here["best_end"],
                active=active_window(s, peak),
                places=ranked,
                details={"shower": s, "here": here},
                source="IMO 2026 Meteor Shower Calendar",
            )
        )
    return events


# ---------- the Moon ----------


def moon_phase_events(start: pd.Timestamp, end: pd.Timestamp) -> list[Event]:
    from skyfield import almanac

    eph = astro._ephemeris()
    t, phase = almanac.find_discrete(_sf([start])[0], _sf([end])[0], almanac.moon_phases(eph))
    out = []
    for ti, ph in zip(t, phase, strict=True):
        when = _utc(ti)
        if ph == 0:
            out.append(
                Event(
                    "moon",
                    "New Moon",
                    when,
                    "The darkest nights of the month: the best week for the Milky Way, "
                    "faint galaxies and meteors.",
                    source="Computed (JPL DE421)",
                )
            )
        elif ph == 2:
            out.append(
                Event(
                    "moon",
                    "Full Moon",
                    when,
                    "The Moon is up all night and floods the sky with light: great for "
                    "the Moon itself, poor for faint objects.",
                    source="Computed (JPL DE421)",
                )
            )
    return out


# ---------- planets ----------


def _transit_tonight(site: Site, key: str, when: pd.Timestamp) -> dict:
    """Where a planet is during the night around `when`: highest moment and direction."""
    night = viewing_night(when, site)
    times = pd.date_range(night["dusk_utc"], night["dawn_utc"], freq="20min")
    b = sky.Observer(site).body(key, times)
    i = int(np.argmax(b["alt"]))
    return {
        "utc": times[i],
        "alt": float(b["alt"][i]),
        "az": float(b["az"][i]),
        "mag": float(b["mag"][i]),
        "con": sky.constellation_name(sky.constellation_at([b["ra"][i]], [b["dec"][i]])[0]),
    }


def opposition_events(start: pd.Timestamp, end: pd.Timestamp, site: Site) -> list[Event]:
    """Outer planets at opposition: opposite the Sun, closest, brightest, up all night."""
    from skyfield import almanac

    eph = astro._ephemeris()
    out = []
    for name, key in OUTER_PLANETS.items():
        f = almanac.oppositions_conjunctions(eph, eph[key])
        t, kind = almanac.find_discrete(_sf([start])[0], _sf([end])[0], f)
        for ti, k in zip(t, kind, strict=True):
            if k != 1:  # 1 = opposition, 0 = conjunction with the Sun (invisible)
                continue
            when = _utc(ti)
            tr = _transit_tonight(site, key, when)
            eye = tr["mag"] <= 5.5
            out.append(
                Event(
                    "planet",
                    f"{name} at opposition",
                    when,
                    (
                        f"{name} is opposite the Sun: at its closest and brightest for the year"
                        + ("." if eye else ", but only in binoculars or a telescope.")
                    ),
                    look=(
                        "Up all night: rises in the east at sunset and is highest around "
                        f"midnight, {sky.where_words(tr['alt'], tr['az'])}, in {tr['con']}."
                    ),
                    best_start=tr["utc"] - pd.Timedelta(hours=2),
                    best_end=tr["utc"] + pd.Timedelta(hours=2),
                    details=tr,
                    source="Computed (JPL DE421)",
                )
            )
    return out


def elongation_events(start: pd.Timestamp, end: pd.Timestamp, site: Site) -> list[Event]:
    """Mercury and Venus at greatest elongation (farthest from the Sun): their best showing,
    in the evening (east of the Sun) or the morning (west of it)."""
    from skyfield.searchlib import find_maxima

    eph = astro._ephemeris()
    earth, sun = eph["earth"], eph["sun"]
    out = []
    for name in ("Mercury", "Venus"):
        planet = eph[name.lower()]

        def elongation(t, planet=planet):
            e = earth.at(t)
            return e.observe(sun).separation_from(e.observe(planet)).degrees

        elongation.step_days = 5  # type: ignore[attr-defined]
        t, values = find_maxima(_sf([start])[0], _sf([end])[0], elongation)
        for ti, v in zip(t, values, strict=True):
            when = _utc(ti)
            evening = _east_of_sun(when, eph[name.lower()])
            seen = _twilight_altitude(site, name.lower(), when, evening)
            where = "west after sunset" if evening else "east before sunrise"
            ok = seen["alt"] >= 5
            out.append(
                Event(
                    "planet",
                    f"{name} at greatest {'evening' if evening else 'morning'} elongation",
                    when,
                    (
                        f"{name} is as far from the Sun as it gets ({v:.0f}°): its best "
                        f"{'evening' if evening else 'morning'} showing, low in the {where}."
                        + ("" if ok else " From California it stays very low this time.")
                    ),
                    look=(
                        f"About {seen['alt']:.0f}° up in the {sky.compass(seen['az'])} as the sky "
                        f"{'darkens' if evening else 'brightens before dawn'} (a fist at arm's "
                        "length is about 10°)."
                    ),
                    best_start=seen["utc"] - pd.Timedelta(minutes=30),
                    best_end=seen["utc"] + pd.Timedelta(minutes=30),
                    details={"elongation": float(v), **seen},
                    source="Computed (JPL DE421)",
                )
            )
    return out


def _east_of_sun(when: pd.Timestamp, planet) -> bool:
    from skyfield.framelib import ecliptic_frame

    eph = astro._ephemeris()
    e = eph["earth"].at(_sf([when])[0])
    _, ls, _ = e.observe(eph["sun"]).apparent().frame_latlon(ecliptic_frame)
    _, lp, _ = e.observe(planet).apparent().frame_latlon(ecliptic_frame)
    return ((lp.degrees - ls.degrees) % 360) < 180


def _twilight_altitude(site: Site, key: str, when: pd.Timestamp, evening: bool) -> dict:
    """Planet altitude when the Sun is 8° down (the sky dark enough to spot a bright planet)
    on the evening or morning closest to `when`."""
    from skyfield import almanac

    eph = astro._ephemeris()
    topos = astro._topos(site)
    day = when.tz_convert(site.timezone).normalize()
    t0, t1 = _sf([day - pd.Timedelta(hours=12)])[0], _sf([day + pd.Timedelta(hours=36)])[0]
    f = almanac.risings_and_settings(eph, eph["sun"], topos, horizon_degrees=-8.0)
    t, rising = almanac.find_discrete(t0, t1, f)
    pick = [ti for ti, r in zip(t, rising, strict=True) if bool(r) != evening]
    moment = _utc(pick[0]) if pick else when
    b = sky.Observer(site).body(key, pd.DatetimeIndex([moment]))
    return {
        "utc": moment,
        "alt": float(b["alt"][0]),
        "az": float(b["az"][0]),
        "mag": float(b["mag"][0]),
    }


# ---------- close pairings ----------


def _separation_series(a_key: str, b, times: pd.DatetimeIndex) -> np.ndarray:
    """Geocentric angle (degrees) between body `a_key` and body/star `b` at each time."""
    eph = astro._ephemeris()
    e = eph["earth"].at(_sf(times))
    pa = e.observe(eph[a_key])
    pb = e.observe(eph[b] if isinstance(b, str) else b)
    return np.atleast_1d(pa.separation_from(pb).degrees)


def _minima(sep: np.ndarray, times: pd.DatetimeIndex, max_sep: float) -> list[tuple[int, float]]:
    out = []
    for i in range(1, len(sep) - 1):
        if sep[i] <= sep[i - 1] and sep[i] < sep[i + 1] and sep[i] <= max_sep:
            out.append((i, float(sep[i])))
    return out


def _best_view(site: Site, keys: list, when: pd.Timestamp) -> dict | None:
    """The darkest moment (Sun ≤ -8°) within a day of `when` with both objects ≥ 8° up; prefers
    the moment closest to `when`."""
    times = pd.date_range(
        when - pd.Timedelta(hours=18), when + pd.Timedelta(hours=18), freq="15min"
    )
    obs = sky.Observer(site)
    sun = obs.body("sun", times)["alt"]
    alts, azs = [], []
    for k in keys:
        if isinstance(k, str):
            b = obs.body(k, times)
            alts.append(b["alt"])
            azs.append(b["az"])
        else:  # (ra, dec) of a star
            a, z = obs.altaz([k[0]], [k[1]], times)
            alts.append(a[:, 0])
            azs.append(z[:, 0])
    low = np.minimum.reduce(alts)
    ok = (sun <= -8) & (low >= 8)
    if not ok.any():
        return None
    idx = np.flatnonzero(ok)
    i = int(idx[np.argmin(np.abs((times[idx] - when).total_seconds()))])
    return {"utc": times[i], "alt": float(np.mean([a[i] for a in alts])), "az": float(azs[0][i])}


def pairing_events(start: pd.Timestamp, end: pd.Timestamp, site: Site) -> list[Event]:
    """The Moon close to a bright planet or star, and two planets close together."""
    from skyfield.api import Star

    cat = sky.load_catalog()
    times = pd.date_range(start, end, freq="1h")
    out = []
    for name, (key, max_sep) in MOON_TARGETS.items():
        sep = _separation_series("moon", key, times)
        for i, s in _minima(sep, times, max_sep):
            view = _best_view(site, ["moon", key], times[i])
            if view:
                out.append(_pairing("Moon", name, times[i], s, view, site))
    for star in MOON_STARS:
        row = cat.stars[cat.stars["name"] == star].iloc[0]
        target = Star(ra_hours=row["ra"] / 15, dec_degrees=row["dec"])
        sep = _separation_series("moon", target, times)
        for i, s in _minima(sep, times, MOON_STAR_MAX_SEP):
            view = _best_view(site, ["moon", (row["ra"], row["dec"])], times[i])
            if view:
                out.append(_pairing("Moon", star, times[i], s, view, site))
    days = pd.date_range(start, end, freq="6h")
    names = list(BRIGHT_PLANETS)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            ka, kb = BRIGHT_PLANETS[names[a]], BRIGHT_PLANETS[names[b]]
            sep = _separation_series(ka, kb, days)
            for i, s in _minima(sep, days, PLANET_PAIR_MAX_SEP):
                view = _best_view(site, [ka, kb], days[i])
                if view:
                    out.append(_pairing(names[a], names[b], days[i], s, view, site))
    return out


def part_of_night(local: pd.Timestamp) -> str:
    h = local.hour
    if 12 <= h < 22:
        return "in the evening"
    if h >= 22 or h < 3:
        return "around midnight"
    return "before dawn"


def apart_words(sep: float) -> str:
    """'1.1° apart, about one finger's width at arm's length' (a finger covers ~1°)."""
    if sep < 0.75:
        return f"{sep:.1f}° apart, less than a finger's width at arm's length"
    fingers = round(sep)
    width = {1: "one finger's width", 2: "two fingers' width", 3: "three fingers' width"}
    return f"{sep:.1f}° apart, about {width.get(fingers, f'{fingers} fingers')} at arm's length"


def _pairing(a: str, b: str, when: pd.Timestamp, sep: float, view: dict, site: Site) -> Event:
    local = view["utc"].tz_convert(site.timezone)
    look = f"{part_of_night(local)}, {sky.where_words(view['alt'], view['az'])}."
    return Event(
        "pairing",
        f"{a} near {b}",
        when,
        f"{a} and {b} appear {apart_words(sep)}.",
        look=look[:1].upper() + look[1:],
        best_start=view["utc"] - pd.Timedelta(minutes=45),
        best_end=view["utc"] + pd.Timedelta(minutes=45),
        details={"separation": sep, "with": [b], **view},
        source="Computed (JPL DE421)",
    )


def merge_moon_pairings(events: list[Event], site: Site) -> list[Event]:
    """One event per night for the Moon: 'The Moon near Jupiter and Regulus', not two."""
    moon = sorted((e for e in events if e.title.startswith("Moon near")), key=lambda e: e.utc)
    rest = [e for e in events if not e.title.startswith("Moon near")]
    groups: list[list[Event]] = []
    for e in moon:
        night = (e.details["utc"] - pd.Timedelta(hours=12)).tz_convert(site.timezone).date()
        if groups and groups[-1][0].details["night"] == night:
            groups[-1].append(e)
        else:
            e.details["night"] = night
            groups.append([e])
    merged = []
    for g in groups:
        first = min(g, key=lambda e: e.details["separation"])
        names = [e.details["with"][0] for e in g]
        listing = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
        closest = first.details["separation"]
        merged.append(
            Event(
                "pairing",
                f"The Moon near {listing}",
                first.utc,
                f"The Moon passes close to {listing} ({first.details['with'][0]} "
                f"{apart_words(closest)}).",
                look=first.look,
                best_start=first.best_start,
                best_end=first.best_end,
                details={**first.details, "with": names},
                source=first.source,
            )
        )
    return rest + merged


# ---------- eclipses and seasons ----------


def eclipse_events(start: pd.Timestamp, end: pd.Timestamp, site: Site) -> list[Event]:
    from skyfield import eclipselib

    eph = astro._ephemeris()
    t, kind, details = eclipselib.lunar_eclipses(_sf([start])[0], _sf([end])[0], eph)
    out = []
    for k, (ti, y) in enumerate(zip(t, kind, strict=True)):
        when = _utc(ti)
        b = sky.Observer(site).body("moon", pd.DatetimeIndex([when]))
        sun = sky.Observer(site).body("sun", pd.DatetimeIndex([when]))["alt"][0]
        seen = b["alt"][0] > 0 and sun < -6
        label = eclipselib.LUNAR_ECLIPSES[y]
        mag = float(details["umbral_magnitude"][k])
        out.append(
            Event(
                "eclipse",
                f"{label} lunar eclipse",
                when,
                (
                    f"A {label.lower()} eclipse of the Moon"
                    + (f" ({mag:.0%} of the Moon's width inside Earth's shadow)" if mag > 0 else "")
                    + (
                        ". Visible from California."
                        if seen
                        else ". Not visible from California "
                        "(the Moon is below the horizon here at mid-eclipse)."
                    )
                ),
                look=(
                    f"At mid-eclipse the Moon is "
                    f"{sky.where_words(float(b['alt'][0]), float(b['az'][0]))}."
                    if seen
                    else ""
                ),
                outlook="" if seen else "Not visible",
                best_start=when - pd.Timedelta(hours=1) if seen else None,
                best_end=when + pd.Timedelta(hours=1) if seen else None,
                details={"umbral_magnitude": mag, "visible": seen},
                source="Computed (JPL DE421)",
            )
        )
    return out


def season_events(start: pd.Timestamp, end: pd.Timestamp) -> list[Event]:
    from skyfield import almanac

    eph = astro._ephemeris()
    t, y = almanac.find_discrete(_sf([start])[0], _sf([end])[0], almanac.seasons(eph))
    text = {
        0: ("March equinox", "Day and night are about equal; spring begins."),
        1: ("June solstice", "The shortest night of the year."),
        2: ("September equinox", "Day and night are about equal; autumn begins."),
        3: ("December solstice", "The longest night of the year: the most hours of darkness."),
    }
    return [
        Event("season", text[int(k)][0], _utc(ti), text[int(k)][1], source="Computed (JPL DE421)")
        for ti, k in zip(t, y, strict=True)
    ]


# ---------- everything ----------


def upcoming(
    now: pd.Timestamp, days: int, site: Site, zenith_sqm: float, places: list[tuple[Site, float]]
) -> list[Event]:
    """All events from `now` to `now + days`, in time order. Meteor showers are included while
    active, even if the peak has passed."""
    start, end = (
        pd.Timestamp(now).tz_convert("UTC"),
        pd.Timestamp(now).tz_convert("UTC") + pd.Timedelta(days=days),
    )
    events = (
        meteor_events(start, end, site, zenith_sqm, places)
        + moon_phase_events(start, end)
        + opposition_events(start, end, site)
        + elongation_events(start, end, site)
        + merge_moon_pairings(pairing_events(start, end, site), site)
        + eclipse_events(start, end, site)
        + season_events(start, end)
    )
    keep = [
        e
        for e in events
        if e.utc >= start - pd.Timedelta(days=1) or (e.active is not None and e.active[1] >= start)
    ]
    return sorted(keep, key=lambda e: e.utc)


def moon_dark_week(new_moon: pd.Timestamp, tz: str) -> tuple[dt.date, dt.date]:
    """The week around New Moon (3 days either side), the darkest stretch of the month."""
    d = new_moon.tz_convert(tz).date()
    return d - dt.timedelta(days=3), d + dt.timedelta(days=3)


def degrees_to_fists(deg: float) -> str:
    """A fist at arm's length is about 10°."""
    n = max(1, round(deg / 10))
    return f"about {n} fist{'s' if n != 1 else ''} up"
