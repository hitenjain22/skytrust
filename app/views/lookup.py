"""Turn the sky engine's output into short, plain sentences for beginners: what to look at, where
and when, and whether it will show through the light here. Pure functions, no Streamlit."""

from __future__ import annotations

import pandas as pd

from skytrust import sky
from skytrust.config import Site

KIND_LABELS = {
    "planet": "Planet",
    "moon": "The Moon",
    "star": "Bright star",
    "pattern": "Star pattern",
    "cluster": "Star cluster",
    "galaxy": "Galaxy",
    "nebula": "Nebula",
}


def cap(text: str) -> str:
    """Capitalise the first letter only ('high in the SE' -> 'High in the SE'); str.capitalize
    would lower-case the rest ('SE' -> 'se', 'Deneb' -> 'deneb')."""
    return text[:1].upper() + text[1:]


def uncap(text: str) -> str:
    """Lower-case the first letter only, to continue a sentence ('The Milky Way is hidden' ->
    'the Milky Way is hidden'); str.lower would also lower-case 'Milky Way'."""
    return text[:1].lower() + text[1:]


def clock(ts: pd.Timestamp | None, tz: str) -> str:
    """'9 PM' / '9:40 PM' (rounded to 5 minutes; people don't need the seconds)."""
    if ts is None:
        return ""
    t = ts.tz_convert(tz).round("5min")
    return t.strftime("%-I %p") if t.minute == 0 else t.strftime("%-I:%M %p")


def when_words(item: sky.SkyItem, dusk: pd.Timestamp, dawn: pd.Timestamp, tz: str) -> str:
    """'All night, highest at 12:40 AM' / 'After dark until 10 PM' / 'From 2 AM'."""
    start, end, best = item.up_from, item.up_until, item.best_utc
    edge = pd.Timedelta(minutes=25)
    early = start is not None and start <= dusk + edge
    late = end is not None and end >= dawn - edge
    if early and late:
        return f"All night, highest at {clock(best, tz)}"
    if early:
        return f"After dark until {clock(end, tz)}"
    if late:
        return f"From {clock(start, tz)} until dawn"
    return f"{clock(start, tz)} to {clock(end, tz)}"


def visible_words(item: sky.SkyItem) -> str:
    if item.visible:
        return "Visible to the naked eye from here"
    if item.kind in ("galaxy", "nebula", "cluster"):
        return "Too faint to see from here: try binoculars or a darker sky"
    return "Washed out from here: needs a darker sky"


def planets(items: list[sky.SkyItem]) -> list[sky.SkyItem]:
    """Planets worth pointing out, brightest first (Uranus only if it shows to the eye)."""
    out = [i for i in items if i.kind == "planet" and (i.visible or i.name != "Uranus")]
    return sorted(out, key=lambda i: i.mag if i.mag is not None else 9)


def pick_pattern(items: list[sky.SkyItem]) -> sky.SkyItem | None:
    """The most recognisable star pattern that shows from here tonight (list order in
    sky.PATTERNS + asterisms is roughly 'easiest first')."""
    order = {name: k for k, (name, _, _) in enumerate(sky.PATTERNS)}
    order |= {"Summer Triangle": -2, "Winter Triangle": -1}
    pats = [i for i in items if i.kind == "pattern" and i.visible]
    return min(pats, key=lambda i: order.get(i.name, 99)) if pats else None


def pick_showpiece(items: list[sky.SkyItem]) -> sky.SkyItem | None:
    """The best deep-sky showpiece up tonight: a visible one if any (brightest first), else
    the brightest one that's up (with 'too faint from here')."""
    deep = [i for i in items if i.kind in ("cluster", "galaxy", "nebula")]
    if not deep:
        return None
    shown = [i for i in deep if i.visible]
    pool = shown or deep
    return min(pool, key=lambda i: i.vis_mag)


def nearest_darker(
    site: Site, places: list[tuple[Site, float]], min_sqm: float
) -> tuple[Site, float, float] | None:
    """The closest place on the list with a sky at least `min_sqm` dark: (place, km, sqm)."""
    from skytrust.lightpollution import haversine_km

    cands = [
        (p, float(haversine_km(site.lat, site.lon, p.lat, p.lon)), q)
        for p, q in places
        if q >= min_sqm and p.id != site.id
    ]
    return min(cands, key=lambda c: c[1]) if cands else None


def milky_way_text(mw: dict, tz: str) -> tuple[str, str]:
    """(where/when line, help line) for the Milky Way card."""
    band = mw.get("band")
    if band is None:
        return "Below the horizon tonight", ""
    ends = band["ends"]
    through = ", ".join(band["through"][:3])
    arc = (
        f"From the {ends[0]} horizon to the {ends[1]}"
        if len(ends) == 2
        else f"Highest in the {sky.compass(band['top_az'])}"
    )
    when = f"best around {clock(mw['best_utc'], tz)}"
    help_ = f"{arc}, passing through {through}." if through else f"{arc}."
    if mw.get("centre_until") is not None and mw["centre_max_alt"] > 10:
        help_ += (
            f" Its brightest part, toward the centre of the galaxy in Sagittarius, is "
            f"{sky.height_words(mw['centre_max_alt'])} in the "
            f"{sky.compass(mw['centre_az'])} until about {clock(mw['centre_until'], tz)}."
        )
    return cap(when), help_


def fmt_count(n: int) -> str:
    if n >= 1000:
        return f"{round(n, -2):,}"
    if n >= 100:
        return f"{round(n, -1):,}"
    return f"{n}"
