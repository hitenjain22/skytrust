"""Events: what's coming up in the California sky, when and where to look from the chosen place,
and for meteor showers, which of the app's places will show the most (with tonight's weather
odds when the peak is within the 7-night forecast)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from skytrust.events import Event
from views import components as ui
from views import lookup
from views.common import Context, esc, upcoming

FILTERS = {
    "All": None,
    "Meteor showers": {"meteor"},
    "Moon & planets": {"moon", "planet", "pairing"},
    "Eclipses & seasons": {"eclipse", "season"},
}
OUTLOOK_COLORS = {
    "Great": "var(--sk-go)",
    "Good": "var(--sk-go)",
    "Fair": "var(--sk-maybe)",
    "Weak": "var(--sk-skip)",
    "Not visible": "var(--sk-skip)",
}

# Viewing advice, quoted from the organisations that publish these events.
TIPS = [
    ("Get away from lights", "“Find an area well away from the city or street lights.”", "NASA"),
    (
        "Give your eyes time",
        "“In less than 30 minutes in the dark, your eyes will adapt and you "
        "will begin to see meteors.”",
        "NASA",
    ),
    (
        "Look away from the radiant",
        "The best views are “45 to 90 degrees away from the radiant”, "
        "where meteors “appear longer and more spectacular.”",
        "NASA",
    ),
    (
        "Lie back",
        "“Lie flat on your back … and look up.” Bring “a sleeping bag, blanket, or lawn chair.”",
        "NASA",
    ),
    (
        "What the numbers mean",
        "A shower's ZHR is the rate an ideal observer would see “in "
        "perfectly clear skies” with the radiant overhead; real rates are lower.",
        "IMO",
    ),
]


def best_moment(e: Event) -> pd.Timestamp | None:
    if e.best_start is None:
        return None
    return e.best_start + (e.best_end - e.best_start) / 2


def viewing_date(e: Event, tz: str) -> pd.Timestamp:
    """The local date to show. Night events are dated by the evening the night starts (a peak at
    1 AM on Sunday belongs to Saturday night), as NASA and the IMO write them."""
    if e.kind == "meteor":
        return pd.Timestamp(e.details["here"]["night"])
    t = (best_moment(e) or e.utc).tz_convert(tz)
    if e.best_start is not None and t.hour < 12:
        t = t - pd.Timedelta(days=1)
    return t.normalize().tz_localize(None)


def when_line(e: Event, day: pd.Timestamp, ctx: Context) -> str:
    tz = ctx.site.timezone
    if e.kind == "meteor":
        return (
            f"Night of {day:%a %b %-d} · best {lookup.clock(e.best_start, tz)}–"
            f"{lookup.clock(e.best_end, tz)} from {esc(ctx.site_label)}"
        )
    mid = best_moment(e)
    if mid is None:
        return f"{e.utc.tz_convert(tz):%a %b %-d, %-I:%M %p}"
    local = mid.tz_convert(tz)
    if local.hour < 12:
        return f"Night of {day:%a %b %-d} · best around {lookup.clock(mid, tz)}"
    return f"{local:%a %b %-d} · best around {lookup.clock(mid, tz)}"


def forecast_odds(ctx: Context, place_id: str, night_date) -> float | None:
    """P(clear night) for a place and night if it's within the live 7-night forecast."""
    if ctx.forecast_for is None:
        return None
    try:
        fc, _ = ctx.forecast_for(place_id)
    except Exception:  # weather is a bonus on this page
        return None
    for n in upcoming(fc, ctx.now_utc):
        if n.night_date == night_date:
            return n.p_usable
    return None


def place_chips(ctx: Context, e: Event, in_range: bool) -> str:
    chips = []
    for p in e.places[:4]:
        odds = forecast_odds(ctx, p["site"].id, p["night"]) if in_range else None
        weather = f" · {odds:.0%} clear" if odds is not None else ""
        rate = "<1" if p["rate"] < 1 else f"{p['rate']:.0f}"
        chips.append(
            f'<span class="sk-chip"><b>{esc(p["site"].name)}</b> · ~{rate}/hr{weather}</span>'
        )
    title = '<div class="sk-eyebrow" style="margin-top:12px">Best places in California</div>'
    return f'{title}<div class="sk-places">{"".join(chips)}</div>'


def event_card(ctx: Context, e: Event) -> str:
    tz = ctx.site.timezone
    day = viewing_date(e, tz)
    now = ctx.now_utc or pd.Timestamp.now(tz="UTC")
    badge = ui.badge(e.outlook, OUTLOOK_COLORS.get(e.outlook)) if e.outlook else ""
    when = when_line(e, day, ctx)
    look = f"<p>{esc(e.look)}</p>" if e.look else ""
    extra = ""
    if e.kind == "meteor":
        in_range = 0 <= (day.date() - now.tz_convert(tz).date()).days <= 6
        extra = place_chips(ctx, e, in_range)
        notes = e.details["shower"].notes
        if notes:
            extra += f'<p class="sk-muted" style="font-size:.84rem">{esc(notes)}</p>'
        if e.active and e.active[0] <= now <= e.active[1]:
            badge += " " + ui.badge("Active now", "var(--sk-accent)")
    return ui.block(
        '<div class="sk-event">',
        f'<div class="sk-date"><div class="d">{day:%-d}</div><div class="m">{day:%b}</div>'
        f'<div class="w">{day:%a}</div></div>',
        "<div>",
        '<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">',
        f"{ui.eyebrow(kind_label(e))}{badge}</div>",
        f"<h4>{esc(e.title)}</h4>",
        f"<p>{esc(e.summary)}</p>",
        look,
        f'<div class="sk-when">{when}</div>',
        extra,
        "</div></div>",
    )


def kind_label(e: Event) -> str:
    return {
        "meteor": "Meteor shower",
        "moon": "Moon",
        "planet": "Planet",
        "pairing": "Close pairing",
        "eclipse": "Eclipse",
        "season": "Season",
    }.get(e.kind, e.kind)


def happening_now(ctx: Context, events: list[Event]) -> str:
    now = ctx.now_utc or pd.Timestamp.now(tz="UTC")
    tz = ctx.site.timezone
    active = [
        e for e in events if e.kind == "meteor" and e.active and e.active[0] <= now <= e.active[1]
    ]
    if not active:
        return ""
    bits = []
    for e in active:
        days = (e.utc.tz_convert(tz).date() - now.tz_convert(tz).date()).days
        peak = (
            "peaks tonight"
            if days == 0
            else f"peaks in {days} days"
            if days > 0
            else "past its peak"
        )
        bits.append(f"<b>{esc(e.details['shower'].name)}</b> ({peak})")
    return (
        f'<div class="sk-card" style="margin:8px 0 4px"><div class="sk-eyebrow">Happening now</div>'
        f'<p style="margin-top:8px">Active meteor showers: {", ".join(bits)}. Any dark, clear '
        "night this week can bring a few.</p></div>"
    )


def render(ctx: Context) -> None:
    tz = ctx.site.timezone
    st.markdown(
        ui.section(
            "Sky events",
            "Meteor showers, the Moon, planets at their best and eclipses over the next four "
            f"months, with times and directions for {esc(ctx.site_label)}.",
            "Coming up",
        ),
        unsafe_allow_html=True,
    )
    events = ctx.service("events")(ctx.site, ctx.sqm_at())
    pick = st.segmented_control(
        "Show", list(FILTERS), default="All", key="ev_filter", label_visibility="collapsed"
    )
    kinds = FILTERS[pick or "All"]
    st.markdown(happening_now(ctx, events), unsafe_allow_html=True)
    now = ctx.now_utc or pd.Timestamp.now(tz="UTC")
    shown = [
        e
        for e in events
        if (kinds is None or e.kind in kinds)
        and (e.utc >= now - pd.Timedelta(hours=12) or e.kind == "meteor")
    ]
    shown = [e for e in shown if not (e.kind == "meteor" and e.active and e.active[1] < now)]
    if not shown:
        st.info("Nothing of this kind in the next four months.")
    html, month = "", None
    for e in shown:
        day = viewing_date(e, tz)
        if (day.year, day.month) != month:
            month = (day.year, day.month)
            html += f'<div class="sk-month">{day:%B %Y}</div>'
        html += event_card(ctx, e)
    st.markdown(html, unsafe_allow_html=True)

    st.markdown(
        ui.section(
            "How to watch", "Advice from the organisations that track these events.", "Tips"
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        ui.grid([ui.card("meteor", src, title, body=esc(text)) for title, text, src in TIPS], n=3),
        unsafe_allow_html=True,
    )
    with st.expander("Where these events come from", icon=":material/info:"):
        st.markdown(
            """
- **Meteor showers**: the International Meteor Organization's *2026 Meteor Shower Calendar*
  (J. Rendtel, ed.), its working list of showers: activity dates, peak (as a solar longitude,
  which repeats every year), radiant, speed and rate. Expected rates use the IMO's formula
  (rate = ZHR × sin(radiant height) ÷ r^(6.5 − limiting magnitude)) with the light pollution and
  moonlight at each place. Notes on parent comets are from the IMO calendar and NASA.
- **Moon phases, oppositions, elongations, pairings, eclipses, seasons**: computed from JPL's
  DE421 ephemeris with Skyfield and checked against published dates (the IMO's 2026 lunar phase
  table, Saturn's 2026 opposition, NASA's 2026 lunar eclipse figures).
- The outlook (Great / Good / Fair / Weak) is the expected rate at the darkest place on the
  list: 30+ an hour is great, 12+ good, 5+ fair.
"""
        )
