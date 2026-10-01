"""Tonight: the answer first (will it be clear, and a picture of the sky you'll actually see),
four facts, what to look at, the week ahead, then details folded away for people who want them."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from skytrust import sky
from views import charts, lookup, outlook
from views import components as ui
from views.common import (  # noqa: F401  (show/unavailable re-exported for other pages)
    VERDICTS,
    Context,
    agreement_text,
    duration,
    esc,
    moon_advice,
    night_summary,
    short_time,
    show,
    unavailable,
    with_ci,
)

GOES_IMAGE = "https://cdn.star.nesdis.noaa.gov/GOES18/ABI/SECTOR/psw/GEOCOLOR/600x600.jpg"
GOES_LOOP = "https://www.star.nesdis.noaa.gov/GOES/sector.php?sat=G18&sector=psw"

RISK_SETTINGS = {
    "Adventurous: go at 30%+": 0.3,
    "Balanced: go at 50%+": 0.5,
    "Cautious: go at 70%+": 0.7,
}


def in_progress(night, now: pd.Timestamp | None) -> bool:
    return now is not None and night.dusk_utc <= now < night.dawn_utc


def view_moment(night, now: pd.Timestamp | None) -> pd.Timestamp:
    """The moment the hero's sky picture shows: right now if the night is under way, else the
    middle of the clearest stretch (or two hours after dark)."""
    if now is not None and night.dusk_utc <= now < night.dawn_utc:
        return now.floor("10min")
    bw = night.best_window
    t = (
        bw.start_utc + (bw.until_utc - bw.start_utc) / 2
        if bw
        else night.dusk_utc + pd.Timedelta(hours=2)
    )
    return min(max(t, night.dusk_utc), night.dawn_utc).floor("10min")


def hero(ctx: Context, night, guide: dict) -> str:
    tz, pal = ctx.site.timezone, ctx.palette
    headline, advice = VERDICTS[night.verdict]
    when = "Tonight, under way" if in_progress(night, ctx.now_utc) else "Tonight"
    date = pd.Timestamp(night.night_date)
    number = "–" if night.p_usable is None else f"{night.p_usable * 100:.0f}<sup>%</sup>"
    note = ""
    tip = moon_advice(night)
    if tip:
        icon = ui.moon_svg(night.moon_phase_deg, 16)
        note = f'<div class="sk-note">{icon}<span>{esc(tip[1])}</span></div>'

    t = view_moment(night, ctx.now_utc)
    svg, info = ctx.service("chart")(ctx.site, t, ctx.sqm_at(), "here", False, True, True)
    counts = guide["stars_visible"]
    moment = "right now" if in_progress(night, ctx.now_utc) else f"at {lookup.clock(t, tz)}"
    caption = (
        f"Your sky {moment}: about {lookup.fmt_count(info['stars'])} stars "
        f"to the naked eye. A natural sky shows ~{lookup.fmt_count(counts['natural'])}."
    )
    return ui.block(
        '<div class="sk-hero sk-rise">',
        "<div>",
        ui.eyebrow(f"{when} · {ctx.site_label} · {date:%a %b %-d}"),
        f'<div class="sk-display" style="margin-top:14px">{number}</div>',
        '<div class="sk-caption">chance of a clear night (3+ clear dark hours in a row)</div>',
        ui.meter(night.p_usable),
        '<div style="margin-top:22px">',
        ui.status(night.verdict.upper(), pal[night.verdict]),
        "</div>",
        f'<div class="sk-headline">{esc(headline)}</div>',
        f'<div class="sk-lede">{esc(advice)} '
        f"{night_summary(night, tz, ctx.settings.clear_threshold)}</div>",
        note,
        "</div>",
        f'<div class="sk-dome">{svg}<p class="sk-caption">{esc(caption)}</p></div>',
        "</div>",
    )


def fact_strip(ctx: Context, night, guide: dict) -> str:
    tz = ctx.site.timezone
    dark = ui.stat(
        "Dark sky",
        f"{short_time(night.dusk_utc, tz)} – {short_time(night.dawn_utc, tz)}",
        f"{duration(night.dusk_utc, night.dawn_utc)} of full darkness",
    )
    bw = night.best_window
    best = ui.stat(
        "Clearest",
        f"{short_time(bw.start_utc, tz)} – {short_time(bw.until_utc, tz)}"
        if bw
        else "No clear hours",
        f"{duration(bw.start_utc, bw.until_utc)} of clear sky in the typical forecast"
        if bw
        else "No dark hour looks clear in the typical forecast",
    )
    moon = ui.stat(
        "Moon",
        ui.phase_name(night.moon_phase_deg),
        f"{night.moon_illum or 0:.0%} lit · {night.moon_free_hours} moon-free dark hour"
        f"{'' if night.moon_free_hours == 1 else 's'}",
        icon=ui.moon_svg(night.moon_phase_deg, 22),
    )
    d = guide["darkness"]
    light = ui.stat("Light pollution", esc(d.title), esc(sky.brightness_words(d.times_natural)))
    return ui.strip([dark, best, moon, light])


def look_up_cards(ctx: Context, night, guide: dict) -> list[str]:
    """Up to four things to look at tonight, chosen from what's actually up and visible here."""
    tz = ctx.site.timezone
    dusk, dawn = night.dusk_utc, night.dawn_utc
    items = guide["items"]
    cards = []
    for p in lookup.planets(items)[:2]:
        cards.append(
            ui.card(
                "planet",
                "Planet",
                p.name,
                where=f"{esc(lookup.cap(p.where))}, in {esc(sky.constellation_name(p.con))}",
                body=esc(lookup.when_words(p, dusk, dawn, tz))
                + ". "
                + (
                    "Shines steadily: brighter than any star nearby."
                    if p.visible
                    else esc(lookup.visible_words(p)) + "."
                ),
                meta=f"magnitude {p.mag:.1f}",
            )
        )
    mw = guide["milky_way"]
    when, help_ = lookup.milky_way_text(mw, tz)
    if mw["visible"]:
        cards.append(
            ui.card(
                "milkyway",
                "The Milky Way",
                "Look for a pale band",
                where=esc(when),
                body=esc(help_),
                meta=esc(mw["looks"]) + ".",
            )
        )
    else:
        places = [(p, ctx.sqm_at(p)) for p in ctx.sites]
        near = lookup.nearest_darker(ctx.site, places, 20.3)
        go = f"Nearest dark sky on the list: {near[0].name}, {near[1]:.0f} km away." if near else ""
        # dark enough here without the Moon, so tonight it's the Moon's fault
        moonlit = not mw["moon_down"] and ctx.sqm_at() >= sky.MILKY_WAY_MIN_SQM
        reason = "Bright moonlight hides it tonight" if moonlit else "City light hides it"
        cards.append(
            ui.card(
                "milkyway",
                "The Milky Way",
                "Not visible from here",
                where=esc(reason),
                body=esc(go),
            )
        )
    pat = lookup.pick_pattern(items)
    if pat:
        cards.append(
            ui.card(
                "pattern",
                "Star pattern",
                pat.name,
                where=esc(f"{lookup.cap(pat.where)} at {lookup.clock(pat.best_utc, tz)}"),
                body=esc(lookup.cap(pat.note)) + ".",
            )
        )
    show_ = lookup.pick_showpiece(items)
    if show_ and len(cards) < 4:
        cards.append(
            ui.card(
                show_.kind,
                ui_kind(show_.kind),
                show_.name,
                where=esc(f"{lookup.cap(show_.where)} at {lookup.clock(show_.best_utc, tz)}"),
                body=esc(show_.note) + ". " + esc(lookup.visible_words(show_)) + ".",
            )
        )
    return cards[:4]


def ui_kind(kind: str) -> str:
    return lookup.KIND_LABELS.get(kind, lookup.cap(kind))


def next_event_line(ctx: Context) -> str:
    """One line pointing at the next meteor-shower peak (the full events list lives on the
    Events tab; computing it here would slow this page down)."""
    from skytrust.events import next_peaks

    now = ctx.now_utc or pd.Timestamp.now(tz="UTC")
    peaks = next_peaks(now, 45)
    if not peaks:
        return ""
    shower, peak = peaks[0]
    tz = ctx.site.timezone
    days = (peak.tz_convert(tz).date() - now.tz_convert(tz).date()).days
    when = "today" if days <= 0 else "tomorrow" if days == 1 else f"in {days} days"
    return (
        f'<p class="sk-lede" style="margin:6px 0 0">Coming up: the <b>{esc(shower.name)}</b> '
        f"meteor shower peaks {when} ({peak.tz_convert(tz):%a %b %-d}), up to "
        f"{shower.zhr:.0f} an hour in a perfect sky. See the Events tab for when and where.</p>"
    )


def reliability_details(night, site_label: str) -> None:
    """The exact backtest numbers behind the forecast, with confidence intervals."""
    record = night.track_record
    if not record:
        st.caption("No backtest metrics available.")
        return
    if record.get("kind") == "unseen":
        o = record["overall"]
        st.markdown(
            f"{esc(site_label)} wasn't part of SkyTrust's test, so it uses the version of the "
            "model built for new places. Tested at airports it had **never seen** (lead "
            f"{night.lead}), its false-clear rate was {with_ci(o, 'false_clear_rate')} and its "
            f"skill vs the seasonal average {with_ci(o, 'bss', 'num')}."
        )
        return
    site, overall = record.get("site"), record["overall"]
    c1, c2 = st.columns(2)
    c1.metric(
        f"False-clear rate at {site_label}",
        with_ci(site, "false_clear_rate"),
        help="In the held-out 2026 test, of the nights the blend said 'go' at this lead, "
        "the share that turned out not usable. 95% week-block bootstrap CI.",
    )
    c2.metric(
        f"Skill at {site_label}",
        with_ci(site, "bss", "num"),
        help="Brier Skill Score: 0 = no better than the seasonal base rate, 1 = perfect.",
    )
    st.caption(
        f"All sites, lead {night.lead}: false-clear {with_ci(overall, 'false_clear_rate')}, "
        f"skill {with_ci(overall, 'bss', 'num')}."
    )


def risk_panel(ctx: Context, night) -> None:
    """Let the user pick their own go threshold and show what that meant in the backtest."""
    if night.p_usable is None or not ctx.metrics:
        return
    curve = next(
        (
            c["curve"]
            for c in ctx.metrics.get("threshold_curves", [])
            if c["label"] == "primary" and c["lead"] == night.lead
        ),
        None,
    )
    if not curve:
        return
    choice = st.select_slider(
        "How much does a wasted trip bother you?",
        options=list(RISK_SETTINGS),
        value="Balanced: go at 50%+",
        key="risk",
        help="Cautious = only go when it's very likely clear (fewer wasted trips, more missed "
        "nights). Adventurous = the reverse.",
    )
    threshold = RISK_SETTINGS[choice]
    row = min(curve, key=lambda r: abs(r["threshold"] - threshold))
    call = "Go" if night.p_usable >= threshold else "Stay home"
    fcr, miss = row["false_clear_rate"], row["miss_rate"]
    st.markdown(
        f"At this setting tonight's call is **{call}**. In the 2026 test at this lead, "
        f"{'–' if fcr is None else f'{fcr:.0%}'} of 'go' calls turned out cloudy and "
        f"{'–' if miss is None else f'{miss:.0%}'} of good nights would have been skipped."
    )


def beginner_guide(ctx: Context) -> None:
    d = ctx.settings.raw["definitions"]
    st.markdown(
        f"""
- **Chance of a clear night** = the chance of at least **{d["min_run_hours"]} clear hours in a
  row** after the sky is fully dark. It's a real probability: when SkyTrust says 70%, about
  7 nights in 10 like it turned out clear in testing.
- **Dark sky** starts when the Sun is {abs(d["sun_altitude_deg"]):.0f}° below the horizon
  (astronomical dusk). Before that the sky is still glowing.
- **Light pollution** is how much city light brightens your sky. In a city you'll see the Moon,
  planets and the brightest stars; from a dark site, thousands of stars and the Milky Way.
- **The Moon** matters as much as clouds for faint things: a bright Moon washes out the Milky
  Way. The Moon itself, planets and bright stars are fine.
- **Let your eyes adjust**: give them about 20-30 minutes away from screens and lights (use a red
  light if you need one).
"""
    )


def render(ctx: Context) -> None:
    if unavailable(ctx):
        return
    fc, tz, pal = ctx.forecast, ctx.site.timezone, ctx.palette
    night = ctx.nights[0]
    guide = ctx.service("sky")(ctx.site, night.dusk_utc, night.dawn_utc, ctx.sqm_at())
    st.markdown(hero(ctx, night, guide), unsafe_allow_html=True)
    if night.models_missing:
        st.info(
            f"Computed without {', '.join(m.upper() for m in night.models_missing)} "
            f"(no data right now); using {', '.join(m.upper() for m in night.models_used)}."
        )
    st.markdown(fact_strip(ctx, night, guide), unsafe_allow_html=True)

    st.markdown(
        ui.section(
            "Look up tonight",
            "Picked from what's above the horizon tonight and bright enough to see from here.",
            "What to see",
        ),
        unsafe_allow_html=True,
    )
    cards = look_up_cards(ctx, night, guide)
    st.markdown(
        ui.grid(cards, n=min(4, max(2, len(cards)))) + next_event_line(ctx), unsafe_allow_html=True
    )

    st.markdown(
        ui.section(
            "The week ahead",
            "Chance of a clear night, and the Moon. The dots show "
            "how well SkyTrust did that many days ahead in testing.",
            "7 nights",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        ui.rows([outlook.night_row(ctx, n, i) for i, n in enumerate(ctx.nights)]),
        unsafe_allow_html=True,
    )

    st.markdown(ui.section("Details", "", "For the curious"), unsafe_allow_html=True)
    with st.expander("Hour by hour tonight", icon=":material/schedule:"):
        go_at, maybe_at = ctx.settings.raw["verdict"]["go"], ctx.settings.raw["verdict"]["maybe"]
        show(charts.night_chart(fc.hourly, night, tz, pal, go_at, maybe_at))
        mid = (go_at + maybe_at) / 2
        st.markdown(
            ui.legend(
                [
                    (f"likely clear (≥ {go_at:.0%})", charts.ink_level(1.0, pal, go_at, maybe_at)),
                    ("could go either way", charts.ink_level(mid, pal, go_at, maybe_at)),
                    (
                        f"likely cloudy (< {maybe_at:.0%})",
                        charts.ink_level(0.0, pal, go_at, maybe_at),
                    ),
                    ("full darkness (shaded)", pal["dark"]),
                    ("Moon above the horizon", pal["moon"]),
                ]
            ),
            unsafe_allow_html=True,
        )
    with st.expander("Should I go? Set your own threshold", icon=":material/tune:"):
        risk_panel(ctx, night)
    with st.expander("What the weather models say", icon=":material/stacked_line_chart:"):
        spread = f" (spread {night.spread:.2f})" if night.spread is not None else ""
        st.markdown(f"**{night.agreement}**{spread}. {agreement_text(night)[1]}")
        show(charts.hourly_cloud(fc.hourly, night, tz, ctx.settings.clear_threshold, pal))
        show(charts.cloud_layers(fc.hourly, night, tz, pal))
        st.markdown(
            f"[See the latest satellite loop]({GOES_LOOP}) (NOAA GOES-West) to watch "
            "which way the clouds are moving."
        )
    with st.expander("How reliable is this forecast here?", icon=":material/verified:"):
        reliability_details(night, ctx.site_label)
    with st.expander("New to stargazing? Start here", icon=":material/school:"):
        beginner_guide(ctx)
    st.caption(
        f"Forecast fetched {fc.fetched_at_utc.tz_convert(tz):%b %-d, %-I:%M %p} ({fc.source}). "
        "Tonight's live forecast is fresher than the 1-day-ahead data the model was tested on, so "
        "this probability is slightly conservative."
    )
