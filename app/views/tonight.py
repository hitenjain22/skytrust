"""Tonight: the answer first (will it be clear, and a picture of the sky you'll actually see),
four facts, what to look at, the week ahead, then details folded away for people who want them."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from skytrust import haze, sky
from views import charts, lookup, outlook
from views import components as ui
from views.common import (  # noqa: F401  (show/unavailable re-exported for other pages)
    CLEAR_P,
    CLOUDY_P,
    VERDICTS,
    Context,
    agreement_text,
    clearest,
    day_label,
    duration,
    esc,
    held_with_ci,
    hour_chances,
    longest_run,
    moon_sentence,
    moon_up_in_dark,
    short_time,
    show,
    station_label,
    unavailable,
    with_ci,
)

GOES_IMAGE = "https://cdn.star.nesdis.noaa.gov/GOES18/ABI/SECTOR/psw/GEOCOLOR/600x600.jpg"
GOES_LOOP = "https://www.star.nesdis.noaa.gov/GOES/sector.php?sat=G18&sector=psw"

# The weather models, by who makes them (a beginner knows "the European model", not "IFS").
MODEL_NAMES = {
    "gfs": "US (GFS)",
    "hrrr": "US short-range (HRRR)",
    "ecmwf": "European (ECMWF)",
    "gem": "Canadian (GEM)",
    "icon": "German (ICON)",
}


def in_progress(night, now: pd.Timestamp | None) -> bool:
    return now is not None and night.dusk_utc <= now < night.dawn_utc


def view_moment(night, now: pd.Timestamp | None) -> pd.Timestamp:
    """The moment the hero's sky picture shows: right now if the night is under way, else the
    middle of the clearest stretch (or two hours after dark)."""
    if now is not None and night.dusk_utc <= now < night.dawn_utc:
        return now.floor("10min")
    span = clearest(night)
    t = span[0] + (span[1] - span[0]) / 2 if span else night.dusk_utc + pd.Timedelta(hours=2)
    return min(max(t, night.dusk_utc), night.dawn_utc).floor("10min")


def story(night, tz: str) -> str:
    """What the sky does tonight, in one plain sentence ("Clear until about 11 PM, then clouds
    move in."), from the chance of clear sky hour by hour."""
    p = hour_chances(night)
    if p.empty:
        return ""
    kind = ["clear" if v >= CLEAR_P else "cloudy" if v < CLOUDY_P else "mixed" for v in p]
    runs: list[list] = []
    for when, k in zip(p.index, kind, strict=True):
        if runs and runs[-1][0] == k:
            runs[-1][2] = when
        else:
            runs.append([k, when, when])
    runs = [r for r in runs if r[2] > r[1] or len(runs) == 1]  # ignore one-hour blips
    if not runs:
        return ""
    first, last = runs[0][0], runs[-1][0]
    if len(runs) == 1:
        return {"clear": "Clear skies all night.", "cloudy": "Cloudy all night.",
                "mixed": "Patchy cloud all night: clear spells are possible."}[first]  # fmt: skip
    # the sky changes between two hourly readings: say the half hour in between, the same
    # boundary the "Clearest" stretch uses
    change = lookup.clock(runs[1][1] - pd.Timedelta(minutes=30), tz)
    if first == "clear" and last in ("cloudy", "mixed"):
        return f"Clear until about {change}, then clouds move in."
    if first in ("cloudy", "mixed") and last == "clear":
        return f"Cloudy at first, clearing around {change}."
    if first == "mixed" and last == "cloudy":
        return f"Patchy cloud at first, then cloudy from about {change}."
    return "On and off: some clear spells, some cloud."


def next_better_night(ctx: Context) -> str:
    """'Saturday looks better (90%).' when a later night this week is much clearer."""
    nights = ctx.nights
    if len(nights) < 2 or nights[0].p_usable is None:
        return ""
    later = [n for n in nights[1:] if n.p_usable is not None]
    if not later:
        return ""
    best = max(later, key=lambda n: n.p_usable)
    if best.p_usable < max(0.6, nights[0].p_usable + 0.25):
        return ""
    i = nights.index(best)
    day = day_label(best.night_date, i, ctx.now_utc, ctx.site.timezone)
    return f"{day} looks better ({best.p_usable:.0%})."


def clear_moonless(night) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """The longest stretch (2 hours or more) of dark hours that are likely clear with the Moon
    down: the window for the Milky Way and faint objects. A thin Moon (< 15% lit) counts as
    down. Each hour stands for the half hour either side of it (as in the cloud labels)."""
    p = hour_chances(night)
    if p.empty:
        return None
    thin = (night.moon_illum or 0.0) < 0.15
    runs_up = [] if thin else moon_up_in_dark(night)
    half = pd.Timedelta(minutes=30)
    good = [
        t
        for t, v in p.items()
        if v >= CLEAR_P and not any(a < t + half and t - half < b for a, b in runs_up)
    ]
    span = longest_run(good, night)
    if span is None or span[1] - span[0] < pd.Timedelta(hours=2):
        return None
    return span


def heads_up(ctx: Context, night, guide: dict, aod: float | None) -> list[tuple[str, str]]:
    """Short notes under the headline, only when they change the plan: the Moon (and when the
    sky is both clear and moonless), smoke or haze, and the early evening before full dark.
    Returns (icon html, text) pairs."""
    tz = ctx.site.timezone
    notes = []
    illum = night.moon_illum or 0.0
    runs = moon_up_in_dark(night)
    up_min = sum((b - a).total_seconds() / 60 for a, b in runs)
    dark_min = (night.dawn_utc - night.dusk_utc).total_seconds() / 60
    milky_way_here = ctx.sqm_at() >= sky.MILKY_WAY_MIN_SQM
    moon_icon = ui.moon_svg(night.moon_phase_deg, 16)
    window = clear_moonless(night) if night.verdict != "Skip" else None
    if window and ctx.now_utc is not None and ctx.now_utc >= window[1]:
        window = None  # already over tonight
    span = f"{short_time(window[0], tz)} – {short_time(window[1], tz)}" if window else ""
    if illum >= 0.15 and runs:
        bright = illum >= 0.5 and up_min >= 0.5 * dark_min
        if window and milky_way_here:
            notes.append((moon_icon, f"Best for the Milky Way: {span}, clear with the Moon down."))
        elif bright:
            text = ("Bright Moon most of the night: the Milky Way and faint galaxies are washed "
                    "out. Planets and the Moon are fine.")  # fmt: skip
            notes.append((moon_icon, text))
    elif milky_way_here and night.verdict != "Skip":
        notes.append((moon_icon, "No Moon in the dark hours: the best kind of night for the "
                                 "Milky Way."))  # fmt: skip
    w = haze.words(aod)
    if w:
        notes.append((ui.icon("haze", 16), f"{w[0]} in the forecast: {w[1]}."))
    evening = early_evening(ctx, night)
    if evening:
        notes.append((ui.icon("dusk", 16), evening))
    return notes


def early_evening(ctx: Context, night) -> str:
    """'Early evening (7:05 – 8:20 PM): 4 of 5 forecasts say clear.' Before full darkness the
    first stars and planets are already out; the tested hourly model covers dark hours only,
    so this just counts the weather forecasts that show a clear sky (mostly <= 20% cloud)."""
    if ctx.now_utc is not None and ctx.now_utc >= night.dusk_utc:
        return ""
    svc = (ctx.services or {}).get("twilight")
    if svc is None or ctx.forecast is None:
        return ""
    first_stars, _ = svc(ctx.site, night.dusk_utc, night.dawn_utc)
    if night.dusk_utc - first_stars < pd.Timedelta(minutes=30):
        return ""
    h = ctx.forecast.hourly
    h = h[(h["time"] >= first_stars.floor("h")) & (h["time"] < night.dusk_utc)]
    cover = h.groupby("model")["cloud_cover"].mean().dropna()
    if cover.empty:
        return ""
    clear = int((cover <= ctx.settings.clear_threshold).sum())
    tz = ctx.site.timezone
    return (f"Early evening ({short_time(first_stars, tz)} – {short_time(night.dusk_utc, tz)}): "
            f"{clear} of {len(cover)} forecasts say clear.")  # fmt: skip


def hero_lede(ctx: Context, night) -> str:
    """The plain-words summary under the headline: what happens tonight, and what to do."""
    tz = ctx.site.timezone
    parts = [story(night, tz)]
    span = clearest(night)
    now = ctx.now_utc
    if night.verdict == "Go":
        late_start = span and span[0] > night.dusk_utc + pd.Timedelta(minutes=45)
        if span and now is not None and span[0] <= now < span[1]:
            parts.append("A good time to be out now.")
        elif late_start and (now is None or now < span[0]):
            when = short_time(span[0], tz)
            if when not in parts[0]:  # the story may already say when it clears
                parts.append(f"Head out for {when}.")
        elif not in_progress(night, now):
            parts.append(f"It's fully dark from {short_time(night.dusk_utc, tz)}.")
    elif night.verdict == "Maybe":
        parts.append("Check again closer to dark.")
    else:
        parts.append(next_better_night(ctx) or "A good night to plan rather than drive out.")
    return " ".join(x for x in parts if x)


def headline_for(night, aod: float | None) -> str:
    """The verdict in words; a clear night with real smoke or haze says so (the chance is about
    clouds only, but a smoky clear night is not the night a beginner expects)."""
    headline, _ = VERDICTS[night.verdict]
    w = haze.words(aod)
    if night.verdict == "Go" and w and w[0] != "Some haze":
        return "Clear but hazy tonight"
    return headline


def hero(ctx: Context, night, guide: dict, aod: float | None = None) -> str:
    tz, pal = ctx.site.timezone, ctx.palette
    headline = headline_for(night, aod)
    when = "Tonight, under way" if in_progress(night, ctx.now_utc) else "Tonight"
    date = pd.Timestamp(night.night_date)
    number = "–" if night.p_usable is None else f"{night.p_usable * 100:.0f}<sup>%</sup>"
    note = "".join(f'<div class="sk-note">{icon}<span>{esc(text)}</span></div>'
                   for icon, text in heads_up(ctx, night, guide, aod))  # fmt: skip

    t = view_moment(night, ctx.now_utc)
    svg, info = ctx.service("chart")(ctx.site, t, ctx.sqm_at(), "here", False, True, True, aod)
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
        '<div class="sk-caption">chance of at least 3 clear hours in a row after dark</div>',
        f'<div class="sk-headline" style="margin-top:22px">'
        f'<span class="sk-dot" style="--c:{pal[night.verdict]}"></span>{esc(headline)}</div>',
        f'<div class="sk-lede">{esc(hero_lede(ctx, night))}</div>',
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
    span = clearest(night)
    best = ui.stat(
        "Clearest",
        f"{short_time(span[0], tz)} – {short_time(span[1], tz)}" if span else "No clear stretch",
        f"{duration(*span)} likely clear" if span else "No hour looks likely to be clear",
    )
    moon = ui.stat(
        "Moon",
        ui.phase_name(night.moon_phase_deg),
        f"{night.moon_illum or 0:.0%} lit · {moon_when(night, tz)}",
        icon=ui.moon_svg(night.moon_phase_deg, 22),
    )
    d = guide["darkness"]
    light = ui.stat("Light pollution", esc(d.title), esc(sky.brightness_words(d.times_natural)))
    return ui.strip([dark, best, moon, light])


def moon_when(night, tz: str) -> str:
    """'rises 10:17 PM' / 'sets 1:05 AM' / 'up all night' / 'down all night' (darkness only)."""
    sentence = moon_sentence(night, tz)  # "The Moon (64% lit) is rising at 10:17 PM."
    if "below the horizon" in sentence:
        return "down all night"
    if "up all night" in sentence:
        return "up all night"
    if "rising at" in sentence:
        return "rises " + sentence.split("rising at ")[1].rstrip(".")
    if "up until" in sentence:
        return "sets " + sentence.split("up until ")[1].rstrip(".")
    return sentence.split(" is ")[1].rstrip(".")  # "up from 9 PM to 2 AM"


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
            )
        )
    mw = guide["milky_way"]
    when, help_ = lookup.milky_way_text(mw, tz)
    if mw["visible"]:
        route = help_.split(". ")[0].rstrip(".") + "." if help_ else ""  # the full text: Sky Guide
        dark_enough = mw["zenith_sqm"] >= ctx.settings.raw["light_pollution"]["dark_sqm"]
        cards.append(
            ui.card(
                "milkyway",
                "The Milky Way",
                "Look for a pale band" if dark_enough else "Faint from here",
                where=esc(when),
                body=esc(route),
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
        f"meteor shower peaks {when} ({peak.tz_convert(tz):%a %b %-d}). Details in Events.</p>"
    )


def days_ahead(lead: int) -> str:
    return "1 day ahead" if lead == 1 else f"{lead} days ahead"


def reliability_details(night, site_label: str) -> None:
    """The exact backtest numbers behind the forecast, with confidence intervals."""
    record = night.track_record
    if not record:
        st.caption("No backtest metrics available.")
        return
    if record.get("kind") == "statewide":
        o = record["overall"]
        lines = [
            f"- **{esc(station_label(s))}**, {s['distance_km']:.0f} km away: “go” calls that held "
            f"{held_with_ci(s)}, skill {with_ci(s, 'bss', 'num')} ({s['n']} nights)"
            for s in record.get("near", [])
        ]
        st.markdown(
            f"{esc(site_label)} has no weather station in SkyTrust's test, so it uses the version "
            "of the model built for any place in California. How it did at the stations nearest "
            "you, each scored as a place it had **never seen** (2026, "
            f"{days_ahead(night.lead)}):\n\n" + "\n".join(lines) + "\n\n"
            f"Across all {record['n_stations']} stations: “go” calls that held {held_with_ci(o)}; "
            f"skill {with_ci(o, 'bss', 'num')} (0 = no better than the season's average, "
            "1 = perfect)."
        )
        return
    if record.get("kind") == "unseen":
        o = record["overall"]
        bss = o.get("bss")
        smaller = f"{bss:.0%} smaller" if bss is not None and bss == bss and bss > 0 else "–"
        st.markdown(
            f"{esc(site_label)} wasn't part of SkyTrust's test, so it uses the version of the "
            "model built for new places. At airports it had **never seen** "
            f"({days_ahead(night.lead)}): “go” calls held {held_with_ci(o)} of the time, and its "
            f"errors were {smaller} than guessing from the season "
            f"(skill {with_ci(o, 'bss', 'num')})."
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
        f"All sites, {days_ahead(night.lead)}: false-clear "
        f"{with_ci(overall, 'false_clear_rate')}, skill {with_ci(overall, 'bss', 'num')}."
    )


def risk_panel(ctx: Context, night) -> None:
    """Pick your own bar for "go" (0-100%) and see tonight's call, and what that bar meant in
    the 2026 test: how many trips would have been clouded out, how many good nights missed."""
    if night.p_usable is None:
        return
    st.caption(
        "Lower = go more often (more wasted trips). Higher = only go when it's very likely clear "
        "(more good nights missed)."
    )
    bar = st.slider(
        "Go when the chance of a clear night is at least",
        min_value=0,
        max_value=100,
        value=50,
        step=5,
        format="%d%%",
        key="risk",
    )
    p = night.p_usable
    call = "go" if p * 100 >= bar else "stay home"
    lines = [f"Tonight is **{p:.0%}**, so at {bar}% the call is: **{call}**."]
    curve = next(
        (c["curve"] for c in (ctx.metrics or {}).get("threshold_curves", [])
         if c["label"] == "primary" and c["lead"] == night.lead),
        None,
    )  # fmt: skip
    if bar == 0:
        lines.append("You'd go out every night: never miss a good one, but every cloudy night "
                     "is a wasted trip.")  # fmt: skip
    elif bar == 100:
        lines.append("You'd never go: no wasted trips, but you'd miss every good night.")
    elif curve:
        row = min(curve, key=lambda r: abs(r["threshold"] - bar / 100))
        fcr, miss = row["false_clear_rate"], row["miss_rate"]
        if fcr is not None and miss is not None:
            lines.append(
                f"In testing (2026), at this setting about **{round(fcr * 10)} in 10** trips "
                f"would have been clouded out, and you'd have skipped **{round(miss * 10)} in "
                "10** good nights."
            )
    st.markdown(" ".join(lines))


def forecast_agreement(fc, night, tz: str, threshold: float) -> str:
    """One plain sentence on how many of the weather forecasts expect a clear sky, hour by
    hour: 'All 5 forecasts expect clear sky until 11 PM; after that they split (2 say clear).'"""
    data = fc.hourly[(fc.hourly["time"] >= night.dusk_utc.floor("h"))
                     & (fc.hourly["time"] <= night.dawn_utc)]  # fmt: skip
    if data.empty:
        return ""
    cover = data.pivot_table(index="time", columns="model", values="cloud_cover")
    n = cover.notna().sum(axis=1)
    clear = (cover <= threshold).sum(axis=1)
    total = int(n.max())
    if (clear == n).all():
        return f"All {total} forecasts expect a clear sky all night."
    if (clear == 0).all():
        return f"All {total} forecasts expect clouds all night."
    agree = (clear == n) | (clear == 0)
    if agree.all():
        flip = clear.index[(clear > 0).to_numpy() != (clear.iloc[0] > 0)][0]
        a, b = ("clear sky", "clouds") if clear.iloc[0] > 0 else ("clouds", "clear sky")
        return (f"All {total} forecasts expect {a} until about {lookup.clock(flip, tz)}, then "
                f"{b}.")  # fmt: skip
    split = clear.index[~agree.to_numpy()][0]
    mid = int(round(clear[~agree].median()))
    if agree.iloc[0]:
        return (f"All {total} agree until about {lookup.clock(split, tz)}; after that only about "
                f"{mid} of {total} say clear.")  # fmt: skip
    return f"They disagree: about {mid} of {total} say clear through the night."


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
    aod = ctx.haze_aod(night.dusk_utc, night.dawn_utc)
    guide = ctx.service("sky")(ctx.site, night.dusk_utc, night.dawn_utc, ctx.sqm_at(), aod)
    st.markdown(hero(ctx, night, guide, aod), unsafe_allow_html=True)
    if night.models_missing:
        st.info(
            f"Computed without {', '.join(m.upper() for m in night.models_missing)} "
            f"(no data right now); using {', '.join(m.upper() for m in night.models_used)}."
        )
    st.markdown(fact_strip(ctx, night, guide), unsafe_allow_html=True)

    st.markdown(
        ui.section(
            "Look up tonight",
            "What's up tonight and bright enough to see from here.",
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
            "The chance of a clear night, night by night. The dots show how far to trust each "
            "number (●●● high, ●○○ low: forecasts blur further ahead).",
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
        st.caption(
            "Bars: the chance each hour is clear. Shaded: full darkness. Strip below: the Moon "
            "is up (its light hides faint stars)."
        )
        show(charts.night_chart(fc.hourly, night, tz, pal, CLEAR_P, CLOUDY_P))
        st.markdown(
            ui.legend(
                [
                    ("likely clear", pal["Go"]),
                    ("could go either way", pal["Maybe"]),
                    ("likely cloudy", pal["Skip"]),
                    ("Moon up", pal["moon"]),
                ]
            ),
            unsafe_allow_html=True,
        )
    with st.expander("Should I go? Set your own bar", icon=":material/tune:"):
        risk_panel(ctx, night)
    with st.expander("Do the weather forecasts agree?", icon=":material/stacked_line_chart:"):
        st.markdown(
            "SkyTrust combines five weather forecasts from different agencies. When they agree, "
            "trust the forecast more. "
            + forecast_agreement(fc, night, tz, ctx.settings.clear_threshold)
        )
        show(charts.model_grid(fc.hourly, night, tz, pal, MODEL_NAMES))
        st.caption(
            "Each square: one forecast's cloud cover (%) for that hour, dark = clear. "
            f"[Satellite picture now]({GOES_LOOP}) (NOAA)."
        )
    with st.expander("How reliable is this forecast here?", icon=":material/verified:"):
        reliability_details(night, ctx.site_label)
    with st.expander("New to stargazing? Start here", icon=":material/school:"):
        beginner_guide(ctx)
    st.caption(
        f"Forecast fetched {fc.fetched_at_utc.tz_convert(tz):%b %-d, %-I:%M %p} ({fc.source}). "
        "Tested on day-old forecasts; tonight's fresher one is, if anything, a little better."
    )
