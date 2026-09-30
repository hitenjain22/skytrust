"""Page 1: tonight. Answer first (verdict + probability + why), then the four facts that matter,
then hour by hour, then the details for people who want them."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from views import charts
from views import components as ui
from views.common import (
    VERDICTS,
    Context,
    agreement_text,
    duration,
    esc,
    moon_advice,
    night_summary,
    short_time,
    with_ci,
)

GOES_IMAGE = "https://cdn.star.nesdis.noaa.gov/GOES18/ABI/SECTOR/psw/GEOCOLOR/600x600.jpg"
GOES_LOOP = "https://www.star.nesdis.noaa.gov/GOES/sector.php?sat=G18&sector=psw"

RISK_SETTINGS = {
    "Adventurous: go at 30%+": 0.3,
    "Balanced: go at 50%+": 0.5,
    "Cautious: go at 70%+": 0.7,
}

PLOT_CONFIG = {"displayModeBar": False, "responsive": True}


def show(fig) -> None:
    st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)


def unavailable(ctx: Context) -> bool:
    """Show a friendly message (and stop) when there's no forecast at all."""
    if ctx.forecast is None or not ctx.nights:
        reason = f" ({ctx.forecast_error})" if ctx.forecast_error else ""
        st.warning(
            f"Live forecasts are unavailable right now and there's no saved copy yet.{reason} "
            "The Track Record and How It Works pages still work."
        )
        return True
    if ctx.forecast.warning:
        st.warning(ctx.forecast.warning)
    return False


def in_progress(night, now: pd.Timestamp | None) -> bool:
    return now is not None and night.dusk_utc <= now < night.dawn_utc


def hero(ctx: Context, night) -> str:
    tz, pal = ctx.site.timezone, ctx.palette
    headline, advice = VERDICTS[night.verdict]
    color = pal[night.verdict]
    title = "Tonight (in progress)" if in_progress(night, ctx.now_utc) else "Tonight"
    note = ""
    tip = moon_advice(night)
    if tip:
        note = f'<div class="sk-note">{ui.moon_svg(night.moon_phase_deg, 16)} {esc(tip[1])}</div>'
    return ui.card(
        ui.block(
            '<div class="sk-hero"><div>',
            ui.ring(night.p_usable, color, "chance of a usable night"),
            "</div><div>",
            f'<div class="sk-eyebrow">{title} · {esc(ctx.site_label)}</div>',
            ui.pill(night.verdict.upper(), color),
            f'<div class="sk-headline">{esc(headline)}</div>',
            f'<div class="sk-lede">{esc(advice)} '
            f"{night_summary(night, tz, ctx.settings.clear_threshold)}</div>",
            note,
            "</div></div>",
        )
    )


def fact_tiles(ctx: Context, night) -> str:
    tz = ctx.site.timezone
    dark = ui.tile(
        "Darkness",
        f"{short_time(night.dusk_utc, tz)} – {short_time(night.dawn_utc, tz)}",
        f"{duration(night.dusk_utc, night.dawn_utc)} of true dark, with the Sun more than "
        f"{abs(ctx.settings.raw['definitions']['sun_altitude_deg']):.0f}° below the horizon.",
    )
    bw = night.best_window
    best = ui.tile(
        "Best clear window",
        f"{short_time(bw.start_utc, tz)} – {short_time(bw.until_utc, tz)}" if bw else "None",
        f"{duration(bw.start_utc, bw.until_utc)} where the typical model shows clear sky."
        if bw
        else "No dark hour is clear in the typical model forecast.",
    )
    moon = ui.tile(
        "Moon",
        f"{ui.phase_name(night.moon_phase_deg)}",
        f"{night.moon_illum or 0:.0%} lit · {night.moon_free_hours} moon-free dark hour"
        f"{'' if night.moon_free_hours == 1 else 's'}",
        icon=ui.moon_svg(night.moon_phase_deg, 40),
    )
    record = night.track_record or {}
    site_rec = record.get("site") if record.get("kind") != "unseen" else None
    rec = site_rec or record.get("overall")
    if rec and rec.get("false_clear_rate") is not None:
        right = 1 - rec["false_clear_rate"]
        where = "here" if site_rec else "at sites it had never seen" if record.get("kind") else ""
        trust = ui.tile(
            "How much to trust it",
            f"{right:.0%} of “go” calls held up",
            f"Share of “go” calls {where} that stayed usable in the 2026 test. "
            f"{esc(agreement_text(night)[1])}",
        )
    else:
        trust = ui.tile("How much to trust it", "–", esc(agreement_text(night)[1]))
    return ui.tiles([dark, best, moon, trust])


def reliability_details(night, site_label: str) -> None:
    """The exact backtest numbers behind the trust tile, with confidence intervals."""
    record = night.track_record
    if not record:
        st.caption("No backtest metrics available.")
        return
    if record.get("kind") == "unseen":
        o = record["overall"]
        st.markdown(
            "No local track record here. At airports the model had **never seen** "
            f"(leave-one-site-out test, lead {night.lead}), its false-clear rate was "
            f"{with_ci(o, 'false_clear_rate')} and its skill vs climatology "
            f"{with_ci(o, 'bss', 'num')}."
        )
        return
    site, overall = record.get("site"), record["overall"]
    c1, c2 = st.columns(2)
    c1.metric(
        f"False-clear rate at {site_label}",
        with_ci(site, "false_clear_rate"),
        help="In the held-out 2026 test period, of the nights the blend said 'go' at this lead, "
        "the share that turned out not usable. 95% week-block bootstrap CI.",
    )
    c2.metric(
        f"Skill vs climatology at {site_label}",
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
        (c["curve"] for c in ctx.metrics.get("threshold_curves", [])
         if c["label"] == "primary" and c["lead"] == night.lead),
        None,
    )  # fmt: skip
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
- **Usable night** = at least **{d["min_run_hours"]} clear hours in a row** after the sky is
  fully dark. That's roughly what an imaging session needs.
- **Chance of a usable night** is a real probability: on nights when SkyTrust says 70%, about
  7 in 10 turned out usable in testing.
- **Darkness** starts when the Sun is {abs(d["sun_altitude_deg"]):.0f}° below the horizon
  (astronomical dusk). Before that the sky is still glowing.
- **The Moon** matters as much as clouds for faint targets. A bright Moon is fine for the Moon
  itself, planets and star clusters.
- **Cloud layers:** thin high cloud (cirrus) is often invisible to the eye but ruins long
  exposures. The layer chart under *What the weather models say* shows it.
- **Models agree / disagree:** SkyTrust combines five weather models. When they disagree,
  treat the forecast with more caution and check again closer to the night.
"""
    )


def render(ctx: Context) -> None:
    if unavailable(ctx):
        return
    fc, tz, pal = ctx.forecast, ctx.site.timezone, ctx.palette
    night = ctx.nights[0]
    st.markdown(hero(ctx, night), unsafe_allow_html=True)
    if night.models_missing:
        st.info(
            f"Computed without {', '.join(m.upper() for m in night.models_missing)} "
            f"(no data right now); using {', '.join(m.upper() for m in night.models_used)}."
        )
    st.markdown(fact_tiles(ctx, night), unsafe_allow_html=True)

    st.subheader("Hour by hour")
    go_at, maybe_at = ctx.settings.raw["verdict"]["go"], ctx.settings.raw["verdict"]["maybe"]
    show(charts.night_chart(fc.hourly, night, tz, pal, go_at, maybe_at))
    st.markdown(
        ui.legend(
            [
                (f"likely clear (≥ {go_at:.0%})", pal["Go"]),
                ("could go either way", pal["Maybe"]),
                (f"likely cloudy (< {maybe_at:.0%})", pal["Skip"]),
                ("astronomical darkness", pal["accent2"]),
                ("Moon above the horizon", pal["moon"]),
                ("best window (outlined)", pal["accent"]),
            ]
        ),
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown("**Should I go?** Set how much a wasted trip bothers you.")
        risk_panel(ctx, night)

    with st.expander("📈 What the weather models say (cloud cover by model and by layer)"):
        badge = {"Models agree": "✅", "Models split": "⚠️"}.get(night.agreement, "ℹ️")
        spread = f" (spread {night.spread:.2f})" if night.spread is not None else ""
        st.markdown(f"{badge} **{night.agreement}**{spread}. {agreement_text(night)[1]}")
        show(charts.hourly_cloud(fc.hourly, night, tz, ctx.settings.clear_threshold, pal))
        show(charts.cloud_layers(fc.hourly, night, tz, pal))
    with st.expander("🎯 Track record behind this forecast"):
        reliability_details(night, ctx.site_label)
    with st.expander("🛰️ Latest satellite view (GOES-West, Pacific Southwest)"):
        st.image(GOES_IMAGE, caption="NOAA/NESDIS STAR GeoColor; refreshes every few minutes.")
        st.markdown(
            f"[Open the animated loop]({GOES_LOOP}) to see which way the clouds are moving."
        )
    with st.expander("🔭 New to this? How to read tonight's forecast"):
        beginner_guide(ctx)
    st.caption(
        f"Forecast fetched {fc.fetched_at_utc.tz_convert(tz):%b %-d, %-I:%M %p} local "
        f"({fc.source}). Tonight's live forecast is fresher than the 1-day-ahead data the model "
        "was tested on, so this probability is slightly conservative."
    )
