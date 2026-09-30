"""Page 1: tonight's probability, verdict, best window, reliability, and hourly detail."""

from __future__ import annotations

import streamlit as st

from views import charts
from views.common import Context, local_time, pct, verdict_badge, with_ci


def unavailable(ctx: Context) -> bool:
    """Show a friendly message (and stop) when there's no forecast at all."""
    if ctx.forecast is None:
        st.warning(
            f"Live forecasts are unavailable right now and there's no saved copy yet. "
            f"({ctx.forecast_error}) The Track Record and Methodology pages still work."
        )
        return True
    if ctx.forecast.warning:
        st.warning(ctx.forecast.warning)
    return False


def reliability_card(night, site_id: str) -> None:
    record = night.track_record
    st.markdown("**How reliable is this?**")
    if not record:
        st.caption("No backtest metrics available.")
        return
    site, overall = record.get("site"), record["overall"]
    c1, c2 = st.columns(2)
    c1.metric(
        f"False-clear rate at {site_id}",
        with_ci(site, "false_clear_rate"),
        help="In the held-out 2026 test period, of the nights the blend said 'go' at this lead, "
        "the share that turned out not usable. 95% week-block bootstrap CI.",
    )
    c2.metric(
        f"Skill vs climatology at {site_id}",
        with_ci(site, "bss", "num"),
        help="Brier Skill Score: 0 = no better than the seasonal base rate, 1 = perfect.",
    )
    st.caption(
        f"All sites, lead {night.lead}: false-clear {with_ci(overall, 'false_clear_rate')}, "
        f"skill {with_ci(overall, 'bss', 'num')}."
    )


def render(ctx: Context) -> None:
    st.header(f"Tonight at {ctx.site.id}")
    if unavailable(ctx):
        return
    fc, tz, pal = ctx.forecast, ctx.site.timezone, ctx.palette
    night = fc.nights[0]
    left, right = st.columns([1, 1])
    with left:
        prob = "–" if night.p_usable is None else f"{night.p_usable:.0%}"
        st.markdown(
            f"<div style='font-size:3.6rem;font-weight:800;line-height:1'>{prob}</div>"
            f"<div style='margin:4px 0 10px 0'>chance of a usable night "
            f"(≥ {ctx.settings.min_run_hours} consecutive clear dark hours)</div>"
            + verdict_badge(night.verdict, pal),
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f"**Dark:** {local_time(night.dusk_utc, tz)} → {local_time(night.dawn_utc, tz)} "
            f"({night.dark_hours} h)"
        )
        if night.best_window:
            bw = night.best_window
            st.markdown(
                f"**Best window:** {local_time(bw.start_utc, tz)} → "
                f"{local_time(bw.until_utc, tz)} ({bw.hours} h)"
            )
        else:
            st.markdown("**Best window:** none. No dark hour is clear by the model median.")
        badge = {"Models agree": "✅", "Models split": "⚠️"}.get(night.agreement, "ℹ️")
        spread = f" (spread {night.spread:.2f})" if night.spread is not None else ""
        st.markdown(f"**{badge} {night.agreement}**{spread}")
        st.markdown(
            f"**Moon:** {pct(night.moon_illum)} illuminated, "
            f"{night.moon_free_hours} moon-free dark hours"
        )
        if night.models_missing:
            st.info(
                f"Computed without {', '.join(m.upper() for m in night.models_missing)} "
                f"(no data right now); using {', '.join(m.upper() for m in night.models_used)}."
            )
    st.divider()
    reliability_card(night, ctx.site.id)
    st.plotly_chart(charts.darkness_timeline(ctx.site, night, tz, pal), width="stretch")
    st.plotly_chart(
        charts.hourly_cloud(fc.hourly, night, tz, ctx.settings.clear_threshold, pal),
        width="stretch",
    )
    st.plotly_chart(charts.cloud_layers(fc.hourly, night, tz, pal), width="stretch")
    st.caption(
        f"Forecast fetched {local_time(fc.fetched_at_utc, tz, '%b %-d, %-I:%M %p')} local "
        f"({fc.source}). Tonight's live forecast is fresher than the lead-1 data the model was "
        "tested on, so this probability is slightly conservative."
    )
