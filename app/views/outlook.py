"""Page 2: the next 7 nights. A week strip of cards (like a weather app), then the hour-by-hour
cloud grid (like a Clear Sky Chart, but with a colour key and 12-hour times)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from views import charts
from views import components as ui
from views.common import Context, day_label, lead_phrase, short_time
from views.tonight import show, unavailable

TRUST_TEXT = {
    "High": "high trust",
    "Medium": "medium trust",
    "Low": "low trust",
    "Unknown": "trust unknown",
}


def day_card(ctx: Context, n, i: int) -> str:
    tz, pal = ctx.site.timezone, ctx.palette
    color = pal[n.verdict]
    bw = n.best_window
    window = (
        f"{short_time(bw.start_utc, tz)}–{short_time(bw.until_utc, tz)}"
        if bw
        else "no clear window"
    )
    return ui.block(
        f'<div class="sk-day" style="--c:{color}">',
        f'<div class="sk-day-name">{day_label(n.night_date, i, ctx.now_utc, tz)}</div>',
        f'<div class="sk-day-date">{pd.Timestamp(n.night_date):%b %-d}</div>',
        f'<div style="margin:8px 0 2px">{ui.moon_svg(n.moon_phase_deg, 30)}</div>',
        f'<div class="sk-day-p">{"–" if n.p_usable is None else f"{n.p_usable:.0%}"}</div>',
        ui.pill(n.verdict.upper(), color),
        f'<div class="sk-day-meta">{window}<br>{n.moon_illum or 0:.0%} Moon<br>',
        f"{ui.trust_dots(n.trust)} {TRUST_TEXT.get(n.trust, '')}</div>",
        "</div>",
    )


def render(ctx: Context) -> None:
    st.header(f"The next 7 nights at {ctx.site_label}")
    if unavailable(ctx):
        return
    nights, tz, pal = ctx.nights, ctx.site.timezone, ctx.palette
    st.caption(
        "Chance of a usable night (3+ clear dark hours in a row). Forecasts further ahead are "
        "measurably less reliable: the dots show how well SkyTrust did at that range in testing."
    )
    cards = "".join(day_card(ctx, n, i) for i, n in enumerate(nights))
    st.markdown(f'<div class="sk-week">{cards}</div>', unsafe_allow_html=True)

    st.subheader("Cloud cover, hour by hour")
    labels = [
        f"{day_label(n.night_date, i, ctx.now_utc, tz)} · {pd.Timestamp(n.night_date):%-m/%-d}"
        for i, n in enumerate(nights)
    ]
    show(charts.outlook_grid(nights, ctx.forecast.hourly, tz, pal, labels))
    st.caption(
        "Each cell is the typical (median) forecast cloud cover for that dark hour, in %. "
        "Dark navy = clear, white = overcast; blank = not astronomically dark."
    )

    with st.expander("📋 Night-by-night details"):
        rows = []
        for i, n in enumerate(nights):
            bw = n.best_window
            skill = (n.track_record or {}).get("overall", {}).get("bss") if n.track_record else None
            rows.append(
                {
                    "night": f"{day_label(n.night_date, i, ctx.now_utc, tz)} "
                    f"{pd.Timestamp(n.night_date):%b %-d}",
                    "forecast range": lead_phrase(n.lead),
                    "chance usable": None if n.p_usable is None else round(100 * n.p_usable),
                    "verdict": n.verdict,
                    "dark": f"{short_time(n.dusk_utc, tz)}–{short_time(n.dawn_utc, tz)}",
                    "best window": (
                        f"{short_time(bw.start_utc, tz)}–{short_time(bw.until_utc, tz)}"
                        if bw
                        else "–"
                    ),
                    "Moon": f"{ui.phase_name(n.moon_phase_deg)} ({n.moon_illum or 0:.0%})",
                    "moon-free dark h": n.moon_free_hours,
                    "models": n.agreement,
                    "trust": f"{n.trust}" + (f" (skill {skill:.2f})" if skill is not None else ""),
                }
            )
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            width="stretch",
            column_config={
                "chance usable": st.column_config.ProgressColumn(
                    "chance usable", format="%d%%", min_value=0, max_value=100
                )
            },
        )
