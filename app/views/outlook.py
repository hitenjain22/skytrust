"""Page 2: the next 7 nights. A list in the style of a 10-day weather forecast (one quiet row per
night with a probability bar), then the hour-by-hour cloud grid with a labelled scale."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from views import charts
from views import components as ui
from views.common import Context, day_label, lead_phrase, short_time
from views.tonight import show, unavailable

TRUST_TEXT = {"High": "high trust", "Medium": "medium trust", "Low": "low trust"}
COLS = "minmax(120px, 1.3fr) 26px minmax(60px, 2fr) 52px 96px"


def night_row(ctx: Context, n, i: int) -> str:
    tz, pal = ctx.site.timezone, ctx.palette
    bw = n.best_window
    window = (
        f"{short_time(bw.start_utc, tz)}–{short_time(bw.until_utc, tz)}"
        if bw
        else "no clear window"
    )
    sub = (
        f"{pd.Timestamp(n.night_date):%b %-d} · {window} · {ui.trust_dots(n.trust)} "
        f'<span class="sk-hide-sm">{TRUST_TEXT.get(n.trust, "")}</span>'
    )
    p = "–" if n.p_usable is None else f"{n.p_usable:.0%}"
    return ui.row(
        [
            f'<div><div class="sk-row-title">{day_label(n.night_date, i, ctx.now_utc, tz)}</div>'
            f'<div class="sk-row-sub">{sub}</div></div>',
            ui.moon_svg(n.moon_phase_deg, 20),
            ui.bar(n.p_usable, pal[n.verdict]),
            f'<div class="sk-row-p">{p}</div>',
            f'<div style="text-align:right">{ui.status(n.verdict.upper(), pal[n.verdict])}</div>',
        ],
        COLS,
    )


def render(ctx: Context) -> None:
    st.header(f"The next 7 nights at {ctx.site_label}")
    if unavailable(ctx):
        return
    nights, tz, pal = ctx.nights, ctx.site.timezone, ctx.palette
    st.caption(
        "Chance of a usable night (3+ clear dark hours in a row). Accuracy fades with distance: "
        "the dots show how well SkyTrust did at that range in testing."
    )
    st.markdown(
        ui.rows([night_row(ctx, n, i) for i, n in enumerate(nights)]), unsafe_allow_html=True
    )

    st.subheader("Cloud cover, hour by hour")
    labels = [
        f"{day_label(n.night_date, i, ctx.now_utc, tz)} · {pd.Timestamp(n.night_date):%-m/%-d}"
        for i, n in enumerate(nights)
    ]
    show(charts.outlook_grid(nights, ctx.forecast.hourly, tz, pal, labels))
    st.caption(
        "Typical (median) forecast cloud cover for each dark hour, in %. Dark = clear, "
        "light = overcast; blank = not astronomically dark."
    )

    with st.expander("Night-by-night details", icon=":material/table_rows:"):
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
