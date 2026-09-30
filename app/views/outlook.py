"""Page 2: the next 7 nights, with a trust level that drops with lead time."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from views import charts
from views.common import Context, local_time, pct, verdict_badge
from views.tonight import unavailable

TRUST_DOTS = {"High": "●●●", "Medium": "●●○", "Low": "●○○", "Unknown": "○○○"}


def render(ctx: Context) -> None:
    st.header(f"7-night outlook at {ctx.site_label}")
    if unavailable(ctx):
        return
    tz, pal = ctx.site.timezone, ctx.palette
    st.plotly_chart(charts.outlook_grid(ctx.forecast, tz, pal), width="stretch")
    st.caption(
        "Trust comes from the backtest: the blend's skill on the held-out 2026 test period at "
        "that lead time. "
        "Forecasts further ahead are measurably less reliable."
    )
    for i, n in enumerate(ctx.forecast.nights):
        with st.container(border=True):
            c1, c2, c3 = st.columns([1.3, 1, 1.4])
            title = "Tonight" if i == 0 else f"{pd.Timestamp(n.night_date):%a %b %-d}"
            c1.markdown(f"**{title}** · {n.lead} day{'s' if n.lead > 1 else ''} ahead")
            c1.markdown(verdict_badge(n.verdict, pal), unsafe_allow_html=True)
            prob = "–" if n.p_usable is None else f"{n.p_usable:.0%}"
            c2.metric("P(usable)", prob)
            if n.best_window:
                bw = n.best_window
                window = (
                    f"{local_time(bw.start_utc, tz)}–{local_time(bw.until_utc, tz)} ({bw.hours} h)"
                )
            else:
                window = "none clear"
            skill = (n.track_record or {}).get("overall", {}).get("bss") if n.track_record else None
            skill_text = f" (skill {skill:.2f})" if skill is not None else ""
            c3.markdown(
                f"Best window: **{window}**  \n"
                f"Moon-free dark hours: **{n.moon_free_hours}** ({pct(n.moon_illum)} moon)  \n"
                f"Trust: **{TRUST_DOTS[n.trust]} {n.trust}**{skill_text}  \n"
                f"{n.agreement}"
            )
