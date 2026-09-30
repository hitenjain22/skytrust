"""Where Tonight: every site ranked by tonight's probability of a usable night, on a map."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from views.common import Context, local_time, pct


def tonight_table(ctx: Context) -> pd.DataFrame:
    rows = []
    for site in ctx.sites:
        forecast, error = ctx.forecast_for(site.id)
        if forecast is None or not forecast.nights:
            rows.append(
                {
                    "site": site.id,
                    "terrain": site.terrain_class,
                    "P(usable)": None,
                    "verdict": "No data",
                    "best window": "–",
                    "moon-free h": None,
                    "lat": site.lat,
                    "lon": site.lon,
                }
            )
            continue
        n = forecast.nights[0]
        window = (
            f"{local_time(n.best_window.start_utc, site.timezone)}–"
            f"{local_time(n.best_window.until_utc, site.timezone)}"
            if n.best_window
            else "none"
        )
        rows.append(
            {
                "site": site.id,
                "terrain": site.terrain_class,
                "P(usable)": n.p_usable,
                "verdict": n.verdict,
                "best window": window,
                "moon-free h": n.moon_free_hours,
                "lat": site.lat,
                "lon": site.lon,
            }
        )
    return pd.DataFrame(rows).sort_values("P(usable)", ascending=False, na_position="last")


def render(ctx: Context) -> None:
    st.header("Where should I go tonight?")
    st.caption("Tonight's probability of a usable night at every site, best first.")
    table = tonight_table(ctx)
    fig = go.Figure(
        go.Scattermap(
            lat=table["lat"],
            lon=table["lon"],
            mode="markers+text",
            text=[f"{s} {pct(p)}" for s, p in zip(table["site"], table["P(usable)"], strict=True)],
            textposition="top right",
            marker={
                "size": 16,
                "color": table["P(usable)"].fillna(0) * 100,
                "cmin": 0,
                "cmax": 100,
                "colorscale": [[0, "#EE6677"], [0.5, "#DDCC77"], [1, "#44AA99"]],
                "colorbar": {"title": "P %", "thickness": 10},
            },
        )
    )
    fig.update_layout(
        map={"style": "carto-darkmatter", "center": {"lat": 38.0, "lon": -120.0}, "zoom": 5},
        height=420,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor=ctx.palette["paper"],
    )
    st.plotly_chart(fig, width="stretch")
    shown = table.drop(columns=["lat", "lon"]).assign(**{"P(usable)": table["P(usable)"].map(pct)})
    st.dataframe(shown, hide_index=True, width="stretch")
