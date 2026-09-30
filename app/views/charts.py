"""Plotly figures for the app, in the Moonlight theme. Times are the site's local time zone,
12-hour clock (24-hour times are a common complaint about astronomy weather apps)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from skytrust.report import display_name

PAD = pd.Timedelta(hours=1)
LABEL_POSITIONS = {
    "TRK": "top right",
    "AUN": "top left",
    "SAC": "bottom left",
    "BIH": "middle right",
    "FAT": "bottom right",
}
FONT = "Inter, system-ui, sans-serif"


def _local(ts, tz: str):
    """Plotly shows naive datetimes as-is, so convert to local wall-clock time first."""
    return (
        pd.DatetimeIndex(ts).tz_convert(tz).tz_localize(None)
        if hasattr(ts, "__len__")
        else (pd.Timestamp(ts).tz_convert(tz).tz_localize(None))
    )


def _layout(
    fig: go.Figure,
    pal: dict,
    title: str = "",
    height: int = 330,
    yaxis: dict | None = None,
    time_axis: bool = False,
    legend: bool = True,
) -> go.Figure:
    fig.update_layout(
        title={"text": title, "font": {"size": 14, "color": pal["text"]}, "x": 0, "xanchor": "left"}
        if title
        else None,
        height=height,
        margin={"l": 8, "r": 8, "t": 44 if title else 12, "b": 8},
        paper_bgcolor=pal["paper"],
        plot_bgcolor=pal["paper"],
        font={"color": pal["text"], "family": FONT, "size": 12},
        legend={"orientation": "h", "y": -0.18, "x": 0, "yanchor": "top", "font": {"size": 11}},
        showlegend=legend,
        hovermode="x unified",
        hoverlabel={
            "bgcolor": pal["surface"],
            "bordercolor": pal["border"],
            "font": {"family": FONT},
        },
    )
    fig.update_xaxes(gridcolor=pal["grid"], zeroline=False, linecolor=pal["border"])
    if time_axis:
        fig.update_xaxes(tickformat="%-I %p", hoverformat="%-I:%M %p")
    fig.update_yaxes(gridcolor=pal["grid"], zeroline=False, **(yaxis or {}))
    return fig


def _night_slice(hourly: pd.DataFrame, night) -> pd.DataFrame:
    return hourly[
        (hourly["time"] >= night.dusk_utc - PAD) & (hourly["time"] <= night.dawn_utc + PAD)
    ]


def median_cover(hourly: pd.DataFrame, column: str = "cloud_cover") -> pd.Series:
    return hourly.pivot_table(index="time", columns="model", values=column).median(axis=1)


def level_color(p: float, pal: dict, go_at: float = 0.7, maybe_at: float = 0.4) -> str:
    if p is None or np.isnan(p):
        return pal["No data"]
    return pal["Go"] if p >= go_at else pal["Maybe"] if p >= maybe_at else pal["Skip"]


# ---------- Tonight: the one chart a beginner needs ----------


def night_chart(hourly: pd.DataFrame, night, tz: str, pal: dict, go_at=0.7, maybe_at=0.4):
    """Hour by hour through the night: the chance each hour is clear (bars, coloured like the
    verdict), astronomical darkness (shaded), when the Moon is up (band on top), and the best
    window (gold outline). Falls back to the model median's clear-sky share without the hourly
    model."""
    data = _night_slice(hourly, night)
    med = median_cover(data)
    if night.hourly_clear is not None and not night.hourly_clear.empty:
        p = night.hourly_clear
        name, hover = (
            "Chance of clear sky",
            "%{y:.0f}% chance clear · typical cloud %{customdata:.0f}%",
        )
    else:
        dark = med[(med.index >= night.dusk_utc.floor("h")) & (med.index <= night.dawn_utc)]
        p = 1 - dark
        name, hover = "Clear sky (model median)", "%{y:.0f}% clear sky · cloud %{customdata:.0f}%"
    cloud = med.reindex(p.index).to_numpy() * 100
    fig = go.Figure()
    start, end = night.dusk_utc - PAD, night.dawn_utc + PAD
    fig.add_vrect(
        x0=_local(night.dusk_utc, tz), x1=_local(night.dawn_utc, tz), fillcolor=pal["dark"],
        line_width=0, layer="below",
    )  # fmt: skip
    if night.best_window:
        bw = night.best_window
        fig.add_shape(
            type="rect", x0=_local(bw.start_utc - pd.Timedelta(minutes=30), tz),
            x1=_local(bw.until_utc, tz), y0=0, y1=100, line={"color": pal["accent"], "width": 1.5},
            fillcolor="rgba(0,0,0,0)", layer="below",
        )  # fmt: skip
    fig.add_trace(
        go.Bar(
            x=_local(p.index, tz),
            y=p.to_numpy() * 100,
            name=name,
            marker={"color": [level_color(v, pal, go_at, maybe_at) for v in p.to_numpy()],
                    "line": {"width": 0}},
            customdata=cloud,
            hovertemplate=hover + "<extra></extra>",
            width=1000 * 60 * 60 * 0.72,
        )
    )  # fmt: skip
    # Moon-up band above the bars
    for a, b in getattr(night, "moon_up", []) or []:
        fig.add_shape(
            type="rect", x0=_local(max(a, start), tz), x1=_local(min(b, end), tz), y0=104, y1=110,
            fillcolor=pal["moon"], line_width=0, opacity=0.85,
        )  # fmt: skip
    if getattr(night, "moon_up", None):
        first = max(night.moon_up[0][0], start)
        fig.add_annotation(
            x=_local(first, tz), y=113, text="Moon up", showarrow=False, xanchor="left",
            font={"size": 11, "color": pal["muted"]},
        )  # fmt: skip
    for when, label, anchor in [
        (night.dusk_utc, "dark", "left"),
        (night.dawn_utc, "dawn", "right"),
    ]:
        fig.add_vline(x=_local(when, tz), line={"color": pal["accent2"], "width": 1, "dash": "dot"})
        fig.add_annotation(
            x=_local(when, tz), y=97, text=f"{label} {when.tz_convert(tz):%-I:%M %p}",
            showarrow=False, xanchor=anchor, xshift=4 if anchor == "left" else -4,
            font={"size": 11, "color": pal["accent2"]},
        )  # fmt: skip
    fig.update_xaxes(type="date", range=[_local(start, tz), _local(end, tz)])
    return _layout(
        fig, pal, height=300, time_axis=True, legend=False,
        yaxis={"range": [0, 118], "tickvals": [0, 25, 50, 75, 100], "ticksuffix": "%",
               "title": {"text": "chance clear", "font": {"size": 11}}},
    )  # fmt: skip


def hourly_cloud(hourly: pd.DataFrame, night, tz: str, threshold: float, pal: dict) -> go.Figure:
    """Each model's cloud cover through the night, the cross-model median, and the clear line."""
    data = _night_slice(hourly, night)
    fig = go.Figure()
    for model, g in data.groupby("model", sort=False):
        fig.add_trace(
            go.Scatter(
                x=_local(g["time"], tz),
                y=g["cloud_cover"] * 100,
                name=model.upper(),
                mode="lines",
                line={"color": pal["models"].get(model), "width": 1.6, "shape": "spline"},
            )
        )
    med = median_cover(data)
    fig.add_trace(
        go.Scatter(
            x=_local(med.index, tz),
            y=med * 100,
            name="Median",
            mode="lines",
            line={"color": pal["models"]["median"], "width": 3, "dash": "dot"},
        )
    )
    fig.add_vrect(
        x0=_local(night.dusk_utc, tz), x1=_local(night.dawn_utc, tz), fillcolor=pal["dark"],
        line_width=0, layer="below",
    )  # fmt: skip
    fig.add_hline(
        y=threshold * 100,
        line_dash="dash",
        line_color=pal["accent"],
        annotation_text=f"clear ≤ {threshold:.0%}",
        annotation_position="bottom right",
        annotation_font_color=pal["accent"],
    )
    return _layout(
        fig, pal, "Cloud cover by weather model (%)", yaxis={"range": [0, 100]}, time_axis=True
    )


def cloud_layers(hourly: pd.DataFrame, night, tz: str, pal: dict) -> go.Figure:
    """Median across models of low / mid / high cloud (high = cirrus, which ruins deep-sky)."""
    data = _night_slice(hourly, night)
    fig = go.Figure()
    for col, key, name in [
        ("cloud_cover_low", "low", "Low (fog, stratus)"),
        ("cloud_cover_mid", "mid", "Mid"),
        ("cloud_cover_high", "high", "High (thin cirrus)"),
    ]:
        med = median_cover(data, col)
        fig.add_trace(
            go.Scatter(
                x=_local(med.index, tz),
                y=med * 100,
                name=name,
                mode="lines",
                line={"width": 2.2, "color": pal["layers"][key], "shape": "spline"},
            )
        )
    fig.add_vrect(
        x0=_local(night.dusk_utc, tz), x1=_local(night.dawn_utc, tz), fillcolor=pal["dark"],
        line_width=0, layer="below",
    )  # fmt: skip
    return _layout(
        fig,
        pal,
        "Cloud layers, median of the models (%)",
        height=280,
        yaxis={"range": [0, 100]},
        time_axis=True,
    )


# ---------- 7 nights ----------


def outlook_grid(nights, hourly: pd.DataFrame, tz: str, pal: dict, labels: list[str]) -> go.Figure:
    """Clear-Sky-Chart-style grid: one row per night, one column per local hour (7 PM-6 AM),
    colour = median forecast cloud cover (navy = clear, white = overcast), with the number in
    every cell and a labelled colour bar. Hours outside astronomical darkness are blank."""
    slots = [19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6]
    med = median_cover(hourly)
    z, text = [], []
    for n in nights:
        dark = pd.date_range(n.dusk_utc.ceil("h"), n.dawn_utc.floor("h"), freq="h")
        local = {t.tz_convert(tz).hour: t for t in dark}
        row = [
            med.get(local[h]) * 100 if h in local and local[h] in med.index else None for h in slots
        ]
        z.append(row)
        text.append(["" if v is None else f"{v:.0f}" for v in row])
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[f"{(h % 12) or 12} {'AM' if h < 12 else 'PM'}" for h in slots],
            y=labels,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 11},
            colorscale=pal["heat"],
            zmin=0,
            zmax=100,
            xgap=3,
            ygap=3,
            colorbar={"title": {"text": "cloud %", "side": "right"}, "thickness": 10,
                      "tickvals": [0, 50, 100], "outlinewidth": 0},
            hovertemplate="%{y}, %{x}: %{z:.0f}% cloud<extra></extra>",
            hoverongaps=False,
        )
    )  # fmt: skip
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, side="top")
    return _layout(fig, pal, height=46 * len(nights) + 60, legend=False)


# ---------- Where ----------


def site_map(table: pd.DataFrame, pal: dict, go_at: float, maybe_at: float) -> go.Figure:
    """One marker per site coloured like its verdict. One trace per site, because a map trace
    can't place each label differently, and neighbours (Auburn / Truckee) would hide each other."""
    fig = go.Figure()
    for _, r in table.iterrows():
        p = float(r["p"]) if not pd.isna(r["p"]) else float("nan")
        label = f"{r['place']}  {'–' if np.isnan(p) else f'{p:.0%}'}"
        fig.add_trace(
            go.Scattermap(
                lat=[r["lat"]],
                lon=[r["lon"]],
                mode="markers+text",
                text=[label],
                textposition=LABEL_POSITIONS.get(r["site"], "top right"),
                textfont={"color": pal["text"], "size": 13},
                marker={"size": 18, "color": level_color(p, pal, go_at, maybe_at), "opacity": 0.95},
                hovertemplate="%{text}<extra></extra>",
            )
        )
    fig.update_layout(
        map={"style": "carto-darkmatter", "center": {"lat": 38.0, "lon": -120.0}, "zoom": 5.3},
        height=420,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor=pal["paper"],
        showlegend=False,
    )
    return fig


# ---------- Track Record ----------


def go_accuracy_by_lead(records: pd.DataFrame, pal: dict) -> go.Figure:
    """Beginner view of skill decay: of the nights SkyTrust said "go", the share that really
    were usable, by days ahead, with 95% CIs, next to the seasonal base rate."""
    o = records[
        (records["label"] == "primary")
        & (records["subset_type"] == "overall")
        & (records["method"].isin(["blend", "climatology"]))
    ].sort_values("lead")
    fig = go.Figure()
    for method, g in o.groupby("method", sort=False):
        ok = 100 * (1 - g["false_clear_rate"])
        is_blend = method == "blend"
        fig.add_trace(
            go.Scatter(
                x=g["lead"],
                y=ok,
                name="SkyTrust" if is_blend else "Guessing from the season",
                mode="lines+markers",
                line={"color": pal["methods"]["blend" if is_blend else "climatology"],
                      "width": 3 if is_blend else 1.5, "dash": "solid" if is_blend else "dot"},
                marker={"size": 8 if is_blend else 5},
                error_y={
                    "type": "data", "symmetric": False, "thickness": 1,
                    "array": (100 * (g["false_clear_rate"] - g["false_clear_rate_lo"])).fillna(0),
                    "arrayminus": (
                        100 * (g["false_clear_rate_hi"] - g["false_clear_rate"])
                    ).fillna(0),
                }
                if is_blend
                else None,
                hovertemplate="%{x} day(s) ahead: %{y:.0f}% of 'go' nights were usable"
                "<extra></extra>",
            )
        )  # fmt: skip
    fig.update_xaxes(title="Days ahead", dtick=1)
    return _layout(
        fig, pal, "When it said “go”, how often the night was usable", height=320,
        yaxis={"range": [50, 100], "ticksuffix": "%"},
    )  # fmt: skip


def false_clear_bars(records: pd.DataFrame, label: str, lead: int, pal: dict) -> go.Figure:
    o = records[
        (records["label"] == label)
        & (records["lead"] == lead)
        & (records["subset_type"] == "overall")
        & records["false_clear_rate"].notna()
    ]
    o = o.sort_values("false_clear_rate", ascending=False)
    colors = [pal["methods"].get(m, pal["muted"]) for m in o["method"]]
    fig = go.Figure(
        go.Bar(
            y=[display_name(m) for m in o["method"]],
            x=o["false_clear_rate"] * 100,
            orientation="h",
            marker_color=colors,
            error_x={
                "type": "data",
                "symmetric": False,
                "color": pal["muted"],
                "array": (o["false_clear_rate_hi"] - o["false_clear_rate"]).fillna(0) * 100,
                "arrayminus": (o["false_clear_rate"] - o["false_clear_rate_lo"]).fillna(0) * 100,
            },
        )
    )
    fig.update_xaxes(title="False-clear rate (%) with 95% CI: of 'go' nights, share not usable")
    return _layout(fig, pal, f"False-clear rate, lead {lead}", height=440, legend=False)


def reliability(
    rows: list[dict], methods: list[str], label: str, lead: int, pal: dict
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect",
            line={"dash": "dash", "color": pal["muted"]},
        )
    )
    for r in rows:
        if r["label"] != label or r["lead"] != lead or r["method"] not in methods:
            continue
        bins = pd.DataFrame(r["bins"])
        bins = bins[bins["n"] > 0]
        fig.add_trace(
            go.Scatter(
                x=bins["mean_p"],
                y=bins["observed"],
                mode="lines+markers",
                name=display_name(r["method"]),
                marker={"size": 6 + 14 * bins["n"] / bins["n"].max()},
                line={"color": pal["methods"].get(r["method"])},
                customdata=bins["n"],
                hovertemplate="forecast %{x:.2f} → observed %{y:.2f} (n=%{customdata})",
            )
        )
    fig.update_xaxes(title="Forecast P(usable)", range=[0, 1])
    fig.update_layout(hovermode="closest")
    return _layout(
        fig,
        pal,
        f"Reliability, lead {lead} (marker size = nights)",
        height=380,
        yaxis={"title": "Observed frequency", "range": [0, 1]},
    )


def skill_by_lead(records: pd.DataFrame, label: str, pal: dict) -> go.Figure:
    o = records[
        (records["label"] == label)
        & (records["subset_type"] == "overall")
        & (records["kind"] == "prob")
    ]
    fig = go.Figure()
    for method, g in o.groupby("method", sort=False):
        g = g.sort_values("lead")
        emphasised = method in pal["methods"]
        fig.add_trace(
            go.Scatter(
                x=g["lead"],
                y=g["bss"],
                mode="lines+markers",
                name=display_name(method),
                line={"width": 3 if emphasised else 1, "color": pal["methods"].get(method)},
                opacity=1 if emphasised else 0.5,
            )
        )
    fig.add_hline(y=0, line_color=pal["muted"])
    fig.update_xaxes(title="Lead time (days ahead)", dtick=1)
    return _layout(
        fig,
        pal,
        "Brier Skill Score vs climatology by lead",
        height=400,
        yaxis={"title": "BSS (higher = better)"},
    )


def value_curves(
    curves: list[dict], methods: list[str], label: str, lead: int, pal: dict
) -> go.Figure:
    fig = go.Figure()
    for c in curves:
        if c["label"] != label or c["lead"] != lead or c["method"] not in methods:
            continue
        fig.add_trace(
            go.Scatter(
                x=c["alpha"],
                y=[None if v is None else v * 100 for v in c["value"]],
                mode="lines",
                name=display_name(c["method"]),
                line={
                    "width": 3 if c["method"] in pal["methods"] else 1.5,
                    "color": pal["methods"].get(c["method"]),
                },
            )
        )
    fig.add_hline(y=0, line_color=pal["muted"])
    fig.update_xaxes(title="Setup effort ÷ value of a good night (α)")
    return _layout(
        fig,
        pal,
        f"Decision value, lead {lead}: % of a perfect forecast's benefit",
        height=360,
        yaxis={"range": [-20, 100], "title": "%"},
    )
