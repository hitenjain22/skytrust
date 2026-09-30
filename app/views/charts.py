"""Plotly figures for the app. Times are shown in the site's local time zone."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from skytrust import astro
from skytrust.report import display_name

PAD = pd.Timedelta(hours=1)


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
    title: str,
    height: int = 330,
    yaxis: dict | None = None,
    time_axis: bool = False,
) -> go.Figure:
    fig.update_layout(
        title={"text": title, "font": {"size": 14}},
        height=height,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        paper_bgcolor=pal["paper"],
        plot_bgcolor=pal["bg"],
        font={"color": pal["text"]},
        legend={"orientation": "h", "y": -0.18, "x": 0, "yanchor": "top"},
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=pal["grid"])
    if time_axis:
        fig.update_xaxes(tickformat="%-I %p")  # "9 PM": one line, so the legend never collides
    fig.update_yaxes(gridcolor=pal["grid"], **(yaxis or {}))
    return fig


def _night_slice(hourly: pd.DataFrame, night) -> pd.DataFrame:
    return hourly[
        (hourly["time"] >= night.dusk_utc - PAD) & (hourly["time"] <= night.dawn_utc + PAD)
    ]


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
                line={"color": pal["models"].get(model), "width": 1.5},
            )
        )
    med = data.pivot_table(index="time", columns="model", values="cloud_cover").median(axis=1)
    fig.add_trace(
        go.Scatter(
            x=_local(med.index, tz),
            y=med * 100,
            name="Median",
            mode="lines+markers",
            line={"color": pal["models"]["median"], "width": 3, "dash": "dot"},
        )
    )
    fig.add_vrect(
        x0=_local(night.dusk_utc, tz),
        x1=_local(night.dawn_utc, tz),
        fillcolor=pal["dark"],
        line_width=0,
        annotation_text="astronomical dark",
        annotation_position="top left",
    )
    fig.add_hline(
        y=threshold * 100,
        line_dash="dash",
        line_color=pal["accent"],
        annotation_text=f"clear ≤ {threshold:.0%}",
        annotation_position="bottom right",
    )
    return _layout(fig, pal, "Cloud cover by model (%)", yaxis={"range": [0, 100]}, time_axis=True)


def cloud_layers(hourly: pd.DataFrame, night, tz: str, pal: dict) -> go.Figure:
    """Median across models of low / mid / high cloud (high = cirrus, which ruins deep-sky)."""
    data = _night_slice(hourly, night)
    fig = go.Figure()
    for col, name, dash in [
        ("cloud_cover_low", "Low", "solid"),
        ("cloud_cover_mid", "Mid", "dash"),
        ("cloud_cover_high", "High (cirrus)", "dot"),
    ]:
        med = data.pivot_table(index="time", columns="model", values=col).median(axis=1)
        fig.add_trace(
            go.Scatter(
                x=_local(med.index, tz),
                y=med * 100,
                name=name,
                mode="lines",
                line={"dash": dash, "width": 2},
            )
        )
    fig.add_vrect(
        x0=_local(night.dusk_utc, tz),
        x1=_local(night.dawn_utc, tz),
        fillcolor=pal["dark"],
        line_width=0,
    )
    return _layout(
        fig,
        pal,
        "Cloud layers, median of models (%)",
        height=280,
        yaxis={"range": [0, 100]},
        time_axis=True,
    )


def true_runs(times: pd.DatetimeIndex, flags) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """(start, end) of each stretch where `flags` is True. A stretch still open at the end
    runs to the last time (e.g. the Moon is still up at the right edge of the chart)."""
    runs, start = [], None
    for t, flag in zip(times, flags, strict=True):
        if flag and start is None:
            start = t
        elif not flag and start is not None:
            runs.append((start, t))
            start = None
    if start is not None:
        runs.append((start, times[-1]))
    return runs


def darkness_timeline(site, night, tz: str, pal: dict) -> go.Figure:
    """Rows for astronomical darkness, Moon above the horizon, and the best clear window."""
    start, end = night.dusk_utc - PAD, night.dawn_utc + PAD
    grid = pd.date_range(start, end, freq="10min")
    up = astro.moon_altitude(site, grid) > 0
    fig = go.Figure()

    def bar(row: str, t0, t1, color: str, text: str = "") -> None:
        fig.add_trace(
            go.Bar(
                y=[row],
                x=[(t1 - t0).total_seconds() * 1000],
                base=[_local(t0, tz)],
                orientation="h",
                marker_color=color,
                showlegend=False,
                hovertext=text,
                hoverinfo="text",
            )
        )

    bar("Dark", night.dusk_utc, night.dawn_utc, pal["accent"], "Astronomical darkness")
    for t0, t1 in true_runs(grid, up):
        bar("Moon up", t0, t1, pal["moon"], "Moon above horizon")
    if night.best_window:
        bw = night.best_window
        bar(
            "Best window",
            bw.start_utc,
            bw.until_utc,
            pal["Go"],
            f"{bw.hours} h clear",
        )
    fig.update_xaxes(type="date", range=[_local(start, tz), _local(end, tz)])
    fig.update_layout(barmode="overlay", bargap=0.35)
    return _layout(fig, pal, "Darkness, Moon, and best window", height=200, time_axis=True)


# ---------- Track Record ----------


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
                "array": (o["false_clear_rate_hi"] - o["false_clear_rate"]).fillna(0) * 100,
                "arrayminus": (o["false_clear_rate"] - o["false_clear_rate_lo"]).fillna(0) * 100,
            },
        )
    )
    fig.update_xaxes(title="False-clear rate (%) with 95% CI: of 'go' nights, share not usable")
    return _layout(fig, pal, f"False-clear rate, lead {lead}", height=420)


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
                opacity=1 if emphasised else 0.6,
            )
        )
    fig.add_hline(y=0, line_color=pal["muted"])
    fig.update_xaxes(title="Lead time (days ahead)", dtick=1)
    return _layout(
        fig,
        pal,
        "Brier Skill Score vs climatology by lead",
        height=380,
        yaxis={"title": "BSS (higher = better)"},
    )


def value_curves(
    curves: list[dict], methods: list[str], label: str, lead: int, pal: dict
) -> go.Figure:
    fig = go.Figure()
    for c in curves:
        if c["label"] != label or c["lead"] != lead or c["method"] not in methods:
            continue
        fig.add_trace(go.Scatter(
            x=c["alpha"], y=[None if v is None else v * 100 for v in c["value"]], mode="lines",
            name=display_name(c["method"]),
            line={"width": 3 if c["method"] in pal["methods"] else 1.5,
                  "color": pal["methods"].get(c["method"])},
        ))  # fmt: skip
    fig.add_hline(y=0, line_color=pal["muted"])
    fig.update_xaxes(title="Setup effort ÷ value of a good night (α)")
    return _layout(fig, pal, f"Decision value, lead {lead}: % of a perfect forecast's benefit",
                   height=360, yaxis={"range": [-20, 100], "title": "%"})  # fmt: skip
