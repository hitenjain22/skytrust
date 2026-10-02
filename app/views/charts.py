"""Plotly figures for the app, in the Observatory design system. Text, grid and background come
from Streamlit's active chart theme (so charts are right in light and dark); only data colours are
set here. Times are the site's local time zone, 12-hour clock (24-hour times are a common
complaint about astronomy weather apps)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from skytrust.report import display_name

PAD = pd.Timedelta(hours=1)
LABEL_POSITIONS = {  # chosen so neighbouring places' labels don't collide on the California map
    "san-francisco": "bottom left",
    "sacramento": "top left",
    "lake-tahoe": "top right",
    "yosemite": "middle right",
    "death-valley": "middle right",
    "santa-barbara": "bottom left",
    "los-angeles": "middle left",
    "big-bear-lake": "top center",
    "joshua-tree": "middle right",
    "TRK": "top right",
    "AUN": "top left",
    "SAC": "bottom left",
    "BIH": "middle right",
    "FAT": "bottom right",
}
FONT = "Geist, system-ui, sans-serif"
# Map labels are drawn by MapLibre from the basemap's own glyph server, which has no Geist;
# asking for it fails (a CORS error per page load) before falling back.
MAP_FONT = "Open Sans Regular"


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
    """Quiet chart chrome: no title box, faint horizontal grid only, no axis lines. Colours for
    text and background are left to the Streamlit theme unless the palette pins them (night
    vision)."""
    font = {"family": FONT, "size": 12} | ({"color": pal["text"]} if pal.get("text") else {})
    fig.update_layout(
        title={"text": title, "font": {"size": 13, "family": FONT}, "x": 0, "xanchor": "left"}
        if title
        else None,
        height=height,
        margin={"l": 4, "r": 4, "t": 40 if title else 24, "b": 4},
        paper_bgcolor=pal["paper"],
        plot_bgcolor=pal["paper"],
        font=font,
        legend={"orientation": "h", "y": -0.2, "x": 0, "yanchor": "top", "font": {"size": 11}},
        showlegend=legend,
        hovermode="x unified",
        hoverlabel={"font": {"family": FONT}}
        | ({"bgcolor": pal["surface"], "bordercolor": pal["border"]} if pal.get("surface") else {}),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showline=False, ticks="")
    if time_axis:
        fig.update_xaxes(tickformat="%-I %p", hoverformat="%-I:%M %p")
    fig.update_yaxes(gridcolor=pal["grid"], zeroline=False, showline=False, **(yaxis or {}))
    return fig


def _night_slice(hourly: pd.DataFrame, night) -> pd.DataFrame:
    return hourly[
        (hourly["time"] >= night.dusk_utc - PAD) & (hourly["time"] <= night.dawn_utc + PAD)
    ]


def median_cover(hourly: pd.DataFrame, column: str = "cloud_cover") -> pd.Series:
    return hourly.pivot_table(index="time", columns="model", values=column).median(axis=1)


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


INK_ALPHA = {"Go": 0.92, "Maybe": 0.55, "Skip": 0.26, "No data": 0.15}


def ink_level(p: float, pal: dict, go_at: float = 0.7, maybe_at: float = 0.4) -> str:
    """One neutral ink at three brightnesses: bright = likely clear, faint = likely cloudy."""
    if p is None or np.isnan(p):
        level = "No data"
    else:
        level = "Go" if p >= go_at else "Maybe" if p >= maybe_at else "Skip"
    return rgba(pal["ink"], INK_ALPHA[level])


def level_color(p: float, pal: dict, go_at: float = 0.7, maybe_at: float = 0.4) -> str:
    if p is None or np.isnan(p):
        return pal["No data"]
    return pal["Go"] if p >= go_at else pal["Maybe"] if p >= maybe_at else pal["Skip"]


# ---------- Tonight: the one chart a beginner needs ----------


def night_chart(hourly: pd.DataFrame, night, tz: str, pal: dict, go_at=0.6, maybe_at=0.35):
    """Hour by hour through the night: the chance each hour is clear (bars coloured likely
    clear / either way / likely cloudy), full darkness shaded, and a strip under the bars for
    when the Moon is up. Labels sit above the plot so they never cover a bar."""
    data = _night_slice(hourly, night)
    med = median_cover(data)
    if night.hourly_clear is not None and not night.hourly_clear.empty:
        p = night.hourly_clear
    else:
        dark = med[(med.index >= night.dusk_utc.floor("h")) & (med.index <= night.dawn_utc)]
        p = 1 - dark
    fig = go.Figure()
    start, end = night.dusk_utc - PAD, night.dawn_utc + PAD
    fig.add_vrect(
        x0=_local(night.dusk_utc, tz),
        x1=_local(night.dawn_utc, tz),
        fillcolor=pal["dark"],
        line_width=0,
        layer="below",
    )
    fig.add_trace(
        go.Bar(
            x=_local(p.index, tz),
            y=p.to_numpy() * 100,
            marker={
                "color": [level_color(v, pal, go_at, maybe_at) for v in p.to_numpy()],
                "line": {"width": 0},
                "cornerradius": 3,
                "opacity": 0.9,
            },
            hovertemplate="%{y:.0f}% chance of clear sky<extra></extra>",
            width=1000 * 60 * 60 * 0.62,
        )
    )
    # the Moon: a strip under the bars, so it can't be mistaken for cloud
    for a, b in getattr(night, "moon_up", []) or []:
        fig.add_shape(type="rect", x0=_local(max(a, start), tz), x1=_local(min(b, end), tz),
                      y0=-13, y1=-7, fillcolor=pal["moon"], line_width=0, opacity=0.75)  # fmt: skip
    if getattr(night, "moon_up", None):
        a = max(night.moon_up[0][0], start)
        fig.add_annotation(x=_local(a, tz), y=-10, text=" Moon up ", showarrow=False,
                           xanchor="right", font={"size": 11, "color": pal["muted"]})  # fmt: skip
    for when, label, anchor in [
        (night.dusk_utc, "Dark", "left"),
        (night.dawn_utc, "Dawn", "right"),
    ]:
        fig.add_vline(x=_local(when, tz), line={"color": pal["accent2"], "width": 1, "dash": "dot"})
        fig.add_annotation(
            x=_local(when, tz), y=1.0, yref="paper", yanchor="bottom",
            text=f"{label} {when.tz_convert(tz):%-I:%M %p}", showarrow=False, xanchor=anchor,
            font={"size": 11, "color": pal["accent2"]},
        )  # fmt: skip
    fig.update_xaxes(type="date", range=[_local(start, tz), _local(end, tz)])
    fig = _layout(
        fig,
        pal,
        height=300,
        time_axis=True,
        legend=False,
        yaxis={
            "range": [-15, 102],
            "tickvals": [0, 25, 50, 75, 100],
            "ticktext": ["0%", "25%", "50%", "75%", "100%"],
            "title": {"text": "chance of clear sky", "font": {"size": 11}},
        },
    )
    fig.update_layout(margin={"l": 4, "r": 4, "t": 26, "b": 4}, hovermode="closest")
    return fig


def model_grid(hourly: pd.DataFrame, night, tz: str, pal: dict, names: dict) -> go.Figure:
    """One row per weather forecast, one square per dark hour, coloured by the cloud cover it
    predicts (dark navy = clear, light grey = overcast) with the number in each square. Shows
    at a glance whether the forecasts agree."""
    data = hourly[
        (hourly["time"] >= night.dusk_utc.floor("h")) & (hourly["time"] <= night.dawn_utc)
    ]
    cover = data.pivot_table(index="time", columns="model", values="cloud_cover") * 100
    models = [m for m in names if m in cover.columns] + [m for m in cover.columns if m not in names]
    z = cover[models].T.to_numpy()
    x = [t.tz_convert(tz).strftime("%-I %p") for t in cover.index]
    y = [names.get(m, m.upper()) for m in models]
    text = [["" if np.isnan(v) else f"{v:.0f}" for v in row] for row in z]
    fig = go.Figure(
        go.Heatmap(
            z=z, x=x, y=y, text=text, texttemplate="%{text}", textfont={"size": 10},
            zmin=0, zmax=100, xgap=2, ygap=2, showscale=False,
            colorscale=[[0, "#16203A"], [0.2, "#26324E"], [0.5, "#5B6684"], [1, "#D9DCE4"]],
            hovertemplate="%{y} · %{x}: %{z:.0f}% cloud<extra></extra>",
        )
    )  # fmt: skip
    fig = _layout(fig, pal, height=60 + 34 * len(models), legend=False)
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)", ticks="")
    fig.update_xaxes(side="top", tickfont={"size": 11})
    fig.update_layout(margin={"l": 4, "r": 4, "t": 30, "b": 4}, hovermode="closest")
    return fig


def site_map(
    table: pd.DataFrame, pal: dict, go_at: float, maybe_at: float, light: bool = False
) -> go.Figure:
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
                textfont={
                    "color": pal.get("text") or ("#2A2A2E" if light else "#D5D5DA"),
                    "size": 12,
                    "family": MAP_FONT,
                },
                marker={"size": 13, "color": level_color(p, pal, go_at, maybe_at), "opacity": 0.95},
                hovertemplate="%{text}<extra></extra>",
            )
        )
    fig.update_layout(
        map={
            "style": "carto-positron" if light else "carto-darkmatter",
            "center": {"lat": 38.1, "lon": -119.9},
            "zoom": 5.9,
        },
        height=360,
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
                line={
                    "color": pal["methods"]["blend" if is_blend else "climatology"],
                    "width": 3 if is_blend else 1.5,
                    "dash": "solid" if is_blend else "dot",
                },
                marker={"size": 8 if is_blend else 5},
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "thickness": 1,
                    "array": (100 * (g["false_clear_rate"] - g["false_clear_rate_lo"])).fillna(0),
                    "arrayminus": (100 * (g["false_clear_rate_hi"] - g["false_clear_rate"])).fillna(
                        0
                    ),
                }
                if is_blend
                else None,
                hovertemplate="%{x} day(s) ahead: %{y:.0f}% of 'go' nights were usable"
                "<extra></extra>",
            )
        )
    fig.update_xaxes(title="Days ahead", dtick=1)
    return _layout(
        fig,
        pal,
        "When it said “go”, how often the night was usable",
        height=320,
        yaxis={"range": [50, 103], "tickvals": [50, 60, 70, 80, 90, 100], "ticksuffix": "%"},
    )


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
        yaxis={"title": "Observed frequency", "range": [-0.02, 1.04]},
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
        yaxis={"range": [-20, 104], "title": "%"},
    )


# ---------- light pollution ----------

# Representative overlay colours (from lightpollution.OVERLAY_RGBA) for the map key.
GLOW_KEY = [
    ("dark: near natural", "rgba(0,0,0,0.0)"),
    ("some light near the horizon", "rgba(130,104,70,0.6)"),
    ("up to 2× natural", "rgba(196,150,80,0.75)"),
    ("Milky Way hidden", "rgba(238,202,128,0.85)"),
    ("city sky", "rgba(252,244,222,0.95)"),
]


def two_line_label(label: str) -> str:
    """ "Name · tier · 75%" -> "Name" over "tier · 75%" (map labels otherwise wrap anywhere)."""
    parts = label.split(" · ")
    return parts[0] if len(parts) == 1 else parts[0] + "<br>" + " · ".join(parts[1:])


def label_anchor(lat: float, lon: float, position: str, zoom: float, px: float = 6.0):
    """A point `px` screen pixels from a marker on the side `position` names ("top left",
    "middle right"...). Web Mercator with MapLibre's 512-pixel tiles: the world is 512·2^zoom
    pixels wide, and a degree of
    latitude covers 1/cos(lat) times more pixels than a degree of longitude."""
    deg = px * 360.0 / (512.0 * 2**zoom)
    vertical, horizontal = (position.split() + ["center"])[:2]
    up = 1.7 * deg * math.cos(math.radians(lat))  # a little more room above/below the dot
    if vertical == "top":
        lat += up
    elif vertical == "bottom":
        lat -= up
    if horizontal == "right":
        lon += deg
    elif horizontal == "left":
        lon -= deg
    return lat, lon


def light_map(
    grid,
    overlay_png: bytes,
    markers: pd.DataFrame,
    pal: dict,
    center: tuple[float, float] = (38.0, -120.0),
    zoom: float = 5.3,
    height: int = 460,
    spots: pd.DataFrame | None = None,
    pickable: pd.DataFrame | None = None,
) -> go.Figure:
    """The atlas as a warm glow over a dark base map (light pollution is a night-time view, so
    the base stays dark in both themes), with site markers and optional darker-sky spots.

    `pickable` (columns lat, lon, id, label, visible): points a click can open. Visible ones are
    faint town dots; invisible ones fill the gaps between towns, so a click anywhere lands on
    the nearest town. The place id travels as the point's customdata.

    `markers` columns: lat, lon, label, color, position. `spots` columns: lat, lon, label and
    optionally position."""
    import base64

    src = "data:image/png;base64," + base64.b64encode(overlay_png).decode()
    corners = [
        [grid.west, grid.north],
        [grid.east, grid.north],
        [grid.east, grid.south],
        [grid.west, grid.south],
    ]
    fig = go.Figure()
    if pickable is not None and len(pickable):
        for visible, sub in pickable.groupby("visible"):
            fig.add_trace(
                go.Scattermap(
                    lat=sub["lat"],
                    lon=sub["lon"],
                    mode="markers",
                    customdata=sub["id"],
                    text=sub["label"],
                    hovertemplate="%{text}<extra></extra>",
                    marker={"size": 3, "color": "#D9DCE4", "opacity": 0.22}
                    if visible
                    else {"size": 12, "color": "#D9DCE4", "opacity": 0.01},
                )
            )
    for _, r in (spots if spots is not None else pd.DataFrame()).iterrows():
        fig.add_trace(
            go.Scattermap(
                lat=[r["lat"]],
                lon=[r["lon"]],
                mode="markers+text",
                text=[r["label"]],
                textposition=r.get("position", "bottom center"),
                textfont={"color": "#CFCFD4", "size": 11, "family": MAP_FONT},
                marker={"size": 9, "color": "#CFCFD4", "symbol": "circle", "opacity": 0.9},
                hovertemplate="%{text}<extra></extra>",
            )
        )
    for _, r in markers.iterrows():
        text = two_line_label(r["label"])
        fig.add_trace(
            go.Scattermap(
                lat=[r["lat"]],
                lon=[r["lon"]],
                mode="markers",
                text=[text],
                customdata=[r.get("site")],  # a click on a place opens it
                marker={"size": 13, "color": r["color"], "opacity": 0.95},
                hovertemplate="%{text}<extra></extra>",
            )
        )
        # The label is drawn from a point nudged away from the dot: map labels get no gap of
        # their own, so the dot covered the end of "Los Angeles". (The map engine trims spaces,
        # so padding the text doesn't work; a text-only trace isn't drawn, hence the invisible
        # marker.)
        lat, lon = label_anchor(r["lat"], r["lon"], r["position"], zoom)
        fig.add_trace(
            go.Scattermap(
                lat=[lat],
                lon=[lon],
                mode="markers+text",
                text=[text],
                textposition=r["position"],
                textfont={"color": "#E6E6EA", "size": 12, "family": MAP_FONT},
                marker={"size": 1, "opacity": 0},
                hoverinfo="skip",
            )
        )
    fig.update_layout(
        map={
            "style": "carto-darkmatter",
            "center": {"lat": center[0], "lon": center[1]},
            "zoom": zoom,
            # keep the view inside the mapped region, so its edge never shows
            "bounds": {
                "west": grid.west,
                "east": grid.east,
                "south": grid.south,
                "north": grid.north,
            },
            "layers": [
                {
                    "sourcetype": "image",
                    "source": src,
                    "coordinates": corners,
                    "below": "traces",
                    "opacity": 0.95,
                }
            ],
        },
        height=height,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor=pal["paper"],
        showlegend=False,
        clickmode="event+select" if pickable is not None else "event",
    )
    return fig


def bortle_bar(shares: dict[str, float], words: dict[str, str] | None = None) -> str:
    """Stacked horizontal bar (HTML) of the area share in each darkness band; `words` relabels
    the atlas's Bortle bands in plain language."""
    tones = {"1–3": 0.14, "4–4.5": 0.38, "5–6": 0.62, "7–9": 0.9}
    name = (lambda k: words.get(k, k)) if words else (lambda k: f"Bortle {k}")
    segs = "".join(
        f'<span title="{name(k)}: {v:.0%}" style="width:{v * 100:.2f}%;'
        f'background:color-mix(in srgb, currentColor {tones[k] * 100:.0f}%, transparent)"></span>'
        for k, v in shares.items()
        if v > 0
    )
    labels = "".join(
        f'<span><span class="sk-swatch" style="--c:color-mix(in srgb, currentColor '
        f'{tones[k] * 100:.0f}%, transparent)"></span>{name(k)} · {v:.0%}</span>'
        for k, v in shares.items()
    )
    return f'<div class="sk-stack">{segs}</div><div class="sk-legend">{labels}</div>'


def dome_polar(glow: dict, pal: dict, height: int = 360) -> go.Figure:
    """Light domes around the horizon: Walker's-law strength per direction on a log scale, with
    1 = Sacramento's dome seen from 50 km. North is up, east to the right, like a compass."""
    sectors = glow["sectors"]
    r = [max(s["dome"], 0.003) for s in sectors]
    alpha = [min(0.95, 0.25 + 0.2 * np.log10(max(v, 0.003) / 0.003)) for v in r]
    fig = go.Figure(
        go.Barpolar(
            r=r,
            theta=[s["direction"] for s in sectors],
            marker={"color": [rgba(pal["accent"], a) for a in alpha], "line": {"width": 0}},
            customdata=[[s["dome"], 100 * s["overhead_share"]] for s in sectors],
            hovertemplate="%{theta}: dome %{customdata[0]:.2f}× · "
            "%{customdata[1]:.0f}% of overhead glow<extra></extra>",
        )
    )
    fig.update_layout(
        polar={
            "bgcolor": "rgba(0,0,0,0)",
            "angularaxis": {
                "direction": "clockwise",
                "rotation": 90,
                "gridcolor": pal["grid"],
                "linecolor": pal["border"],
                "tickfont": {"size": 11},
            },
            "radialaxis": {
                "type": "log",
                "range": [-2.5, 2],
                "gridcolor": pal["grid"],
                "tickvals": [0.1, 1, 10],
                "ticktext": ["0.1×", "1×", "10×"],
                "tickfont": {"size": 10, "color": pal["muted"]},
                "angle": 45,
                "tickangle": 45,
                "showline": False,
            },
        },
        height=height,
        margin={"l": 30, "r": 30, "t": 20, "b": 20},
        paper_bgcolor=pal["paper"],
        font={"family": FONT, "size": 12} | ({"color": pal["text"]} if pal.get("text") else {}),
        showlegend=False,
    )
    return fig


# ---------- statewide accuracy ----------

# Dim to bright starlight gold: low skill recedes into the dark map, high skill stands out. One
# hue with rising lightness reads the same for colour-blind viewers.
SKILL_SCALE = [[0.0, "#4B4231"], [0.5, "#A88A4E"], [1.0, "#F6DFA6"]]


def station_skill_map(
    stations: pd.DataFrame, pal: dict, here: tuple[float, float, str] | None = None
) -> go.Figure:
    """Every network station coloured by forecast skill (BSS, scored as a never-seen place).

    `stations` columns: lat, lon, label (hover text), bss."""
    fig = go.Figure(
        go.Scattermap(
            lat=stations["lat"],
            lon=stations["lon"],
            mode="markers",
            text=stations["label"],
            hovertemplate="%{text}<extra></extra>",
            marker={
                "size": 13,
                "color": stations["bss"].clip(0, 1),
                "colorscale": SKILL_SCALE,
                "cmin": 0,
                "cmax": 1,
                "opacity": 0.95,
                "colorbar": {
                    "title": {"text": "Skill", "font": {"size": 11}},
                    "tickvals": [0, 0.5, 1],
                    "ticktext": ["0 (season)", "0.5", "1 (perfect)"],
                    "thickness": 10,
                    "len": 0.5,
                    "x": 0.98,
                    "tickfont": {"size": 10},
                },
            },
        )
    )
    if here is not None:
        fig.add_trace(
            go.Scattermap(
                lat=[here[0]],
                lon=[here[1]],
                mode="markers+text",
                text=[here[2]],
                textposition="top right",
                textfont={"color": "#E6E6EA", "size": 12, "family": MAP_FONT},
                marker={"size": 11, "color": pal["accent"], "symbol": "circle"},
                hovertemplate="%{text}<extra></extra>",
            )
        )
    fig.update_layout(
        map={"style": "carto-darkmatter", "center": {"lat": 37.2, "lon": -119.4}, "zoom": 4.6},
        height=520,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor=pal["paper"],
        showlegend=False,
    )
    return fig
