"""Where to go: every site ranked by tonight's conditions (clear sky, dark site, Moon down), a
light-pollution map of the region, and a light-pollution report for the selected location with
the darkest skies nearby."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from skytrust import lightpollution
from views import charts
from views import components as ui
from views.common import (
    Context,
    conditions,
    maps_link,
    place_detail,
    place_name,
    short_time,
    upcoming,
)
from views.tonight import show

CLIMATOLOGY = Path(__file__).resolve().parents[2] / "artifacts" / "climatology.json"
COLS = "28px minmax(0, 2fr) minmax(90px, 1.1fr) 132px 236px"
NEAR_COLS = "minmax(0, 2fr) minmax(0, 1.6fr) 64px 110px"


# ---------- tonight, every site ----------


def site_rows(ctx: Context) -> pd.DataFrame:
    rows = []
    for site in ctx.sites:
        forecast, _ = ctx.forecast_for(site.id)
        nights = upcoming(forecast, ctx.now_utc)
        light = ctx.light_at(site)
        here = (light or {}).get("here") or {}
        row = {
            "site": site.id,
            "place": place_name(site),
            "area": place_detail(site),
            "lat": site.lat,
            "lon": site.lon,
            "bortle": here.get("bortle"),
            "sqm": here.get("sqm"),
            "ratio": here.get("ratio"),
            "p": None,
            "verdict": "No data",
            "window": "–",
            "moon_free": None,
            "illum": None,
            "conds": [],
            "met": 0,
        }
        if nights:
            n = nights[0]
            bw = n.best_window
            conds = conditions(n, light, ctx.settings)
            row |= {
                "p": n.p_usable,
                "verdict": n.verdict,
                "window": (
                    f"{short_time(bw.start_utc, site.timezone)}–"
                    f"{short_time(bw.until_utc, site.timezone)}"
                    if bw
                    else "no clear window"
                ),
                "moon_free": n.moon_free_hours,
                "illum": n.moon_illum,
                "conds": conds,
                "met": sum(1 for _, ok, _ in conds if ok),
            }
        rows.append(row)
    df = pd.DataFrame(rows)
    df["_p"] = df["p"].astype(float).fillna(-1)
    df["_sqm"] = df["sqm"].astype(float).fillna(0)
    return df.sort_values(["met", "_p", "_sqm"], ascending=False).drop(columns=["_p", "_sqm"])


def condition_chips(conds) -> str:
    chips = []
    for name, ok, detail in conds:
        mark = "✓" if ok else "–" if ok is None else "✕"
        tone = 100 if ok else 45
        chips.append(
            f'<span class="sk-cond" title="{ui.esc(detail)}" '
            f'style="color:color-mix(in srgb, currentColor {tone}%, transparent)">'
            f"{mark} {ui.esc(name)}</span>"
        )
    return f'<div class="sk-conds">{"".join(chips)}</div>'


def rank_row(i: int, r: pd.Series, pal: dict) -> str:
    color = pal[r["verdict"]]
    p = None if pd.isna(r["p"]) else float(r["p"])
    sky = "–"
    if r["bortle"]:
        sky = (
            f'<div class="sk-row-title">Bortle {ui.esc(r["bortle"])}</div>'
            f'<div class="sk-row-sub">{r["sqm"]:.2f} mag/arcsec²</div>'
        )
    chance = "–" if p is None else f"{p:.0%}"
    return ui.row(
        [
            f'<div class="sk-rank">{i:02d}</div>',
            f'<div class="sk-main"><div class="sk-row-title">{ui.esc(r["place"])}</div>'
            f'<div class="sk-row-sub">{ui.esc(r["area"])} · clearest {ui.esc(r["window"])}'
            "</div></div>",
            f'<div class="sk-grow">{ui.bar(p, color)}'
            '<div class="sk-row-sub" style="margin-top:6px">'
            f"{chance} chance clear</div></div>",
            f"<div>{sky}</div>",
            condition_chips(r["conds"]),
        ],
        COLS,
    )


def marker_color(met: int, pal: dict) -> str:
    return pal["Go"] if met == 3 else pal["Maybe"] if met == 2 else pal["Skip"]


def marker_label(r: pd.Series) -> str:
    chance = "–" if pd.isna(r["p"]) else f"{r['p']:.0%}"
    return f"{r['place']} · B{r['bortle']} · {chance}"


# ---------- light pollution at the selected place ----------


def light_strip(report: dict, sigma: float) -> str:
    h = report["here"]
    added = f"{h['ratio']:.0%}" if h["ratio"] < 10 else f"{h['ratio']:.0f}×"
    return ui.strip(
        [
            ui.stat(
                "Sky quality",
                f"{h['sqm']:.2f} <span class='sk-muted' style='font-size:.8em'>"
                f"± {sigma:.2f}</span>",
                "mag/arcsec² at the zenith (higher = darker; 22.0 is a natural sky)",
            ),
            ui.stat("Bortle class", f"Class {ui.esc(h['bortle'])}", ui.esc(h["bortle_title"])),
            ui.stat(
                "Artificial light",
                added,
                "added on top of the natural sky brightness"
                + (f", so the sky is {1 + h['ratio']:.0f}× brighter" if h["ratio"] >= 1 else ""),
            ),
            ui.stat("Atlas level", ui.esc(h["level_name"]), ui.esc(h["level_text"])),
        ]
    )


def nearby_rows(report: dict) -> str:
    items = []
    seen = set()
    entries = [(f"Darkest within {r} km", d) for r, d in report["darkest"].items()]
    if report.get("nearest_dark"):
        entries.append(("Nearest Bortle 1–3 sky", report["nearest_dark"]))
    for title, d in entries:
        if not d:
            continue
        key = (round(d["lat"], 3), round(d["lon"], 3))
        if key in seen:
            continue
        seen.add(key)
        where = (
            "right here"
            if d["distance_km"] < 1.5
            else f"{d['distance_km']:.0f} km {d['direction']}"
        )
        link = maps_link(d["lat"], d["lon"])
        items.append(
            ui.row(
                [
                    f'<div class="sk-main"><div class="sk-row-title">{ui.esc(title)}</div>'
                    f'<div class="sk-row-sub">{where} · straight-line distance</div></div>',
                    f'<div><div class="sk-row-title">Bortle {ui.esc(d["bortle"])}</div>'
                    f'<div class="sk-row-sub">{ui.esc(d["bortle_title"])}</div></div>',
                    f'<div class="sk-row-p">{d["sqm"]:.2f}</div>',
                    f'<div style="text-align:right"><a href="{link}" target="_blank" '
                    'rel="noopener">Open map ↗</a></div>',
                ],
                NEAR_COLS,
            )
        )
    return ui.rows(items)


def methodology(ctx: Context) -> None:
    cfg = ctx.settings.raw["light_pollution"]
    nat_ucd, nat_sqm = cfg["natural_ucd"], cfg["natural_sqm"]
    st.markdown(
        f"""
**Source.** The *World Atlas of Artificial Night Sky Brightness* (Falchi et al., 2016,
*Science Advances* 2:e1600377; data doi:10.5880/GFZ.1.4.2016.001, CC BY-NC 4.0). It models the
artificial glow of the zenith sky on a 30-arcsecond grid (about 0.7 × 0.9 km here) from VIIRS
satellite measurements of upward light and a model of how that light scatters in the atmosphere.

**How the numbers are made.** The paper assumes a natural sky of {nat_ucd:.0f} µcd/m²
({nat_sqm:.1f} mag/arcsec²). Adding the atlas's artificial brightness gives the total, in the
sky-quality-meter unit astronomers use: SQM = {nat_sqm:.1f} − 2.5·log₁₀((artificial +
{nat_ucd:.0f}) / {nat_ucd:.0f}). The atlas was checked against sky quality meter readings with a
standard deviation of **±{cfg["sigma_sqm"]} mag/arcsec²**; that is the ± shown above. Bortle
classes use the SQM ranges tabulated for the Bortle (2001) scale, which is a visual scale, so the
class is approximate.

**What it can't tell you.**
- The satellite data are from 2014–2015. Skies have brightened since (LED conversions and
  growth), so treat values near cities as a best case.
- It's the sky straight up. Domes of light from cities show near the horizon even at dark sites.
- "Darkest nearby" is straight-line distance on a ~1 km grid: it ignores roads, land access,
  terrain blocking the horizon, and local lights.
- Tonight's conditions combine this map with the cloud forecast and the Moon; the cloud part is
  what SkyTrust's track record measures.
"""
    )


def month_rates(month: int, path: Path = CLIMATOLOGY) -> tuple[dict[str, float], str]:
    """Long-term share of usable nights per site for a calendar month, and the years it covers
    (from artifacts/climatology.json)."""
    try:
        table = json.loads(path.read_text())
        tables, (first, last) = table["tables"]["primary"], table["period"]
    except (OSError, KeyError, ValueError):
        return {}, ""
    rates = {site: m[str(month)]["rate"] for site, m in tables.items() if str(month) in m}
    return rates, f"{first[:4]}–{last[:4]}"


def climate_tip(ctx: Context) -> str:
    now = ctx.now_utc if ctx.now_utc is not None else pd.Timestamp.now(tz="UTC")
    month = now.tz_convert(ctx.site.timezone).month
    rates, years = month_rates(month)
    names = {s.id: place_name(s) for s in ctx.sites}
    ranked = sorted(((r, s) for s, r in rates.items() if s in names), reverse=True)
    if not ranked:
        return ""
    listing = ", ".join(f"{names[s]} {r:.0%}" for r, s in ranked)
    return (
        f"In a typical {now.tz_convert(ctx.site.timezone):%B} ({years} observations), the share "
        f"of usable nights was: {listing}."
    )


# ---------- page ----------


def local_map(ctx: Context, report: dict, overlay: bytes) -> None:
    pal = ctx.palette
    spots = [
        (d["lat"], d["lon"], f"{d['sqm']:.2f} · B{d['bortle']}")
        for d in [*report["darkest"].values(), report.get("nearest_dark")]
        if d and d["distance_km"] >= 1.5
    ]
    spot_df = pd.DataFrame(spots, columns=["lat", "lon", "label"]).drop_duplicates(["lat", "lon"])
    here = pd.DataFrame(
        {
            "lat": [ctx.site.lat],
            "lon": [ctx.site.lon],
            "label": [ctx.site_label],
            "color": [pal["accent"]],
            "position": ["top right"],
        }
    )
    fig = charts.light_map(
        ctx.light, overlay, here, pal, center=(ctx.site.lat, ctx.site.lon), zoom=7.4, height=380,
        spots=spot_df,
    )  # fmt: skip
    show(fig)


def all_sites_table(table: pd.DataFrame) -> None:
    shown = pd.DataFrame(
        {
            "place": table["place"],
            "area": table["area"],
            "chance clear": table["p"].map(lambda p: None if pd.isna(p) else round(100 * p)),
            "verdict": table["verdict"],
            "best window": table["window"],
            "Bortle": table["bortle"],
            "SQM (mag/arcsec²)": table["sqm"].map(lambda m: None if pd.isna(m) else round(m, 2)),
            "added light vs natural": table["ratio"].map(
                lambda r: None if pd.isna(r) else f"{r:.0%}"
            ),
            "moon-free dark h": table["moon_free"],
            "conditions met": table["met"].map(lambda m: f"{m}/3"),
        }
    )
    st.dataframe(
        shown,
        hide_index=True,
        width="stretch",
        column_config={
            "chance clear": st.column_config.ProgressColumn(
                "chance clear", format="%d%%", min_value=0, max_value=100
            )
        },
    )


def render(ctx: Context) -> None:
    st.header("Where should I go tonight?")
    st.caption(
        "A good deep-sky night needs three things: a clear sky, a dark site, and the Moon down. "
        "Sites are ranked by how many they meet tonight, then by the chance of clear sky."
    )
    table = site_rows(ctx)
    pal = ctx.palette
    rows = [rank_row(i + 1, r, pal) for i, (_, r) in enumerate(table.iterrows())]
    st.markdown(ui.rows(rows), unsafe_allow_html=True)

    path = lightpollution.OVERLAY_PATH
    overlay = path.read_bytes() if path.exists() else None
    if ctx.light is not None and overlay is not None:
        st.subheader("Light pollution across the region")
        markers = pd.DataFrame(
            {
                "lat": table["lat"],
                "lon": table["lon"],
                "label": [marker_label(r) for _, r in table.iterrows()],
                "color": [marker_color(m, pal) for m in table["met"]],
                "position": [charts.LABEL_POSITIONS.get(s, "top right") for s in table["site"]],
            }
        )
        show(charts.light_map(ctx.light, overlay, markers, pal))
        key = [("marker: all 3 conditions", pal["Go"]), ("2 of 3", pal["Maybe"]),
               ("1 or none", pal["Skip"])]  # fmt: skip
        st.markdown(ui.legend(charts.GLOW_KEY) + ui.legend(key), unsafe_allow_html=True)

    report = ctx.light_at()
    st.subheader(f"Light pollution at {ctx.site_label}")
    if report is None:
        st.info("No light-pollution data for this location (outside the mapped region).")
    else:
        sigma = ctx.settings.raw["light_pollution"]["sigma_sqm"]
        st.markdown(light_strip(report, sigma), unsafe_allow_html=True)
        st.markdown(
            f'<p class="sk-lede" style="margin-top:4px">What you can expect to see: '
            f"{ui.esc(report['here']['bortle_text'])}.</p>",
            unsafe_allow_html=True,
        )
        st.markdown(ui.eyebrow("Darker skies nearby"), unsafe_allow_html=True)
        st.markdown(nearby_rows(report), unsafe_allow_html=True)
        st.markdown(
            ui.eyebrow(f"Sky darkness within {report['shares_radius_km']} km")
            + charts.bortle_bar(report["shares"]),
            unsafe_allow_html=True,
        )
        if overlay is not None:
            local_map(ctx, report, overlay)

    with st.expander("How light pollution is measured, and its limits", icon=":material/info:"):
        methodology(ctx)
    with st.expander("All sites as a table", icon=":material/table_rows:"):
        all_sites_table(table)
    tip = climate_tip(ctx)
    if tip:
        st.caption(tip)
