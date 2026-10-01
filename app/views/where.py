"""Where to go: every site ranked by tonight's conditions (clear sky, dark site, Moon down), a
light-pollution map of the region, and a light-pollution report for the selected location with
the darkest skies nearby."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
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
        light = ctx.light_at(site, glow=False)
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
    if pd.notna(r["bortle"]) and pd.notna(r["sqm"]):
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
    sky = f" · B{r['bortle']}" if pd.notna(r["bortle"]) else ""
    return f"{r['place']}{sky} · {chance}"


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
            change_stat(report),
        ]
    )


def change_stat(report: dict) -> str:
    ch = report.get("change")
    h = report["here"]
    if not ch:
        return ui.stat("Atlas level", ui.esc(h["level_name"]), ui.esc(h["level_text"]))
    pct = ch["artificial_ratio"] - 1
    direction = "more" if pct >= 0 else "less"
    if abs(ch["delta_mag"]) < 0.005:
        trend = "no change"
    else:  # magnitudes run backwards: a lower number is a brighter sky
        trend = "brighter" if ch["delta_mag"] < 0 else "darker"
    return ui.stat(
        "Since 2015",
        f"{ch['delta_mag']:+.2f} mag <span class='sk-muted' style='font-size:.8em'>{trend}</span>",
        f"{abs(pct):.0%} {direction} artificial light than the 2015 atlas "
        f"(was {ch['sqm_base']:.2f}, Bortle {ui.esc(ch['bortle_base'])})",
    )


def glow_rows(glow: dict) -> str:
    items = []
    for c in glow["cities"]:
        items.append(
            ui.row(
                [
                    f'<div class="sk-main"><div class="sk-row-title">{ui.esc(c["name"])}</div>'
                    f'<div class="sk-row-sub">{c["distance_km"]:.0f} km {c["direction"]} · '
                    f"population {c['population']:,}</div></div>",
                    f'<div class="sk-row-p">{c["strength"]:.2f}×</div>',
                    f'<div class="sk-row-sub" style="text-align:right">{c["share"]:.0%} of the '
                    "dome light</div>",
                ],
                "minmax(0, 2fr) 72px 140px",
            )
        )
    if glow["scattered_share"] >= 0.01:
        items.append(
            ui.row(
                [
                    '<div class="sk-main"><div class="sk-row-title">Scattered lights</div>'
                    '<div class="sk-row-sub">roads, farms and buildings outside towns</div></div>',
                    "<div></div>",
                    f'<div class="sk-row-sub" style="text-align:right">'
                    f"{glow['scattered_share']:.0%} of the dome light</div>",
                ],
                "minmax(0, 2fr) 72px 140px",
            )
        )
    return ui.rows(items)


def glow_summary(glow: dict) -> str:
    """One sentence: where to point, and the biggest dome."""
    quarter = f"{glow['darkest_quarter'][0]}–{glow['darkest_quarter'][-1]}"
    text = f"The darkest part of the horizon is <b>{quarter}</b>: frame your targets there."
    if glow["cities"]:
        c = glow["cities"][0]
        text += (
            f" The strongest light dome is {ui.esc(c['name'])}, {c['distance_km']:.0f} km "
            f"{c['direction']}."
        )
    shares = [s["overhead_share"] for s in glow["sectors"]]
    n = len(shares)
    wedge = max(range(n), key=lambda i: shares[i - 1] + shares[i] + shares[(i + 1) % n])
    share = shares[wedge - 1] + shares[wedge] + shares[(wedge + 1) % n]
    text += (
        f" Of the artificial glow overhead, {share:.0%} comes from the "
        f"{glow['sectors'][wedge]['direction']} (a 67° wedge)."
    )
    return f'<p class="sk-lede" style="margin:6px 0 4px">{text}</p>'


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
    model = (ctx.light_data or {}).get("model") or {}
    cv = model.get("spatial_cv", {})
    checks = model.get("validation_2025_atlas") or {}
    val, val0 = checks.get("updated", {}), checks.get("atlas_2016", {})
    ratio = model.get("ratio", {})
    year = model.get("year", cfg.get("year"))
    nan = float("nan")
    st.markdown(
        f"""
**1 · The calibrated atlas.** The *World Atlas of Artificial Night Sky Brightness* (Falchi et al.,
2016, *Science Advances* 2:e1600377; data doi:10.5880/GFZ.1.4.2016.001, CC BY-NC 4.0) models the
artificial glow of the zenith sky on a 30″ grid (~0.7 × 0.9 km here) and was checked against sky
quality meter readings with a standard deviation of ±{cfg["sigma_sqm"]} mag/arcsec². Its lights
are from 2014–2015.

**2 · Brought up to {year}.** NASA's Black Marble annual night lights (VNP46A4 / VJ146A4, CC0;
GeoTIFFs from lightpollutionmap.info) give the lights for 2015 and {year}. SkyTrust learns how
light spreads through the air by fitting the atlas from the 2015 lights: each place's glow is a
sum over every light within 300 km of its brightness × a kernel K(distance), with K fitted by
non-negative least squares. Held out region by region, that reproduces the atlas to
{cv.get("rmse_mag", nan):.3f} mag RMS ({cv.get("share_within_0_15", nan):.0%} of places within
0.15). The {year} sky is then *atlas × modelled {year} glow ÷ modelled 2015 glow*, so the atlas
keeps its calibration and altitude handling and only the change in lights comes from the model.
In lit areas the median change is ×{ratio.get("median_lit", nan):.2f}.

**3 · Checked against an independent {year} model.** David Lorenz's {year} re-calculation of the
atlas agrees with the updated grid within one of its zones in
{val.get("lit_within_one_zone", nan):.0%} of lit places (the 2015 atlas alone:
{val0.get("lit_within_one_zone", nan):.0%}). In lit areas it reads
{val.get("lit_bias_mag", nan):.2f} mag brighter than SkyTrust on average, so treat city values as
a best case.

**4 · Units.** Natural sky {nat_ucd:.0f} µcd/m² = {nat_sqm:.1f} mag/arcsec² (the paper's value),
so SQM = {nat_sqm:.1f} − 2.5·log₁₀((artificial + {nat_ucd:.0f}) / {nat_ucd:.0f}). Bortle classes
use the SQM ranges tabulated for the Bortle (2001) scale, a visual scale, so the class is
approximate.

**5 · City glow.** The light domes on the horizon use Walker's law (Walker 1977, *PASP* 89:405):
the glow toward a city falls off as distance^−2.5. Each light within 300 km is weighted that way
(lights within 2 km are local lighting; closer than 10 km count as 10 km, the range where the law
was measured), grouped into 16 directions and named after the town it falls in (GeoNames, places
of 1,000+ people). Strengths are relative: 1× is Sacramento's dome seen from 50 km.

**What it can't tell you.**
- Satellites are nearly blind to the blue light of white LEDs, so they understate brightening:
  citizen observations found skies brightening 9.6% a year in 2011–2022 (Kyba et al., 2023,
  *Science* 379:265), faster than satellites show.
- It models the sky straight up and relative dome strength, not exact horizon brightness.
- "Darkest nearby" is straight-line distance on a ~1 km grid: it ignores roads, land access,
  terrain blocking the horizon, and local lights.
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


def label_side(lat: float, lon: float, site_lat: float, site_lon: float) -> str:
    """Put a spot's label on the side facing away from the site, so it never covers the site's
    own marker and label."""
    dy, dx = lat - site_lat, (lon - site_lon) * np.cos(np.radians(site_lat))
    if abs(dx) > abs(dy):
        return "middle right" if dx > 0 else "middle left"
    return "top center" if dy > 0 else "bottom center"


def map_spots(report: dict, site_lat: float, site_lon: float, min_gap_km: float = 4.0):
    """The darker-sky spots for the local map, darkest first, skipping any within `min_gap_km`
    of one already shown (their dots and labels would sit on top of each other)."""
    found = [d for d in [*report["darkest"].values(), report.get("nearest_dark")]
             if d and d["distance_km"] >= 1.5]  # fmt: skip
    kept: list[dict] = []
    for d in sorted(found, key=lambda d: -d["sqm"]):
        near = lightpollution.haversine_km(
            d["lat"],
            d["lon"],
            np.array([k["lat"] for k in kept]),
            np.array([k["lon"] for k in kept]),
        )
        if not kept or float(np.min(near)) >= min_gap_km:
            kept.append(d)
    return pd.DataFrame(
        {
            "lat": [d["lat"] for d in kept],
            "lon": [d["lon"] for d in kept],
            "label": [f"{d['sqm']:.2f} · B{d['bortle']}" for d in kept],
            "position": [label_side(d["lat"], d["lon"], site_lat, site_lon) for d in kept],
        }
    )


def local_map(ctx: Context, report: dict, overlay: bytes) -> None:
    pal = ctx.palette
    spot_df = map_spots(report, ctx.site.lat, ctx.site.lon)
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
        glow = report.get("glow")
        if glow:
            st.subheader("City glow: light domes on the horizon")
            st.markdown(glow_summary(glow), unsafe_allow_html=True)
            left, right = st.columns([1, 1.15], gap="large", vertical_alignment="center")
            with left:
                show(charts.dome_polar(glow, ctx.palette))
            with right:
                st.markdown(ui.eyebrow("Biggest light domes") + glow_rows(glow),
                            unsafe_allow_html=True)  # fmt: skip
                st.caption(
                    "Dome strength by Walker's law (glow ∝ light ÷ distance²·⁵), where 1× is "
                    "Sacramento's dome seen from 50 km. Total for this spot: "
                    f"{glow['dome_total']:.2f}×."
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
