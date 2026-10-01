"""Where to go: every place ranked by tonight's conditions (clear sky, dark sky, Moon down), a
light-pollution map of California, and for the chosen place how dark it is in plain words, where
the nearest darker skies are, and which towns light up its horizon."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from skytrust import lightpollution, sky
from views import charts, lookup
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

COLS = "28px minmax(0, 2fr) minmax(90px, 1.1fr) 150px 236px"
NEAR_COLS = "minmax(0, 2fr) minmax(0, 1.6fr) 110px"

# The light-pollution atlas's area shares come in Bortle bands; shown in the plain tiers they
# correspond to (sky.TIERS uses the same SQM boundaries).
BAND_WORDS = {
    "1–3": "Very dark",
    "4–4.5": "Dark",
    "5–6": "Suburban / bright",
    "7–9": "Bright / city",
}


# ---------- tonight, every place ----------


def ranked_sites(ctx: Context) -> list:
    """The featured places, plus the chosen place when it isn't one of them (any town or ZIP
    code typed in the menu), so people can see where their own sky stands."""
    sites = list(ctx.sites)
    if ctx.site.id not in {s.id for s in sites}:
        sites.append(ctx.site)
    return sites


def _forecast(ctx: Context, site):
    from skytrust import live

    if live.is_custom(site):  # exact coordinates travel as lat/lon, not as an id
        return ctx.forecast_for(live.CUSTOM_ID, site.lat, site.lon, site.name)
    return ctx.forecast_for(site.id)


def site_rows(ctx: Context) -> pd.DataFrame:
    rows = []
    for site in ranked_sites(ctx):
        forecast, _ = _forecast(ctx, site)
        nights = upcoming(forecast, ctx.now_utc)
        sqm = ctx.sqm_at(site) if ctx.light is not None else None
        d = sky.darkness(sqm) if sqm is not None else None
        row = {
            "site": site.id,
            "place": place_name(site),
            "yours": site.id == ctx.site.id,
            "area": place_detail(site),
            "lat": site.lat,
            "lon": site.lon,
            "sqm": sqm,
            "tier": d.title if d else None,
            "times": d.times_natural if d else None,
            "p": None,
            "verdict": "No data",
            "window": "–",
            "moon_free": None,
            "conds": [],
            "met": 0,
        }
        if nights:
            n = nights[0]
            bw = n.best_window
            conds = conditions(n, sqm, ctx.settings)
            row |= {
                "p": n.p_usable,
                "verdict": n.verdict,
                "window": window_text(bw, site.timezone),
                "moon_free": n.moon_free_hours,
                "conds": conds,
                "met": sum(1 for _, ok, _ in conds if ok),
            }
        rows.append(row)
    df = pd.DataFrame(rows)
    df["_p"] = df["p"].astype(float).fillna(-1)
    df["_sqm"] = df["sqm"].astype(float).fillna(0)
    return df.sort_values(["met", "_p", "_sqm"], ascending=False).drop(columns=["_p", "_sqm"])


def window_text(bw, tz: str) -> str:
    if not bw:
        return "no clear window"
    return f"{short_time(bw.start_utc, tz)}–{short_time(bw.until_utc, tz)}"


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


def times_words(times: float) -> str:
    return (
        "natural sky"
        if times < 1.05
        else f"{times:.1f}× natural"
        if times < 10
        else f"{times:.0f}× natural"
    )


def rank_row(i: int, r: pd.Series, pal: dict) -> str:
    color = pal[r["verdict"]]
    p = None if pd.isna(r["p"]) else float(r["p"])
    sky_cell = "–"
    if r["tier"] is not None and not pd.isna(r["times"]):
        sky_cell = (
            f'<div class="sk-row-title">{ui.esc(r["tier"])} sky</div>'
            f'<div class="sk-row-sub">{times_words(float(r["times"]))}</div>'
        )
    chance = "–" if p is None else f"{p:.0%}"
    return ui.row(
        [
            f'<div class="sk-rank">{i:02d}</div>',
            f'<div class="sk-main"><div class="sk-row-title">{ui.esc(r["place"])}'
            + (
                ' <span class="sk-badge" style="--c:var(--sk-accent)">You</span>'
                if r.get("yours", False)
                else ""
            )
            + "</div>"
            f'<div class="sk-row-sub">{ui.esc(r["area"])} · clearest {ui.esc(r["window"])}'
            "</div></div>",
            f'<div class="sk-grow">{ui.bar(p, color)}'
            f'<div class="sk-row-sub" style="margin-top:6px">{chance} chance clear</div></div>',
            f"<div>{sky_cell}</div>",
            condition_chips(r["conds"]),
        ],
        COLS,
    )


def marker_color(met: int, pal: dict) -> str:
    return pal["Go"] if met == 3 else pal["Maybe"] if met == 2 else pal["Skip"]


def your_label_side(table: pd.DataFrame, r: pd.Series) -> str:
    """The chosen place's label goes on the side away from the nearest other place, so the two
    labels don't hide each other on the map."""
    others = table[~table["yours"]]
    d = lightpollution.haversine_km(r["lat"], r["lon"], others["lat"].to_numpy(),
                                    others["lon"].to_numpy())  # fmt: skip
    near = others.iloc[int(np.argmin(d))]
    return label_side(r["lat"], r["lon"], near["lat"], near["lon"])


def marker_label(r: pd.Series) -> str:
    chance = "–" if pd.isna(r["p"]) else f"{r['p']:.0%}"
    tier = f" · {r['tier'].lower()}" if isinstance(r.get("tier"), str) else ""
    return f"{r['place']}{tier} · {chance}"


# ---------- light pollution at the selected place ----------


def light_strip(report: dict, d: sky.Darkness) -> str:
    h = report["here"]
    return ui.strip(
        [
            ui.stat("How dark", f"{ui.esc(d.title)}", ui.esc(d.verdict) + ui.darkness_scale(d.key)),
            ui.stat(
                "Sky brightness",
                ui.esc(times_words(d.times_natural)),
                "compared with a natural sky, from the 2025 light-pollution map",
            ),
            ui.stat(
                "Faintest stars",
                f"magnitude {d.nelm:.1f}",
                "what a typical eye picks out overhead on a moonless night "
                f"(about {h['sqm']:.2f} mag/arcsec²)",
            ),
            change_stat(report),
        ]
    )


def change_stat(report: dict) -> str:
    ch = report.get("change")
    if not ch:
        return ui.stat("Since 2015", "–", "no 2015 comparison here")
    pct = ch["artificial_ratio"] - 1
    if abs(ch["delta_mag"]) < 0.005:
        trend = "about the same"
    else:  # magnitudes run backwards: a lower number is a brighter sky
        trend = "brighter" if ch["delta_mag"] < 0 else "darker"
    return ui.stat(
        "Since 2015",
        f"{ui.esc(lookup.cap(trend))}",
        f"{abs(pct):.0%} {'more' if pct >= 0 else 'less'} artificial light than in the 2015 atlas "
        f"({ch['delta_mag']:+.2f} mag)",
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
                    "glow on the horizon</div>",
                ],
                "minmax(0, 2fr) 72px 150px",
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
                    f"{glow['scattered_share']:.0%} of the glow on the horizon</div>",
                ],
                "minmax(0, 2fr) 72px 150px",
            )
        )
    return ui.rows(items)


def glow_summary(glow: dict) -> str:
    """One sentence: where to point, and the biggest dome."""
    quarter = f"{glow['darkest_quarter'][0]}–{glow['darkest_quarter'][-1]}"
    text = (
        f"The darkest part of the horizon is <b>{quarter}</b>: point your camera or telescope "
        "there."
    )
    if glow["cities"]:
        c = glow["cities"][0]
        text += (
            f" The biggest glow on the horizon comes from {ui.esc(c['name'])}, "
            f"{c['distance_km']:.0f} km {c['direction']}."
        )
    return f'<p class="sk-lede" style="margin:6px 0 4px">{text}</p>'


def nearby_rows(report: dict) -> str:
    items = []
    seen = set()
    entries = [(f"Darkest spot within {r} km", d) for r, d in report["darkest"].items()]
    if report.get("nearest_dark"):
        entries.insert(0, ("Nearest very dark sky", report["nearest_dark"]))
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
        tier = sky.darkness(d["sqm"])
        items.append(
            ui.row(
                [
                    f'<div class="sk-main"><div class="sk-row-title">{ui.esc(title)}</div>'
                    f'<div class="sk-row-sub">{where} · straight-line distance</div></div>',
                    f'<div><div class="sk-row-title">{ui.esc(tier.title)} sky</div>'
                    f'<div class="sk-row-sub">{times_words(tier.times_natural)}</div></div>',
                    f'<div style="text-align:right"><a href="{maps_link(d["lat"], d["lon"])}" '
                    'target="_blank" rel="noopener">Open map ↗</a></div>',
                ],
                NEAR_COLS,
            )
        )
    return ui.rows(items)


def area_bar(shares: dict[str, float]) -> str:
    """Area within the radius in each darkness tier, as a stacked bar with plain labels."""
    return charts.bortle_bar(shares, BAND_WORDS)


def methodology(ctx: Context) -> None:
    cfg = ctx.settings.raw["light_pollution"]
    model = (ctx.light_data or {}).get("model") or {}
    cv = model.get("spatial_cv", {})
    checks = model.get("validation_2025_atlas") or {}
    val, val0 = checks.get("updated", {}), checks.get("atlas_2016", {})
    ratio = model.get("ratio", {})
    year = model.get("year", cfg.get("year"))
    nan = float("nan")
    st.markdown(
        f"""
**How dark, in plain words.** The five steps (very dark, dark, suburban, bright, city) are the
sky-brightness ranges of the Bortle scale grouped for beginners: very dark = classes 1–3 (the
Milky Way bright and detailed), dark = 4–4.5, suburban = 5, bright = 6–7, city = 8–9.
"× natural" is how many times brighter than an unpolluted sky.

**1 · The calibrated atlas.** The *World Atlas of Artificial Night Sky Brightness* (Falchi et al.,
2016, *Science Advances* 2:e1600377; data doi:10.5880/GFZ.1.4.2016.001, CC BY-NC 4.0) maps the
artificial glow overhead on a ~1 km grid, checked against sky-meter readings (±{cfg["sigma_sqm"]}
mag/arcsec²). Its lights are from 2014–2015.

**2 · Brought up to {year}.** NASA's Black Marble night lights (VNP46A4 / VJ146A4, CC0) give the
lights for 2015 and {year}. SkyTrust learns how light spreads through the air by fitting the atlas
from the 2015 lights, then scales the atlas by the change in lights. Held out region by region,
that reproduces the atlas to {cv.get("rmse_mag", nan):.3f} mag. In lit areas the median change is
×{ratio.get("median_lit", nan):.2f}.

**3 · Checked independently.** David Lorenz's {year} atlas agrees with the updated map within one
of its zones in {val.get("lit_within_one_zone", nan):.0%} of lit places (the 2015 atlas alone:
{val0.get("lit_within_one_zone", nan):.0%}), and reads {val.get("lit_bias_mag", nan):.2f} mag
brighter in lit areas, so treat city values as a best case.

**4 · City glow.** The glow on the horizon toward each town uses Walker's law (brightness falls
as distance^−2.5); strengths are relative, 1× being Sacramento's glow seen from 50 km.

**What it can't tell you.** Satellites barely see the blue light of white LEDs, so they understate
brightening (citizen observers found skies brightening 9.6% a year in 2011–2022; Kyba et al.,
2023, *Science* 379:265). Distances are straight lines on a ~1 km grid and ignore roads, terrain
and nearby lights.
"""
    )


def all_sites_table(table: pd.DataFrame) -> None:
    shown = pd.DataFrame(
        {
            "place": table["place"],
            "area": table["area"],
            "chance clear": table["p"].map(lambda p: None if pd.isna(p) else round(100 * p)),
            "verdict": table["verdict"],
            "clearest": table["window"],
            "sky": table["tier"],
            "× natural": table["times"].map(lambda t: None if pd.isna(t) else round(float(t), 1)),
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


# ---------- maps ----------


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
    found = [
        d
        for d in [*report["darkest"].values(), report.get("nearest_dark")]
        if d and d["distance_km"] >= 1.5
    ]
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
            "label": [f"{sky.darkness(d['sqm']).title.lower()} sky" for d in kept],
            "position": [label_side(d["lat"], d["lon"], site_lat, site_lon) for d in kept],
        }
    )


def local_map(ctx: Context, report: dict, overlay: bytes) -> None:
    pal = ctx.palette
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
        ctx.light,
        overlay,
        here,
        pal,
        center=(ctx.site.lat, ctx.site.lon),
        zoom=7.4,
        height=380,
        spots=map_spots(report, ctx.site.lat, ctx.site.lon),
    )
    show(fig)


# ---------- page ----------


def render(ctx: Context) -> None:
    st.markdown(
        ui.section(
            "Where should I go tonight?",
            "A good night needs three things: a clear sky, a dark sky, and the Moon down. Places "
            "are ranked by how many they have tonight, then by the chance of clear sky.",
            "Places",
        ),
        unsafe_allow_html=True,
    )
    table = site_rows(ctx)
    pal = ctx.palette
    rows = [rank_row(i + 1, r, pal) for i, (_, r) in enumerate(table.iterrows())]
    st.markdown(ui.rows(rows), unsafe_allow_html=True)

    path = lightpollution.OVERLAY_PATH
    overlay = path.read_bytes() if path.exists() else None
    if ctx.light is not None and overlay is not None:
        st.markdown(
            ui.section(
                "Light pollution across California",
                "Brighter colours = more city "
                "light in the sky (2025). The dots are the places above.",
                "Map",
            ),
            unsafe_allow_html=True,
        )
        markers = pd.DataFrame(
            {
                "lat": table["lat"],
                "lon": table["lon"],
                "label": [marker_label(r) for _, r in table.iterrows()],
                "color": [
                    pal["accent"] if y else marker_color(m, pal)
                    for m, y in zip(table["met"], table["yours"], strict=True)
                ],  # fmt: skip
                "position": [
                    your_label_side(table, r)
                    if r.get("yours", False)
                    else charts.LABEL_POSITIONS.get(r["site"], "top right")
                    for _, r in table.iterrows()
                ],  # fmt: skip
            }
        )
        # whole state: San Francisco and Lake Tahoe to Joshua Tree and Los Angeles
        show(
            charts.light_map(
                ctx.light, overlay, markers, pal, center=(36.4, -118.9), zoom=4.85, height=620
            )
        )
        key = [
            ("dot: all 3 tonight", pal["Go"]),
            ("2 of 3", pal["Maybe"]),
            ("1 or none", pal["Skip"]),
        ]
        st.markdown(ui.legend(charts.GLOW_KEY) + ui.legend(key), unsafe_allow_html=True)

    report = ctx.light_at()
    st.markdown(
        ui.section(f"How dark is it at {ctx.site_label}?", "", "Light pollution"),
        unsafe_allow_html=True,
    )
    if report is None:
        st.info("No light-pollution data for this location (outside the mapped region).")
    else:
        d = ctx.darkness()
        st.markdown(light_strip(report, d), unsafe_allow_html=True)
        st.markdown(
            f'<p class="sk-lede" style="margin-top:4px">What you can expect on a moonless night: '
            f"{ui.esc(lookup.uncap(d.milky_way))}.</p>",
            unsafe_allow_html=True,
        )
        st.markdown(ui.eyebrow("Darker skies nearby") + nearby_rows(report), unsafe_allow_html=True)
        st.markdown(
            ui.eyebrow(f"How dark the land is within {report['shares_radius_km']} km")
            + area_bar(report["shares"]),
            unsafe_allow_html=True,
        )
        glow = report.get("glow")
        if glow:
            st.markdown(
                ui.section(
                    "Glow on the horizon",
                    "Which towns light up which part of the "
                    "horizon from here: matters most for photos low in the sky.",
                    "City glow",
                ),
                unsafe_allow_html=True,
            )
            st.markdown(glow_summary(glow), unsafe_allow_html=True)
            left, right = st.columns([1, 1.15], gap="large", vertical_alignment="center")
            with left:
                show(charts.dome_polar(glow, ctx.palette))
            with right:
                st.markdown(ui.eyebrow("Biggest glows") + glow_rows(glow), unsafe_allow_html=True)
                st.caption(
                    "Strength by Walker's law (glow ∝ light ÷ distance²·⁵); 1× is Sacramento's "
                    f"glow seen from 50 km. Total here: {glow['dome_total']:.2f}×."
                )
        if overlay is not None:
            local_map(ctx, report, overlay)

    with st.expander("How light pollution is measured, and its limits", icon=":material/info:"):
        methodology(ctx)
    with st.expander("All places as a table", icon=":material/table_rows:"):
        all_sites_table(table)
