"""Where to go: every site ranked by tonight's chance of a usable night, as a list and a map."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from views import charts
from views import components as ui
from views.common import Context, place_detail, place_name, short_time, upcoming
from views.tonight import show

CLIMATOLOGY = Path(__file__).resolve().parents[2] / "artifacts" / "climatology.json"


def tonight_table(ctx: Context) -> pd.DataFrame:
    rows = []
    for site in ctx.sites:
        forecast, _ = ctx.forecast_for(site.id)
        nights = upcoming(forecast, ctx.now_utc)
        base = {
            "site": site.id,
            "place": place_name(site),
            "area": place_detail(site),
            "lat": site.lat,
            "lon": site.lon,
        }
        if not nights:
            rows.append(
                base | {"p": None, "verdict": "No data", "window": "–", "moon_free": None,
                        "illum": None, "phase": None}
            )  # fmt: skip
            continue
        n = nights[0]
        bw = n.best_window
        window = (
            f"{short_time(bw.start_utc, site.timezone)}–{short_time(bw.until_utc, site.timezone)}"
            if bw
            else "no clear window"
        )
        rows.append(
            base
            | {
                "p": n.p_usable,
                "verdict": n.verdict,
                "window": window,
                "moon_free": n.moon_free_hours,
                "illum": n.moon_illum,
                "phase": n.moon_phase_deg,
            }
        )
    return pd.DataFrame(rows).sort_values("p", ascending=False, na_position="last")


COLS = "28px minmax(0, 2.2fr) minmax(60px, 1.6fr) 52px 96px"


def rank_row(i: int, r: pd.Series, pal: dict) -> str:
    color = pal[r["verdict"]]
    p = None if pd.isna(r["p"]) else float(r["p"])
    moon_free = "" if pd.isna(r["moon_free"]) else f" · {int(r['moon_free'])} moon-free h"
    return ui.row(
        [
            f'<div class="sk-rank">{i:02d}</div>',
            f'<div><div class="sk-row-title">{ui.esc(r["place"])}</div>'
            f'<div class="sk-row-sub">{ui.esc(r["area"])} · {ui.esc(r["window"])}{moon_free}'
            "</div></div>",
            ui.bar(p, color),
            f'<div class="sk-row-p">{"–" if p is None else f"{p:.0%}"}</div>',
            f'<div style="text-align:right">{ui.status(r["verdict"].upper(), color)}</div>',
        ],
        COLS,
    )


def render(ctx: Context) -> None:
    st.header("Where should I go tonight?")
    st.caption("Tonight's chance of a usable night at every site, best first.")
    table = tonight_table(ctx)
    pal = ctx.palette
    rows = [rank_row(i + 1, r, pal) for i, (_, r) in enumerate(table.iterrows())]
    st.markdown(ui.rows(rows), unsafe_allow_html=True)
    v = ctx.settings.raw["verdict"]
    light = st.context.theme.type == "light"
    show(charts.site_map(table, pal, v["go"], v["maybe"], light=light))
    with st.expander("All sites as a table", icon=":material/table_rows:"):
        shown = table.assign(
            chance=table["p"].map(lambda p: None if pd.isna(p) else round(100 * p)),
            moon=table["illum"].map(lambda x: "–" if pd.isna(x) else f"{x:.0%}"),
        )[["place", "area", "chance", "verdict", "window", "moon_free", "moon"]]
        st.dataframe(
            shown.rename(columns={"window": "best window", "moon_free": "moon-free h"}),
            hide_index=True,
            width="stretch",
            column_config={
                "chance": st.column_config.ProgressColumn(
                    "chance usable", format="%d%%", min_value=0, max_value=100
                )
            },
        )
    tip = climate_tip(ctx)
    if tip:
        st.caption(tip)


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
