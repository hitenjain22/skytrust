"""Accuracy across California: how the forecast for any place did at 32 weather stations in all
ten NWS regions of the state, each scored as a place SkyTrust had never seen. Every number comes
from artifacts/statewide.json (src/skytrust/statewide.py); the sentences are templates.

The JSON is read directly: importing skytrust.statewide would pull in scikit-learn, which the
app must never load."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

from skytrust import inference
from views import charts
from views import components as ui
from views.common import Context, held_with_ci, show, station_label, with_ci

LEAD = 1  # headline numbers: one day ahead, like the rest of the Accuracy page


def load() -> dict | None:
    path = inference.STATEWIDE_PATH
    return json.loads(path.read_text()) if path.exists() else None


def _verdict(paired: dict | None, better: str, worse: str, same: str) -> str:
    if not paired:
        return same
    if paired["significant"]:
        return better if paired["brier_diff"] < 0 else worse
    return same


def headline(entry: dict, method: str) -> str:
    sub = entry["subsets"]["all"]
    m = sub["methods"][method]
    vs_avg = _verdict(
        sub["paired"].get("statewide_minus_equal_weight"),
        "Better, with 95% confidence",
        "Worse, with 95% confidence",
        "No clear difference",
    )
    return ui.strip(
        [
            ui.stat(
                "“Go” calls that held",
                f"{1 - m['false_clear_rate']:.0%}",
                f"one day ahead, at {sub['n_stations']} stations as never-seen places "
                f"({held_with_ci(m)}, {sub['n']:,} station-nights in 2026)",
                big=True,
            ),
            ui.stat(
                "Skill vs the season",
                f"{m['bss']:.2f}",
                f"0 = no better than each station's seasonal rate, 1 = perfect "
                f"({with_ci(m, 'bss', 'num')})",
                big=True,
            ),
            ui.stat(
                "Versus the simple average",
                vs_avg,
                "of the five weather models on the same nights (Brier score, week-block bootstrap)",
                big=True,
            ),
        ],
        three=True,
    )


def near_rows(record: dict | None) -> str:
    if not record or not record.get("near"):
        return ""
    rows = [
        ui.row(
            [
                f'<div class="sk-main"><div class="sk-row-title">{ui.esc(station_label(s))}</div>'
                f'<div class="sk-row-sub">{s["distance_km"]:.0f} km away · {s["n"]} nights in '
                "2026</div></div>",
                f'<div><div class="sk-row-title">{held_with_ci(s)}</div>'
                '<div class="sk-row-sub">“go” calls that held</div></div>',
                f'<div><div class="sk-row-title">{with_ci(s, "bss", "num")}</div>'
                '<div class="sk-row-sub">skill</div></div>',
            ],
            "minmax(0,1.6fr) minmax(0,1fr) minmax(0,1fr)",
        )
        for s in record["near"]
    ]
    return ui.rows(rows)


def station_table(result: dict, entry: dict, method: str) -> pd.DataFrame:
    rows = []
    for s in result["stations"]:
        rec = entry["stations"].get(s["id"])
        if rec is None:
            continue
        m = rec["methods"][method]
        rows.append(
            {
                "lat": s["lat"],
                "lon": s["lon"],
                "bss": m["bss"],
                "label": (
                    f"{s['id']} · {result['region_names'].get(s['region'], s['region'])}<br>"
                    f"skill {m['bss']:.2f} · “go” calls held {1 - m['false_clear_rate']:.0%} "
                    f"· {rec['n']} nights"
                ),
            }
        )
    return pd.DataFrame(rows)


def region_table(result: dict, entry: dict, method: str) -> pd.DataFrame:
    counts = pd.Series([s["region"] for s in result["stations"]]).value_counts()
    rows = []
    for region, rec in entry["regions"].items():
        m, avg = rec["methods"][method], rec["methods"]["equal_weight"]
        rows.append(
            {
                "region": result["region_names"].get(region, region),
                "stations": int(counts.get(region, 0)),
                "nights": rec["n"],
                "skill": round(m["bss"], 2),
                "skill, simple average": round(avg["bss"], 2),
                "“go” calls held": None
                if np.isnan(m["false_clear_rate"])
                else round(100 * (1 - m["false_clear_rate"])),
            }
        )
    return pd.DataFrame(rows).sort_values("skill", ascending=False)


def crosschecks(result: dict, entry: dict, method: str) -> list[str]:
    """Plain sentences, each comparing the headline with a different way of measuring it."""
    out = []
    sub = entry["subsets"]["all"]
    loso = sub["paired"].get("loro_minus_loso")
    if loso:
        out.append(
            "Holding out one station at a time instead of a whole region changes the Brier "
            f"score by {-loso['brier_diff']:+.3f} (95% range {-loso['hi']:+.3f} to "
            f"{-loso['lo']:+.3f}): {'a real' if loso['significant'] else 'no clear'} "
            "difference, so the result doesn't hinge on how the stations were held out."
        )
    new = entry["subsets"]["new_stations"]
    airport = new["paired"].get("statewide_minus_airport_geo")
    if airport:
        verdict = _verdict(
            airport, "more accurate than", "less accurate than", "about as accurate as"
        )
        out.append(
            f"At the {new['n_stations']} stations the old model never saw, the statewide model "
            f"was {verdict} the earlier one trained on five Northern California airports "
            f"(Brier difference {airport['brier_diff']:+.3f}, 95% range {airport['lo']:+.3f} to "
            f"{airport['hi']:+.3f})."
        )
    for label, name in [("asos", "airport ceilometers only"), ("era5", "the ERA5 reanalysis only")]:
        e = next((x for x in result["labels"].get(label, []) if x["lead"] == LEAD), None)
        if e:
            m = e["subsets"]["all"]["methods"]["statewide_loro"]
            out.append(
                f"Judged by {name}: skill {with_ci(m, 'bss', 'num')}, “go” calls held "
                f"{held_with_ci(m)}."
            )
    return out


def render(ctx: Context) -> None:
    st.markdown(
        ui.section(
            "Across California",
            "The five airports above are all in Northern California's interior. To check every "
            "kind of place (the coast, the deserts, the mountains, Southern California), the "
            "forecast for places without their own record was also tested at weather stations "
            "in all ten National Weather Service regions of the state, each scored as if "
            "SkyTrust had never seen it.",
            "Statewide test",
        ),
        unsafe_allow_html=True,
    )
    result = load()
    if result is None:
        st.info(
            "The statewide test is still running: forecasts for 2024–2026 at 32 weather "
            "stations are being downloaded within the weather service's free limits. This "
            "section fills in when it's done.",
            icon=":material/hourglass_top:",
        )
        return
    method = result.get("shipped_method", "statewide_loro")
    entry = next(e for e in result["labels"]["primary"] if e["lead"] == LEAD)
    st.markdown(headline(entry, method), unsafe_allow_html=True)
    record = inference.place_record(ctx.site.lat, ctx.site.lon, LEAD)
    rows = near_rows(record)
    if rows:
        st.markdown(
            ui.eyebrow(f"Nearest stations to {ctx.site_label}") + rows, unsafe_allow_html=True
        )
    show(
        charts.station_skill_map(
            station_table(result, entry, method),
            ctx.palette,
            here=(ctx.site.lat, ctx.site.lon, ctx.site_label),
        )
    )
    st.caption(
        "Each dot is a weather station, brighter = more skilful one day ahead. Skill compares "
        "the forecast with simply guessing each station's usual rate for the month."
    )
    with st.expander("By region, and how the result was cross-checked", icon=":material/rule:"):
        st.dataframe(region_table(result, entry, method), hide_index=True, width="stretch")
        for line in crosschecks(result, entry, method):
            st.markdown(f"- {line}")
