"""Live (prospective) verification: forecasts logged daily before the outcome, scored later."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from skytrust.report import display_name

BENCHMARKS = {
    "nbm_frac_clear": "NOAA NBM",
    "ens_ecmwf": "ECMWF ensemble",
    "ens_gefs": "GEFS ensemble",
}


def _pct(x) -> str:
    return "–" if x is None or pd.isna(x) else f"{x:.0%}"


def lead_table(summary: dict) -> pd.DataFrame:
    rows = {}
    for entry in summary["by_lead"]:
        if entry["lead"] == "all":
            continue
        m = entry["methods"]
        blend = m.get("blend", {})
        row = {
            "nights": entry["n"],
            "'Go' calls that were usable": (
                f"{blend.get('go_calls_usable', 0)}/{blend.get('go_calls', 0)}"
            ),
            "false-clear (live)": _pct(blend.get("false_clear_rate")),
            "false-clear (backtest)": _pct((entry.get("backtest") or {}).get("false_clear_rate")),
            "Brier: SkyTrust": f"{blend.get('brier', float('nan')):.3f}",
        }
        for key, name in BENCHMARKS.items():
            if key in m:
                row[f"Brier: {name}"] = f"{m[key]['brier']:.3f}"
        rows[f"{entry['lead']} day{'s' if entry['lead'] > 1 else ''} ahead"] = row
    out = pd.DataFrame(rows).T
    out.index.name = "lead"
    return out


def render(summary: dict | None) -> None:
    st.subheader("Live verification")
    st.caption(
        "Every afternoon SkyTrust's forecasts for the next 7 nights are saved *before* anyone "
        "knows the outcome. Once the observations are published (about a week later) they're "
        "scored with exactly the backtest's method, next to NOAA's National Blend of Models "
        "and the raw ECMWF/GEFS ensembles. Nothing here can be tuned after the fact."
    )
    if summary is None:
        st.info("The live verification record isn't reachable right now.")
        return
    if not summary.get("n_verified"):
        st.info(
            f"Logging since {summary.get('first_issue') or summary['forward_start']}: "
            f"{summary['n_logged']} forecasts saved so far. Each night becomes scorable "
            f"{summary['verify_after_days']} days after it happens, and results appear here "
            "automatically."
        )
        return
    pooled = next(e for e in summary["by_lead"] if e["lead"] == "all")
    blend = pooled["methods"]["blend"]
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Nights verified",
        f"{summary['n_verified']}",
        help=f"through {summary.get('last_verified_night')}",
    )
    c2.metric("'Go' calls that were usable", f"{blend['go_calls_usable']}/{blend['go_calls']}")
    c3.metric("Skill vs climatology (live)", f"{blend.get('bss', float('nan')):.2f}")
    st.dataframe(lead_table(summary), width="stretch")
    st.caption(
        f"Scored with {display_name('blend')} probabilities as shown in the app at the time. "
        "Confidence intervals appear once 8+ weeks of nights are verified."
    )
