"""Page 3: how often has it been wrong? A plain-English report card first, then the live
forward test, then the full backtest for the data-curious. Every number is read from
artifacts/metrics.json (or the forward-test summary); nothing is typed by hand."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from skytrust import report
from views import charts, forward_panel, statewide_view
from views import components as ui
from views.common import Context
from views.tonight import show


def _share(rec: pd.Series | None) -> str:
    return (
        "–"
        if rec is None or pd.isna(rec.get("false_clear_rate"))
        else f"{1 - rec['false_clear_rate']:.0%}"
    )


def report_card(v: report.MetricsView) -> str:
    lead = v.lead_list("primary")[0]
    blend, clim = v.rec("primary", lead, "blend"), v.rec("primary", lead, "climatology")
    cards = []
    if blend is not None:
        ci = ""
        if not pd.isna(blend.get("false_clear_rate_lo")):
            lo, hi = 1 - blend["false_clear_rate_hi"], 1 - blend["false_clear_rate_lo"]
            ci = f" (95% range {lo:.0%}–{hi:.0%})"
        cards.append(
            ui.stat(
                "“Go” calls that held",
                _share(blend),
                f"of the nights it called “go” one day ahead were usable{ci}. Guessing from the "
                f"season alone: {_share(clim)}.",
                big=True,
            )
        )
    leads = v.lead_list("primary")
    wins = [
        ld for ld in leads
        if (d := v.diff("primary", ld, "blend", "nbm_lr")) is not None
        and d["significant"] and d["brier_diff"] < 0
    ]  # fmt: skip
    if any(v.diff("primary", ld, "blend", "nbm_lr") is not None for ld in leads):
        cards.append(
            ui.stat(
                "Versus NOAA",
                f"{len(wins)}<span class='sk-muted'>/{len(leads)}</span>",
                f"forecast ranges ({leads[0]}–{leads[-1]} days ahead) where it beat NOAA’s own "
                "National Blend of Models with 95% confidence, on the same nights.",
                big=True,
            )
        )
    sat = v.rec("goes", lead, "blend_primary")
    if sat is not None:
        cards.append(
            ui.stat(
                "Satellite check",
                _share(sat),
                "of its “go” calls were confirmed by the GOES-18 weather satellite, an "
                "independent check the model never trained on.",
                big=True,
            )
        )
    elif blend is not None:
        cards.append(
            ui.stat(
                "Skill score",
                f"{blend['bss']:.2f}",
                "vs the seasonal average (0 = no better, 1 = perfect), one day ahead.",
                big=True,
            )
        )
    return ui.strip(cards, three=True)


def render(ctx: Context, standalone: bool = True) -> None:
    if standalone:
        st.header("How often has it been wrong?")
    if not ctx.metrics:
        st.warning("No backtest metrics found (artifacts/metrics.json).")
        return
    v = report.MetricsView(ctx.metrics)
    meta = ctx.metrics["meta"]
    st.caption(
        f"Trained on {meta['train_period'][0]} → {meta['train_period'][1]}; tested once on "
        f"{meta['test_period'][0]} → {meta['test_period'][1]}, nights the model never saw. "
        "95% confidence intervals resample whole weeks."
    )
    st.markdown(report_card(v), unsafe_allow_html=True)
    show(charts.go_accuracy_by_lead(v.records, ctx.palette))
    st.caption(
        "Accuracy fades the further ahead you look, which is why the week ahead on Tonight "
        "shows a trust level for each night. Error bars: 95% confidence intervals."
    )

    with st.container(key="panel_forward"):
        forward_panel.render(ctx.forward_summary)
    statewide_view.render(ctx)

    st.markdown(
        ui.section(
            "The full backtest",
            "For the data-curious: every method, every truth source, every forecast range.",
            "Details",
        ),
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    names = {"primary": "Primary", "asos": "ASOS only", "era5": "ERA5 only", "goes": "Satellite"}
    label = c1.radio(
        "Truth label",
        report.labels_in(v),
        horizontal=True,
        key="label",
        format_func=names.get,
        help=(
            "ASOS can't see cirrus above 12,000 ft; ERA5 is a reanalysis. "
            "Primary = the cloudier of the two. Satellite = GOES-18 cloud mask, an independent "
            "check that sees high cloud."
        ),
    )
    leads = v.lead_list(label)
    lead = (
        c2.segmented_control(
            "Days ahead (lead)", leads, default=leads[0], required=True, key="lead"
        )
        or leads[0]
    )

    with st.container(key="panel_takeaways"):
        st.markdown(ui.eyebrow("Takeaways"), unsafe_allow_html=True)
        for line in report.takeaways(v, label, lead):
            st.markdown(f"- {line}")

    pal = ctx.palette
    best = v.best_single(label, lead)
    shown = [m for m in ["climatology", best, "equal_weight", "nbm_lr", "blend"] if m]
    tab1, tab2, tab3, tab4 = st.tabs(["False clears", "Calibration", "Skill by range", "Value"])
    with tab1:
        show(charts.false_clear_bars(v.records, label, lead, pal))
    with tab2:
        show(charts.reliability(ctx.metrics["reliability"], shown, label, lead, pal))
        st.caption("Points on the diagonal mean the probabilities can be taken at face value.")
    with tab3:
        show(charts.skill_by_lead(v.records, label, pal))
    with tab4:
        if ctx.metrics.get("value_curves"):
            show(charts.value_curves(ctx.metrics["value_curves"], shown, label, lead, pal))
            st.caption(
                "Relative value: if a good night is worth 5× your setup effort (α = 0.2), this is "
                "the share of a perfect forecast's benefit you'd get by going out when P ≥ α."
            )

    with st.expander("By site and season", icon=":material/table_rows:"):
        st.dataframe(report.breakdown_table(v, label, lead, "site"), width="stretch")
        st.dataframe(report.breakdown_table(v, label, lead, "season"), width="stretch")
        st.caption("Subsets spanning fewer than 8 weeks get no confidence interval.")
    if label in ("primary", "era5") and best == "ecmwf_lr":
        st.info(
            "ECMWF is the best single model here. ERA5 is produced by ECMWF, so part of that "
            "may be shared model physics rather than real-world skill."
        )
