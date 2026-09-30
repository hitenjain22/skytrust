"""Page 3: the backtest. Everything here is read from artifacts/metrics.json."""

from __future__ import annotations

import streamlit as st

from skytrust import report
from views import charts, forward_panel
from views.common import Context


def render(ctx: Context) -> None:
    st.header("Track record: how often has it been wrong?")
    if not ctx.metrics:
        st.warning("No backtest metrics found (artifacts/metrics.json).")
        return
    v = report.MetricsView(ctx.metrics)
    meta = ctx.metrics["meta"]
    st.caption(
        f"Trained on {meta['train_period'][0]} → {meta['train_period'][1]}; tested once on "
        f"{meta['test_period'][0]} → {meta['test_period'][1]}, a period the models never saw. "
        "95% confidence intervals resample whole weeks."
    )
    forward_panel.render(ctx.forward_summary)
    st.divider()
    st.subheader("Backtest")
    c1, c2 = st.columns(2)
    label = c1.radio(
        "Truth label",
        ["primary", "asos", "era5"],
        horizontal=True,
        key="label",
        format_func=lambda x: {"primary": "Primary", "asos": "ASOS only", "era5": "ERA5 only"}[x],
        help=(
            "ASOS can't see cirrus above 12,000 ft; ERA5 is a reanalysis. "
            "Primary = the cloudier of the two."
        ),
    )
    leads = v.lead_list(label)
    lead = c2.select_slider("Days ahead (lead)", options=leads, value=leads[0], key="lead")

    st.subheader("Takeaways")
    for line in report.takeaways(v, label, lead):
        st.markdown(f"- {line}")

    st.plotly_chart(charts.false_clear_bars(v.records, label, lead, ctx.palette), width="stretch")
    best = v.best_single(label, lead)
    shown = [m for m in ["climatology", best, "equal_weight", "nbm_lr", "blend"] if m]
    st.plotly_chart(
        charts.reliability(ctx.metrics["reliability"], shown, label, lead, ctx.palette),
        width="stretch",
    )
    st.plotly_chart(charts.skill_by_lead(v.records, label, ctx.palette), width="stretch")

    st.subheader("By site and season")
    st.dataframe(report.breakdown_table(v, label, lead, "site"), width="stretch")
    st.dataframe(report.breakdown_table(v, label, lead, "season"), width="stretch")
    st.caption("Subsets spanning fewer than 8 weeks get no confidence interval.")
    if label in ("primary", "era5") and best == "ecmwf_lr":
        st.info(
            "ECMWF is the best single model here. ERA5 is produced by ECMWF, so part of that "
            "may be shared model physics rather than real-world skill."
        )
