"""Accuracy: how often SkyTrust has been wrong (the track record), then how it works."""

from __future__ import annotations

import streamlit as st

from views import components as ui
from views import methodology, track_record
from views.common import Context


def render(ctx: Context) -> None:
    st.markdown(
        ui.section(
            "How often has it been wrong?",
            "SkyTrust grades itself: tested on nights it never saw, and every new forecast is "
            "logged in public before the night.",
            "Accuracy",
        ),
        unsafe_allow_html=True,
    )
    track_record.render(ctx, standalone=False)
    st.markdown(
        ui.section(
            "How it works", "The thirty-second version, a glossary, and the full method.", "Method"
        ),
        unsafe_allow_html=True,
    )
    methodology.render(ctx, standalone=False)
