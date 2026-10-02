"""Sky Guide: a live chart of tonight's sky from the chosen place, what's up at the chosen time and
where to look, the Milky Way, and where to look when the Moon is up."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from skytrust import haze, sky
from views import components as ui
from views import lookup, skylive
from views.common import Context, esc


def render(ctx: Context) -> None:
    tz = ctx.site.timezone
    dusk, dawn = ctx.service("night")(ctx.site)
    sqm = ctx.sqm_at()
    aod = ctx.haze_aod(dusk, dawn)
    guide = ctx.service("sky")(ctx.site, dusk, dawn, sqm, aod)
    d = guide["darkness"]
    w = haze.words(aod)
    smoke = f" {w[0]} tonight: {w[1]}." if w else ""
    st.markdown(
        ui.section(
            f"The sky tonight over {ctx.site_label}",
            f"{esc(d.title)} sky: {esc(lookup.uncap(d.verdict))}.{esc(smoke)} Drag the time, "
            "tap a name to find it.",
            f"Sky guide · {dusk.tz_convert(tz):%a %b %-d}",
        ),
        unsafe_allow_html=True,
    )
    # open on now once it's fully dark, otherwise an hour after dark (twilight is one drag away)
    start = dusk + pd.Timedelta(hours=1)
    if ctx.now_utc is not None and dusk <= ctx.now_utc <= dawn:
        start = ctx.now_utc
    # the chart, the time slider and the "Up at ..." list are drawn in the browser, so the sky
    # follows the slider while it is dragged (views/skylive.py)
    skylive.render(ctx.service("live")(ctx.site, dusk, dawn, sqm, min(start, dawn), aod))

    mw = guide["milky_way"]
    when, help_ = lookup.milky_way_text(mw, tz)
    dark_enough = mw["zenith_sqm"] >= ctx.settings.raw["light_pollution"]["dark_sqm"]
    mw_title = (
        ("Visible from here" if dark_enough else "Faint from here")
        if mw["visible"]
        else "Not visible from here tonight"
    )
    mw_body = esc(help_) + " " + esc(mw["looks"]) + "."
    cards = [ui.card("milkyway", "The Milky Way", mw_title, where=esc(when), body=mw_body)]
    om = guide["opposite_moon"]
    if om:
        names = [it.name for it, _, _ in om["objects"]]
        listing = ", ".join(names[:3]) if names else "the stars on that side"
        cards.append(
            ui.card(
                "moon",
                "While the Moon is up",
                f"Face {sky.compass_words(om['face'])}",
                where=esc(
                    f"At {lookup.clock(om['utc'], tz)} the Moon ({om['illum']:.0%} lit) is "
                    f"{sky.where_words(om['moon_alt'], om['moon_az'])}"
                ),
                body=esc(
                    "Its glare is weakest away from it: the darkest part of the sky is "
                    f"{sky.where_words(om['darkest_alt'], om['darkest_az'])}. Look there for "
                    f"{listing}."
                ),
            )
        )
    else:
        cards.append(
            ui.card(
                "moon",
                "The Moon",
                "Down during the dark hours",
                where="The whole sky is dark tonight",
                body="No moonlight: the best time for the Milky Way and faint objects.",
            )
        )
    st.markdown(ui.grid(cards, n=2), unsafe_allow_html=True)

    with st.expander("How the chart and the visibility are worked out", icon=":material/info:"):
        st.markdown(
            """
- **Positions**: stars (Hipparcos), planets and the Moon (JPL DE421), computed with Skyfield
  for your place and time. Center = overhead, rim = horizon.
- **What you can see** depends on how bright the sky is in that direction: light pollution
  (the atlas overhead, spread over the sky by each town's light dome, which is the amber glow on
  the rim), moonlight (Krisciunas & Schaefer 1991) and twilight, brightest on the side of the
  set Sun (Patat et al. 2006; Schaefer 1998). The faintest visible star follows Schaefer (1990).
- **Air**: stars low down shine through more air, and smoke or haze (the CAMS forecast) dims
  every star further.
- Galaxies and nebulae need about half a magnitude darker sky than a star of the same
  brightness. Experienced observers can see up to a magnitude fainter than shown.
"""
        )
