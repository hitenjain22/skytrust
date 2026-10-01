"""Sky Guide: a live chart of tonight's sky from the chosen place, what's up at the chosen time and
where to look, the Milky Way, and where to look when the Moon is up."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from skytrust import sky
from views import components as ui
from views import lookup, skylive
from views.common import Context, esc


def render(ctx: Context) -> None:
    tz = ctx.site.timezone
    dusk, dawn = ctx.service("night")(ctx.site)
    sqm = ctx.sqm_at()
    guide = ctx.service("sky")(ctx.site, dusk, dawn, sqm)
    d = guide["darkness"]
    st.markdown(
        ui.section(
            f"The sky tonight over {ctx.site_label}",
            f"{esc(d.title)} sky here "
            f"({esc(lookup.uncap(sky.brightness_words(d.times_natural)))}): "
            f"{esc(lookup.uncap(d.verdict))}. Slide through the night to see what's up.",
            f"Sky guide · {dusk.tz_convert(tz):%a %b %-d}",
        ),
        unsafe_allow_html=True,
    )
    start = dusk + pd.Timedelta(hours=2)
    if ctx.now_utc is not None and dusk <= ctx.now_utc <= dawn:
        start = ctx.now_utc  # the night is under way: open on now
    # the chart, the time slider and the "Up at ..." list are drawn in the browser, so the sky
    # follows the slider while it is dragged (views/skylive.py)
    skylive.render(ctx.service("live")(ctx.site, dusk, dawn, sqm, min(start, dawn)))

    mw = guide["milky_way"]
    when, help_ = lookup.milky_way_text(mw, tz)
    mw_title = "Visible from here" if mw["visible"] else "Not visible from here tonight"
    mw_body = (
        (esc(help_) + " " + esc(mw["looks"]) + ".")
        if mw["visible"]
        else (esc(help_) + " From here, " + esc(lookup.uncap(mw["looks"])) + ".")
    )
    cards = [ui.card("milkyway", "The Milky Way", mw_title, where=esc(when), body=mw_body)]
    om = guide["opposite_moon"]
    if om:
        names = [it.name for it, _, _ in om["objects"]]
        listing = ", ".join(names[:4]) if names else "the stars on that side"
        cards.append(
            ui.card(
                "moon",
                "With your back to the Moon",
                f"Face {sky.compass_words(om['face'])}",
                where=esc(
                    f"At {lookup.clock(om['utc'], tz)}, the Moon ({om['illum']:.0%} lit) is "
                    f"{sky.where_words(om['moon_alt'], om['moon_az'])}"
                ),
                body=esc(
                    "The darkest part of the sky is "
                    f"{sky.where_words(om['darkest_alt'], om['darkest_az'])}, away from the "
                    f"Moon's glare. Look there for {listing}."
                ),
                meta="Moonlight is scattered least about 90° from the Moon "
                "(Krisciunas & Schaefer 1991).",
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
    counts = guide["stars_visible"]
    cards.append(
        ui.card(
            "star",
            "Stars you can see",
            f"About {lookup.fmt_count(counts['here'])}",
            where=f"of ~{lookup.fmt_count(counts['natural'])} in a natural sky",
            body=esc(
                f"At the darkest moment tonight ({lookup.clock(counts['utc'], tz)}), the faintest "
                f"stars a typical eye picks out overhead are magnitude {d.nelm:.1f} "
                "(bigger numbers are fainter)."
            ),
            extra=ui.darkness_scale(d.key),
        )
    )
    st.markdown(ui.grid(cards, n=3), unsafe_allow_html=True)

    with st.expander("How the chart and the visibility are worked out", icon=":material/info:"):
        st.markdown(
            """
- **Positions** of the stars (ESA's Hipparcos catalogue), planets and Moon (JPL's DE421
  ephemeris) are computed for your place and time with Skyfield. The chart is a planisphere:
  the centre is straight overhead, the rim is the horizon.
- **What you can see** depends on how bright the sky is. The light-pollution map gives the sky
  brightness overhead; the Moon's glow is added with the Krisciunas & Schaefer (1991) model of
  scattered moonlight; and the faintest visible star follows Schaefer's (1990) formula for a
  typical observer. Stars low down are dimmed by the extra air they shine through.
- **Clusters, galaxies and nebulae** are harder to see than a star of the same brightness; they
  need about half a magnitude darker sky, a rule set so the Bortle scale's descriptions come out
  right (Andromeda barely visible from a city edge, Triangulum only from dark sites).
- Experienced observers with fully dark-adapted eyes can do up to a magnitude better than shown.
"""
        )
