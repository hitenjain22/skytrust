"""Sky Guide: a live chart of tonight's sky from the chosen place, what's up at the chosen time and
where to look, the Milky Way, and where to look when the Moon is up."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from skytrust import sky
from views import components as ui
from views import lookup
from views.common import Context, esc

MODES = {"Your sky": "here", "A perfectly dark sky": "dark"}
KIND_ORDER = {
    "moon": 0,
    "planet": 1,
    "star": 2,
    "pattern": 3,
    "cluster": 4,
    "galaxy": 4,
    "nebula": 4,
}


def time_options(dusk: pd.Timestamp, dawn: pd.Timestamp, tz: str) -> dict[str, pd.Timestamp]:
    """'Dusk', each whole hour of darkness, 'Dawn' -> UTC time."""
    out = {f"Dusk {lookup.clock(dusk, tz)}": dusk}
    t = dusk.tz_convert(tz).ceil("h")
    while t < dawn.tz_convert(tz) - pd.Timedelta(minutes=20):
        if (t - dusk.tz_convert(tz)) >= pd.Timedelta(minutes=20):
            out[lookup.clock(t, tz)] = t.tz_convert("UTC")
        t += pd.Timedelta(hours=1)
    out[f"Dawn {lookup.clock(dawn, tz)}"] = dawn - pd.Timedelta(minutes=5)
    return out


def default_choice(options: dict[str, pd.Timestamp], now: pd.Timestamp | None) -> str:
    """Now if the night is under way, else about two hours after dark."""
    keys, times = list(options), list(options.values())
    target = times[0] + pd.Timedelta(hours=2)
    if now is not None and times[0] <= now <= times[-1]:
        target = now
    return keys[int(np.argmin([abs((t - target).total_seconds()) for t in times]))]


def up_now(guide: dict, i: int, zenith_sqm: float) -> list[tuple[sky.SkyItem, float, float, bool]]:
    """Everything in the guide that is above the horizon at time index i, with its position then
    and whether it shows to the eye at that moment."""
    out = []
    for it in guide["items"]:
        a, z = float(it.track_alt[i]), float(it.track_az[i])
        if a <= 3:
            continue
        seen = it.kind == "moon" or sky.is_visible(
            a, z, it.vis_mag, it.extended, zenith_sqm, guide["moon"], i
        )
        out.append((it, a, z, seen))
    out.sort(key=lambda r: (KIND_ORDER.get(r[0].kind, 5), r[0].mag if r[0].mag is not None else 9))
    return out


def object_row(it: sky.SkyItem, alt: float, az: float, seen: bool) -> str:
    where = sky.where_words(alt, az)
    tag = (
        '<span class="sk-badge" style="--c:var(--sk-go)">Visible</span>'
        if seen
        else '<span class="sk-badge" title="Too faint to see from here now">Too faint</span>'
    )
    if it.kind == "moon":
        sub = it.note  # "75% lit"
    elif it.kind in ("pattern", "cluster", "galaxy", "nebula") and it.note:
        sub = lookup.cap(it.note)
    else:
        sub = lookup.KIND_LABELS.get(it.kind, "")
    return ui.row(
        [
            f'<div class="sk-main"><div class="sk-row-title">{esc(it.name)}</div>'
            f'<div class="sk-row-sub">{esc(sub)}</div></div>',
            f'<div class="sk-row-sub" style="font-size:.86rem">{esc(lookup.cap(where))}</div>',
            f'<div style="text-align:right">{tag}</div>',
        ],
        "minmax(0,1.4fr) minmax(0,1fr) 96px",
    )


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
            f"{esc(lookup.uncap(d.verdict))}. Pick a time to see what's up.",
            f"Sky guide · {dusk.tz_convert(tz):%a %b %-d}",
        ),
        unsafe_allow_html=True,
    )
    options = time_options(dusk, dawn, tz)
    c1, c2 = st.columns([3, 1.3], vertical_alignment="bottom", gap="large")
    choice = c1.select_slider(
        "Time tonight", list(options), value=default_choice(options, ctx.now_utc), key="sky_time"
    )
    mode_label = c2.segmented_control("Show", list(MODES), default="Your sky", key="sky_mode")
    t = options[choice]
    mode = MODES[mode_label or "Your sky"]
    i = int(np.argmin(np.abs((guide["times"] - t).total_seconds())))

    left, right = st.columns([1.15, 1], gap="large")
    with left:
        svg, info = ctx.service("chart")(ctx.site, t.floor("min"), sqm, mode, True, True, False)
        stars = lookup.fmt_count(info["stars"])
        what = "you can see" if mode == "here" else "a perfectly dark sky would show"
        st.markdown(
            f'<div class="sk-chart">{svg}</div><div class="sk-chart-help">About {stars} stars '
            f"{what} at {lookup.clock(t, tz)}. Hold it overhead with north at the top, or turn it "
            "so the direction you face is at the bottom.</div>",
            unsafe_allow_html=True,
        )
    with right:
        rows = up_now(guide, i, sqm if mode == "here" else sky.NATURAL_SQM)
        if mode == "dark":
            rows = [(it, a, z, True) for it, a, z, _ in rows]
        st.markdown(ui.eyebrow(f"Up at {lookup.clock(t, tz)}"), unsafe_allow_html=True)
        best = [r for r in rows if r[0].kind in ("moon", "planet")]
        stars_ = [r for r in rows if r[0].kind == "star"][:4]
        shapes = [r for r in rows if r[0].kind == "pattern"][:4]
        deep = [r for r in rows if r[0].kind in ("cluster", "galaxy", "nebula")][:4]
        html = ""
        for title, group in [
            ("Moon and planets", best),
            ("Brightest stars", stars_),
            ("Star patterns", shapes),
            ("Clusters, galaxies, nebulae", deep),
        ]:
            if group:
                html += f'<div class="sk-eyebrow" style="margin-top:14px">{title}</div>'
                html += ui.rows([object_row(*r) for r in group])
        st.markdown(html, unsafe_allow_html=True)

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
