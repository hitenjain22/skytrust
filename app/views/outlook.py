"""The week ahead on Tonight: one row per night, like a 10-day weather forecast (the chance of a
clear night, the Moon's phase, and how far that night's number can be trusted)."""

from __future__ import annotations

import pandas as pd

from skytrust import haze
from views import components as ui
from views.common import Context, clearest, day_label, short_time

TRUST_TEXT = {"High": "high trust", "Medium": "medium trust", "Low": "low trust"}
COLS = "minmax(120px, 1.3fr) 26px minmax(60px, 2fr) 52px 96px"


def night_row(ctx: Context, n, i: int) -> str:
    tz, pal = ctx.site.timezone, ctx.palette
    span = clearest(n)
    window = f"{short_time(span[0], tz)}–{short_time(span[1], tz)}" if span else "no clear window"
    w = haze.words(ctx.haze_aod(n.dusk_utc, n.dawn_utc))
    smoke = f" · {ui.icon('haze', 12)} hazy" if w and w[0] != "Some haze" else ""
    sub = (
        f"{pd.Timestamp(n.night_date):%b %-d} · {window}{smoke} · {ui.trust_dots(n.trust)} "
        f'<span class="sk-hide-sm">{TRUST_TEXT.get(n.trust, "")}</span>'
    )
    p = "–" if n.p_usable is None else f"{n.p_usable:.0%}"
    day = day_label(n.night_date, i, ctx.now_utc, tz)
    return ui.row(
        [
            f'<div class="sk-main"><div class="sk-row-title">{day}</div>'
            f'<div class="sk-row-sub">{sub}</div></div>',
            ui.moon_svg(n.moon_phase_deg, 20),
            f'<div class="sk-grow">{ui.bar(n.p_usable, pal[n.verdict])}</div>',
            f'<div class="sk-row-p">{p}</div>',
            f'<div style="text-align:right">{ui.status(n.verdict.upper(), pal[n.verdict])}</div>',
        ],
        COLS,
    )
