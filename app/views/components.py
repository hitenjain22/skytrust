"""Small HTML building blocks (cards, tiles, the probability ring, the Moon icon).

Each function returns an HTML string for `st.markdown(..., unsafe_allow_html=True)`. Streamlit's
Markdown parser turns indented lines into code blocks and blank lines into paragraph breaks, so
every snippet is collapsed onto one line by `block()`. Anything a user can type (a custom
location's name) goes through `esc()`.
"""

from __future__ import annotations

import html
import math


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def block(*parts: str) -> str:
    """Join HTML fragments into one line (see module docstring). Only indentation is removed:
    a trailing space can be the gap between two words."""
    return "".join(line.lstrip() for part in parts for line in part.splitlines())


# ---------- the Moon ----------

PHASE_NAMES = [
    (22.5, "New Moon"),
    (67.5, "Waxing crescent"),
    (112.5, "First quarter"),
    (157.5, "Waxing gibbous"),
    (202.5, "Full Moon"),
    (247.5, "Waning gibbous"),
    (292.5, "Last quarter"),
    (337.5, "Waning crescent"),
    (360.1, "New Moon"),
]


def phase_name(phase_deg: float | None) -> str:
    if phase_deg is None or math.isnan(phase_deg):
        return "Moon"
    return next(name for limit, name in PHASE_NAMES if phase_deg % 360 < limit)


def moon_svg(phase_deg: float | None, size: int = 44) -> str:
    """The Moon as seen from the northern hemisphere: lit on the right while waxing.

    The lit area is bounded by the disk's edge (a half circle) and the terminator, an ellipse
    whose half-width is r * |cos(phase)|. A crescent's terminator bulges toward the lit edge, a
    gibbous Moon's away from it; the SVG arc sweep flags encode exactly that.
    """
    r, c = size / 2 - 1, size / 2
    if phase_deg is None or math.isnan(phase_deg):
        phase_deg = 180.0
    phi = math.radians(phase_deg % 360)
    rx = abs(math.cos(phi)) * r
    waxing = (phase_deg % 360) < 180
    crescent = math.cos(phi) > 0  # less than half lit
    edge_sweep = 1 if waxing else 0  # top -> bottom along the lit edge
    term_sweep = (0 if crescent else 1) if waxing else (1 if crescent else 0)
    top, bottom = f"{c:.2f},{c - r:.2f}", f"{c:.2f},{c + r:.2f}"
    lit = (
        f"M{top} A{r:.2f},{r:.2f} 0 0 {edge_sweep} {bottom} "
        f"A{rx:.2f},{r:.2f} 0 0 {term_sweep} {top} Z"
    )
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" role="img" '
        f'aria-label="{phase_name(phase_deg)}">'
        f'<circle cx="{c}" cy="{c}" r="{r:.2f}" fill="var(--sk-moon_shadow)" '
        f'stroke="var(--sk-border)" stroke-width="1"/>'
        f'<path d="{lit}" fill="var(--sk-moon)"/></svg>'
    )


# ---------- pieces ----------


def pill(text: str, color: str) -> str:
    return f'<span class="sk-pill" style="--c:{color}">{esc(text)}</span>'


def chip(text: str) -> str:
    return f'<span class="sk-chip">{text}</span>'


def ring(p: float | None, color: str, sub: str) -> str:
    """Donut gauge filled to `p` (0-1), with the percentage in the middle."""
    value = "–" if p is None else f"{p:.0%}"
    fill = 0 if p is None else round(p * 100, 1)
    return block(
        f'<div class="sk-ring" style="--p:{fill};--c:{color}"><div class="sk-ring-in"><div>',
        f'<div class="sk-ring-num">{value}</div><div class="sk-ring-sub">{esc(sub)}</div>',
        "</div></div></div>",
    )


def tile(label: str, value: str, sub: str = "", icon: str = "") -> str:
    head = f'<div class="sk-eyebrow">{esc(label)}</div>'
    body = f'<div class="sk-tile-value">{value}</div>'
    if icon:
        body = f'<div class="sk-tile-row">{icon}<div>{body}</div></div>'
    return f'<div class="sk-tile">{head}{body}<div class="sk-tile-sub">{sub}</div></div>'


def tiles(items: list[str]) -> str:
    return f'<div class="sk-tiles">{"".join(items)}</div>'


def card(inner: str) -> str:
    return f'<div class="sk-card">{inner}</div>'


def legend(items: list[tuple[str, str]]) -> str:
    """Colour key shown next to every colour-coded chart (the classic Clear Sky Chart flaw is
    colour with no key)."""
    spans = "".join(
        f'<span><span class="sk-swatch" style="--c:{c}"></span>{esc(t)}</span>' for t, c in items
    )
    return f'<div class="sk-legend">{spans}</div>'


def stat(number: str, text: str, sub: str = "") -> str:
    return (
        f'<div class="sk-card"><div class="sk-stat-num">{number}</div>'
        f'<div class="sk-stat-text">{text}</div><div class="sk-stat-sub">{sub}</div></div>'
    )


def trust_dots(level: str) -> str:
    filled = {"High": 3, "Medium": 2, "Low": 1}.get(level, 0)
    return f'<span class="sk-dots">{"●" * filled}{"○" * (3 - filled)}</span>'
