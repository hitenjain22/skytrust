"""Small HTML building blocks for the Observatory design system.

Each function returns an HTML string for `st.markdown(..., unsafe_allow_html=True)`. Streamlit's
Markdown parser turns indented lines into code blocks and blank lines into paragraph breaks, so
every snippet is collapsed onto one line by `block()`. Anything a user can type (a custom
location's name) goes through `esc()`. Colours come from `currentColor` (see theme.py), except
status dots, which take a palette colour.
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
    gibbous Moon's away from it; the SVG arc sweep flags encode exactly that. Drawn in the
    current text colour, so it matches whichever theme is showing.
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
        f'<circle cx="{c}" cy="{c}" r="{r:.2f}" fill="currentColor" fill-opacity=".12" '
        f'stroke="currentColor" stroke-opacity=".22" stroke-width="1"/>'
        f'<path d="{lit}" fill="currentColor" fill-opacity=".86"/></svg>'
    )


# ---------- pieces ----------


def status(text: str, color: str) -> str:
    """A quiet status pill: hairline outline, text in the body colour, meaning in a small dot."""
    dot = f'<span class="sk-dot" style="--c:{color}"></span>'
    return f'<span class="sk-status">{dot}{esc(text)}</span>'


# Older name used by some callers.
pill = status


def meter(p: float | None) -> str:
    """Thin track filled to p (0-1)."""
    width = 0 if p is None else max(0.0, min(1.0, p)) * 100
    fill = f'<span style="width:{width:.1f}%"></span>'
    return f'<div class="sk-meter" aria-hidden="true">{fill}</div>'


def bar(p: float | None, color: str) -> str:
    width = 0 if p is None else max(0.0, min(1.0, p)) * 100
    return f'<div class="sk-bar"><span style="width:{width:.1f}%;--c:{color}"></span></div>'


def stat(label: str, value: str, sub: str = "", big: bool = False, icon: str = "") -> str:
    cls = "sk-stat-big" if big else "sk-stat-value"
    return block(
        '<div class="sk-stat">',
        f'<div class="sk-eyebrow">{esc(label)}</div>',
        f'<div class="{cls}">{icon}<span>{value}</span></div>',
        f'<div class="sk-stat-sub">{sub}</div>' if sub else "",
        "</div>",
    )


def strip(items: list[str], three: bool = False, five: bool = False) -> str:
    cls = " three" if three else " five" if five else ""
    return f'<div class="sk-strip{cls} sk-rise sk-d2">{"".join(items)}</div>'


def row(cells: list[str], cols: str) -> str:
    """One list row; `cols` is a CSS grid-template-columns value."""
    return f'<div class="sk-row" style="--cols:{cols}">{"".join(cells)}</div>'


def rows(items: list[str]) -> str:
    return f'<div class="sk-list">{"".join(items)}</div>'


def legend(items: list[tuple[str, str]]) -> str:
    """Colour key shown under every colour-coded chart (the classic Clear Sky Chart flaw is
    colour with no key)."""
    spans = "".join(
        f'<span><span class="sk-swatch" style="--c:{c}"></span>{esc(t)}</span>' for t, c in items
    )
    return f'<div class="sk-legend">{spans}</div>'


def eyebrow(text: str) -> str:
    return f'<div class="sk-eyebrow">{esc(text)}</div>'


def trust_dots(level: str) -> str:
    filled = {"High": 3, "Medium": 2, "Low": 1}.get(level, 0)
    dots = "●" * filled + "○" * (3 - filled)
    return f'<span class="sk-dots" title="{esc(level)} trust">{dots}</span>'
