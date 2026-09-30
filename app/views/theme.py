"""The "Moonlight" design system: colour palettes (normal + red night vision) and the global CSS.

Every custom HTML element reads its colours from CSS variables set here, so switching to night
vision recolours the whole page (cards, moon icon, pills) as well as the Plotly charts, which read
the same palette dict.
"""

from __future__ import annotations

import streamlit as st

MOON = {
    "bg": "#070B18",
    "paper": "rgba(0,0,0,0)",
    "surface": "#111831",
    "surface2": "#172044",
    "border": "#24305A",
    "text": "#E6E8F2",
    "muted": "#98A2C3",
    "grid": "rgba(152,162,195,0.14)",
    "accent": "#E8C872",  # moonlight gold
    "accent2": "#8FB3FF",  # lunar blue
    "Go": "#5ED3A5",
    "Maybe": "#F2C35B",
    "Skip": "#F28482",
    "No data": "#6B7394",
    "dark": "rgba(143,179,255,0.10)",
    "twilight": "rgba(143,179,255,0.04)",
    "moon": "#F4EFD8",
    "moon_shadow": "#1B2346",
    # Paul Tol's colour-blind-safe palette (lighter variants for the dark background).
    "models": {
        "gfs": "#88CCEE",
        "hrrr": "#DDCC77",
        "ecmwf": "#44AA99",
        "gem": "#AA99DD",
        "icon": "#CC6677",
        "median": "#FFFFFF",
    },
    "layers": {"low": "#8FB3FF", "mid": "#C3A6FF", "high": "#E8C872"},
    "methods": {
        "blend": "#E8C872",
        "equal_weight": "#8FB3FF",
        "climatology": "#7A83A6",
        "nbm_lr": "#C792EA",
        "equal_weight_cal": "#5FB7D9",
    },
    "heat": [[0, "#0B1633"], [0.2, "#1F3B73"], [0.5, "#5F7FB8"], [0.8, "#B8C4DE"], [1, "#F1F3F9"]],
}

# Astronomers use dim red light to keep their eyes dark-adapted.
NIGHT = {
    "bg": "#000000",
    "paper": "rgba(0,0,0,0)",
    "surface": "#0A0000",
    "surface2": "#140000",
    "border": "#3A0000",
    "text": "#FF3B30",
    "muted": "#A3261F",
    "grid": "rgba(255,59,48,0.15)",
    "accent": "#FF3B30",
    "accent2": "#C0392B",
    "Go": "#FF5A4F",
    "Maybe": "#C0392B",
    "Skip": "#7F1D1D",
    "No data": "#5A1A1A",
    "dark": "rgba(255,59,48,0.10)",
    "twilight": "rgba(255,59,48,0.04)",
    "moon": "#FF8A80",
    "moon_shadow": "#2A0000",
    "models": {
        k: c
        for k, c in zip(
            ["gfs", "hrrr", "ecmwf", "gem", "icon"],
            ["#FF6B60", "#E04B40", "#C0392B", "#A93226", "#8B1E1E"],
            strict=True,
        )
    }
    | {"median": "#FF3B30"},
    "layers": {"low": "#FF6B60", "mid": "#C0392B", "high": "#8B1E1E"},
    "methods": {
        "blend": "#FF3B30",
        "equal_weight": "#C0392B",
        "climatology": "#6B1A1A",
        "nbm_lr": "#A93226",
    },
    "heat": [[0, "#000000"], [0.5, "#5A0F0A"], [1, "#FF3B30"]],
}

# Backwards-compatible name used by older code and tests.
DAY = MOON


def palette(night_vision: bool) -> dict:
    return NIGHT if night_vision else MOON


def _variables(pal: dict) -> str:
    keys = ["bg", "surface", "surface2", "border", "text", "muted", "accent", "accent2", "moon",
            "moon_shadow", "Go", "Maybe", "Skip"]  # fmt: skip
    return "".join(f"--sk-{k.lower().replace(' ', '-')}:{pal[k]};" for k in keys)


# A few hundred pixels of hand-placed "stars" tiled across the page, under two soft glows (the
# Moon's halo top right, a faint blue sky glow top left). Pure CSS: no images to load.
_STARS = [
    f"radial-gradient({r}px {r}px at {x}px {y}px, rgba(255,255,255,{a}) 50%, transparent 51%)"
    for x, y, r, a in [
        (23, 41, 1, 0.55), (91, 187, 1, 0.35), (147, 67, 1.4, 0.6), (211, 223, 1, 0.3),
        (263, 119, 1, 0.45), (57, 271, 1.2, 0.4), (301, 31, 1, 0.3), (181, 301, 1, 0.5),
        (331, 211, 1.5, 0.55), (121, 131, 1, 0.25), (237, 17, 1, 0.4), (17, 157, 1, 0.3),
    ]
]  # fmt: skip


def css(pal: dict, night_vision: bool) -> str:
    # One background-size / background-repeat entry per layer: CSS silently cycles a shorter list,
    # which once tiled the soft glows like stars and striped the page.
    layers, sizes, repeats = [], [], []
    if not night_vision:
        stars = list(_STARS)
        glows = [
            "radial-gradient(900px 520px at 88% -8%, rgba(232,200,114,0.10), transparent 62%)",
            "radial-gradient(800px 480px at -8% 6%, rgba(143,179,255,0.08), transparent 60%)",
        ]
        layers += stars + glows
        sizes += ["360px 330px"] * len(stars) + ["100% 100%"] * len(glows)
        repeats += ["repeat"] * len(stars) + ["no-repeat"] * len(glows)
    image = ", ".join(layers) if layers else "none"
    size = ", ".join(sizes) if sizes else "auto"
    repeat = ", ".join(repeats) if repeats else "no-repeat"
    return f"""
<style>
:root {{ {_variables(pal)} }}
.stApp {{
  background-color: var(--sk-bg);
  background-image: {image};
  background-size: {size};
  background-repeat: {repeat};
  background-attachment: fixed;
}}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stMainBlockContainer"], .block-container {{
  max-width: 1120px; padding-top: 2.2rem; padding-bottom: 3rem;
}}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
/* ---------- cards ---------- */
.sk-card {{
  background: linear-gradient(180deg, color-mix(in srgb, var(--sk-surface2) 88%, transparent),
              color-mix(in srgb, var(--sk-surface) 92%, transparent));
  border: 1px solid var(--sk-border); border-radius: 18px; padding: 18px 20px;
  box-shadow: 0 12px 32px rgba(0,0,0,0.28);
}}
.sk-eyebrow {{ font-size: .72rem; font-weight: 600; letter-spacing: .12em; text-transform: uppercase;
  color: var(--sk-muted); margin-bottom: 6px; }}
.sk-muted {{ color: var(--sk-muted); }}
.sk-brand {{ display:flex; align-items:center; gap:12px; margin-bottom: 4px; }}
.sk-brand-name {{ font-family: Fraunces, serif; font-size: 1.7rem; font-weight: 600; line-height:1;
  color: var(--sk-text); }}
.sk-brand-tag {{ color: var(--sk-muted); font-size: .92rem; }}
/* ---------- hero ---------- */
.sk-hero {{ display:grid; grid-template-columns: 190px 1fr; gap: 28px; align-items:center; }}
.sk-ring {{ width:176px; height:176px; border-radius:50%; display:grid; place-items:center;
  background: conic-gradient(var(--c) calc(var(--p) * 1%), rgba(255,255,255,0.07) 0);
  box-shadow: 0 0 46px color-mix(in srgb, var(--c) 30%, transparent); }}
.sk-ring-in {{ width:146px; height:146px; border-radius:50%; background: var(--sk-surface);
  display:grid; place-items:center; text-align:center; padding: 0 10px; }}
.sk-ring-num {{ font-family: Fraunces, serif; font-size: 2.9rem; font-weight: 600; line-height:1;
  color: var(--sk-text); }}
.sk-ring-sub {{ font-size: .72rem; color: var(--sk-muted); line-height:1.25; margin-top:4px; }}
.sk-headline {{ font-family: Fraunces, serif; font-size: 1.85rem; font-weight: 600; line-height:1.15;
  margin: 8px 0 8px 0; color: var(--sk-text); }}
.sk-lede {{ font-size: 1.02rem; line-height: 1.55; color: var(--sk-text); opacity:.92; }}
.sk-pill {{ display:inline-block; padding: 3px 12px; border-radius: 999px; font-weight: 700;
  font-size: .78rem; letter-spacing: .08em; color: #0B1020; background: var(--c); }}
.sk-chip {{ display:inline-flex; align-items:center; gap:6px; padding: 4px 10px; margin: 8px 6px 0 0;
  border-radius: 999px; font-size: .82rem; color: var(--sk-text);
  background: color-mix(in srgb, var(--sk-surface2) 80%, transparent);
  border: 1px solid var(--sk-border); }}
.sk-note {{ margin-top: 12px; padding: 10px 12px; border-radius: 12px; font-size: .9rem;
  background: color-mix(in srgb, var(--sk-accent) 10%, transparent);
  border: 1px solid color-mix(in srgb, var(--sk-accent) 35%, transparent); color: var(--sk-text); }}
/* ---------- tiles ---------- */
.sk-tiles {{ display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 12px; margin: 14px 0 4px; }}
.sk-tile {{ background: color-mix(in srgb, var(--sk-surface) 92%, transparent);
  border: 1px solid var(--sk-border); border-radius: 16px; padding: 14px 16px; min-height: 118px; }}
.sk-tile-value {{ font-size: 1.25rem; font-weight: 600; color: var(--sk-text); line-height:1.25; }}
.sk-tile-sub {{ font-size: .82rem; color: var(--sk-muted); margin-top: 6px; line-height:1.35; }}
.sk-tile-row {{ display:flex; align-items:center; gap: 12px; }}
/* ---------- week strip ---------- */
.sk-week {{ display:grid; grid-template-columns: repeat(7, minmax(0,1fr)); gap: 10px; margin: 6px 0 8px; }}
.sk-day {{ background: color-mix(in srgb, var(--sk-surface) 92%, transparent); border: 1px solid var(--sk-border);
  border-top: 3px solid var(--c); border-radius: 16px; padding: 12px 10px; text-align:center; }}
.sk-day-name {{ font-weight: 600; font-size: .92rem; color: var(--sk-text); }}
.sk-day-date {{ font-size: .75rem; color: var(--sk-muted); }}
.sk-day-p {{ font-family: Fraunces, serif; font-size: 1.7rem; font-weight: 600; color: var(--sk-text);
  margin: 4px 0 2px; }}
.sk-day-meta {{ font-size: .74rem; color: var(--sk-muted); line-height: 1.35; margin-top: 6px; }}
.sk-dots {{ letter-spacing: 2px; color: var(--sk-accent); }}
/* ---------- ranked list ---------- */
.sk-rank {{ display:grid; grid-template-columns: 44px 1fr auto; gap: 14px; align-items:center;
  padding: 12px 16px; margin-bottom: 10px; border-radius: 16px; border: 1px solid var(--sk-border);
  border-left: 4px solid var(--c); background: color-mix(in srgb, var(--sk-surface) 92%, transparent); }}
.sk-rank-n {{ font-family: Fraunces, serif; font-size: 1.5rem; color: var(--sk-muted); text-align:center; }}
.sk-rank-name {{ font-weight: 600; font-size: 1.02rem; color: var(--sk-text); }}
.sk-rank-p {{ font-family: Fraunces, serif; font-size: 1.6rem; font-weight: 600; color: var(--sk-text);
  text-align:right; }}
/* ---------- stats ---------- */
.sk-stats {{ display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 12px; margin: 8px 0 6px; }}
.sk-stat-num {{ font-family: Fraunces, serif; font-size: 2.3rem; font-weight: 600; color: var(--sk-accent);
  line-height: 1.05; }}
.sk-stat-text {{ font-size: .92rem; color: var(--sk-text); margin-top: 6px; line-height: 1.4; }}
.sk-stat-sub {{ font-size: .78rem; color: var(--sk-muted); margin-top: 6px; }}
.sk-steps {{ display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 12px; }}
.sk-step-n {{ width: 30px; height: 30px; border-radius: 50%; display:grid; place-items:center;
  font-weight: 700; color: #0B1020; background: var(--sk-accent); margin-bottom: 10px; }}
.sk-legend {{ display:flex; flex-wrap: wrap; gap: 14px; font-size: .8rem; color: var(--sk-muted); margin: 2px 0 6px; }}
.sk-swatch {{ display:inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 6px;
  vertical-align: -1px; background: var(--c); }}
/* ---------- responsive ---------- */
@media (max-width: 900px) {{
  .sk-tiles {{ grid-template-columns: repeat(2, minmax(0,1fr)); }}
  .sk-week {{ grid-template-columns: repeat(4, minmax(0,1fr)); }}
  .sk-stats, .sk-steps {{ grid-template-columns: 1fr; }}
}}
@media (max-width: 640px) {{
  .sk-hero {{ grid-template-columns: 1fr; justify-items: center; text-align: center; gap: 14px; }}
  .sk-headline {{ font-size: 1.5rem; }}
  .sk-week {{ grid-template-columns: repeat(2, minmax(0,1fr)); }}
  [data-testid="stMainBlockContainer"], .block-container {{ padding-top: 1.2rem; }}
}}
{_night_overrides() if night_vision else ""}
</style>
"""


def _night_overrides() -> str:
    """Red-only display: recolour Streamlit's own widgets too, and drop every glow."""
    return """
.stApp, [data-testid="stSidebar"] { background: #000 !important; }
[data-testid="stHeader"] { background: transparent !important; }
/* Streamlit paints a few widget parts in the theme's gold; make those red too. */
[data-testid="stCheckbox"] label > div:first-of-type { background-color: #7f1d1d !important; }
[data-testid="stButtonGroup"] button[aria-checked="true"] { background-color: rgba(255,59,48,0.14) !important; }
[data-testid="stSlider"] [role="group"] div { background-image: none !important; }
[data-testid="stSlider"] [role="group"] div[role="slider"],
[data-testid="stSlider"] [role="group"] > div > div { background-color: #ff3b30 !important; }
.stApp *, [data-testid="stSidebar"] * { color: #ff3b30 !important; border-color: #3a0000 !important; }
.sk-pill, .sk-step-n { color: #000 !important; }
.sk-ring, .sk-card { box-shadow: none !important; }
img { filter: grayscale(1) sepia(1) saturate(6) hue-rotate(-50deg) brightness(.6); }
"""


def inject(pal: dict, night_vision: bool) -> None:
    st.markdown(css(pal, night_vision), unsafe_allow_html=True)
