"""The "Observatory" design system: tokens, palettes, and the global stylesheet.

Principles (from studying Vercel's Geist, Linear's 2025 refresh, Apple Weather and published
motion guidelines):

- Monochrome structure, colour only for meaning. Status is a small muted dot (sage / ochre /
  clay), never a large fill. One quiet accent (champagne) marks SkyTrust's own series in charts.
- Soft ends: graphite and paper instead of pure black and white.
- Every custom colour is derived from `currentColor` (borders 11%, surfaces 4%, secondary text
  60%), so the same HTML is correct in the light and the dark theme without Python needing to
  know which one the browser is showing.
- Motion: transform/opacity only, strong ease-out, 30-80 ms stagger, under ~300 ms for UI;
  hover effects only on precise pointers; reduced motion keeps only gentle fades.
"""

from __future__ import annotations

import streamlit as st

EASE = "cubic-bezier(0.23, 1, 0.32, 1)"

# Data colours for charts. Plotly draws text, grids and backgrounds from Streamlit's active theme;
# these mid-lightness values stay readable on both the light and the dark background.
MONO = {
    "paper": "rgba(0,0,0,0)",
    "text": None,  # let the Streamlit chart theme decide
    "muted": "#8B8B94",
    "grid": "rgba(139,139,148,0.16)",
    "border": "rgba(139,139,148,0.30)",
    "surface": None,
    "ink": "#8E8E97",
    "accent": "#B5A27A",  # champagne: SkyTrust's own series
    "accent2": "#8B8B94",
    "Go": "#5E9E80",
    "Maybe": "#C29A4A",
    "Skip": "#C46A62",
    "No data": "#8B8B94",
    "dark": "rgba(139,139,148,0.10)",
    "moon": "#B4B4BB",
    "models": {
        "gfs": "#5B8DB8",
        "hrrr": "#B8923F",
        "ecmwf": "#4E9A7E",
        "gem": "#8C73B8",
        "icon": "#B86B6B",
        "median": "#8B8B94",
    },
    "layers": {"low": "#6F8FAF", "mid": "#8C7BAE", "high": "#B39A5E"},
    "methods": {
        "blend": "#B5A27A",
        "equal_weight": "#7F8FA6",
        "climatology": "#8B8B94",
        "nbm_lr": "#9C84B5",
        "equal_weight_cal": "#6FA0A0",
    },
    "heat": [[0, "#2A2C31"], [0.35, "#4B4E56"], [0.7, "#8E9099"], [1, "#DADBE0"]],
}

# Older names kept for callers and tests.
MOON = DAY = MONO


def palette() -> dict:
    return MONO


def css(pal: dict) -> str:
    status = "".join(f"--sk-{k.lower().replace(' ', '-')}:{pal[k]};" for k in
                     ["Go", "Maybe", "Skip", "No data", "accent"])  # fmt: skip
    return f"""
<style>
:root {{ {status} --sk-ease: {EASE}; }}
/* ---------- canvas ---------- */
/* Streamlit's header is a full-width strip over the top of the page: it sat on the tab row and
   swallowed clicks. The theme follows the system setting, so its menu isn't needed. */
[data-testid="stHeader"] {{ display: none; }}
[data-testid="stDecoration"] {{ display: none; }}
[data-testid="stMainBlockContainer"], .block-container {{
  max-width: 1080px; padding-top: 1.25rem; padding-bottom: 4rem;
}}
h1, h2, h3, h4 {{ letter-spacing: -0.025em; }}
/* ---------- tokens derived from the text colour ---------- */
.sk-muted {{ color: color-mix(in srgb, currentColor 60%, transparent); }}
.sk-mono {{ font-family: "Geist Mono", ui-monospace, monospace; font-feature-settings: "tnum"; }}
.sk-eyebrow {{ font-family: "Geist Mono", ui-monospace, monospace; font-size: .72rem; font-weight: 500;
  letter-spacing: .08em; text-transform: uppercase; color: color-mix(in srgb, currentColor 55%, transparent); }}
/* ---------- sticky top bar ----------
   Streamlit wraps each container in a wrapper only as tall as the container, and sticky only
   works inside its parent, so the wrapper (whose parent is the whole page column) is what sticks. */
[data-testid="stLayoutWrapper"]:has(> .st-key-topbar) {{ position: sticky; top: 0; z-index: 50; }}
.st-key-topbar {{
  position: relative; padding: 10px 0 8px; margin-bottom: 10px;
  backdrop-filter: saturate(1.3) blur(14px); -webkit-backdrop-filter: saturate(1.3) blur(14px);
  --sk-bar-solid: rgba(20,20,22,.96); background: rgba(20,20,22,.72);
  border-bottom: 1px solid color-mix(in srgb, currentColor 7%, transparent);
}}
@media (prefers-color-scheme: light) {{
  .st-key-topbar {{ --sk-bar-solid: rgba(246,246,244,.96); background: rgba(246,246,244,.72); }}
}}
.sk-brand {{ display:flex; align-items:center; gap:10px; font-weight: 600; font-size: 1.02rem;
  letter-spacing: -0.02em; white-space: nowrap; }}
.sk-brand small {{ font-weight: 400; color: color-mix(in srgb, currentColor 50%, transparent); font-size: .8rem;
  letter-spacing: 0; }}
/* navigation: text tabs with an underline, not pills */
.st-key-topbar [data-testid="stButtonGroup"] button {{
  border: 0 !important; background: transparent !important; border-radius: 0 !important;
  color: color-mix(in srgb, currentColor 58%, transparent) !important;
  min-height: 40px; padding: 8px 12px !important; margin: 0; cursor: pointer;
  box-shadow: inset 0 -1px 0 transparent;
  transition: color 150ms ease, box-shadow 200ms var(--sk-ease), transform 160ms var(--sk-ease);
}}
.st-key-topbar [data-testid="stButtonGroup"] button[aria-checked="true"] {{
  color: inherit !important; box-shadow: inset 0 -2px 0 currentColor;
}}
.st-key-topbar [data-testid="stButtonGroup"] button:active {{ transform: scale(0.97); }}
@media (hover: hover) and (pointer: fine) {{
  .st-key-topbar [data-testid="stButtonGroup"] button:hover {{ color: inherit !important; }}
}}
/* ---------- hero ---------- */
.sk-hero {{ display:grid; grid-template-columns: minmax(220px, 300px) 1fr; gap: 48px; align-items: start;
  padding: 16px 0 6px; }}
.sk-display {{ font-size: clamp(4.2rem, 11vw, 6.6rem); font-weight: 350; line-height: .9; letter-spacing: -0.055em;
  font-feature-settings: "tnum"; }}
.sk-display sup {{ font-size: .42em; font-weight: 400; letter-spacing: -0.02em; vertical-align: .95em; margin-left: 2px; }}
.sk-caption {{ margin-top: 10px; font-size: .9rem; color: color-mix(in srgb, currentColor 60%, transparent); }}
.sk-meter {{ height: 3px; border-radius: 3px; margin-top: 16px; background: color-mix(in srgb, currentColor 10%, transparent);
  overflow: hidden; }}
.sk-meter > span {{ display:block; height:100%; border-radius: 3px; background: currentColor; opacity: .85;
  transform-origin: left; animation: sk-grow 700ms var(--sk-ease) both 120ms; }}
.sk-headline {{ font-size: clamp(1.55rem, 3vw, 2.05rem); font-weight: 500; letter-spacing: -0.03em; line-height: 1.15;
  margin: 14px 0 10px; }}
.sk-lede {{ font-size: 1.02rem; line-height: 1.6; max-width: 62ch; color: color-mix(in srgb, currentColor 78%, transparent); }}
.sk-lede b {{ font-weight: 550; }}
.sk-status {{ display:inline-flex; align-items:center; gap:8px; padding: 4px 10px 4px 9px; border-radius: 999px;
  font-size: .8rem; font-weight: 500; border: 1px solid color-mix(in srgb, currentColor 14%, transparent); }}
.sk-dot {{ width: 7px; height: 7px; border-radius: 50%; background: var(--c); flex: none;
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--c) 18%, transparent); }}
.sk-note {{ display:flex; gap:10px; align-items:flex-start; margin-top: 16px; font-size: .9rem; line-height: 1.5;
  color: color-mix(in srgb, currentColor 72%, transparent); max-width: 64ch; }}
.sk-note svg {{ flex: none; margin-top: 2px; }}
/* ---------- stat strip (hairline-divided, no boxes) ---------- */
.sk-strip {{ display:grid; grid-template-columns: repeat(4, minmax(0,1fr));
  border-top: 1px solid color-mix(in srgb, currentColor 11%, transparent);
  border-bottom: 1px solid color-mix(in srgb, currentColor 11%, transparent); margin: 26px 0 8px; }}
.sk-strip.three {{ grid-template-columns: repeat(3, minmax(0,1fr)); }}
.sk-stat {{ padding: 18px 20px 18px 0; }}
.sk-stat + .sk-stat {{ padding-left: 20px; border-left: 1px solid color-mix(in srgb, currentColor 11%, transparent); }}
.sk-stat-value {{ font-size: 1.22rem; font-weight: 500; letter-spacing: -0.02em; margin-top: 8px; line-height: 1.25;
  display:flex; align-items:center; gap: 10px; }}
.sk-stat-big {{ font-size: clamp(2.2rem, 5vw, 3rem); font-weight: 350; letter-spacing: -0.045em; line-height: 1;
  margin-top: 10px; font-feature-settings: "tnum"; }}
.sk-stat-sub {{ font-size: .84rem; line-height: 1.45; margin-top: 8px; color: color-mix(in srgb, currentColor 58%, transparent); }}
/* ---------- list rows (Apple-Weather-style) ---------- */
.sk-list {{ border-top: 1px solid color-mix(in srgb, currentColor 11%, transparent); margin: 8px 0 22px; }}
.sk-row {{ display:grid; grid-template-columns: var(--cols); gap: 16px; align-items:center; padding: 14px 8px;
  border-bottom: 1px solid color-mix(in srgb, currentColor 11%, transparent);
  transition: background-color 150ms ease; }}
@media (hover: hover) and (pointer: fine) {{
  .sk-row:hover {{ background: color-mix(in srgb, currentColor 4%, transparent); }}
}}
.sk-row-title {{ font-weight: 500; letter-spacing: -0.01em; }}
.sk-row-sub {{ font-size: .8rem; margin-top: 2px; color: color-mix(in srgb, currentColor 55%, transparent); }}
.sk-row-p {{ font-size: 1.15rem; font-weight: 450; text-align: right; font-feature-settings: "tnum"; letter-spacing: -0.02em; }}
.sk-bar {{ position: relative; height: 4px; border-radius: 4px; background: color-mix(in srgb, currentColor 10%, transparent); }}
.sk-bar > span {{ position:absolute; inset: 0 auto 0 0; border-radius: 4px; background: var(--c);
  transform-origin: left; animation: sk-grow 600ms var(--sk-ease) both; }}
.sk-rank {{ font-family: "Geist Mono", ui-monospace, monospace; font-size: .8rem;
  color: color-mix(in srgb, currentColor 45%, transparent); }}
.sk-dots {{ font-family: "Geist Mono", ui-monospace, monospace; letter-spacing: 1px; font-size: .78rem;
  color: color-mix(in srgb, currentColor 60%, transparent); white-space: nowrap; }}
.sk-conds {{ display:flex; flex-wrap: wrap; gap: 4px 12px; justify-content: flex-end; font-size: .8rem;
  font-weight: 500; white-space: nowrap; }}
.sk-stack {{ display:flex; height: 8px; border-radius: 8px; overflow: hidden; margin: 10px 0 8px;
  background: color-mix(in srgb, currentColor 6%, transparent); }}
.sk-stack > span {{ display:block; height: 100%; transform-origin: left; animation: sk-grow 700ms var(--sk-ease) both; }}
.sk-strip.five {{ grid-template-columns: repeat(5, minmax(0,1fr)); }}
/* ---------- editorial steps + glossary ---------- */
.sk-steps {{ display:grid; grid-template-columns: repeat(3, minmax(0,1fr));
  border-top: 1px solid color-mix(in srgb, currentColor 11%, transparent); }}
.sk-step {{ padding: 20px 22px 8px 0; }}
.sk-step + .sk-step {{ padding-left: 22px; border-left: 1px solid color-mix(in srgb, currentColor 11%, transparent); }}
.sk-step-n {{ font-family: "Geist Mono", ui-monospace, monospace; font-size: .75rem;
  color: color-mix(in srgb, currentColor 50%, transparent); }}
.sk-step-t {{ font-weight: 500; font-size: 1.05rem; letter-spacing: -0.02em; margin: 10px 0 6px; }}
.sk-step-b {{ font-size: .9rem; line-height: 1.55; color: color-mix(in srgb, currentColor 65%, transparent); }}
.sk-gloss {{ display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); column-gap: 40px; margin: 0; }}
.sk-gloss > div {{ padding: 14px 0; border-top: 1px solid color-mix(in srgb, currentColor 11%, transparent); }}
.sk-gloss dt {{ font-weight: 500; font-size: .95rem; }}
.sk-gloss dd {{ margin: 4px 0 0; font-size: .88rem; line-height: 1.55; color: color-mix(in srgb, currentColor 62%, transparent); }}
.sk-legend {{ display:flex; flex-wrap: wrap; gap: 6px 18px; font-size: .74rem; margin: 2px 0 8px;
  font-family: "Geist Mono", ui-monospace, monospace; color: color-mix(in srgb, currentColor 58%, transparent); }}
.sk-swatch {{ display:inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px;
  vertical-align: -1px; background: var(--c); box-shadow: inset 0 0 0 1px color-mix(in srgb, currentColor 22%, transparent); }}
/* ---------- Streamlit elements, quieted ---------- */
[data-testid="stExpander"] details {{ border: 0; border-top: 1px solid color-mix(in srgb, currentColor 11%, transparent);
  border-radius: 0; background: transparent; }}
[data-testid="stExpander"] summary {{ padding-left: 2px; padding-right: 2px; }}
[data-testid="stTabs"] [data-baseweb="tab-list"] {{ gap: 18px; }}
/* keyed sections (st.container(key="panel_...")): a hairline instead of a box */
[class*="st-key-panel"] {{ border-top: 1px solid color-mix(in srgb, currentColor 11%, transparent);
  padding-top: 22px; margin-top: 14px; }}
/* info notes in the neutral voice; warnings and errors keep their colour so problems stand out */
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) {{
  background: color-mix(in srgb, currentColor 4%, transparent) !important; color: inherit !important;
  border: 1px solid color-mix(in srgb, currentColor 11%, transparent); }}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) * {{ color: inherit !important; }}
[data-testid="stMetricValue"] {{ letter-spacing: -0.03em; }}
code {{ color: inherit !important; background: color-mix(in srgb, currentColor 7%, transparent) !important;
  font-family: "Geist Mono", ui-monospace, monospace; font-size: .85em; }}
/* ---------- motion ---------- */
@keyframes sk-rise {{ from {{ opacity: 0; transform: translateY(8px); }} to {{ opacity: 1; transform: none; }} }}
@keyframes sk-fade {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
@keyframes sk-grow {{ from {{ transform: scaleX(0); }} to {{ transform: scaleX(1); }} }}
.sk-rise {{ animation: sk-rise 460ms var(--sk-ease) both; }}
.sk-d1 {{ animation-delay: 60ms; }} .sk-d2 {{ animation-delay: 120ms; }} .sk-d3 {{ animation-delay: 180ms; }}
.sk-row {{ animation: sk-rise 420ms var(--sk-ease) both; }}
.sk-row:nth-child(2) {{ animation-delay: 40ms; }} .sk-row:nth-child(3) {{ animation-delay: 80ms; }}
.sk-row:nth-child(4) {{ animation-delay: 120ms; }} .sk-row:nth-child(5) {{ animation-delay: 160ms; }}
.sk-row:nth-child(6) {{ animation-delay: 200ms; }} .sk-row:nth-child(7) {{ animation-delay: 240ms; }}
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {{ animation: sk-fade 500ms ease both 150ms; }}
/* ---------- scroll motion (CSS scroll timelines; ~83% of browsers, the rest just show content)
   Patterns from Apple / Linear / Vercel: sections fade up as they enter, the hero drifts up and
   fades as it leaves, the top bar firms up once you scroll, and a hairline tracks progress. */
[data-testid="stMain"] {{ scroll-behavior: smooth; }}
@keyframes sk-enter {{ from {{ opacity: 0; transform: translateY(24px); }} to {{ opacity: 1; transform: none; }} }}
@keyframes sk-leave {{ to {{ opacity: .25; transform: translateY(-18px) scale(.985); }} }}
@keyframes sk-firm {{ to {{ background: var(--sk-bar-solid); border-bottom-color: color-mix(in srgb, currentColor 14%, transparent);
  box-shadow: 0 8px 24px -18px rgba(0,0,0,.45); }} }}
@keyframes sk-progress {{ from {{ transform: scaleX(0); }} to {{ transform: scaleX(1); }} }}
.st-key-topbar::after {{ content: ""; position: absolute; left: 0; right: 0; bottom: -1px; height: 1px;
  background: currentColor; opacity: .45; transform: scaleX(0); transform-origin: left; }}
@media (prefers-reduced-motion: no-preference) {{
  @supports (animation-timeline: view()) {{
    [data-testid="stMainBlockContainer"] [data-testid="stElementContainer"]:not(.st-key-topbar *),
    .sk-reveal, .sk-row {{
      animation: sk-enter linear both;
      animation-timeline: view();
      animation-range: entry 0% entry 32%;
      animation-delay: 0s;
    }}
    .sk-hero {{
      animation: sk-leave linear both;
      animation-timeline: view();
      animation-range: exit 10% exit 100%;
    }}
    .st-key-topbar {{
      animation: sk-firm linear both;
      animation-timeline: scroll(nearest block);
      animation-range: 0 140px;
    }}
    .st-key-topbar::after {{
      animation: sk-progress linear both;
      animation-timeline: scroll(nearest block);
    }}
  }}
}}
@media (prefers-reduced-motion: reduce) {{
  .sk-rise, .sk-row, .sk-reveal, [data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {{
    animation: sk-fade 200ms ease both !important; }}
  .sk-meter > span, .sk-bar > span, .sk-stack > span {{ animation: none; }}
  [data-testid="stMain"] {{ scroll-behavior: auto; }}
}}
/* ---------- small screens ---------- */
@media (max-width: 860px) {{
  /* the bar stacks into several rows on a phone; pinned, it would eat a quarter of the screen */
  [data-testid="stLayoutWrapper"]:has(> .st-key-topbar) {{ position: static; }}
  .st-key-topbar {{ backdrop-filter: none; -webkit-backdrop-filter: none; }}
  .st-key-topbar::after {{ display: none; }}
  .sk-hero {{ grid-template-columns: 1fr; gap: 10px; }}
  .sk-strip, .sk-strip.three {{ grid-template-columns: repeat(2, minmax(0,1fr)); }}
  .sk-stat:nth-child(3) {{ padding-left: 0; border-left: 0; }}
  .sk-stat:nth-child(n+3) {{ border-top: 1px solid color-mix(in srgb, currentColor 11%, transparent); }}
  .sk-steps {{ grid-template-columns: 1fr; }}
  .sk-step + .sk-step {{ padding-left: 0; border-left: 0; border-top: 1px solid color-mix(in srgb, currentColor 11%, transparent); }}
  .sk-gloss {{ grid-template-columns: 1fr; }}
  .sk-hide-sm {{ display: none; }}
}}
@media (max-width: 640px) {{
  /* list rows: a rigid grid would crush the name column; wrap instead (name first, details below) */
  .sk-row {{ display: flex; flex-wrap: wrap; align-items: center; gap: 8px 14px; }}
  .sk-row > * {{ flex: 0 0 auto; }}
  .sk-row > .sk-main {{ flex: 1 1 60%; min-width: 0; }}
  .sk-row > .sk-grow {{ flex: 1 1 38%; min-width: 90px; }}
  .sk-conds {{ justify-content: flex-start; }}
}}
</style>
"""


def inject(pal: dict) -> None:
    st.markdown(css(pal), unsafe_allow_html=True)
