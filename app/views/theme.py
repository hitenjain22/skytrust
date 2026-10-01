"""The "Nightfall" design system: tokens, palettes, and the global stylesheet.

Principles:

- A night sky, not a black screen: deep blue-black with a faint glow at the top and a sparse
  dust of stars behind the page; warm paper in light mode. One accent, starlight gold, for
  what is SkyTrust's own (the hero number, highlights, our series in charts).
- Colour only where it means something: status is a soft dot (mint / amber / coral), planets are
  gold, stars keep their real colours from their B-V index in the sky charts.
- Structure from hairlines and quiet cards, every tint derived from `currentColor` (borders 10%,
  surfaces 3-5%, secondary text 60%), so the same HTML is right in both themes.
- Type: Instrument Serif for headings and big numbers, Geist for reading, Geist Mono for small
  labels. Generous spacing, short lines (<= 68ch).
- Motion: transform/opacity only, strong ease-out, short staggers; reduced motion keeps fades.
"""

from __future__ import annotations

import streamlit as st

EASE = "cubic-bezier(0.23, 1, 0.32, 1)"
SERIF = '"Instrument Serif", Georgia, serif'
MONO_FONT = '"Geist Mono", ui-monospace, monospace'

# Data colours for charts and status. Plotly draws text and grids from Streamlit's theme; these
# mid-lightness values read on both the night and the paper background.
NIGHTFALL = {
    "paper": "rgba(0,0,0,0)",
    "text": None,  # let the Streamlit chart theme decide
    "muted": "#8A93A6",
    "grid": "rgba(138,147,166,0.16)",
    "border": "rgba(138,147,166,0.30)",
    "surface": None,
    "ink": "#8E97AA",
    "accent": "#E6BE74",  # starlight gold: SkyTrust's own
    "accent2": "#8A93A6",
    "Go": "#6FC3A0",
    "Maybe": "#E2B65F",
    "Skip": "#E2847A",
    "No data": "#8A93A6",
    "dark": "rgba(138,147,166,0.10)",
    "moon": "#C9CCD6",
    "planet": "#F0C987",
    "models": {
        "gfs": "#6E9BD1",
        "hrrr": "#D1A04A",
        "ecmwf": "#5FB394",
        "gem": "#9E86CF",
        "icon": "#D07F7F",
        "median": "#8A93A6",
    },
    "layers": {"low": "#7FA2C8", "mid": "#9C8CC4", "high": "#CFAE69"},
    "methods": {
        "blend": "#E6BE74",
        "equal_weight": "#7F93B3",
        "climatology": "#8A93A6",
        "nbm_lr": "#A58FCB",
        "equal_weight_cal": "#72B0B0",
    },
    "heat": [[0, "#1B2338"], [0.35, "#3B4664"], [0.7, "#8C95AD"], [1, "#E1E4EC"]],
}

# Older names kept for callers and tests.
MONO = MOON = DAY = NIGHTFALL


def palette() -> dict:
    return NIGHTFALL


def _line(pct: int = 10) -> str:
    return f"color-mix(in srgb, currentColor {pct}%, transparent)"


def css(pal: dict) -> str:
    status = "".join(f"--sk-{k.lower().replace(' ', '-')}:{pal[k]};" for k in
                     ["Go", "Maybe", "Skip", "No data", "accent", "planet"])  # fmt: skip
    hair = _line(10)
    return f"""
<style>
:root {{ {status} --sk-ease: {EASE}; --sk-hair: {hair}; }}
/* on paper the starlight gold and the status colours need more depth to stay readable */
@media (prefers-color-scheme: light) {{
  :root {{ --sk-accent: #94661C; --sk-go: #2E8A63; --sk-maybe: #A87412; --sk-skip: #B9564A; }}
}}
/* ---------- canvas: a night sky, not a black screen ---------- */
[data-testid="stHeader"], [data-testid="stDecoration"] {{ display: none; }}
@media (prefers-color-scheme: dark) {{
  .stApp {{
    background:
      radial-gradient(1100px 520px at 78% -8%, rgba(86,112,178,.20), transparent 62%),
      radial-gradient(900px 420px at -6% 4%, rgba(201,160,92,.07), transparent 60%),
      radial-gradient(1.2px 1.2px at 12% 18%, rgba(255,255,255,.55), transparent 60%),
      radial-gradient(1px 1px at 33% 9%, rgba(255,255,255,.45), transparent 60%),
      radial-gradient(1.4px 1.4px at 58% 22%, rgba(255,240,220,.5), transparent 60%),
      radial-gradient(1px 1px at 84% 31%, rgba(220,230,255,.45), transparent 60%),
      radial-gradient(1px 1px at 22% 46%, rgba(255,255,255,.3), transparent 60%),
      radial-gradient(1.2px 1.2px at 71% 54%, rgba(255,255,255,.3), transparent 60%),
      radial-gradient(1px 1px at 46% 73%, rgba(255,255,255,.25), transparent 60%),
      radial-gradient(1px 1px at 91% 81%, rgba(255,255,255,.25), transparent 60%),
      #0A0F1C;
    background-attachment: fixed;
  }}
}}
@media (prefers-color-scheme: light) {{
  .stApp {{ background: radial-gradient(1100px 520px at 78% -8%, rgba(176,140,74,.10), transparent 62%),
    #F6F4EE; background-attachment: fixed; }}
}}
[data-testid="stMainBlockContainer"], .block-container {{
  max-width: 1120px; padding-top: 1rem; padding-bottom: 4rem;
}}
h1, h2, h3 {{ font-family: {SERIF}; font-weight: 400 !important; letter-spacing: -0.01em; }}
h1 {{ line-height: 1.05; }}
h2, h3 {{ margin-top: .6rem; }}
/* ---------- small tokens ---------- */
.sk-muted {{ color: {_line(60)}; }}
.sk-mono {{ font-family: {MONO_FONT}; font-feature-settings: "tnum"; }}
.sk-eyebrow {{ font-family: {MONO_FONT}; font-size: .7rem; font-weight: 500; letter-spacing: .1em;
  text-transform: uppercase; color: {_line(56)}; }}
.sk-gold {{ color: var(--sk-accent); }}
.sk-section {{ margin: 34px 0 10px; }}
.sk-section h2 {{ margin: 6px 0 4px; font-size: 1.9rem; }}
.sk-section p {{ margin: 0; color: {_line(64)}; max-width: 68ch; line-height: 1.55; }}
/* ---------- sticky top bar ---------- */
[data-testid="stLayoutWrapper"]:has(> .st-key-topbar) {{ position: sticky; top: 0; z-index: 50; }}
.st-key-topbar {{
  position: relative; padding: 10px 0 8px; margin-bottom: 6px;
  backdrop-filter: saturate(1.4) blur(16px); -webkit-backdrop-filter: saturate(1.4) blur(16px);
  --sk-bar-solid: rgba(10,15,28,.94); background: rgba(10,15,28,.66);
  border-bottom: 1px solid {_line(7)};
}}
@media (prefers-color-scheme: light) {{
  .st-key-topbar {{ --sk-bar-solid: rgba(246,244,238,.96); background: rgba(246,244,238,.72); }}
}}
.sk-brand {{ display:flex; align-items:center; gap:10px; font-family: {SERIF}; font-size: 1.45rem;
  letter-spacing: -0.01em; white-space: nowrap; line-height: 1; }}
.sk-brand svg {{ color: var(--sk-accent); }}
.st-key-topbar [data-testid="stButtonGroup"] button {{
  border: 0 !important; background: transparent !important; border-radius: 999px !important;
  color: {_line(62)} !important; min-height: 38px; padding: 6px 14px !important; margin: 0;
  cursor: pointer; transition: color 150ms ease, background-color 200ms var(--sk-ease), transform 160ms var(--sk-ease);
}}
.st-key-topbar [data-testid="stButtonGroup"] button[aria-checked="true"] {{
  color: inherit !important; background: {_line(9)} !important;
}}
.st-key-topbar [data-testid="stButtonGroup"] button:active {{ transform: scale(0.97); }}
@media (hover: hover) and (pointer: fine) {{
  .st-key-topbar [data-testid="stButtonGroup"] button:hover {{ color: inherit !important; }}
}}
/* ---------- cards ---------- */
.sk-card {{ position: relative; border-radius: 18px; padding: 20px 22px;
  background: linear-gradient(180deg, {_line(5)}, {_line(2)});
  border: 1px solid {_line(9)};
  box-shadow: inset 0 1px 0 {_line(6)}; }}
.sk-card h4 {{ font-family: {SERIF}; font-weight: 400; font-size: 1.35rem; margin: 6px 0 4px; letter-spacing: 0; }}
.sk-card p {{ margin: 6px 0 0; font-size: .92rem; line-height: 1.5; color: {_line(72)}; }}
.sk-card .sk-meta {{ font-size: .8rem; color: {_line(55)}; margin-top: 10px; }}
.sk-grid {{ display: grid; gap: 14px; grid-template-columns: repeat(var(--n, 3), minmax(0, 1fr)); margin: 12px 0 8px; }}
/* ---------- hero ---------- */
.sk-hero {{ display:grid; grid-template-columns: minmax(0, 1.08fr) minmax(0, 0.92fr); gap: 34px;
  align-items: center; padding: 26px 30px; border-radius: 26px; margin-top: 6px;
  background: linear-gradient(160deg, {_line(6)}, {_line(1)} 60%);
  border: 1px solid {_line(9)}; box-shadow: inset 0 1px 0 {_line(7)}; overflow: hidden; }}
.sk-display {{ font-family: {SERIF}; font-size: clamp(4.6rem, 12vw, 7.4rem); line-height: .86;
  letter-spacing: -0.03em; color: var(--sk-accent); font-feature-settings: "tnum"; }}
.sk-display sup {{ font-size: .4em; vertical-align: 1.05em; margin-left: 4px; color: {_line(70)}; }}
.sk-caption {{ margin-top: 8px; font-size: .92rem; color: {_line(62)}; }}
.sk-headline {{ font-family: {SERIF}; font-size: clamp(1.7rem, 3.2vw, 2.3rem); line-height: 1.1;
  margin: 16px 0 8px; }}
.sk-lede {{ font-size: 1.0rem; line-height: 1.6; max-width: 60ch; color: {_line(76)}; }}
.sk-lede b {{ font-weight: 550; }}
.sk-dome {{ display:flex; flex-direction: column; align-items: center; gap: 8px; }}
.sk-dome svg {{ width: 100%; max-width: 380px; height: auto; display: block; }}
.sk-dome .sk-caption {{ text-align: center; margin: 0; font-size: .82rem; max-width: 40ch; }}
.sk-meter {{ height: 4px; border-radius: 4px; margin-top: 18px; background: {_line(10)}; overflow: hidden; max-width: 320px; }}
.sk-meter > span {{ display:block; height:100%; border-radius: 4px; background: var(--sk-accent);
  transform-origin: left; animation: sk-grow 800ms var(--sk-ease) both 120ms; }}
.sk-status {{ display:inline-flex; align-items:center; gap:8px; padding: 5px 12px 5px 10px; border-radius: 999px;
  font-size: .78rem; font-weight: 600; letter-spacing: .04em; border: 1px solid {_line(14)}; }}
.sk-dot {{ width: 7px; height: 7px; border-radius: 50%; background: var(--c); flex: none;
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--c) 22%, transparent); }}
.sk-note {{ display:flex; gap:10px; align-items:flex-start; margin-top: 14px; font-size: .9rem; line-height: 1.5;
  color: {_line(72)}; max-width: 62ch; }}
.sk-note svg {{ flex: none; margin-top: 2px; }}
/* ---------- fact strip ---------- */
.sk-strip {{ display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); margin: 16px 0 6px; gap: 0;
  border-top: 1px solid var(--sk-hair); border-bottom: 1px solid var(--sk-hair); }}
.sk-strip.three {{ grid-template-columns: repeat(3, minmax(0,1fr)); }}
.sk-strip.five {{ grid-template-columns: repeat(5, minmax(0,1fr)); }}
.sk-stat {{ padding: 16px 18px 16px 0; }}
.sk-stat + .sk-stat {{ padding-left: 18px; border-left: 1px solid var(--sk-hair); }}
.sk-stat-value {{ font-size: 1.2rem; font-weight: 500; letter-spacing: -0.015em; margin-top: 8px; line-height: 1.25;
  display:flex; align-items:center; gap: 10px; }}
.sk-stat-big {{ font-family: {SERIF}; font-size: clamp(2.4rem, 5vw, 3.2rem); line-height: 1; margin-top: 10px;
  font-feature-settings: "tnum"; }}
.sk-stat-sub {{ font-size: .83rem; line-height: 1.45; margin-top: 7px; color: {_line(58)}; }}
/* ---------- look-up cards (what to see) ---------- */
.sk-look {{ display:flex; flex-direction: column; gap: 4px; }}
.sk-look .sk-kind {{ display:flex; align-items:center; gap: 8px; }}
.sk-look .sk-where {{ font-size: 1.02rem; font-weight: 500; margin-top: 2px; }}
.sk-glyph {{ width: 26px; height: 26px; display:inline-grid; place-items:center; border-radius: 50%;
  background: {_line(7)}; color: var(--sk-accent); font-size: .9rem; flex: none; }}
.sk-badge {{ display:inline-flex; align-items:center; gap: 6px; font-size: .72rem; font-weight: 600;
  letter-spacing: .05em; text-transform: uppercase; padding: 3px 9px; border-radius: 999px;
  color: var(--c, inherit); background: color-mix(in srgb, var(--c, currentColor) 14%, transparent); }}
/* ---------- list rows ---------- */
.sk-list {{ border-top: 1px solid var(--sk-hair); margin: 8px 0 18px; }}
.sk-row {{ display:grid; grid-template-columns: var(--cols); gap: 16px; align-items:center; padding: 13px 8px;
  border-bottom: 1px solid var(--sk-hair); transition: background-color 150ms ease; border-radius: 6px; }}
@media (hover: hover) and (pointer: fine) {{ .sk-row:hover {{ background: {_line(4)}; }} }}
.sk-row-title {{ font-weight: 500; letter-spacing: -0.01em; }}
.sk-row-sub {{ font-size: .8rem; margin-top: 2px; color: {_line(56)}; }}
.sk-row-p {{ font-size: 1.15rem; font-weight: 450; text-align: right; font-feature-settings: "tnum"; letter-spacing: -0.02em; }}
.sk-bar {{ position: relative; height: 4px; border-radius: 4px; background: {_line(10)}; }}
.sk-bar > span {{ position:absolute; inset: 0 auto 0 0; border-radius: 4px; background: var(--c);
  transform-origin: left; animation: sk-grow 600ms var(--sk-ease) both; }}
.sk-rank {{ font-family: {MONO_FONT}; font-size: .78rem; color: {_line(45)}; }}
.sk-dots {{ font-family: {MONO_FONT}; letter-spacing: 1px; font-size: .78rem; color: {_line(60)}; white-space: nowrap; }}
.sk-conds {{ display:flex; flex-wrap: wrap; gap: 4px 12px; justify-content: flex-end; font-size: .8rem;
  font-weight: 500; white-space: nowrap; }}
.sk-stack {{ display:flex; height: 8px; border-radius: 8px; overflow: hidden; margin: 10px 0 8px; background: {_line(6)}; }}
.sk-stack > span {{ display:block; height: 100%; transform-origin: left; animation: sk-grow 700ms var(--sk-ease) both; }}
/* ---------- darkness scale (light pollution in plain words) ---------- */
.sk-scale {{ display:grid; grid-template-columns: repeat(5, minmax(0,1fr)); gap: 4px; margin: 10px 0 6px; }}
.sk-scale > div {{ height: 6px; border-radius: 6px; background: {_line(10)}; }}
.sk-scale > div.on {{ background: var(--sk-accent); }}
.sk-scale-labels {{ display:grid; grid-template-columns: repeat(5, minmax(0,1fr)); gap: 4px;
  font-size: .7rem; color: {_line(52)}; font-family: {MONO_FONT}; white-space: nowrap; }}
.sk-scale-labels span {{ grid-row: 1; }}
/* the sky chart stays in view while the list beside it scrolls */
@media (min-width: 901px) {{
  [data-testid="stColumn"]:has(.sk-chart) {{ position: sticky; top: 84px; align-self: flex-start; }}
}}
/* ---------- sky chart ---------- */
.sk-chart {{ display:flex; justify-content: center; }}
.sk-chart svg {{ width: 100%; max-width: 660px; height: auto; display:block; }}
.sk-chart-help {{ font-size: .8rem; color: {_line(56)}; text-align: center; margin-top: 6px; }}
/* ---------- events ---------- */
.sk-event {{ display:grid; grid-template-columns: 74px minmax(0,1fr); gap: 18px; padding: 18px 20px;
  border-radius: 18px; border: 1px solid {_line(9)}; background: linear-gradient(180deg, {_line(4)}, {_line(1)});
  margin: 10px 0; }}
.sk-date {{ text-align: center; border-right: 1px solid var(--sk-hair); padding-right: 14px; }}
.sk-date .d {{ font-family: {SERIF}; font-size: 2.3rem; line-height: 1; }}
.sk-date .m {{ font-family: {MONO_FONT}; font-size: .7rem; letter-spacing: .1em; text-transform: uppercase; color: {_line(58)}; margin-top: 4px; }}
.sk-date .w {{ font-size: .72rem; color: {_line(50)}; margin-top: 2px; }}
.sk-event h4 {{ font-family: {SERIF}; font-weight: 400; font-size: 1.45rem; margin: 2px 0 2px; letter-spacing: 0; }}
.sk-event p {{ margin: 6px 0 0; font-size: .93rem; line-height: 1.55; color: {_line(74)}; max-width: 70ch; }}
.sk-event .sk-when {{ font-size: .86rem; margin-top: 8px; color: {_line(64)}; }}
.sk-month {{ font-family: {SERIF}; font-size: 1.5rem; margin: 26px 0 2px; }}
.sk-places {{ display:flex; flex-wrap: wrap; gap: 6px 8px; margin-top: 10px; }}
.sk-chip {{ font-size: .78rem; padding: 4px 10px; border-radius: 999px; border: 1px solid {_line(12)}; white-space: nowrap; }}
.sk-chip b {{ font-weight: 600; }}
/* ---------- steps + glossary ---------- */
.sk-steps {{ display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); border-top: 1px solid var(--sk-hair); }}
.sk-step {{ padding: 20px 22px 8px 0; }}
.sk-step + .sk-step {{ padding-left: 22px; border-left: 1px solid var(--sk-hair); }}
.sk-step-n {{ font-family: {MONO_FONT}; font-size: .75rem; color: {_line(50)}; }}
.sk-step-t {{ font-weight: 500; font-size: 1.05rem; letter-spacing: -0.02em; margin: 10px 0 6px; }}
.sk-step-b {{ font-size: .9rem; line-height: 1.55; color: {_line(65)}; }}
.sk-gloss {{ display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); column-gap: 40px; margin: 0; }}
.sk-gloss > div {{ padding: 14px 0; border-top: 1px solid var(--sk-hair); }}
.sk-gloss dt {{ font-weight: 500; font-size: .95rem; }}
.sk-gloss dd {{ margin: 4px 0 0; font-size: .88rem; line-height: 1.55; color: {_line(62)}; }}
.sk-legend {{ display:flex; flex-wrap: wrap; gap: 6px 18px; font-size: .74rem; margin: 2px 0 8px;
  font-family: {MONO_FONT}; color: {_line(58)}; }}
.sk-swatch {{ display:inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 6px;
  vertical-align: -1px; background: var(--c); box-shadow: inset 0 0 0 1px {_line(22)}; }}
/* ---------- Streamlit elements, quieted ---------- */
[data-testid="stExpander"] details {{ border: 1px solid {_line(9)}; border-radius: 14px; background: {_line(2)}; }}
[data-testid="stExpander"] summary {{ padding-left: 14px; padding-right: 14px; }}
[data-testid="stTabs"] [data-baseweb="tab-list"] {{ gap: 18px; }}
[class*="st-key-panel"] {{ border-top: 1px solid var(--sk-hair); padding-top: 20px; margin-top: 14px; }}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) {{
  background: {_line(4)} !important; color: inherit !important; border: 1px solid {_line(10)}; }}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) * {{ color: inherit !important; }}
[data-testid="stMetricValue"] {{ letter-spacing: -0.03em; }}
code {{ color: inherit !important; background: {_line(7)} !important; font-family: {MONO_FONT}; font-size: .85em; }}
/* ---------- motion ---------- */
@keyframes sk-rise {{ from {{ opacity: 0; transform: translateY(8px); }} to {{ opacity: 1; transform: none; }} }}
@keyframes sk-fade {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
@keyframes sk-grow {{ from {{ transform: scaleX(0); }} to {{ transform: scaleX(1); }} }}
@keyframes sk-twinkle {{ 0%, 100% {{ opacity: var(--o, .9); }} 50% {{ opacity: calc(var(--o, .9) * .55); }} }}
.sk-rise {{ animation: sk-rise 480ms var(--sk-ease) both; }}
.sk-d1 {{ animation-delay: 60ms; }} .sk-d2 {{ animation-delay: 120ms; }} .sk-d3 {{ animation-delay: 180ms; }}
.sk-row, .sk-event, .sk-card {{ animation: sk-rise 440ms var(--sk-ease) both; }}
.sk-grid > :nth-child(2) {{ animation-delay: 50ms; }} .sk-grid > :nth-child(3) {{ animation-delay: 100ms; }}
.sk-grid > :nth-child(4) {{ animation-delay: 150ms; }}
.sk-tw {{ animation: sk-twinkle 4.5s ease-in-out infinite; }}
.sk-tw:nth-of-type(3n) {{ animation-duration: 6s; animation-delay: 1.2s; }}
.sk-tw:nth-of-type(5n) {{ animation-duration: 3.6s; animation-delay: .6s; }}
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {{ animation: sk-fade 500ms ease both 150ms; }}
[data-testid="stMain"] {{ scroll-behavior: smooth; }}
@keyframes sk-enter {{ from {{ opacity: 0; transform: translateY(22px); }} to {{ opacity: 1; transform: none; }} }}
@keyframes sk-firm {{ to {{ background: var(--sk-bar-solid); border-bottom-color: {_line(13)};
  box-shadow: 0 10px 28px -20px rgba(0,0,0,.55); }} }}
@media (prefers-reduced-motion: no-preference) {{
  @supports (animation-timeline: view()) {{
    [data-testid="stMainBlockContainer"] [data-testid="stElementContainer"]:not(.st-key-topbar *),
    .sk-row, .sk-event {{
      animation: sk-enter linear both; animation-timeline: view();
      animation-range: entry 0% entry 30%; animation-delay: 0s;
    }}
    .st-key-topbar {{ animation: sk-firm linear both; animation-timeline: scroll(nearest block); animation-range: 0 140px; }}
  }}
}}
@media (prefers-reduced-motion: reduce) {{
  .sk-rise, .sk-row, .sk-event, .sk-card, [data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {{
    animation: sk-fade 200ms ease both !important; }}
  .sk-tw {{ animation: none; }}
  .sk-meter > span, .sk-bar > span, .sk-stack > span {{ animation: none; }}
  [data-testid="stMain"] {{ scroll-behavior: auto; }}
}}
/* ---------- small screens ---------- */
@media (max-width: 900px) {{
  [data-testid="stLayoutWrapper"]:has(> .st-key-topbar) {{ position: static; }}
  .st-key-topbar {{ backdrop-filter: none; -webkit-backdrop-filter: none; }}
  .sk-hero {{ grid-template-columns: 1fr; gap: 18px; padding: 22px 20px; }}
  .sk-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  .sk-strip, .sk-strip.three, .sk-strip.five {{ grid-template-columns: repeat(2, minmax(0,1fr)); }}
  .sk-stat:nth-child(odd) {{ padding-left: 0; border-left: 0; }}
  .sk-stat:nth-child(n+3) {{ border-top: 1px solid var(--sk-hair); }}
  .sk-steps {{ grid-template-columns: 1fr; }}
  .sk-step + .sk-step {{ padding-left: 0; border-left: 0; border-top: 1px solid var(--sk-hair); }}
  .sk-gloss {{ grid-template-columns: 1fr; }}
  .sk-hide-sm {{ display: none; }}
}}
@media (max-width: 640px) {{
  .sk-grid {{ grid-template-columns: 1fr; }}
  .sk-event {{ grid-template-columns: 56px minmax(0,1fr); gap: 12px; padding: 16px; }}
  .sk-date .d {{ font-size: 1.8rem; }}
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
