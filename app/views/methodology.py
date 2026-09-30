"""Page 4: how it works. A 30-second version and a glossary for beginners, then the full
methodology (definitions from config), caveats and attribution."""

from __future__ import annotations

import streamlit as st

from skytrust import report
from views import components as ui
from views.common import Context

GLOSSARY = {
    "Astronomical darkness": "When the Sun is more than 18° below the horizon, the last "
    "twilight glow is gone. Faint targets need this.",
    "Usable night": "At least three clear dark hours in a row: roughly what an imaging session "
    "needs.",
    "False clear": "The forecast said clear, you drove out, and it clouded over. The mistake "
    "that costs astrophotographers the most, so SkyTrust counts it.",
    "Probability": "70% means that on nights like this, about 7 in 10 turned out usable in "
    "testing. SkyTrust checks that these percentages can be taken at face value.",
    "Weather model": "A computer simulation of the atmosphere (GFS, ECMWF, ...). Different models "
    "often disagree about clouds, which is why SkyTrust combines five.",
    "Cirrus": "Thin, high ice cloud. Often invisible to the eye at night but it dims stars and "
    "ruins long exposures.",
    "Moon phase": "How much of the Moon is lit. Near full Moon the sky is too bright for faint "
    "galaxies and nebulae; the Moon, planets and clusters are fine.",
    "Lead time": "How many days ahead the forecast is. Accuracy drops the further ahead you look.",
    "Skill score": "How much better than simply guessing the seasonal average (0 = no better, "
    "1 = perfect).",
}


def thirty_seconds(ctx: Context) -> str:
    n_models = len(ctx.settings.models)
    split = ctx.settings.raw["split"]
    steps = [
        ("Ask several forecasts", f"SkyTrust reads {n_models} major weather models for your spot "
         "(refreshed hourly): how cloudy each thinks every dark hour will be."),
        ("Blend them the smart way", "A statistical model, trained on two years of what the "
         "models said versus what the sky actually did, weighs them into one honest probability."),
        ("Grade itself in public", f"It was tested on {split['test_start']:%B}–"
         f"{split['test_end']:%B %Y}, nights it never saw, and keeps logging every forecast "
         "before the night to score later."),
    ]  # fmt: skip
    cells = "".join(
        f'<div class="sk-step"><div class="sk-step-n">0{i}</div>'
        f'<div class="sk-step-t">{ui.esc(t)}</div><div class="sk-step-b">{ui.esc(b)}</div></div>'
        for i, (t, b) in enumerate(steps, 1)
    )
    return f'<div class="sk-steps sk-rise">{cells}</div>'


def sections(text: str) -> None:
    """First section in full; every later "### " section folded into an expander, so the page
    stays scannable while keeping all the detail."""
    first, *rest = text.strip().split("\n### ")
    st.markdown(first.replace("### ", "#### ", 1))  # sits under "The full methodology"
    for part in rest:
        title, _, body = part.partition("\n")
        with st.expander(title.strip(), icon=":material/chevron_right:"):
            st.markdown(body)


def render(ctx: Context) -> None:
    d = ctx.settings.raw["definitions"]
    split = ctx.settings.raw["split"]
    clim = ctx.settings.raw["climatology"]
    forward_start = ctx.settings.raw["forward"]["start_date"]
    models = ", ".join(f"{m.name}" for m in ctx.settings.models)
    st.header("How SkyTrust works")
    st.markdown(thirty_seconds(ctx), unsafe_allow_html=True)
    st.subheader("Glossary")
    terms = "".join(
        f"<div><dt>{ui.esc(t)}</dt><dd>{ui.esc(m)}</dd></div>" for t, m in GLOSSARY.items()
    )
    st.markdown(f'<dl class="sk-gloss sk-reveal">{terms}</dl>', unsafe_allow_html=True)
    st.subheader("The full methodology")
    text = f"""
### The problem
Astrophotographers routinely check five or six forecasts before driving out, and the most common
complaint is the **false clear**: the forecast said clear, you set up, and it clouded over. No app
tells you how often it has been wrong *at your location*. SkyTrust measures that.

### Definitions
- **Night:** keyed by the local date of the evening it starts.
- **Dark hours:** whole hours between astronomical dusk and dawn (Sun more than
  {abs(d["sun_altitude_deg"]):.0f}° below the horizon).
- **Clear hour:** cloud cover ≤ {d["clear_threshold"]:.0%}.
- **Usable night:** at least **{d["min_run_hours"]} consecutive** clear dark hours; a missing hour
  breaks the run. Nights with more than {d["max_missing_frac"]:.0%} of hours missing are excluded,
  never counted as cloudy.

### What counts as "what actually happened"
- **ASOS** airport observations (Iowa Environmental Mesonet): real, but the ceilometer only sees
  cloud below 12,000 ft, so it misses cirrus.
- **ERA5** reanalysis (via Open-Meteo): sees every layer, but it's a model on a ~28 km grid.
- **Primary label:** each hour counts as clear only if **both** say clear. Results are also shown
  for each source alone.
- **GOES-18 satellite** cloud mask (NOAA): a real observation that also sees high cloud, used as
  an independent check on the other two and on the model (Track Record → Satellite).

### The forecasts and the blend
Five models ({models}) via Open-Meteo. For each night and lead time we compute each model's
clear fraction, longest clear run, and mean cover, plus how much the models disagree. A
**logistic regression per lead time** combines them with site, month, and night length into one
calibrated probability.

### Avoiding look-ahead bias
The backtest uses Open-Meteo's **Previous Runs** archive: the value each model predicted about
24·d hours before each hour. The "historical forecast" archive stitches together the freshest runs,
which would quietly let the backtest see the future and overstate how well you can plan ahead.

### A fair test
Training data: **{split["train_start"]} → {split["train_end"]}**. Test data: **{split["test_start"]}
→ {split["test_end"]}**, used once, after every choice was made. Tuning uses time-ordered
cross-validation inside the training years. A random split would leak (tomorrow's weather resembles
today's). Since {forward_start}, every live forecast is also logged before the night and scored
afterwards (the forward test on the Track Record page).

### How to read the numbers
- **Brier score:** average squared error of the probability (lower is better).
- **Brier Skill Score:** improvement over the seasonal base rate, measured over
  {clim["start"].year}–{clim["end"].year} (0 = none, 1 = perfect).
- **False-clear rate:** of the nights it said "go" (P ≥ {ctx.settings.raw["decision_threshold"]}),
  the share that weren't usable.
- **95% intervals** resample whole calendar weeks, because neighbouring nights share weather.
- **Verdicts:** Go ≥ {ctx.settings.raw["verdict"]["go"]:.0%}, Maybe ≥
  {ctx.settings.raw["verdict"]["maybe"]:.0%}, otherwise Skip.
"""
    sections(text)
    caveats = list(report.CAVEATS)
    if ctx.metrics:
        caveats = report.data_caveats(report.MetricsView(ctx.metrics)) + caveats
    with st.expander("Caveats: what this can't tell you", icon=":material/info:"):
        for c in caveats:
            st.markdown(f"- {c}")
    st.markdown("""
#### Data and credits
Weather data by [Open-Meteo.com](https://open-meteo.com/), licensed CC BY 4.0. ASOS
observations courtesy of the [Iowa Environmental Mesonet](https://mesonet.agron.iastate.edu/),
Iowa State University. Satellite cloud mask and imagery: NOAA GOES-18 (public AWS bucket
`noaa-goes18`; NOAA/NESDIS STAR). Astronomy by [Skyfield](https://rhodesmill.org/skyfield/)
with JPL's DE421 ephemeris.
""")
