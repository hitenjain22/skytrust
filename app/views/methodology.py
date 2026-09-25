"""Page 4: plain-English methodology, definitions (from config), caveats, attribution."""

from __future__ import annotations

import streamlit as st

from skytrust import report
from views.common import Context


def render(ctx: Context) -> None:
    d = ctx.settings.raw["definitions"]
    split = ctx.settings.raw["split"]
    models = ", ".join(f"{m.name}" for m in ctx.settings.models)
    st.header("Methodology")
    st.markdown(f"""
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
onward**, used once, after every choice was made. Tuning uses time-ordered cross-validation inside
the training years. A random split would leak (tomorrow's weather resembles today's).

### How to read the numbers
- **Brier score:** average squared error of the probability (lower is better).
- **Brier Skill Score:** improvement over the seasonal base rate (0 = none, 1 = perfect).
- **False-clear rate:** of the nights it said "go" (P ≥ {ctx.settings.raw["decision_threshold"]}),
  the share that weren't usable.
- **95% intervals** resample whole calendar weeks, because neighbouring nights share weather.
- **Verdicts:** Go ≥ {ctx.settings.raw["verdict"]["go"]:.0%}, Maybe ≥
  {ctx.settings.raw["verdict"]["maybe"]:.0%}, otherwise Skip.
""")
    st.markdown("### Caveats")
    caveats = list(report.CAVEATS)
    if ctx.metrics:
        caveats = report.data_caveats(report.MetricsView(ctx.metrics)) + caveats
    for c in caveats:
        st.markdown(f"- {c}")
    st.markdown("""
### Data and credits
Weather data by [Open-Meteo.com](https://open-meteo.com/), licensed CC BY 4.0. ASOS
observations courtesy of the [Iowa Environmental Mesonet](https://mesonet.agron.iastate.edu/),
Iowa State University. Astronomy by [Skyfield](https://rhodesmill.org/skyfield/) with JPL's
DE421 ephemeris.
""")
