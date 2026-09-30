# Learning Notes

How to use this file: read **Start here** first (it's the whole project in a few pages), then
the module entries below it when you want the details. Numbers are deliberately not copied here,
so they can't go stale: see [RESULTS.md](RESULTS.md) and [RESUME_BULLETS.md](RESUME_BULLETS.md).

---

# Start here

## The 2-minute explanation

> Astrophotographers check five or six forecast apps before driving out, because forecasts
> disagree and none tells you how often it's been wrong. The #1 complaint is the "false clear":
> it said clear, you set up, it clouded over. I built SkyTrust to measure that.
>
> First, a backtest: for every night since January 2024 at five California sites, I compared
> what five weather models forecast one to seven days ahead with what the sky actually did.
> "Actually did" is tricky: airport ceilometers can't see high cirrus, and reanalysis is a
> model, so I count an hour as clear only if both agree. I used archived forecasts at fixed lead
> times so the backtest can't peek at information a user wouldn't have had.
>
> Then a blend: a logistic regression per lead time that combines all the models, how much they
> disagree, and site and season. I trained on 2024-25, tuned with time-ordered cross-validation,
> and tested exactly once on 2026 with week-block bootstrap confidence intervals.
>
> The blend beats every single model at every lead time and cuts the night-before false-clear
> rate by a lot versus climatology. Honestly, it only beats a plain average of the models at some
> lead times, because most of the gain comes from averaging at all. The app shows tonight's
> probability *and* how reliable that number has been at your site, and the trust indicator drops
> for nights further out because I measured how skill decays.

## The data flow in 8 steps

1. **Sites** (`sites.py`): five airports; coordinates come from IEM metadata, never typed.
2. **Raw data** (`data/`): ASOS reports, ERA5 reanalysis, archived forecasts, fetched
   once through one HTTP client with retries and cached to disk.
3. **Astronomy** (`astro.py`): each night's astronomical dark window, in UTC.
4. **Rules** (`nightly.py`): clear hour, consecutive runs, usable night, missing-data rule.
5. **Labels + features** (`labels.py`, `features.py`): what happened vs what was forecast,
   per night; joined into `dataset.parquet` and validated (`dataset.py`).
6. **Baselines + blend** (`baselines.py`, `blend.py`): trained on 2024-25 only, and the
   blend exported to JSON.
7. **Evaluation** (`evaluate.py`, `report.py`): scored once on 2026 with CIs, written to
   `metrics.json`, RESULTS.md, and the README by code.
8. **Live** (`live.py`, `app/`): the same features on today's forecasts, the same JSON model,
   plus the measured track record.

## Questions you'll most likely get (and where the answer lives)

| Question | Short answer | Details |
|---|---|---|
| What is a "usable night" and why that definition? | ≥ 3 consecutive clear dark hours; imaging needs continuous time, not scattered clear hours | nightly.py |
| How do you know what the sky actually did? | ASOS + ERA5, clear only if both agree; each covers the other's blind spot | labels.py |
| What's look-ahead bias and how did you avoid it? | Using info you wouldn't have had; Previous Runs = forecasts actually issued d days ahead | SPEC 4.7, DECISIONS |
| Why a time-based split? | Tomorrow resembles today; random splits leak. CV folds are grouped by date too | modeling.py |
| How do you prove no test data reached training? | Overlap assertion + a test that spies on every `fit()` call | modeling.py, blend.py |
| Why Brier score / BSS rather than accuracy? | Scores the probability itself; BSS = improvement over the seasonal base rate | evaluate.py |
| What's a block bootstrap and why? | Resample whole weeks; nights are correlated, so single-night resampling overstates confidence | evaluate.py |
| Why logistic regression? | Small data, calibrated probabilities, explainable coefficients; calibration check passed | blend.py |
| The blend barely beats a simple average. Failure? | No: an honest result. Averaging does most of the work; learned weights add a little, mostly at lead 1 | blend.py, RESULTS |
| Why do forecasts look bad against ASOS alone? | Models forecast total cloud incl. cirrus; ASOS can't see cirrus | RESULTS §6 |
| What happens if the weather API is down? | Last good forecast with an "as of" banner; tested | live.py, app |
| How do you keep README numbers honest? | Generated from metrics.json by code, with the commit hash | report.py |

## Bugs we hit, and what each taught (good interview stories)

- **NaN handling changed in pandas 3.** `stack()` stopped dropping NaN, so an all-missing HRRR
  column failed a range check. *Lesson:* a test caught it before real data did; treat missing
  as missing, never as a value.
- **A permanently empty month re-downloaded forever.** ECMWF's archive starts in February 2024,
  so January never "finished" and was never cached. *Lesson:* know which gaps are permanent.
- **GFS reported −1% and 101% cloud.** Rare rounding artifacts. *Lesson:* validate at the
  boundary, and decide deliberately (clip within 1 point, reject anything worse).
- **Tuning drifted to the edge of the grid.** Flat validation curves plus noise. *Lesson:* treat
  near-ties as ties and prefer the simpler model.
- **Two lines in a figure shared a colour.** *Lesson:* test what the reader sees, not just
  that the code runs.
- **The app crashed only when the Moon was up at the chart's edge**, found by screenshotting the
  real app on a real night. *Lesson:* fixtures can't cover every real condition; run the real
  thing.
- **Every results file said "-dirty".** Training rewrote tracked model files right before
  evaluation recorded the commit. *Lesson:* define precisely what "the code that produced this"
  means.
- **The app took 10 s to start.** 3 s was importing a library it never used at run time.
  *Lesson:* measure before optimizing; the biggest cost is often something you don't need.

## Glossary (consistent with SPEC §18)

- **Lead time:** how far ahead a forecast was made. Lead 1 ≈ issued 24 h before the valid time.
- **Look-ahead bias:** using information that wouldn't have been available when the decision
  was made; it makes backtests look better than reality.
- **Brier score:** mean squared error of probability forecasts vs 0/1 outcomes (lower is better).
- **Brier Skill Score (BSS):** 1 − Brier / Brier(climatology). 0 = no better than climatology,
  1 = perfect, negative = worse.
- **Calibration / reliability:** when the model says 70%, does it happen ~70% of the time?
- **Log loss:** penalizes confident wrong predictions heavily.
- **Climatology:** the historical base rate for that site and month.
- **False-clear rate:** of the nights we said "go", the share that weren't usable.
- **Reanalysis (ERA5):** a best estimate of past weather from a model constrained by observations.
- **METAR / ASOS:** standardized automated airport weather reports; sky cover in oktas (eighths).
- **Block bootstrap:** resampling whole weeks so intervals respect night-to-night correlation.
- **TimeSeriesSplit:** cross-validation that always trains on the past and validates on the future.
- **Logistic regression:** a probability from a weighted sum of features; coefficients show reliance.

---

# Module notes

## Project setup (`pyproject.toml`, `Makefile`, CI)

**What:** `pyproject.toml` pins every dependency to an exact version; `uv sync` builds an identical
environment anywhere. `ruff` lints/formats, `pytest` runs offline tests, and GitHub Actions runs
both on every push.

**Why:** Reproducibility. If a number in RESULTS.md changes, it should be because the data or code
changed, not because scikit-learn silently upgraded. CI proves the tests pass on a clean machine,
not just on this laptop.

**Key concept:** *Tests must not hit the network by default.* Live APIs change, go down, and
rate-limit. We record real responses once (`tests/fixtures/`) and test the parsing against those.
`pytest -m network` is the opt-in for live contract checks.

**Interview questions:**
1. Why pin exact versions instead of `>=`?
2. Why are your tests offline? How do you know your code still works against the real API?
3. What does CI run, and what would a red build tell you?

---

## `skytrust/data/http.py`: the shared HTTP client

**What:** Every request in the project goes through `HttpClient.get()`. It adds a 30 s timeout and a
User-Agent, waits politely between calls, retries 429/5xx and connection errors up to 5 times with
exponential backoff + jitter, and raises exactly two error types: `SourceUnavailableError`
(couldn't get an answer) or `BadResponseError` (got an answer, but it's garbage).

**Why:** Free public APIs (IEM especially) rate-limit. During Phase 0 IEM returned HTTP 429 several
times in a row and the retry logic recovered each time. Two typed errors let callers decide:
"source down → use cached/last-good data" vs "bad payload → don't cache it, fail loudly".

**Key concept:** *Exponential backoff with jitter.* After failure *n*, wait a random time in
[0, base·2ⁿ]. Doubling gives the server room to recover; randomness stops many clients from
retrying at the exact same instant (a "thundering herd").

**Interview questions:**
1. Why retry a 503 but not a 400?
2. What is jitter and what problem does it solve?
3. How do you test retry logic without actually waiting? (Answer: inject `sleep` and a fake session.)

---

## `skytrust/data/cache.py`: disk cache

**What:** Saves each raw response to `data/raw/{source}/{site}/{model}/{start}_{end}.{ext}`. A
second run reads from disk, making zero network calls.

**Why:** The backfill is thousands of requests; re-running it on every code change would be slow
and rude to free services. Writes are atomic (write temp file, then rename) so a crash never leaves
a half-file that later *looks* cached. Payloads are validated *before* caching.

**Interview questions:**
1. What happens if the program crashes mid-download? Why doesn't that corrupt the cache?
2. How do you prove a complete cache makes zero network calls? (A test with a mock client asserts
   `get` is never called.)

---

## `skytrust/data/iem.py`: ASOS observations → hourly cloud cover

**What:** Downloads METAR reports, converts sky codes (CLR/FEW/SCT/BKN/OVC/VV) to cloud fractions
via okta midpoints, takes the max over layers per report, then the max over all reports in each
hour's window (H−30 min, H+30 min].

**Why it's built that way:**
- **Max over layers:** METAR amounts are cumulative, so the top layer's amount *is* total coverage.
- **Nearest report to the hour (not max):** the spec originally took the max over all reports in
  the window. But AUN reports 3×/hour and SAC 1×, so "max" made AUN look cloudier purely because it
  reports more. Using the report nearest H gives every station one reading per hour, which keeps the
  labels comparable. Ties go to the cloudier report. `max` is kept as a sensitivity option.
- **`report_type=3,4`:** IEM by default includes 5-minute readings; a max over 12 readings per
  hour would be much stricter than intended, and stricter at some stations than others.
- **The window trick:** "t is in (H−30, H+30]" rearranges to "H = ceil_to_hour(t − 30 min)". That
  gives each report exactly one hour with a single vectorised line instead of a loop.

**Key concept:** *Know your ground truth's blind spots.* ASOS ceilometers only see clouds below
12,000 ft, so "CLR" can still mean a sky full of cirrus. That's why the primary label also uses ERA5.
We also found FAT is human-augmented and *does* report high clouds, and AUN (an AWOS) reports 3×
per hour, so the ASOS label isn't equally strict everywhere.

**Interview questions:**
1. Why use the report nearest the top of the hour instead of the max over the hour? What's the trade-off?
2. What can't ASOS see, and how does your design compensate?
3. An obs at exactly H−30 min: which hour does it belong to, and why?
4. Why did you pass `report_type=3,4` to IEM?

---

## `skytrust/sites.py` + `validate-sites`

**What:** Looks up each candidate station in IEM's CA_ASOS metadata (coordinates and elevation
come from IEM, never hand-typed), downloads its history one year at a time, and measures the share
of night hours with a valid sky report. Stations ≥ 85 % go into `config/sites.yaml`.

**Why:** A backtest is only as good as its labels. A station with big gaps would silently drop
nights, and if gaps cluster in bad weather (outages during storms) they'd bias the base rate.

**Interview questions:**
1. Why not type the coordinates yourself?
2. Why might missing data be *non-random*, and why does that matter?

---

## Approximation to remember: what "lead d" means (SPEC 4.7)

`cloud_cover_previous_day1` at 03:00 UTC is the value predicted ~24 h before *03:00*, and at 06:00 it's
~24 h before *06:00*. So one night's lead-1 hours come from slightly different model runs. That's
close to "the forecast you'd check the day before", but not identical to looking at one forecast at
5 pm. Also: 3-hourly models (ECMWF 0.25°) are interpolated to hourly by Open-Meteo, so their
"consecutive clear hours" are partly an interpolation artifact.

---

## `skytrust/astro.py`: dusk, dawn, dark hours, the Moon

**What:** For each site and night, finds astronomical dusk and dawn (Sun centre 18° below the
horizon), lists the whole UTC hours in between ("dark hours"), and computes Moon illumination,
Moon altitude, and moonrise/moonset.

**Why it's built that way:**
- **Search by event, not by formula:** Skyfield's `find_discrete` samples "is the Sun below −18°?"
  hourly over the whole date range and then narrows each change down to the exact moment. One call
  gives every dusk and dawn for a year (~5 s per site-year).
- **Night key:** a dusk is assigned to the *local* date it happens on, so "night of Jan 10" =
  dusk on the evening of Jan 10 local, even though that's already Jan 11 in UTC.
- **Dark hours built in UTC:** UTC has no daylight saving, so every hour exists exactly once.
  On spring-forward night the local clock jumps from 01:00 to 03:00; in UTC the hours are still
  consecutive, so there are no duplicates or gaps (tested for 4 DST nights).
- **Offline ephemeris:** a committed 880 KB excerpt of JPL DE421 covering 2023–2030.
- **Verified independently:** dusk/dawn match the `astral` library (different algorithm) within
  2 minutes on both solstices.

**Key concept:** *Astronomical twilight.* Until the Sun is 18° below the horizon, scattered sunlight
still brightens the sky enough to wash out faint galaxies and nebulae. That's why the "dark window"
is shorter than sunset→sunrise: at SAC it's 5 hours in June and 11 in December.

**Interview questions:**
1. Why is a night keyed by the local evening date, and how do you handle it crossing midnight UTC?
2. How do you guarantee no duplicated/missing hours on daylight-saving nights?
3. How did you verify your dusk times are right? Why is a *different* library a better check?
4. What happens at a latitude with no astronomical darkness?

---

## `skytrust/data/openmeteo.py`: forecasts and ERA5

**What:** Downloads Previous Runs forecasts (per site × model, only the leads that model has) and
ERA5 reanalysis, month by month, through the shared HTTP client and disk cache. `parse_hourly`
turns any Open-Meteo response into a UTC-indexed DataFrame of cloud *fractions* (0–1).

**Why it's built that way:**
- **Convert percent → fraction in exactly one place,** so no other module has to remember which
  scale a number is on.
- **Handle both key styles:** multi-model responses name columns `cloud_cover_gfs_global`;
  single-model ones just `cloud_cover`. The parser accepts both.
- **Nulls are "missing", never 0 %.** A forecast that doesn't exist must not look like a clear sky.
  (A test caught a bug here: pandas 3 keeps NaN when stacking, which broke the range check on
  HRRR's all-null long leads.)
- **Don't cache what isn't finished:** if a chunk's last day has no data (ERA5's ~6-day lag), it's
  used but not cached, so a later run fetches the completed version.

**Interview questions:**
1. Why do you treat a null forecast differently from 0 % cloud?
2. How do you avoid caching a month that the provider hasn't finished publishing?
3. Why pass the station elevation to Open-Meteo?

---

## `skytrust/data/backfill.py` + `python -m skytrust fetch`

**What:** Works out which dates each source needs for a range of nights and runs the fetchers for
every site. It prints how many real network requests were made.

**Why:** *Idempotent* commands: running `fetch` twice gives the same result, and the second run
makes **0** requests (verified). If one site fails (outage/rate limit), the others continue and a
re-run resumes from the cache instead of starting over.

**Interview questions:**
1. What does "idempotent" mean and why does it matter for data pipelines?
2. How do you *prove* the second run didn't touch the network?

---

## `skytrust/nightly.py`: the night rules, written once

**What:** Turns an hourly cloud series into per-night numbers over each night's dark hours:
missing share, share of clear hours, longest run of consecutive clear hours, mean cover, and the
usable-night verdict. Labels (observed cloud), features (forecast cloud), and the live forecast
all call this same code.

**Why it's built that way:**
- **One definition, used everywhere.** If "usable" meant something slightly different for
  forecasts than for observations, every accuracy number would be quietly wrong.
- **"Don't know" ≠ "cloudy".** If > 25 % of a night's hours are missing, the result is NaN, not
  False. A missing hour inside a night still breaks a clear run (conservative).
- **Two implementations, cross-checked.** A plain loop (`is_usable`) you can read in 10 seconds,
  and a vectorised version for 35k rows. A property test runs both on 200 random nights with gaps
  and asserts they agree on every night.
- **The run-length trick:** mark each hour clear/not clear. Every non-clear hour starts a new
  "block" (a running count of non-clear hours). All clear hours sharing a block number are
  consecutive, so the biggest block size is the longest clear run.

**Interview questions:**
1. Why is a night with too much missing data NaN instead of "not usable"?
2. Explain how you find the longest consecutive clear run without a Python loop.
3. How do you know the fast version is correct? (Property test against the simple version.)

---

## `skytrust/labels.py`: three versions of the truth

**What:** For each night: `usable_asos` (airport ceilometer), `usable_era5` (reanalysis), and
`usable_primary` (per hour, the *cloudier* of the two). Plus exclusion reasons, how many hours
were filled from only one source, and ERA5 low/mid/high cloud means for analysis.

**Why:** Each source has a blind spot. ASOS can't see above 12,000 ft (cirrus); ERA5 is a
coarse model. Taking the per-hour max means an hour only counts as clear if *both* say so. The
data shows it matters: at 4 of 5 sites ~20 % of nights are "usable" to ASOS but not to ERA5, and
on those nights ERA5's cloud is mostly high cloud. That's the blind spot, measured, not assumed.
FAT is the interesting exception: its human observers *do* report cirrus, so there ASOS sometimes
catches cloud ERA5 misses.

**Key concept:** *Label noise and sensitivity analysis.* When ground truth is imperfect, you
don't pretend it's perfect. You pick a principled primary label and report every result under
the alternatives too, so a reader can see whether conclusions depend on the choice.

**Interview questions:**
1. Why not just use ASOS observations as the truth? Why not just ERA5?
2. What does taking the max of the two do to the base rate, and why is that the conservative choice?
3. What did you find at Fresno, and why?

---

## `skytrust/features.py`: what the forecasts said

**What:** For each night, lead (1–7 days), and model: forecast clear fraction, longest forecast
clear run (hours and as a share of the night), mean forecast cover, whether the forecast itself
meets the usable rule, and missing share. Across models: `spread_frac_clear` (how much they
disagree) and `n_models_available`.

**Why:** These are the inputs the blend will learn from. The model spread is there because
forecast *disagreement* is itself information: when models disagree, confidence should drop.
Models that don't forecast that far (HRRR beyond day 1, ICON at day 7) get NaN, never 0.

**Interview questions:**
1. Why include model spread as a feature?
2. Why is a missing forecast NaN rather than 0 % cloud?
3. How do you avoid look-ahead bias in these features? (Previous Runs lead-d values, not the
   freshest forecast.)

---

## `skytrust/dataset.py` + `report.py` (DATA_QUALITY.md)

**What:** Joins astronomy + labels + features into one row per (site, night, lead) → 34,895
rows, saved as `data/processed/dataset.parquet` (committed, ~1 MB). Before saving, it checks the
invariants: unique keys, fractions in [0, 1], dark hours 4–13, no naive timestamps, labels are
nullable booleans. It refuses to save if any fail. `DATA_QUALITY.md` is generated from the dataset.

**Why:**
- **Validation at the boundary:** a bug that produces, say, 1.2 as a cloud fraction is caught
  before it can poison the model.
- **Deterministic:** rebuilding from the same cache gives a byte-identical file (checked with MD5).
- **Keep problem rows, flag them:** nights with no forecasts or no ERA5 stay in the dataset with a
  reason, so gaps are visible instead of silently dropped.
- **The time split is a column:** `split` is set from config (train 2024–25, test 2026+), so
  every later step uses the same split.

**Interview questions:**
1. What checks run before the dataset is saved, and why there?
2. How would you prove your dataset build is reproducible?
3. What's the base rate of usable nights, and how does it vary by season and site? What does
   that mean for a "climatology" baseline?

---

## `skytrust/modeling.py`: leakage-safe training

**What:** The train/test split with an overlap check, cross-validation folds that respect time,
and the tuned logistic-regression pipeline (standardize → L2 logistic regression, optionally with
median imputation + missing-value indicators for the blend).

**Why it's built that way:**
- **Time-based split, enforced:** train = 2024–25, test = 2026. `split_train_test` raises if any
  training night is on/after the first test night. A test spies on every `fit()` call and proves
  no test row is ever passed in.
- **CV on dates, not rows:** folds are built from unique night dates, so one night's 5 sites
  never straddle a train/validation boundary.
- **Choosing C:** we try 13 values and pick the one with the best average validation log loss.
  When several are equally good (within 0.0001), we prefer the simplest (most regularized) model.

**Key concept:** *Leakage.* Any path by which information from the evaluation period influences
training makes results look better than reality. Random splits leak (tomorrow's weather resembles
today's); row-based folds leak (sites share weather); picking the "best" model on test leaks.

**Interview questions:**
1. Why not a random train/test split?
2. How do you *prove* no test data reached training?
3. What does C control in logistic regression, and how did you choose it?

---

## `skytrust/baselines.py`: what the blend has to beat

**What:** Five reference forecasts, all scored on the same test nights:
B1 climatology (training base rate for that site and month), B2 persistence (what happened d
nights ago), B3 each model's own forecast as a yes/no, B4 each model recalibrated by logistic
regression *with the same site/month/dark-hours context the blend gets* (so blend vs B4 isolates
the value of combining models), B5 the plain average of all models' forecast clear fraction.

**Why:** A number like "Brier 0.10" means nothing alone. Baselines answer: better than just
knowing the season? Better than the best single model? Better than naive averaging?
The "best single model" is picked using *training* data only.

**What we found (see RESULTS.md for numbers):** Persistence is worse than climatology (cloudy
spells don't last a predictable number of days). Every forecast model beats climatology by a wide
margin. And the naive equal-weight average beats every individually calibrated model at most
leads. That's the classic ensemble effect: different models make partly independent errors, and
averaging cancels some of them. That also sets a high bar for the learned blend.

**Interview questions:**
1. Why is climatology the reference for the skill score?
2. Why did persistence do so badly here?
3. Why can an *uncalibrated* average beat *calibrated* single models?

---

## `skytrust/evaluate.py`: metrics and honest uncertainty

**What:** For each method, label, lead, and subset (overall / site / season): Brier score, Brier
Skill Score, log loss, AUC, false-clear rate, miss rate, accuracy, each with a 95% CI; paired
differences between key methods; reliability tables.

**Key concepts:**
- **Brier score** = mean of (forecast probability − outcome)². Lower is better; always saying 50%
  scores 0.25. **BSS** = 1 − Brier/Brier(climatology): the share of climatology's error removed.
- **False-clear rate** = of the nights we said "go", how many weren't usable. The #1 complaint.
- **Block bootstrap:** resample whole calendar weeks (not single nights) because neighbouring
  nights share weather; resampling nights would make intervals look falsely narrow.
- **Paired comparison:** both methods are scored on the *same* resampled weeks, so the CI of the
  difference removes the week-to-week noise they share. That's why a small Brier difference can
  still be clearly significant.
- **Implementation trick:** each bootstrap replicate is a weight per row (how many times its week
  was drawn), so all 1,000 replicates are computed at once with a matrix product. The weighted AUC
  is vectorized too and tested against scikit-learn with sample weights.

**Interview questions:**
1. Why is Brier score a better headline than accuracy for probability forecasts?
2. What's a block bootstrap and why do you need it here?
3. Your CI for the difference is narrower than the CIs of each method. Why?

---

## `skytrust/report.py` (RESULTS.md) + `figures.py`

**What:** `skytrust report` turns metrics.json into RESULTS.md, three figures, and the README
headline block. Summary sentences are templates filled with numbers from metrics.json; the
ECMWF/ERA5 warning appears automatically when ECMWF comes out best under an ERA5-based label.

**Why:** The honesty rule: no number is ever typed by hand, so the write-up can't drift from the
code. The metrics carry the git commit that produced them (marked `-dirty` if the code was
uncommitted). Figures are byte-reproducible (no timestamps embedded).

**Interview questions:**
1. How do you guarantee the numbers in your README match your code?
2. What does the reliability diagram tell you that the Brier score doesn't?
3. Under the ASOS-only label the forecasts look bad. Is that a problem with the models or the label?

---

## `skytrust/blend.py`: the learned blend

**What:** For each lead (1–7 days), a logistic regression that takes every available model's
forecast (clear fraction, longest clear run, mean cover), how much the models disagree (spread),
and context (site, month, dark hours), and outputs P(usable night).

**Why it's built that way:**
- **Logistic regression, not a fancier model:** ~3,500 training rows, and we need calibrated
  probabilities plus coefficients we can explain. It was already well calibrated (out-of-fold
  calibration error ≤ 0.048), so the isotonic step was never triggered.
- **One model per lead:** skill and the best mix of models change with lead time (the blend leans
  much harder on ECMWF at 7 days than at 1 day), and HRRR/ICON don't exist at long leads.
- **Missing values:** median imputation + a "was missing" indicator, so the model can learn
  whether missingness itself means something, rather than silently pretending.
- **Exported to JSON, run with numpy:** the file stores feature order, medians, scaler means and
  scales, coefficients, the calibration decision, the training period, library versions, and the
  git commit. A test shows the numpy version matches scikit-learn to within 1e-9. The app
  therefore runs *exactly* the model that was evaluated, and there's no pickle (pickles can run
  arbitrary code and break across library versions).
- **Fair comparison:** B4 (single model) gets the same context features, so blend vs best single
  measures only the value of combining models.

**What we found (numbers in RESULTS.md):** the blend beats the best single model at every lead.
Against the simple average of the models it's only clearly better at 2 of 7 leads, so most of the
benefit comes from averaging several models at all, and learning the weights adds a little on
top (mostly at lead 1, where it also cuts the false-clear rate the most). Negative spread
coefficient = when models disagree, the blend is less confident the night will be usable.

**Interview questions:**
1. Why logistic regression instead of gradient boosting or a neural net?
2. How did you decide whether to calibrate? What would you have done if the check failed?
3. Why export to JSON instead of pickling, and how do you know the export is correct?
4. The blend barely beats a simple average at most leads. Is that a failure? (No: it's an honest
   result. It says the ensemble effect does most of the work, and it tells a user how much to trust
   a learned weighting. Knowing that is useful in itself.)
5. What does a negative coefficient on model spread mean?

---

## `skytrust/live.py`: tonight and the next 7 nights

**What:** Fetches the live forecast for all five models, finds the next 7 nights, gives each a
lead time, computes *exactly* the training features over each night's dark hours, and runs the
lead-matched blend. Adds a verdict (Go / Maybe / Skip), the best clear window, whether the models
agree, the backtest track record at that site and lead, a trust level, and the Moon.

**Why it's built that way:**
- **Same code as training:** features come from `nightly.summarize_nights` and the model from the
  JSON artifact. If live features were computed differently, the probability would be meaningless.
- **Pure core, thin shell:** `build_forecast` takes a payload and a time and returns a forecast,
  so tests can freeze time and replay a recorded API response.
- **Graceful failure:** a live failure falls back to the last good copy with an "as of" banner;
  a missing model is named rather than hidden.
- **Honest trust:** the outlook's trust level is the model's *measured* skill at that lead in the
  test year. Day 7 really is much less reliable than tonight, and the app says so.

**Interview questions:**
1. How do you decide which trained model (lead) to use for a given night?
2. What happens if Open-Meteo is down? If one model is missing?
3. Why is tonight's probability "slightly conservative"?

---

## `app/` (Streamlit) and `skytrust/inference.py`

**What:** Four pages: Tonight, 7-Night Outlook, Track Record (all numbers from metrics.json,
with a label toggle), Methodology (definitions pulled from config). Plotly charts, dark theme, a
red night-vision mode.

**Why:**
- **The app only reads.** Precomputed JSON models + metrics + the live forecast. It never trains,
  so it's fast and can't drift from the evaluated results.
- **Start-up time matters.** Profiling showed 3 of 10 seconds went to importing scikit-learn, which
  the app doesn't need. Splitting numpy-only inference into its own module halved cold start;
  a test keeps it that way.
- **Never crash:** each page is wrapped; API failure paths are covered by Streamlit AppTest tests.

**Interview questions:**
1. How did you find and fix the slow start-up? (Measure first: time each stage, then remove the
   biggest avoidable cost.)
2. How do you test a Streamlit app automatically?
3. Why does the app not use pickle or scikit-learn at run time?

---

## `skytrust/forward.py` + `.github/workflows/forward.yml`: the live forward test

**What:** Every afternoon a scheduled GitHub Action saves SkyTrust's forecasts for the next 7
nights at every site, plus three reference forecasts (NOAA's National Blend of Models, and the raw
ECMWF and GEFS ensembles). About 9 days later, when the observations exist, it scores them with
exactly the same label and metric code as the backtest, and the app shows the running record.

**Why it matters:** A backtest, however careful, is evaluated on data the developer has seen.
Every time you look at test results and then change something, the test set leaks a little
("the garden of forking paths"). A forward test records predictions *before* the outcome exists,
so it can't be tuned after the fact. It's the difference between "this would have worked" and
"this worked". It also measures the live "tonight" forecast directly, which the backtest could
only approximate with lead-1 data.

**Key concepts:**
- **Prospective vs retrospective evaluation.** Pre-registration in science works the same way.
- **Append-only log:** the first forecast logged for a night on a given day is the one that counts.
- **Raw ensemble probability:** if 40 of 51 ECMWF members predict a usable night, P = 78%. It's
  the natural "no statistics" probabilistic baseline for a learned blend.

**Interview questions:**
1. Your backtest already had a held-out test year. Why build a forward test?
2. How do you make sure the logged forecasts can't be changed later?
3. Why compare against NOAA's NBM and raw ensembles, and what would it mean if they won?

---

## NOAA's National Blend of Models (NBM) as the benchmark

**What:** NBM is NOAA's operational forecast made by statistically blending many models, the
professional version of SkyTrust's idea. It's now scored on the same test nights three ways:
its own clear/not-clear rule, its raw clear fraction used as a probability, and a calibrated
version with the same site/season context the blend gets (the fair comparison).

**Result (numbers in RESULTS.md):** SkyTrust's blend beats calibrated NBM at most lead times,
with paired confidence intervals excluding zero, and the report states where it doesn't. Giving
the blend NBM as an extra input didn't help, so the shipped model stays as it is.

**Why it matters:** "Better than the individual models" could just mean the models are weak.
Beating (or honestly matching) a national weather service's operational blend is a much harder,
more meaningful bar. Keeping NBM *out* of the shipped model preserved the integrity of the live
forward test, which must score one fixed model.

**Interview questions:**
1. Why is NBM a better benchmark than the individual models?
2. Why calibrate NBM before comparing? (Its raw output is a deterministic cloud forecast, not a
   probability; comparing a probability to it uncalibrated would flatter your model.)
3. NBM didn't help as an input. What does that suggest? (Its information is already contained in
   the models the blend sees; it's largely built from them.)

---

## Methods batch: where the skill comes from, and what it's worth

**B6, the calibrated equal-weight average.** Same model family as the blend but with *one*
shared weight for all forecast models. Comparing blend vs B6 isolates the value of learning a
weight per model; B6 vs B5 isolates calibration. The data said: combining models is most of the
value, and learned weights add little. That's the well-known *forecast combination puzzle*
(simple averages are hard to beat because estimated weights are noisy). Reporting it plainly is
more convincing than claiming a fancy model did the work.

**Brier decomposition.** Brier = reliability − resolution + uncertainty. Reliability: do the
probabilities mean what they say? Resolution: do they separate good nights from bad? Uncertainty:
how hard the period was (same for every method). A method can improve by being more honest or by
discriminating better; the decomposition shows which.

**Decision value (cost-loss model).** Setting up costs effort C; skipping a good night loses L.
With α = C/L, going out when P ≥ α is the best policy for a calibrated forecast. Relative value
= share of a perfect forecast's benefit (over the best fixed habit) that acting on the forecast
delivers. It turns "BSS 0.63" into "you'd get about half the benefit of a perfect forecast", and
it differs by user, which is why the app has a risk slider.

**Interview questions:**
1. Your blend barely beats a calibrated average. Why keep it, and why report that?
2. What does the Brier decomposition tell you that the Brier score doesn't?
3. Explain relative economic value to a non-statistician. Why does it depend on the user?

---

## Long-term climatology, robustness grid, walk-forward

**Long-term climatology.** A skill score is only as good as its reference. Two years of history
gives ~60 nights per site-month, which is a noisy baseline that's easy to beat. Labelling 20
years (2004–2023) with the same code gives a stable, standard reference (weather services use
30-year normals). The 20-year history download also surfaced a real data bug: a stray carriage
return inside one 2008 Truckee report, plus a snow depth written as `9"`, which looks like a CSV
quote. Real data is messy; the parser now handles it and a test pins it down.

**Robustness grid.** "Clear ≤ 20 %, 3 hours" are sensible but arbitrary. Re-running the entire
pipeline for 9 combinations shows whether the conclusions depend on that choice. If they only held
for one threshold, that would be a red flag (and a sign of cherry-picking).

**Walk-forward evaluation.** The standard in operational forecasting and quant finance: pretend
it's January 2025, train on the past, forecast the month; step forward a month and repeat. Every
prediction is out-of-sample, and the month-by-month scores reveal whether skill is stable or just
good on average.

**Interview questions:**
1. Why does the reference climatology matter for a skill score?
2. How do you show your conclusions aren't an artifact of your chosen thresholds?
3. What's the difference between a frozen-model test and a walk-forward evaluation, and why do both?
