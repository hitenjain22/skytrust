# Decision Log

Format: date · decision · alternatives considered · why. Newest at the bottom.

---

### 2026-09-24 · Use the Previous Runs API, not the Historical Forecast archive, for backtest forecasts
- **Alternatives:** Open-Meteo Historical Forecast API (longer, denser archive).
- **Why:** The historical-forecast archive stitches together the *freshest* run for each hour, so its
  values were effectively forecast minutes-to-hours before they happened. Backtesting on that would
  overstate how well you can plan a night 1–7 days ahead. That's **look-ahead bias**. Previous Runs
  gives the value predicted ~24·d hours before, which is what a user would actually have seen.
  (Set by SPEC 4.7; recorded here because it's a core design decision.)

### 2026-09-24 · Python 3.12 via `uv`
- **Alternatives:** system Python 3.9 (too old for spec's 3.11+), Homebrew/pyenv (not installed).
- **Why:** `uv` installs a user-local Python and manages the venv + lockfile with one tool, with no
  admin rights needed, and it's the same on a fresh machine and in CI. Current numpy (2.5.3)
  requires Python ≥ 3.12, so `requires-python = ">=3.12"` (still satisfies spec's "3.11+").

### 2026-09-24 · GFS = `gfs_global`, not `gfs_seamless` (proposed, pending approval)
- **Alternatives:** `gfs_seamless`.
- **Why:** Probing showed `gfs_seamless` lead 1 is *identical* to HRRR (100 % of hours). Including
  both would feed the blend the same forecast twice and misattribute HRRR's skill to "GFS".

### 2026-09-24 · ECMWF = `ecmwf_ifs025`; drop NAM and ECMWF 9 km (proposed, pending approval)
- **Alternatives:** `ecmwf_ifs` (9 km HRES), `ncep_nam_conus`.
- **Why:** Neither has data in the training period (2024–2025) at usable fill: NAM starts 2025-09
  (lead 1 only, 74 % fill), ECMWF 9 km has ~57 % fill from 2025-01. SPEC 6.1 says to fall back to
  0.25° if HRES isn't available. GEM and ICON added as the optional models (cheap, full coverage),
  giving 5 models so we're safely above the "≥ 3 models" success criterion.

### 2026-09-24 · ERA5 requested with explicit `models=era5`
- **Alternatives:** the archive API default ("best_match").
- **Why:** best_match mixes in IFS HRES, a forecast model, which would make the label even more
  ECMWF-flavoured and break "ERA5 = reanalysis" as documented.

### 2026-09-24 · IEM `report_type=3,4` (routine + SPECI only)
- **Alternatives:** IEM default (includes 5-minute MADIS HFMETAR rows).
- **Why:** SPEC 4.6(a) takes the max over all obs in each hour. With 5-minute data that's a max
  over ~12 readings, far more conservative than the "routine + SPECI" the spec describes, and
  inconsistent between stations that do/don't have high-frequency feeds.

### 2026-09-24 · Phase 0 site validation uses proxy night hours (local 23:00–03:00)
- **Alternatives:** implement Skyfield dark windows now (Phase 1 scope).
- **Why:** Keeps Phase 0 in scope. 23:00–03:00 local is inside astronomical darkness year-round
  at 36–40°N (June solstice at TRK: dusk ≈ 22:35 PDT, dawn ≈ 03:50 PDT). Coverage will be
  re-checked with true dark windows in Phase 2's data-quality report.

### 2026-09-24 · Phase 0 implements http/cache/iem ahead of Phase 1
- **Why:** `validate-sites` needs to download a full ASOS history per station anyway; writing it
  through the real client + cache means that data is already cached for Phase 1 and the
  window-assignment logic is tested once, not duplicated in a throwaway script.

### 2026-09-25 · Phase 0 checkpoint approvals
- **Models approved:** gfs_global, ncep_hrrr_conus, ecmwf_ifs025, gem_seamless, icon_seamless.
  NAM and ECMWF 9 km dropped (no usable training-period data).
- **Sites approved:** SAC, FAT, AUN, TRK, BIH (all pass; no substitutions).

### 2026-09-25 · ASOS hourly value = report nearest the top of the hour (changes SPEC 4.6a; approved by Hiten)
- **Alternatives:** (A) SPEC's original max over all reports in (H−30, H+30]; (B) nearest report.
- **Evidence:** reports per hour window differ by station (AUN 3, TRK 2, SAC/FAT/BIH 1), and TRK's
  schedule changed during the archive. Under (A) a station looks cloudier just because it reports
  more often, so labels aren't comparable across sites or stable over time.
- **Decision:** (B). Nearest valid report to H; ties go to the cloudier report (keeps a conservative
  lean); reports with no parseable sky layer are skipped. Window is unchanged. (A) stays implemented
  as `asos_hour_aggregation: max` for a sensitivity check in RESULTS.
- **Trade-off:** (B) can miss a brief cloud SPECI 20–30 min off the hour. Accepted: ERA5 in the
  primary label is the backstop, and consistency across sites matters more for a backtest.

### 2026-09-25 · Commit a 2023–2030 excerpt of DE421 (880 KB) instead of downloading at runtime
- **Alternatives:** let Skyfield download the full 17 MB `de421.bsp` on first use; commit the full file.
- **Why:** Tests must be offline (SPEC 12) and the deployed app must not depend on JPL's server.
  `python -m jplephem excerpt 2023/1/1 2031/1/1` keeps only the date range we need. Covers the
  whole backtest and the live app until 2030. The timescale uses Skyfield's builtin tables for
  the same reason.

### 2026-09-25 · Open-Meteo cached in calendar-month chunks; incomplete chunks are not cached
- **Alternatives:** one request for the full history (works, see DATA_NOTES); yearly chunks.
- **Why:** Monthly files are ~2 weighted API calls each, and a daily update only re-downloads the
  current month. A chunk is cached only if its last day has data, so an ERA5 month fetched
  before it's fully published is never frozen into the cache with holes. The ERA5 fetch end is capped at
  today − 8 days (measured lag ~6) so normal runs always produce complete, cacheable chunks.

### 2026-09-25 · `fetch --end` is a *night* date; sources are fetched through end + 1 day
- **Why:** A night's dark hours run past midnight into the next UTC day, so labelling night N
  needs data from day N+1. Default last night = today − 2 days, so every hour is in the past.

### 2026-09-25 · Clip GFS −1 / 101 % values to 0 / 100
- **Alternatives:** treat as missing (NaN); reject the whole chunk; leave as is.
- **Why:** They're off by exactly one point, rare (289 of 3.7 M values), and physically mean
  "clear" / "overcast". NaN would punch artificial holes in consecutive-hour runs; leaving them
  would break the 0–1 validation. The tolerance is 1 point so real corruption still fails loudly.

### 2026-09-25 · Night rules implemented once (`nightly.py`), shared by labels, features, and live
- **Alternatives:** separate implementations in labels.py and features.py.
- **Why:** "clear", "consecutive", "usable", and the 25 % missing rule must mean exactly the same
  thing for observations and forecasts, or the comparison is unfair. A simple loop version
  (`is_usable`) and a vectorised version (`summarize_nights`) are cross-checked by a property test
  on 200 random nights with gaps.
- Small interpretation choices (SPEC doesn't pin these down; none changes a §4 definition):
  - `frac_clear` = clear hours / *available* dark hours (missing hours aren't counted as cloudy).
    `longest_clear_run_frac` = run / *all* dark hours, as SPEC 4.7 writes it.
  - A 1e-9 tolerance on the threshold so 20 % / 100 counts as clear despite float rounding.
  - The 25 % missing rule is also applied to the ASOS-only and ERA5-only labels, each on its own
    source, so all three labels use the same exclusion logic.
  - `spread_frac_clear` = population std (ddof = 0) across available models; NaN if < 2 models.

### 2026-09-25 · Dataset keeps pre-archive and ERA5-lag nights (flagged), rather than dropping them
- **Why:** Dropping rows silently hides coverage problems. Nights before 2024-01-19 have
  `n_models_available = 0`; the last ~7 nights have `exclusion_primary = era5_missing`. Modeling
  code filters them explicitly; DATA_QUALITY.md reports them. The dataset's last night is set by
  ASOS + forecasts (not ERA5), so ASOS-only labels still cover the most recent nights.

### 2026-09-25 · Store ERA5 low/mid/high means per night (analysis only, not model features)
- **Why:** SPEC 4.6b asks for them; they test the cirrus hypothesis directly. Result (DATA_QUALITY
  §7): on nights ASOS calls usable but ERA5 doesn't, ERA5's cloud is mostly high cloud (52–67 %)
  with little low cloud (6–22 %). This is the ASOS 12,000 ft blind spot, measured.

### 2026-09-25 · Site coverage re-checked on true dark windows (closes the Phase 0 proxy decision)
- ASOS is missing 0.0–1.3 % of astronomical-dark hours per site; 0–18 nights per site exceed 25 %.
  The Phase 0 proxy (≥ 98.6 %) was consistent with this.

### 2026-09-25 · One common evaluation set per (label, lead)
- **Alternatives:** score each method on whatever rows it can predict.
- **Why:** Comparisons (and the paired bootstrap) are only fair on identical rows. Test rows are
  kept if the label is known, *every* model that forecasts that lead has features, and the
  persistence outcome is known (≈ 1,200–1,280 site-nights per lead; RESULTS shows exact n).

### 2026-09-25 · "Best single model" is chosen by training CV log loss, never by test results
- **Why:** Picking the model that happened to score best on test and then comparing the blend
  against it would be selection on the test set, which makes the comparison look harder or easier
  than it really is. The choice is made inside training data, exactly like the blend's tuning.

### 2026-09-25 · CV folds split on unique night dates (expanding window, 4 folds)
- **Why:** With 5 sites per night, a row-based TimeSeriesSplit could put SAC's Jan 3 in
  training and FAT's Jan 3 in validation. Same-night weather is strongly correlated across sites,
  so that leaks. Folds are built on dates, then mapped to rows (tested).

### 2026-09-25 · C chosen with a 1e-4 log-loss tie tolerance, preferring stronger regularization
- **Evidence:** CV curves for the 3-feature single-model regressions are flat above C ≈ 1 to the
  5th decimal; without a tolerance, noise pushed choices to the grid edge (C = 1000) with a warning.
- **Decision:** Among C values within 1e-4 of the best CV loss, take the smallest (simplest model).
  Grid widened to 10^-3 … 10^3. A grid-edge warning still fires if the true optimum is at an edge.

### 2026-09-25 · Metric conventions
- Log loss uses probabilities clipped to [0.001, 0.999] (a 0/1 forecast would otherwise be
  infinite). Hard yes/no predictors (persistence, rules) get no log loss or AUC; per SPEC, the
  B3 rules get confusion-matrix metrics only. Persistence keeps Brier/BSS as the classic reference.
- False-clear / miss rates at P ≥ 0.5 (config `decision_threshold`).
- Bootstrap = 1,000 resamples of ISO calendar weeks (all sites move together), implemented as
  replicate weights so every method uses identical resamples (paired). Subset CIs (site/season)
  reuse the same replicates; RESULTS shows the number of weeks behind each subset.

### 2026-09-25 · B4 uses only that model's own features (as SPEC 8.2 states)
- **Note for Phase 4:** the blend also gets site / month / dark-hours features. To show how much of
  any gain comes from *combining models* vs from those context features, Phase 4 will add an
  ablation (blend without context features). No change to B4 itself.

### 2026-09-25 · Test-time tuning knobs live in config; tests use a lighter copy
- The C grid and tie tolerance are in `settings.yaml` (`modeling`). The offline tests use a
  `fast_settings` fixture (2 C values, 2 folds, 100 resamples, leads 1/2/7) so the suite stays
  under 30 s. Production numbers always use the full settings.

### 2026-09-25 · README headline block is generated
- `skytrust report` rewrites only the text between `<!-- RESULTS:START -->` and
  `<!-- RESULTS:END -->`, from metrics.json, with the commit hash (`-dirty` if uncommitted).

### 2026-09-25 · Phase 3 checkpoint: resolve the open items the simplest, most consistent way (Hiten)
Hiten asked for the path that is easiest to understand and gives the most consistent results.
1. **Tiny subsets get no CI.** A bootstrap over a handful of weekly blocks is unreliable, so any
   subset spanning fewer than `min_weeks_for_ci` = 8 weeks reports a point estimate only
   ("too few weeks for a CI"). Applied in `evaluate.py`, so the app and reports behave the same.
2. **B4 (single model) now gets the blend's context features** (site one-hot, month sin/cos,
   dark hours). The blend and the best single model then differ *only* in how many weather
   models they see, so "blend − best single" isolates the value of combining models. This
   replaces the planned ablation. (A small, deliberate extension of SPEC 8.2's B4.) Effect on B4
   itself was tiny (e.g. ECMWF lead-1 BSS 0.514 → 0.515), i.e. context adds little on its own.
3. **2026 being cloudier** is a property of the data, not a defect. Climatology stays computed
   from the training years (standard practice); the generated caveat states the shift.

### 2026-09-25 · Blend design (SPEC 8.3)
- One model per lead, per label (primary = shipped; ASOS/ERA5 = sensitivity only, saved under
  `artifacts/sensitivity/`). Inputs: every model that forecasts that lead × 3 features, model
  spread (if ≥ 2 models), site one-hot, month sin/cos, dark hours. Median imputation + missing
  indicators (needed for ECMWF's first two weeks of 2024), standardize, L2 logistic regression,
  C by date-based CV. Training rows need ≥ 1 model available.
- **Evaluated through the exported JSON**, via the numpy loader: the test set scores exactly the
  file the app will use. `evaluate` refuses an artifact trained on/after the test start.
- **Calibration rule:** compute out-of-fold predictions on the training folds; apply an isotonic
  map (fitted on those OOF predictions) only if their 10-bin expected calibration error > 0.05.
  This mirrors sklearn's `CalibratedClassifierCV(ensemble=False)`, which can't be used directly
  because TimeSeriesSplit folds aren't a partition of the rows.
- **Decision + evidence:** OOF calibration error ranged 0.013–0.048 across all 21 blends (every
  label × lead), all under 0.05, so **no calibration was applied**. Logistic regression was
  already well calibrated. (Per-lead values are in RESULTS §8 and in each artifact.)
- Optional HistGradientBoosting comparison: not done. SPEC makes it optional "only after
  everything else is done"; logistic regression is the shipped model.

### 2026-09-25 · Phase 4 result (one final test evaluation of the blend)
- The blend beats the best single model at 7/7 leads and the equal-weight average at 2/7 leads
  (95% paired week-block CIs). RESULTS states the 5 leads where it does *not* beat the equal-weight
  average plainly. No thresholds, subsets, or features were changed after seeing test results.

### 2026-09-25 · Live forecast design (SPEC 9)
- `live.build_forecast` is a pure function (payload + time → 7 nights), so every case (lead
  assignment, missing model, no data) is tested offline on a recorded payload; `get_forecast`
  adds the network and a last-good copy on disk (`data/live_cache/{site}.json`, gitignored).
- Lead = `max(1, ceil((dusk − now)/24 h))`, capped at 7, exactly as SPEC 9.3. Consequence worth
  knowing: late in the evening, *tomorrow* night's dusk is < 24 h away, so it is also lead 1. That
  matches how lead 1 was defined in training (values issued ~24 h before each hour).
- "Models agree / split": split if tonight's spread > the 75th percentile of the training-period
  spread at that lead (stored in each artifact as `training.spread_threshold`).
- Outlook trust level from the blend's test BSS at that lead: High ≥ 0.5, Medium ≥ 0.3, else Low
  (config `live.trust_bss`). Best window = longest run of dark hours whose cross-model median
  cover is clear.
- If a model is missing live, the blend still runs (median imputation, as in training) and the UI
  names the missing model. If every model is missing, the night shows "No data", never a guess.

### 2026-09-25 · App structure
- Single entry file with a sidebar page selector (not a multipage folder): simplest to read,
  and each page is a small `render(ctx)` function that's easy to smoke-test with AppTest.
- Every page render is wrapped: an error shows a message and the other pages keep working.
  Live forecasts are cached 60 min (`st.cache_data`); failures aren't cached, so the next load
  retries. API down → last good copy with an "as of" banner; no copy → friendly message, and
  Track Record / Methodology still work (both tested).
- Night vision = CSS + a red palette for charts (red light preserves dark adaptation).

### 2026-09-25 · Inference split from training (performance)
- **Evidence:** the Tonight page's cold render was 10.4 s; 3.2 s of it was importing scikit-learn
  through `blend.py`, which the app never needs (it runs the JSON model with numpy).
- **Decision:** `inference.py` (load + predict, numpy only) and `design.py` (feature matrix,
  pandas only) hold everything the app/CLI use; `blend.py` keeps training and re-exports the
  inference names. A test asserts the live import path never loads sklearn/matplotlib/scipy.
  Cold render → 5.2 s including ~2.5 s of Open-Meteo network time; cached loads 0.4 s.

### 2026-09-25 · Test tiers
- `pytest` (quick loop) skips `@pytest.mark.slow` Streamlit AppTest smoke tests; `make test` and
  CI run everything offline (`-m "not network"`). The spec's < 30 s target applies to the quick
  loop; measured times on this laptop vary ~1.7× with background load (Spotlight indexing).

### 2026-09-29 · Phase 6 (ship)
- **Screenshots** for the README were captured from the real running app with headless Chrome
  driven over the DevTools protocol (a throwaway script, not part of the repo). Doing this found
  two real bugs that fixture-based tests missed (Moon still up at the chart edge crashed the
  timeline; best window displayed past dawn). Both fixed with regression tests.
- **Resume bullets** are generated by `skytrust report` into `docs/RESUME_BULLETS.md`, with every
  number taken from metrics.json (same honesty rule as RESULTS/README).
- **Deployment:** Streamlit Community Cloud from `uv.lock`; entry `app/streamlit_app.py`;
  Python 3.12; developer toolbar hidden for visitors; the app falls back to `src/` on the import
  path if the host installs dependencies but not the package. Steps in `docs/DEPLOY.md`.
- **Deep links** `?page=track-record&site=BIH` make pages shareable (and screenshot-able).
- **Deployed:** https://skytrust.streamlit.app/ (verified as a logged-out visitor).

### 2026-09-29 · Freeze the test period: 2026-01-01 → 2026-08-31 (changes SPEC 8.1; Hiten delegated the choice)
- **Problem:** SPEC's test period was "2026-01-01 → latest labeled night", so re-running
  `make all` on a later day changed the metrics (blend BSS 0.627 → 0.629 on 2026-09-29).
  Conclusions were unchanged, but README / RESUME_BULLETS / app numbers would drift.
- **Alternatives:** (A) freeze `split.test_end`; (B) keep it rolling.
- **Decision: A**, better for a portfolio (numbers on the resume always match the repo and app;
  anyone re-running gets identical results) and standard practice (fixed evaluation period,
  refreshed deliberately). Training data and models are unaffected; nights after the end are
  `split = "none"` and the dataset build stops there by default.
- **Why Aug 31 and not Sep 23 (the last night in the earlier results):** a first attempt froze
  2026-09-23 and still didn't reproduce. ERA5's ~6-day publication lag meant the last week before
  Sep 23 was unlabeled when the results were first computed (24 Sep) but labeled by 29 Sep. The
  end date must be one whose data is *final*. A month boundary well past the lag is final for
  good and easy to explain ("tested on Jan–Aug 2026"). Chosen on data completeness only.
- **Refreshing later:** move `split.test_end` (≥ 2 weeks back), run `make all`, commit. The git
  history then shows exactly when and why the numbers changed.

### 2026-09-29 · Prospective (forward) verification as the untouched holdout
- **Problem it fixes:** development touched the 2026 test period more than once (the B4 context
  change and the test-end freeze came after seeing test results). Both were principled, but only
  an evaluation that *couldn't* be seen during development is fully clean.
- **Design:** a scheduled GitHub Action runs daily at 22:00 UTC (3 PM Pacific, a realistic
  decision time). It logs every live forecast (5 sites × 7 nights) with the blend and code version,
  plus three references computed from the same moment's data: NOAA's National Blend of Models
  (NBM), and raw-ensemble probabilities (share of ECMWF's 51 / GEFS's 31 members predicting a
  usable night). About 9 days later (ERA5 lag + margin) the same job labels those nights with the
  backtest's label code and scores everything with the backtest's metric code.
- **Integrity:** append-only log keyed on (issue date, site, night); the first forecast issued for
  a night on a given day wins, so re-runs can't overwrite history. Only issues on/after
  `forward.start_date` (2026-09-30, the day after development stopped touching it) count.
- **Where it lives:** the `forward-data` branch (bot-owned), so daily bot commits never conflict
  with human pushes to `main`. The app reads `summary.json` from that branch (hourly cache) and
  degrades gracefully if it's unreachable.
- **Weekly API contract tests** (`pytest -m network`, scheduled) guard the assumptions the forward
  test depends on. Finding while writing them: Open-Meteo's docs now show `gfs_global_011` /
  `gfs_global_025`, but the API rejects those and still serves `gfs_global`; the test will flag it
  if that ever flips.

### 2026-09-30 · NOAA NBM as a benchmark, not a blend input
- **Design:** NBM is fetched and featurized like the five members (`benchmark_models` in config),
  scored three ways: its own usable rule (hard), its raw clear fraction as a probability, and
  calibrated with exactly B4's context features (the fair head-to-head). The shipped blend's inputs,
  including the model spread, use members only, so the forward test's model is unchanged (the
  retrained artifacts differ only in their timestamp/commit lines, checked).
- **Evaluation set** now also requires NBM present (same nights for every method). That removed
  50 lead-1 nights (NBM lead-1 gaps) and nothing at other leads; headline blend BSS 0.640 → 0.631.
- **Research variant "blend + NBM input"** (trained on the ~15 months NBM exists): not
  significantly different from the shipped blend at lead 1 (ΔBrier +0.0005, CI spans 0), so NBM
  is *not* added to the shipped model. RESULTS regenerates this comparison every run.
- **Result:** SkyTrust's blend beats NOAA's calibrated NBM at 5 of 7 leads (paired week-block CIs
  exclude zero); leads 5 and 7 are not significant, and RESULTS says so plainly.

### 2026-09-30 · Methods batch 1: B6, Brier decomposition, decision value
- **B6 = calibrated equal-weight average** (logistic regression on the members' mean clear
  fraction + the same context as B4/blend). It splits the blend's edge into *calibration*
  (B6 − B5) and *learned per-model weights* (blend − B6). Result: averaging beats the best single
  model at 6/7 leads, calibration adds significantly at 1/7, learned weights at 1/7. The blend is
  essentially a well-calibrated average (the "forecast combination puzzle"). The shipped model is
  unchanged (the forward test scores one fixed model); stated plainly in RESULTS.
- **Murphy Brier decomposition** (reliability / resolution / uncertainty, 10 bins) per method.
- **Relative economic value** (Richardson 2000) over cost/loss ratios α = 0.05…0.95, with the
  decision rule "go when P ≥ α" (what a user of a calibrated forecast would do), plus per-threshold
  false-clear / miss rates stored in metrics.json for the app's risk slider.

### 2026-09-30 · Long-term climatology (2004-2023) as the skill reference
- **Problem:** the reference used 2 training years (~60 nights per site-month); a noisy reference
  can flatter every skill score. Forecast verification normally uses a long climatology.
- **Decision:** label every 2004-2023 night at every site with the same label code (ASOS + ERA5
  history, 20 station-years and 240 ERA5 months per site), keep station-years with ≥ 80 % labeled
  nights (station outages would otherwise skew months), and use P(usable) per (site, month) as B1.
  The 2-year version stays as B1b for comparison. A code guard refuses a reference period that
  reaches the training period. Needed the DE421 excerpt extended back to 2003 (3 MB, committed).
- **Found while fetching:** a 2008 Truckee METAR contains a stray carriage return inside a remark
  (`RMK FIRST 4\r 24HR MAX…`), which crashed pandas' C parser, and remarks use `"` as an inch mark
  (`NO SN 9"`). Parser fixed (CR → space, quoting off; IEM's `onlycomma` never quotes); cached files
  parse identically; regression test added.

### 2026-09-30 · Robustness grid over the cloud definitions
- Clear threshold ∈ {10, 20, 30 %} × minimum run ∈ {2, 3, 4 h}: each cell re-runs labels,
  features, blend training, baselines and paired bootstraps. Each cell is scored against its own
  training-years climatology (the long-term one was built with the default definition), so every
  cell is internally consistent. Astronomy/raw inputs are computed once and reused.

### 2026-09-30 · Walk-forward (rolling-origin) evaluation from 2025-01
- Every month, every model (blend, B6, calibrated NBM) is refit, including C, on all nights
  before that month and forecasts that month; the long-term climatology is the fixed reference.
  Gives out-of-sample evidence for 2025 too and a month-by-month stability check. It complements,
  not replaces, the frozen-model test (which evaluates the exact shipped model).

### 2026-09-30 · Site-agnostic blend, leave-one-site-out test, custom locations
- **Geo blend:** same pipeline as the blend minus the site one-hot, trained on all sites'
  training years and exported to `artifacts/geo/`. Used only for custom locations; the airport
  forecasts keep the site-aware blend.
- **Evidence:** leave-one-site-out. Each site is held out, the geo blend is trained on the other
  sites' training years only, then scored on the held-out site's test period, on the same nights as
  the shipped blend (RESULTS §12). A spy test proves the held-out site never reaches training.
- **Custom locations** are limited to the Pacific-time West (lat 32–49, lon −125…−114): the app
  displays Pacific time and that's the region the models were validated on. Elevation comes from
  Open-Meteo's terrain model. Instead of a local track record, the app shows the unseen-site skill.
- **Where Tonight:** all sites ranked by tonight's probability, on a map (MapLibre dark tiles).

### 2026-09-30 · UI/engineering batch
- Night-by-hour outlook grid (Clear Sky Chart style), the latest GOES-West GeoColor image with a
  link to the animated loop, an About section, Paul Tol colour-blind-safe palettes in figures and
  app, and Dependabot for weekly dependency/Actions update PRs (tested by CI).

### 2026-09-30 · Hourly P(clear) model
- The nightly blend answers "will tonight be usable?"; planning also needs "which hours?". Each
  dark hour at lead d is one example: every member model's forecast cover for that hour, their
  mean and spread, where the hour falls in the night (0 = dusk, 1 = dawn), month and site.
  Target: the hour is clear under the primary truth (the hourly quantity the nightly rule is
  built from). Same tuned L2 logistic regression and JSON/numpy export as the blend.
- Evaluated once on the frozen test period against an hourly climatology (training-years rate
  per site and month) and the "share of models forecasting clear" baseline, with week-block CIs.
  The app shows it as per-hour bars; it is additional information, and the nightly verdict still
  comes from the nightly blend.

### 2026-09-30 · Memory-lean ASOS loading
- Labelling 20 years at AUN (a station with many special reports) read every METAR column and
  pushed an 8 GB laptop into 14 GB of swap. `load_asos` now reads only the time and sky-cover
  columns, dedupes on them, and stores the sky codes as categoricals: 14 MB for the same data.
  A regression test checks the labels are identical to the old loader's.

### 2026-09-30 · GOES-18 satellite cloud mask as an independent truth (Hiten approved adding h5py)
- **Problem:** both ground truths have a known blind spot. ASOS can't see above 12,000 ft; ERA5
  is a model. The primary label (max of the two) was designed around that, but nothing tested it
  against a real observation that sees high cloud.
- **Decision:** add the GOES-18 Clear Sky Mask (AWS open data, 5-minute CONUS scans) as a fourth
  label, `usable_goes`, built with exactly the same clear / usable / missing rules. It does **not**
  change the primary label (SPEC §4 is unchanged); it's used three ways:
  1. **Referee** (DATA_QUALITY §7): on nights where ASOS and ERA5 disagree, which side does the
     satellite take? This tests the "ASOS misses cirrus" explanation directly.
  2. **Label sensitivity** (RESULTS §6): a blend trained on the GOES label, like the ASOS-only and
     ERA5-only ones. GOES skill is measured against its own 2024–25 rate (no 20-year satellite
     record exists for this sensor).
  3. **Cross-truth check** (RESULTS §6): the *shipped* blend, trained on the primary label only,
     scored against the GOES truth it never saw, head-to-head with NOAA's NBM.
- **How:** geometry from the GOES-R Product User Guide (tested on its worked example), 5 × 5
  good-quality-pixel box per site, first scan of each dark hour. `h5py` reads the NetCDF4 files
  (reading NetCDF4 without it would need a heavier stack such as xarray + netCDF4). Labels that
  lack data (e.g. GOES before it's fetched) are skipped by training and evaluation automatically.

### 2026-09-30 · Static type checking (Hiten approved adding mypy)
- mypy runs in CI next to ruff, so type hints are checked, not decorative.

