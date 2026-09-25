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
