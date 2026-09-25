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
