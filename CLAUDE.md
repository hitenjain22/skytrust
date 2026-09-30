# CLAUDE.md: working rules for SkyTrust

**SPEC.md is the source of truth. Read it before any work.** This file is a condensed reminder.

## Rules
1. Work in phases (SPEC §13). At each phase end: run `uv run pytest`, report (built / verified / uncertain / next), **wait for "go"**.
2. Never fabricate data, API behavior, model IDs, or results. Verify against docs/probes; record differences in `docs/DATA_NOTES.md`.
3. Every number in README/RESULTS comes from code in this repo (`skytrust report`). No hand-typed metrics.
4. **Ask before:** new dependencies (allowed list: SPEC §14), changing §4 definitions, changing the train/test split, expanding scope.
5. Log decisions in `docs/DECISIONS.md`; add a `docs/LEARNING.md` entry per finished module (what, why, key concept, 2–3 interview Qs).
6. Tests are offline by default (fixtures in `tests/fixtures/`); live tests use `@pytest.mark.network`.
7. Bugs: failing test first, then fix, keep the test.
8. Small commits, ≥ 1 per task. Remind Hiten to push at every checkpoint.
9. Readable code over clever code. Type hints, docstrings explain *why*, `logging` not `print` (except CLI).

## Definitions (SPEC §4; values live in `config/settings.yaml`)
- All timestamps tz-aware UTC internally; display `America/Los_Angeles`.
- **Night** keyed by local date of the evening it begins.
- **Dark window:** sun < −18° (Skyfield, DE421). **Dark hours:** top-of-hour UTC H with dusk ≤ H ≤ dawn.
- **Clear hour:** cloud fraction ≤ 0.20. **Usable night:** ≥ 3 consecutive clear dark hours; a missing hour breaks a run.
- **ASOS label:** routine+SPECI obs in (H−30, H+30]; okta midpoints; max over layers; hourly value = report **nearest H** (ties → cloudier; approved change 2026-09-25, `max` kept for sensitivity). Blind above 12,000 ft (except human-augmented FAT).
- **ERA5 label:** Open-Meteo archive, `models=era5`, cloud_cover/100. Reanalysis, ECMWF-related, ~6-day lag.
- **Primary label:** per hour `max(asos, era5)`; > 25 % missing in either source → exclude night (log reason).
- **Features:** Previous Runs `cloud_cover_previous_day{d}`; > 25 % missing → NaN.
- **Split:** train 2024-01-01 → 2025-12-31, test 2026-01-01 → 2026-08-31 (`split.test_end`, frozen 2026-09-29; data must be final). Test set used once per final model.

## Commands
- `uv sync` · `uv run pytest` · `uv run ruff check . && uv run ruff format --check .`
- `uv run python -m skytrust validate-sites` · `fetch --source {asos,era5,prevruns}` · `build-dataset` · `train` · `evaluate` · `report` (`make train report`; `make all` rebuilds everything)
- Core rules live in `src/skytrust/nightly.py` (clear / run / usable / missing). Reuse it; don't reimplement.
- Offline tests use the `fast_settings` fixture (light C grid/folds/resamples). Keep the suite < 30 s.
- The shipped models are `artifacts/model_lead{d}.json`; load them only via `inference.load_artifact` + `inference.predict_proba` (numpy). Never pickle. The app/CLI import path must not load sklearn (tested).
- App: `make app`; CLI: `make tonight SITE=BIH`. Quick tests: `pytest`; everything offline: `make test`.
- Live app: https://skytrust.streamlit.app/ (redeploys on every push to main).
