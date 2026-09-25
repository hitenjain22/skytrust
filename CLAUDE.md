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
- **ASOS label:** routine+SPECI obs in (H−30, H+30]; okta midpoints; max over layers, then over obs. Blind above 12,000 ft.
- **ERA5 label:** Open-Meteo archive, `models=era5`, cloud_cover/100. Reanalysis, ECMWF-related, ~6-day lag.
- **Primary label:** per hour `max(asos, era5)`; > 25 % missing in either source → exclude night (log reason).
- **Features:** Previous Runs `cloud_cover_previous_day{d}`; > 25 % missing → NaN.
- **Split:** train 2024-01-01 → 2025-12-31, test 2026-01-01 → latest. Test set used once per final model.

## Commands
- `uv sync` · `uv run pytest` · `uv run ruff check . && uv run ruff format --check .`
- `uv run python -m skytrust validate-sites`
