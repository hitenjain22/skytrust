from __future__ import annotations

import copy
import dataclasses
import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from skytrust.config import Settings, load_settings
from synthetic import SYNTH_SITES, write_synthetic_cache

FIXTURES = Path(__file__).parent / "fixtures"
SYNTH_FIRST, SYNTH_LAST = dt.date(2025, 12, 1), dt.date(2026, 1, 31)


@pytest.fixture(scope="session", autouse=True)
def _no_real_climatology(tmp_path_factory):
    """The repo's 20-year climatology table is built from real observations at the real sites.
    Tests run on synthetic weather, so they must never pick it up by accident; a test that needs
    a long-term table passes or patches one explicitly. Session scope so it is in place before
    any module-scoped fixture trains or evaluates."""
    from skytrust import climatology

    mp = pytest.MonkeyPatch()
    mp.setattr(climatology, "TABLE_PATH", tmp_path_factory.mktemp("clim") / "absent.json")
    yield
    mp.undo()


@pytest.fixture(scope="session")
def settings() -> Settings:
    return load_settings()


@pytest.fixture(scope="session")
def fast_settings(settings) -> Settings:
    """Same settings with a 2-value C grid, 2 CV folds, 100 bootstrap resamples, and evaluation
    at leads 1, 2, 7 (all models / HRRR gone / longest lead). Exercises the same code paths as
    production but keeps the offline suite fast (SPEC 12: < 30 s)."""
    raw = copy.deepcopy(settings.raw)
    raw["leads"] = [1, 2, 7]
    raw["modeling"]["c_grid_log10"] = [-1, 1, 2]
    raw["split"]["cv_folds"] = 2
    raw["bootstrap_resamples"] = 100
    raw["min_weeks_for_ci"] = 2  # the synthetic test month only spans ~5 weeks
    return dataclasses.replace(settings, raw=raw)


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture(scope="session")
def synthetic_built(tmp_path_factory, settings) -> tuple[pd.DataFrame, Path]:
    """Dataset built end-to-end from a fake raw cache: 2 sites x 62 nights spanning the
    train/test boundary (Dec 2025 = train, Jan 2026 = test)."""
    from skytrust import dataset

    root = tmp_path_factory.mktemp("raw")
    write_synthetic_cache(root, settings.forecast_models, SYNTH_FIRST, SYNTH_LAST)
    return dataset.build_dataset(settings, SYNTH_SITES, SYNTH_FIRST, SYNTH_LAST, root=root), root
