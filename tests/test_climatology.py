"""Long-term climatology: coverage filter, rate tables, leakage guard, and its use as B1."""

from __future__ import annotations

import copy
import dataclasses
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from skytrust import baselines, climatology
from skytrust.errors import LeakageError
from synthetic import SYNTH_SITES


def frame(site="SAC", years=(2010, 2011), coverage=(1.0, 0.5)):
    rows = []
    for year, cov in zip(years, coverage, strict=True):
        dates = pd.date_range(f"{year}-01-01", f"{year}-12-31").date
        for i, d in enumerate(dates):
            labeled = i < int(cov * len(dates))
            usable = pd.NA if not labeled else (i % 2 == 0)
            rows.append(
                {
                    "site": site,
                    "night_date": d,
                    "usable_primary": usable,
                    "usable_asos": usable,
                    "usable_era5": usable,
                }
            )
    df = pd.DataFrame(rows)
    for c in climatology.LABEL_COLUMNS.values():
        df[c] = df[c].astype("boolean")
    dates = pd.to_datetime(df["night_date"])
    return df.assign(year=dates.dt.year, month=dates.dt.month)


def test_years_with_poor_coverage_are_dropped():
    years = climatology.usable_years(frame(), min_coverage=0.8)
    assert years == {"SAC": [2010]}


def test_rate_tables_use_only_usable_years():
    df = frame()
    tables = climatology.rate_tables(df, {"SAC": [2010]})
    jan = tables["primary"]["SAC"][1]
    assert jan["n"] == 31 and jan["rate"] == pytest.approx(16 / 31)  # alternating, starts True


def test_reference_period_must_end_before_training(settings):
    raw = copy.deepcopy(settings.raw)
    raw["climatology"]["end"] = dt.date(2024, 6, 30)
    with pytest.raises(LeakageError):
        climatology.reference_period(dataclasses.replace(settings, raw=raw))
    first, last = climatology.reference_period(settings)
    assert last < settings.raw["split"]["train_start"] and first < last


def test_save_and_load_round_trip(tmp_path):
    df = frame()
    table = {
        "period": ["2010-01-01", "2011-12-31"],
        "min_year_coverage": 0.8,
        "years_used": {"SAC": [2010]},
        "tables": climatology.rate_tables(df, {"SAC": [2010]}),
    }
    climatology.save(table, df, tmp_path / "c.json", tmp_path / "c.parquet")
    loaded = climatology.load_table("primary", tmp_path / "c.json")
    assert loaded.loc[("SAC", 1), "n"] == 31
    assert climatology.load_table("primary", tmp_path / "missing.json") is None


def test_label_reference_nights_on_synthetic_cache(settings, synthetic_built):
    """Same label code as the dataset, on a reference period inside the synthetic cache."""
    _, root = synthetic_built
    raw = copy.deepcopy(settings.raw)
    raw["climatology"].update(start=dt.date(2025, 12, 1), end=dt.date(2025, 12, 20))
    raw["split"]["train_start"] = dt.date(2025, 12, 21)  # keep the leakage guard satisfied
    s = dataclasses.replace(settings, raw=raw)
    df = climatology.label_reference_nights(s, SYNTH_SITES, root)
    assert len(df) == 2 * 20 and df["usable_primary"].notna().mean() > 0.9
    table, _ = climatology.build(s, SYNTH_SITES, root)
    assert set(table["tables"]["primary"]) == {"SAC", "BIH"}


def test_baselines_use_long_term_table_when_present(synthetic_built, fast_settings, monkeypatch):
    table = pd.DataFrame(
        {"site": ["SAC", "BIH"] * 12, "month": np.repeat(range(1, 13), 2), "rate": 0.42, "n": 600}
    ).set_index(["site", "month"])
    monkeypatch.setattr(climatology, "load_table", lambda label="primary", path=None: table)
    result = baselines.run_baselines(synthetic_built[0], "primary", 1, fast_settings)
    assert np.allclose(result.methods["climatology"].p, 0.42)
    assert result.methods["climatology"].info["source"] == "long_term"
    assert "climatology_train" in result.methods  # training-years version kept for comparison
