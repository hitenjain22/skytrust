"""Dataset build (end-to-end on a synthetic raw cache) and the SPEC 12.1 validation rules."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from conftest import SYNTH_FIRST as FIRST
from conftest import SYNTH_LAST as LAST
from skytrust import __main__ as cli
from skytrust import dataset, report
from synthetic import SYNTH_SITES


@pytest.fixture
def built(synthetic_built):
    return synthetic_built


def test_shape_keys_and_splits(built, settings):
    df, _ = built
    n_nights = (LAST - FIRST).days + 1
    assert len(df) == len(SYNTH_SITES) * n_nights * len(settings.raw["leads"])
    assert not df.duplicated(dataset.KEYS).any()
    assert set(df["split"]) == {"train", "test"}
    assert (pd.to_datetime(df.loc[df["split"] == "train", "night_date"]) <= "2025-12-31").all()
    assert set(df["season"]) == {"DJF"}


def test_built_dataset_passes_validation(built):
    dataset.validate_dataset(built[0])


def test_labels_and_features_are_populated_and_related(built):
    df, _ = built
    lead1 = df[df["lead"] == 1]
    assert lead1["usable_primary"].notna().mean() > 0.95
    assert (lead1["n_models_available"] == 5).mean() > 0.95
    # Forecasts are truth + noise, so forecast clear fraction should track the label.
    usable = lead1["usable_primary"].astype("float")
    corr = np.corrcoef(lead1["ecmwf_frac_clear"].fillna(0), usable.fillna(0))[0, 1]
    assert corr > 0.5


def test_lead_availability_mirrors_model_leads(built):
    df, _ = built
    assert df.loc[df["lead"] == 2, "hrrr_frac_clear"].isna().all()
    assert df.loc[df["lead"] == 7, "icon_frac_clear"].isna().all()
    assert (df.loc[df["lead"] == 7, "n_models_available"] <= 3).all()


def test_labels_repeat_across_leads(built):
    df, _ = built
    per_night = df.groupby(["site", "night_date"])["usable_primary"].nunique(dropna=False)
    assert (per_night == 1).all()


def test_cached_last_night_is_min_of_sources_minus_one(built, settings):
    _, root = built
    # Synthetic data runs through LAST + 1 day 23:00 -> last complete night is LAST.
    assert dataset.cached_last_night(SYNTH_SITES, settings, root) == LAST


def test_empty_cache_raises(tmp_path, settings):
    with pytest.raises(dataset.DatasetValidationError):
        dataset.cached_last_night(SYNTH_SITES, settings, tmp_path)


def test_save_and_load_round_trip(built, tmp_path):
    df, _ = built
    path = dataset.save_dataset(df, tmp_path / "d.parquet")
    back = dataset.load_dataset(path)
    assert back.shape == df.shape
    assert str(back["usable_primary"].dtype) == "boolean"
    assert str(back["dusk_utc"].dt.tz) == "UTC"


# ---------- validation rules on hand-made frames ----------


def small_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "site": ["SAC", "SAC"],
            "night_date": [dt.date(2025, 1, 1), dt.date(2025, 1, 2)],
            "lead": [1, 1],
            "dark_hours": [10, 10],
            "dusk_utc": pd.to_datetime(["2025-01-02 02:30", "2025-01-03 02:30"], utc=True),
            "dawn_utc": pd.to_datetime(["2025-01-02 13:30", "2025-01-03 13:30"], utc=True),
            "gfs_frac_clear": [0.5, 0.9],
            "usable_primary": pd.array([True, pd.NA], dtype="boolean"),
            "usable_asos": pd.array([True, False], dtype="boolean"),
            "usable_era5": pd.array([True, False], dtype="boolean"),
        }
    )


def test_validation_accepts_clean_frame():
    dataset.validate_dataset(small_frame())


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.assign(night_date=[dt.date(2025, 1, 1)] * 2), "duplicate"),
        (lambda d: d.assign(gfs_frac_clear=[0.5, 1.2]), "values outside"),
        (lambda d: d.assign(dark_hours=[10, 15]), "dark_hours"),
        (lambda d: d.assign(dusk_utc=d["dusk_utc"].dt.tz_localize(None)), "naive"),
        (lambda d: d.assign(usable_asos=[1.0, 0.0]), "nullable boolean"),
    ],
)
def test_validation_rejects(mutate, message):
    with pytest.raises(dataset.DatasetValidationError, match=message):
        dataset.validate_dataset(mutate(small_frame()))


def test_save_refuses_invalid_dataset(tmp_path):
    bad = small_frame().assign(gfs_frac_clear=[0.5, 1.2])
    with pytest.raises(dataset.DatasetValidationError):
        dataset.save_dataset(bad, tmp_path / "bad.parquet")
    assert not (tmp_path / "bad.parquet").exists()


def test_assign_split(settings):
    dates = pd.Series([dt.date(2023, 12, 31), dt.date(2025, 12, 31), dt.date(2026, 1, 1)])
    assert dataset.assign_split(dates, settings).tolist() == ["none", "train", "test"]


# ---------- report + CLI ----------


def test_data_quality_report_renders_every_section(built, settings):
    df, _ = built
    md = report.data_quality_markdown(df, settings, dt.datetime(2026, 1, 1, tzinfo=dt.UTC))
    for heading in ["## 1. Overview", "## 5. Base rate", "## 7. Do ASOS and ERA5 agree?", "## 9."]:
        assert heading in md
    assert "| model |" in md
    assert "SAC" in md and "BIH" in md


def test_test_sufficiency_flags_small_test_set(built):
    n, verdict = report.test_sufficiency(built[0])
    assert n < report.MIN_TEST_NIGHTS_LEAD1 and verdict.startswith("⚠ FLAG")


def test_md_table_and_rate_helpers():
    table = report.md_table(pd.DataFrame({"a": ["1"], "b": ["2"]}, index=pd.Index(["x"], name="k")))
    assert table.splitlines()[0] == "| k | a | b |"
    assert report.rate_with_n(pd.Series([True, False, pd.NA], dtype="boolean")) == "50.0% (n=2)"
    assert report.rate_with_n(pd.Series([pd.NA], dtype="boolean")) == "–"


def test_cli_build_dataset(monkeypatch, built, tmp_path, capsys):
    df, _ = built
    monkeypatch.setattr(dataset, "build_dataset", lambda *a, **k: df)
    monkeypatch.setattr(dataset, "save_dataset", lambda d: tmp_path / "d.parquet")
    monkeypatch.setattr(report, "write_data_quality", lambda d, s: tmp_path / "q.md")
    assert cli.main(["build-dataset"]) == 0
    out = capsys.readouterr().out
    assert "site-nights" in out and "Test nights at lead 1" in out


def test_nights_after_test_end_are_outside_the_split(settings):
    """Frozen evaluation period (DECISIONS 2026-09-29): later nights are neither train nor test."""
    end = settings.raw["split"]["test_end"]
    dates = pd.Series([end, end + dt.timedelta(days=1)])
    assert dataset.assign_split(dates, settings).tolist() == ["test", "none"]


def test_default_dataset_end_is_capped_at_test_end(settings, monkeypatch):
    monkeypatch.setattr(dataset, "cached_last_night", lambda *a, **k: dt.date(2030, 1, 1))
    assert dataset.default_last_night(SYNTH_SITES, settings) == settings.raw["split"]["test_end"]
    monkeypatch.setattr(dataset, "cached_last_night", lambda *a, **k: dt.date(2026, 3, 1))
    assert dataset.default_last_night(SYNTH_SITES, settings) == dt.date(2026, 3, 1)
