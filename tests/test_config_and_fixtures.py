"""Config validation + contract checks on the recorded Open-Meteo fixtures.

The fixture checks pin down the API facts recorded in docs/DATA_NOTES.md, so if a
re-captured fixture ever changes shape, these tests say exactly which assumption broke.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from skytrust.config import load_settings, load_sites
from skytrust.data.iem import year_chunks


def test_settings_load_with_spec_defaults(settings):
    assert settings.clear_threshold == 0.20
    assert settings.min_run_hours == 3
    assert settings.max_missing_frac == 0.25
    assert settings.sources["era5_model"] == "era5"
    assert settings.raw["split"]["train_end"] < settings.raw["split"]["test_start"]


def test_bad_clear_threshold_rejected(tmp_path):
    text = (load_settings.__globals__["CONFIG_DIR"] / "settings.yaml").read_text()
    bad = tmp_path / "s.yaml"
    bad.write_text(text.replace("clear_threshold: 0.20", "clear_threshold: 20"))
    with pytest.raises(ValueError, match="0-1 scale"):
        load_settings(bad)


def test_sites_yaml_has_required_fields():
    sites = load_sites()
    assert len(sites) >= 5
    for s in sites:
        assert 32 < s.lat < 42 and -125 < s.lon < -114  # California
        assert s.timezone == "America/Los_Angeles"


def test_year_chunks_cover_range_exclusive_end():
    chunks = year_chunks(dt.date(2024, 1, 1), dt.date(2025, 3, 5))
    assert [c[0].date() for c in chunks] == [dt.date(2024, 1, 1), dt.date(2025, 1, 1)]
    assert chunks[-1][1].date() == dt.date(2025, 3, 6)
    assert all(c[0].tzinfo is dt.UTC for c in chunks)


def load(fixtures_dir, name):
    return json.loads((fixtures_dir / name).read_text())


def test_prevruns_fixture_keys_are_suffixed_per_model(fixtures_dir, settings):
    h = load(fixtures_dir, "openmeteo_prevruns_SAC_20250308_20250310.json")["hourly"]
    assert len(h["time"]) == 72  # 3 days hourly, UTC
    for m in settings.models:
        for d in m.leads:
            values = h[f"cloud_cover_previous_day{d}_{m.id}"]
            assert any(v is not None for v in values), (m.id, d)


def test_hrrr_has_only_lead_1(fixtures_dir):
    h = load(fixtures_dir, "openmeteo_prevruns_SAC_20250308_20250310.json")["hourly"]
    assert any(v is not None for v in h["cloud_cover_previous_day1_ncep_hrrr_conus"])
    assert all(v is None for v in h["cloud_cover_previous_day2_ncep_hrrr_conus"])


def test_era5_fixture_has_layers_in_percent(fixtures_dir):
    d = load(fixtures_dir, "openmeteo_era5_SAC_20250308_20250310.json")
    assert d["hourly_units"]["cloud_cover"] == "%"
    for k in ["cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high"]:
        assert all(0 <= v <= 100 for v in d["hourly"][k] if v is not None)


def test_live_fixture_has_layers_for_every_model(fixtures_dir, settings):
    h = load(fixtures_dir, "openmeteo_forecast_SAC_live.json")["hourly"]
    assert len(h["time"]) == 8 * 24
    for m in settings.models:
        assert f"cloud_cover_high_{m.id}" in h
