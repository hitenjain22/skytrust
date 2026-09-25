from __future__ import annotations

import copy
import datetime as dt
import json
from unittest.mock import MagicMock

import pandas as pd
import pytest

from skytrust.config import ModelSpec, Site
from skytrust.data import cache, openmeteo
from skytrust.data.http import BadResponseError

SAC = Site("SAC", "Sacramento", 38.5069, -121.495, 8.0, "valley", "America/Los_Angeles")
GFS = ModelSpec("gfs", "gfs_global", "GFS", (1, 2, 3, 4, 5, 6, 7))
HRRR = ModelSpec("hrrr", "ncep_hrrr_conus", "HRRR", (1,))


@pytest.fixture
def prevruns(fixtures_dir):
    return json.loads((fixtures_dir / "openmeteo_prevruns_SAC_20250308_20250310.json").read_text())


@pytest.fixture
def era5(fixtures_dir):
    return json.loads((fixtures_dir / "openmeteo_era5_SAC_20250308_20250310.json").read_text())


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(openmeteo, "raw_path", lambda *a, **k: cache.raw_path(*a, root=tmp_path))
    return tmp_path


def test_month_chunks():
    chunks = openmeteo.month_chunks(dt.date(2024, 1, 15), dt.date(2024, 3, 3))
    assert chunks == [
        (dt.date(2024, 1, 15), dt.date(2024, 1, 31)),
        (dt.date(2024, 2, 1), dt.date(2024, 2, 29)),  # leap year
        (dt.date(2024, 3, 1), dt.date(2024, 3, 3)),
    ]
    assert openmeteo.month_chunks(dt.date(2024, 5, 2), dt.date(2024, 5, 1)) == []


@pytest.mark.parametrize(
    "model_id", ["gfs_global", "ecmwf_ifs025", "gem_seamless", "icon_seamless"]
)
def test_parse_prevruns_fixture_suffixed_keys(prevruns, model_id):
    variables = [openmeteo.prevruns_var(d) for d in range(1, 7)]
    df = openmeteo.parse_hourly(prevruns, variables, model_id)
    assert str(df.index.tz) == "UTC" and len(df) == 72
    assert df.stack().between(0, 1).all()  # converted from percent
    assert df.notna().all().all()


def test_hrrr_long_leads_are_all_nan(prevruns):
    df = openmeteo.parse_hourly(prevruns, ["cloud_cover_previous_day2"], "ncep_hrrr_conus")
    assert df["cloud_cover_previous_day2"].isna().all()


def test_parse_era5_fixture_unsuffixed(era5):
    df = openmeteo.parse_hourly(era5, openmeteo.ERA5_VARS)
    assert list(df.columns) == openmeteo.ERA5_VARS
    assert df.max().max() <= 1.0
    raw_first = era5["hourly"]["cloud_cover"][0]
    assert df["cloud_cover"].iloc[0] == pytest.approx(raw_first / 100)


def test_parse_live_fixture(fixtures_dir):
    payload = json.loads((fixtures_dir / "openmeteo_forecast_SAC_live.json").read_text())
    cols = openmeteo.LIVE_CLOUD_VARS + openmeteo.LIVE_EXTRA_VARS
    df = openmeteo.parse_hourly(payload, cols, "gfs_global")
    assert len(df) == 192
    assert df["cloud_cover"].between(0, 1).all()
    assert df["temperature_2m"].abs().max() > 1  # non-cloud vars are NOT divided by 100


def test_missing_variable_becomes_nan_column(era5):
    df = openmeteo.parse_hourly(era5, ["cloud_cover", "cloud_cover_previous_day9"])
    assert df["cloud_cover_previous_day9"].isna().all()


def _mutate(payload, fn):
    p = copy.deepcopy(payload)
    fn(p)
    return p


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.pop("hourly"),
        lambda p: p["hourly"].update(time=[]),
        lambda p: p["hourly"]["cloud_cover"].pop(),
        lambda p: p["hourly"]["cloud_cover"].__setitem__(0, 150),
        lambda p: p.update(timezone="America/Los_Angeles"),
    ],
    ids=["no-hourly", "no-times", "length-mismatch", "out-of-range", "not-utc"],
)
def test_malformed_payloads_raise(era5, mutation):
    with pytest.raises(BadResponseError):
        openmeteo.parse_hourly(_mutate(era5, mutation), ["cloud_cover"])


def test_complete_cache_makes_zero_network_calls(tmp_cache, era5):
    path = cache.raw_path("era5", "SAC", None, "20250308", "20250310", "json", root=tmp_cache)
    cache.write_cached(path, json.dumps(era5))
    client = MagicMock()
    settings = MagicMock(
        sources={"era5_url": "https://x.test", "era5_model": "era5", "settled_after_days": 30}
    )
    openmeteo.fetch_era5(client, settings, SAC, dt.date(2025, 3, 8), dt.date(2025, 3, 10))
    client.get_json.assert_not_called()


def test_fetch_writes_cache_when_complete(tmp_cache, era5):
    client = MagicMock()
    client.get_json.return_value = era5
    settings = MagicMock(
        sources={"era5_url": "https://x.test", "era5_model": "era5", "settled_after_days": 30}
    )
    openmeteo.fetch_era5(client, settings, SAC, dt.date(2025, 3, 8), dt.date(2025, 3, 10))
    params = client.get_json.call_args.args[1]
    assert params["models"] == "era5" and params["elevation"] == 8.0
    assert params["start_date"] == "2025-03-08" and params["end_date"] == "2025-03-10"
    assert len(list(tmp_cache.rglob("*.json"))) == 1


def test_fetch_does_not_cache_unpublished_trailing_day(tmp_cache, era5):
    def blank_last_day(p):
        for k in openmeteo.ERA5_VARS:
            p["hourly"][k][-24:] = [None] * 24

    client = MagicMock()
    client.get_json.return_value = _mutate(era5, blank_last_day)
    settings = MagicMock(
        sources={"era5_url": "https://x.test", "era5_model": "era5", "settled_after_days": 30}
    )
    recent = dt.date(2025, 3, 12)  # chunk ended 2 days ago -> may still be filling in
    openmeteo.fetch_era5(
        client, settings, SAC, dt.date(2025, 3, 8), dt.date(2025, 3, 10), today=recent
    )
    assert list(tmp_cache.rglob("*.json")) == []


def test_fetch_prevruns_requests_only_the_models_leads(tmp_cache, prevruns):
    client = MagicMock()
    client.get_json.return_value = prevruns
    settings = MagicMock(sources={"prevruns_url": "https://x.test", "settled_after_days": 30})
    n = openmeteo.fetch_prevruns(
        client, settings, SAC, HRRR, dt.date(2025, 3, 8), dt.date(2025, 3, 10)
    )
    assert n == 1
    assert client.get_json.call_args.args[1]["hourly"] == "cloud_cover_previous_day1"


def test_load_prevruns_later_chunk_wins_on_overlap(tmp_path, prevruns):
    folder = tmp_path / "prevruns" / "SAC" / "gfs_global"
    folder.mkdir(parents=True)
    older = _mutate(
        prevruns, lambda p: p["hourly"]["cloud_cover_previous_day1_gfs_global"].__setitem__(0, 0)
    )
    newer = _mutate(
        prevruns, lambda p: p["hourly"]["cloud_cover_previous_day1_gfs_global"].__setitem__(0, 100)
    )
    (folder / "20250308_20250309.json").write_text(json.dumps(older))
    (folder / "20250308_20250310.json").write_text(json.dumps(newer))
    df = openmeteo.load_prevruns("SAC", GFS, root=tmp_path)
    assert df.index.is_unique and df.index.is_monotonic_increasing
    assert list(df.columns) == [f"lead{d}" for d in range(1, 8)]
    assert df["lead1"].iloc[0] == 1.0


def test_load_empty_cache_returns_empty_frame(tmp_path):
    df = openmeteo.load_era5("SAC", root=tmp_path)
    assert df.empty and list(df.columns) == openmeteo.ERA5_VARS


def test_fetch_live_requests_all_models_in_one_call(settings):
    client = MagicMock()
    openmeteo.fetch_live(client, settings, SAC)
    params = client.get_json.call_args.args[1]
    assert params["models"].split(",") == [m.id for m in settings.models]
    assert params["forecast_days"] == 8 and "cloud_cover_high" in params["hourly"]


def test_no_naive_timestamps_anywhere(era5):
    df = openmeteo.parse_hourly(era5, ["cloud_cover"])
    assert isinstance(df.index.dtype, pd.DatetimeTZDtype)
    assert str(df.index.tz) == "UTC"


def test_old_chunk_with_permanent_nulls_is_still_cached(tmp_cache, era5):
    """Regression: ECMWF's archive starts 2024-02-04, so its January 2024 chunk is empty
    forever. It must be cached anyway, or every run re-downloads it."""

    def blank_everything(p):
        for k in openmeteo.ERA5_VARS:
            p["hourly"][k] = [None] * len(p["hourly"]["time"])

    client = MagicMock()
    client.get_json.return_value = _mutate(era5, blank_everything)
    settings = MagicMock(sources={"era5_url": "https://x.test", "era5_model": "era5",
                                  "settled_after_days": 30})  # fmt: skip
    today = dt.date(2025, 6, 1)  # chunk is ~3 months old -> settled
    openmeteo.fetch_era5(
        client, settings, SAC, dt.date(2025, 3, 8), dt.date(2025, 3, 10), today=today
    )
    assert len(list(tmp_cache.rglob("*.json"))) == 1
