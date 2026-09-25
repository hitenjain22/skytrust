from __future__ import annotations

import datetime as dt
import json
import math
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from skytrust.data import iem
from skytrust.data.http import BadResponseError

DEFAULT = {"CLR": 0.0, "SKC": 0.0, "FEW": 0.1875, "SCT": 0.4375, "BKN": 0.75, "OVC": 1.0, "VV": 1.0}
CONSERVATIVE = {**DEFAULT, "FEW": 0.25}


@pytest.mark.parametrize(
    ("code", "expected_default", "expected_conservative"),
    [
        ("CLR", 0.0, 0.0),
        ("SKC", 0.0, 0.0),
        ("FEW", 0.1875, 0.25),
        ("SCT", 0.4375, 0.4375),
        ("BKN", 0.75, 0.75),
        ("OVC", 1.0, 1.0),
        ("VV", 1.0, 1.0),
        ("VV ", 1.0, 1.0),  # IEM really sends "VV " with a trailing space
    ],
)
def test_sky_code_mapping_both_modes(code, expected_default, expected_conservative):
    assert iem.sky_code_to_fraction(code, DEFAULT) == expected_default
    assert iem.sky_code_to_fraction(code, CONSERVATIVE) == expected_conservative


@pytest.mark.parametrize("code", ["M", "", None, "///"])
def test_missing_codes_are_nan(code):
    assert math.isnan(iem.sky_code_to_fraction(code, DEFAULT))


def test_config_mappings_match_spec_table(settings):
    modes = settings.raw["definitions"]["sky_cover_mapping"]
    assert modes["default"]["FEW"] == 0.1875
    assert modes["conservative"]["FEW"] == 0.25
    # FEW must be "clear" by default and "not clear" in conservative mode.
    assert modes["default"]["FEW"] <= settings.clear_threshold < modes["conservative"]["FEW"]


def obs(rows: list[tuple[str, str, str, str, str]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["valid", "skyc1", "skyc2", "skyc3", "skyc4"])
    df["valid"] = pd.to_datetime(df["valid"], utc=True)
    return df


def test_observation_cover_is_max_over_layers_and_nan_when_all_missing():
    df = obs(
        [
            ("2025-01-01 00:53", "FEW", "SCT", "BKN", "M"),
            ("2025-01-01 01:53", "VV ", "M", "M", "M"),
            ("2025-01-01 02:53", "M", "M", "M", "M"),
            ("2025-01-01 03:53", "CLR", "M", "M", "M"),
        ]
    )
    cover = iem.observation_cover(df, DEFAULT).to_numpy()
    assert cover[0] == 0.75
    assert cover[1] == 1.0
    assert np.isnan(cover[2])
    assert cover[3] == 0.0


def test_window_includes_h_plus_30_and_excludes_h_minus_30():
    h = pd.Timestamp("2025-01-01 05:00", tz="UTC")
    valid = pd.Series(
        [h + pd.Timedelta(minutes=30), h - pd.Timedelta(minutes=30), h - pd.Timedelta(minutes=29)]
    )
    hours = iem.assign_hour(valid)
    assert hours.iloc[0] == h  # H+30 included
    assert hours.iloc[1] == h - pd.Timedelta(hours=1)  # H-30 excluded (belongs to previous hour)
    assert hours.iloc[2] == h


H5 = pd.Timestamp("2025-01-01 05:00", tz="UTC")


def test_max_method_takes_max_over_window():
    df = obs(
        [
            ("2025-01-01 04:53", "CLR", "M", "M", "M"),  # routine
            ("2025-01-01 05:12", "BKN", "M", "M", "M"),  # SPECI in the same window
        ]
    )
    assert iem.hourly_cover(df, DEFAULT, method="max").loc[H5] == 0.75


def test_nearest_method_uses_report_closest_to_top_of_hour():
    df = obs(
        [
            ("2025-01-01 04:53", "CLR", "M", "M", "M"),  # 7 min from 05:00 -> chosen
            ("2025-01-01 05:12", "BKN", "M", "M", "M"),  # 12 min away
            ("2025-01-01 05:25", "OVC", "M", "M", "M"),
        ]
    )
    assert iem.hourly_cover(df, DEFAULT).loc[H5] == 0.0


def test_nearest_ties_go_to_the_cloudier_report():
    df = obs(
        [
            ("2025-01-01 04:50", "CLR", "M", "M", "M"),
            ("2025-01-01 05:10", "SCT", "M", "M", "M"),
        ]
    )
    assert iem.hourly_cover(df, DEFAULT).loc[H5] == 0.4375


def test_nearest_skips_reports_without_sky_data():
    df = obs(
        [
            ("2025-01-01 05:00", "M", "M", "M", "M"),  # nearest, but no sky layers
            ("2025-01-01 05:20", "BKN", "M", "M", "M"),
        ]
    )
    assert iem.hourly_cover(df, DEFAULT).loc[H5] == 0.75


def test_unknown_method_rejected():
    with pytest.raises(ValueError):
        iem.hourly_cover(obs([("2025-01-01 04:53", "CLR", "M", "M", "M")]), DEFAULT, "mean")


def test_hour_with_only_missing_obs_is_nan():
    df = obs([("2025-01-01 04:53", "M", "M", "M", "M")])
    assert np.isnan(iem.hourly_cover(df, DEFAULT).iloc[0])


def test_parse_recorded_asos_fixture(fixtures_dir):
    df = iem.parse_asos_csv((fixtures_dir / "iem_asos_SAC_20250309_20250311.csv").read_text())
    assert len(df) > 40
    assert str(df["valid"].dt.tz) == "UTC"
    hourly = iem.hourly_cover(df, DEFAULT)
    assert hourly.dropna().between(0, 1).all()
    assert hourly.index.minute.unique().tolist() == [0]


@pytest.mark.parametrize("text", ["", "station,valid\nSAC,2025-01-01 00:53\n"])
def test_bad_asos_csv_raises(text):
    with pytest.raises(BadResponseError):
        iem.parse_asos_csv(text)


def test_parse_network_fixture(fixtures_dir):
    payload = json.loads((fixtures_dir / "iem_network_CA_ASOS_subset.geojson").read_text())
    meta = iem.parse_network_geojson(payload, ["SAC", "AUN", "NOPE"])
    assert set(meta) == {"SAC", "AUN"}
    assert meta["SAC"]["lat"] == pytest.approx(38.5069)  # GeoJSON is [lon, lat]
    assert meta["SAC"]["lon"] == pytest.approx(-121.495)
    assert meta["AUN"]["is_awos"] is True
    assert meta["SAC"]["is_awos"] is False


def test_proxy_night_hours_handle_dst_without_duplicates():
    # Spring-forward night 2025-03-08 -> 03-09: local 02:00 does not exist.
    hours = iem.proxy_night_hours(
        dt.date(2025, 3, 8), dt.date(2025, 3, 8), "America/Los_Angeles", [23, 0, 1, 2, 3]
    )
    assert hours.is_unique
    assert len(hours) == 4
    assert hours.min() == pd.Timestamp("2025-03-09 07:00", tz="UTC")  # 23:00 PST


def test_night_coverage():
    idx = pd.date_range("2025-01-01 07:00", periods=4, freq="h", tz="UTC")
    hourly = pd.Series([0.0, np.nan, 1.0], index=idx[:3])
    assert iem.night_coverage(hourly, idx) == 0.5
    assert math.isnan(iem.night_coverage(hourly, idx[:0]))


def test_fetch_asos_uses_cache_and_makes_zero_calls(tmp_path, monkeypatch, fixtures_dir):
    text = (fixtures_dir / "iem_asos_SAC_20250309_20250311.csv").read_text()
    monkeypatch.setattr(iem, "raw_path", lambda *a, **k: tmp_path / "cached.csv")
    (tmp_path / "cached.csv").write_text(text)
    client = MagicMock()
    start = dt.datetime(2025, 3, 9, tzinfo=dt.UTC)
    out = iem.fetch_asos_csv(client, "https://x.test", "SAC", start, start, [3, 4])
    assert out == text
    client.get.assert_not_called()


def test_fetch_asos_writes_cache_and_sends_report_types(tmp_path, monkeypatch, fixtures_dir):
    text = (fixtures_dir / "iem_asos_SAC_20250309_20250311.csv").read_text()
    monkeypatch.setattr(iem, "raw_path", lambda *a, **k: tmp_path / "new.csv")
    client = MagicMock()
    client.get.return_value.text = text
    start = dt.datetime(2025, 3, 9, tzinfo=dt.UTC)
    iem.fetch_asos_csv(client, "https://x.test", "SAC", start, start, [3, 4])
    params = client.get.call_args.args[1]
    assert ("report_type", 3) in params and ("report_type", 4) in params
    assert (tmp_path / "new.csv").read_text() == text
