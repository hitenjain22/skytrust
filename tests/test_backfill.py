from __future__ import annotations

import datetime as dt
from unittest.mock import MagicMock

import pytest

from skytrust import __main__ as cli
from skytrust.config import Site
from skytrust.data import backfill, iem, openmeteo
from skytrust.data.http import SourceUnavailableError

SITES = (
    Site("SAC", "a", 38.5, -121.5, 8.0, "valley", "America/Los_Angeles"),
    Site("BIH", "b", 37.4, -118.4, 1263.0, "desert", "America/Los_Angeles"),
)
TODAY = dt.date(2026, 9, 25)


def test_default_last_night_and_era5_cap(settings):
    assert backfill.default_last_night(settings, TODAY) == dt.date(2026, 9, 23)
    # ERA5 is capped by its publication lag...
    assert backfill.era5_end(settings, dt.date(2026, 9, 23), TODAY) == dt.date(2026, 9, 17)
    # ...but an old end date is just end + 1 (the last night's morning).
    assert backfill.era5_end(settings, dt.date(2025, 1, 1), TODAY) == dt.date(2025, 1, 2)


def test_one_site_failing_does_not_stop_others(settings, monkeypatch):
    calls = []

    def fake_era5(client, s, site, start, end, refresh, today=None):
        calls.append(site.id)
        if site.id == "SAC":
            raise SourceUnavailableError("down")
        return 1

    monkeypatch.setattr(openmeteo, "fetch_era5", fake_era5)
    client = MagicMock(n_requests=0)
    summary = backfill.fetch_source(
        client, settings, SITES, "era5", dt.date(2025, 1, 1), dt.date(2025, 1, 31), TODAY
    )
    assert calls == ["SAC", "BIH"]
    assert len(summary.failures) == 1 and summary.failures[0].startswith("SAC")


def test_prevruns_fetches_every_model_through_next_morning(settings, monkeypatch):
    seen = []
    monkeypatch.setattr(
        openmeteo,
        "fetch_prevruns",
        lambda c, s, site, m, a, b, r, today=None: seen.append((m.id, b)),
    )
    backfill.fetch_source(
        MagicMock(n_requests=0), settings, SITES[:1], "prevruns",
        dt.date(2025, 1, 1), dt.date(2025, 1, 31), TODAY,
    )  # fmt: skip
    # Blend members and benchmarks (NOAA NBM) are both downloaded.
    assert [m for m, _ in seen] == [m.id for m in settings.forecast_models]
    assert all(end == dt.date(2025, 2, 1) for _, end in seen)


def test_asos_goes_through_range_fetch(settings, monkeypatch):
    seen = []
    monkeypatch.setattr(iem, "fetch_asos_range", lambda c, u, st, a, b, rt, r: seen.append(st))
    backfill.fetch_source(
        MagicMock(n_requests=0), settings, SITES, "asos", dt.date(2025, 1, 1),
        dt.date(2025, 1, 2), TODAY,
    )  # fmt: skip
    assert seen == ["SAC", "BIH"]


def test_unknown_source_rejected(settings):
    with pytest.raises(ValueError):
        backfill.fetch_source(MagicMock(), settings, SITES, "gfs", TODAY, TODAY, TODAY)


def test_cli_fetch_reports_network_requests(monkeypatch, capsys):
    summary = backfill.FetchSummary("era5", dt.date(2024, 1, 1), dt.date(2026, 9, 23), 0, [])
    monkeypatch.setattr(backfill, "fetch_source", lambda *a, **k: summary)
    assert cli.main(["fetch", "--source", "era5", "--site", "sac"]) == 0
    assert "network requests: 0" in capsys.readouterr().out


def test_cli_fetch_unknown_site(capsys):
    assert cli.main(["fetch", "--source", "era5", "--site", "XYZ"]) == 2


def test_load_asos_dedupes_overlapping_chunks(tmp_path, fixtures_dir):
    folder = tmp_path / "asos" / "SAC" / "na"
    folder.mkdir(parents=True)
    text = (fixtures_dir / "iem_asos_SAC_20250309_20250311.csv").read_text()
    (folder / "20250309_20250310.csv").write_text(text)
    (folder / "20250309_20250311.csv").write_text(text)
    df = iem.load_asos("SAC", root=tmp_path)
    assert df["valid"].is_unique and df["valid"].is_monotonic_increasing
    assert iem.load_asos("NONE", root=tmp_path).empty


def test_fetch_asos_range_fetches_through_next_day(monkeypatch, fixtures_dir):
    text = (fixtures_dir / "iem_asos_SAC_20250309_20250311.csv").read_text()
    windows = []

    def fake_fetch(client, url, st, lo, hi, rt, refresh):
        windows.append((lo, hi))
        return text

    monkeypatch.setattr(iem, "fetch_asos_csv", fake_fetch)
    iem.fetch_asos_range(MagicMock(), "u", "SAC", dt.date(2024, 12, 1), dt.date(2025, 3, 9), [3, 4])
    assert len(windows) == 2  # one request per calendar year
    assert windows[-1][1] == dt.datetime(2025, 3, 11, tzinfo=dt.UTC)  # exclusive end
