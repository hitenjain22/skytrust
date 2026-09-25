"""Offline tests for site validation and the CLI, driven by recorded fixtures."""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
from unittest.mock import MagicMock

import yaml

from skytrust import __main__ as cli
from skytrust import sites
from skytrust.data import iem


def fake_client(fixtures_dir) -> MagicMock:
    client = MagicMock()
    client.get_json.return_value = json.loads(
        (fixtures_dir / "iem_network_CA_ASOS_subset.geojson").read_text()
    )
    client.get.return_value.text = (fixtures_dir / "iem_asos_SAC_20250309_20250311.csv").read_text()
    return client


def test_validate_sites_offline(fixtures_dir, settings, tmp_path, monkeypatch):
    monkeypatch.setattr(iem, "raw_path", lambda *a, **k: tmp_path / "x" / f"{a[1]}_{a[3]}.csv")
    # Fixture ends 2025-03-11 00:00Z, so only the night of 03-09 (morning = 03-10 UTC) is complete.
    short = dataclasses.replace(settings, history_start=dt.date(2025, 3, 9))
    checks = sites.validate_sites(fake_client(fixtures_dir), short, dt.date(2025, 3, 9))
    assert [c.id for c in checks] == ["SAC", "FAT", "AUN", "TRK", "BIH"]
    # Every station is fed the same SAC fixture, so all should pass with full coverage.
    assert all(c.passed and c.coverage_total > 0.85 for c in checks)
    aun = next(c for c in checks if c.id == "AUN")
    assert "AWOS" in aun.note

    out = sites.write_sites_yaml(checks, tmp_path / "sites.yaml")
    written = yaml.safe_load(out.read_text())["sites"]
    assert written[0]["lat"] == 38.5069 and written[0]["terrain_class"] == "Central Valley floor"


def test_missing_station_fails(settings):
    check = sites.check_site(MagicMock(), settings, "ZZZ", "nowhere", None, dt.date(2025, 1, 1))
    assert not check.passed and "not found" in check.note


def test_failed_sites_are_not_written(tmp_path):
    bad = sites.SiteCheck("ZZZ", "x", None, {}, 0.0, False)
    written = yaml.safe_load(sites.write_sites_yaml([bad], tmp_path / "s.yaml").read_text())
    assert written["sites"] == []


def test_cli_validate_sites_reports_and_exit_code(monkeypatch, capsys):
    good = sites.SiteCheck("SAC", "valley", {"name": "n"}, {2025: 0.99}, 0.99, True)
    bad = sites.SiteCheck("TRK", "mtn", {"name": "n"}, {2025: 0.5}, 0.5, False)
    monkeypatch.setattr(sites, "validate_sites", lambda *a, **k: [good, bad])
    monkeypatch.setattr(sites, "write_sites_yaml", lambda checks: "sites.yaml")
    assert cli.main(["validate-sites"]) == 1
    out = capsys.readouterr().out
    assert "SAC  PASS" in out and "TRK  FAIL" in out and "ask before substituting" in out
