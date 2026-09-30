"""Forward (prospective) verification: benchmark parsing, append-only log, and scoring."""

from __future__ import annotations

import copy
import dataclasses
import datetime as dt
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from conftest import FIXTURES
from skytrust import __main__ as cli
from skytrust import astro, forward, live
from skytrust.config import load_sites
from skytrust.nightly import NightRules

RULES = NightRules(clear_threshold=0.20, min_run_hours=3, max_missing_frac=0.25)


@pytest.fixture(scope="module")
def sac():
    return next(s for s in load_sites() if s.id == "SAC")


@pytest.fixture(scope="module")
def ensemble_payload():
    return json.loads((FIXTURES / "openmeteo_ensemble_SAC_live.json").read_text())


@pytest.fixture(scope="module")
def nbm_payload():
    return json.loads((FIXTURES / "openmeteo_nbm_SAC_live.json").read_text())


def fake_forecast(site, first: dt.date, n: int = 6):
    """Minimal stand-in exposing what benchmark_columns needs: nights with dusk/dawn."""
    w = astro.dark_windows(site, first, first + dt.timedelta(days=n - 1))
    nights = [
        SimpleNamespace(night_date=d, dusk_utc=r.dusk_utc, dawn_utc=r.dawn_utc)
        for d, r in w.iterrows()
    ]
    return SimpleNamespace(nights=nights, site=site)


# ---------- benchmarks ----------


def test_ensemble_members_parsed_including_control(ensemble_payload):
    ecmwf = forward.ensemble_members(ensemble_payload, "ecmwf_ifs025_ensemble")
    gefs = forward.ensemble_members(ensemble_payload, "ncep_gefs025")
    assert len(ecmwf) == 51 and "control" in ecmwf and "member01" in ecmwf
    assert len(gefs) == 31
    assert all(s.dropna().between(0, 1).all() for s in ecmwf.values())


def test_ensemble_probability_is_share_of_usable_members():
    hours = pd.date_range("2026-01-02 03:00", periods=5, freq="h", tz="UTC")
    nh = pd.DataFrame({"night_date": dt.date(2026, 1, 1), "hour": hours})
    members = {
        "a": pd.Series([0.0] * 5, index=hours),  # usable
        "b": pd.Series([0.0, 0.0, 0.0, 0.9, 0.9], index=hours),  # usable
        "c": pd.Series([0.9] * 5, index=hours),  # not usable
        "d": pd.Series([np.nan] * 5, index=hours),  # no data -> doesn't vote
    }
    p, n = forward.ensemble_p_usable(members, nh, RULES)
    assert p.iloc[0] == pytest.approx(2 / 3) and n.iloc[0] == 3


def test_benchmark_columns_from_recorded_payloads(sac, ensemble_payload, nbm_payload, settings):
    fc = fake_forecast(sac, dt.date(2026, 9, 30))
    cols = forward.benchmark_columns(
        sac, fc, {"nbm": nbm_payload, "ensemble": ensemble_payload}, settings
    )
    assert len(cols) == 6
    for c in ["nbm_frac_clear", "nbm_pred_usable", "ens_ecmwf_p_usable", "ens_gefs_p_usable"]:
        assert cols[c].notna().all() and cols[c].between(0, 1).all(), c
    assert (cols["ens_ecmwf_members"] == 51).all() and (cols["ens_gefs_members"] == 31).all()


def test_benchmark_failure_leaves_columns_absent(sac, settings):
    cols = forward.benchmark_columns(sac, fake_forecast(sac, dt.date(2026, 9, 30)), {}, settings)
    assert cols.empty or cols.columns.empty


# ---------- logging ----------


@pytest.fixture(scope="module")
def live_forecast(sac, settings):
    payload = json.loads((FIXTURES / "openmeteo_forecast_SAC_live.json").read_text())
    now = pd.Timestamp("2026-09-25 01:00", tz="UTC")
    return live.build_forecast(sac, payload, now, now, settings)


def test_forecast_rows_capture_what_the_app_showed(live_forecast):
    rows = forward.forecast_rows(live_forecast, pd.Timestamp("2026-09-25 01:00", tz="UTC"))
    assert len(rows) == 7 and rows["lead"].tolist() == [1, 2, 3, 4, 5, 6, 7]
    assert str(rows["issue_date"].iloc[0]) == "2026-09-24"  # local (PDT) date of issuance
    assert {"gfs_frac_clear", "hrrr_frac_clear", "ecmwf_mean_cover"} <= set(rows.columns)
    assert rows["p_usable"].between(0, 1).all() and rows["blend_version"].notna().all()


def test_append_log_is_append_only_and_first_forecast_wins(tmp_path):
    path = tmp_path / "log.csv"
    day1 = pd.DataFrame(
        {
            "issue_date": ["2026-10-01"] * 2,
            "site": ["SAC", "BIH"],
            "night_date": ["2026-10-01"] * 2,
            "p_usable": [0.3, 0.8],
        }
    )
    assert forward.append_log(path, day1) == 2
    rerun = day1.assign(p_usable=[0.9, 0.1])  # same day, same nights: must NOT overwrite
    assert forward.append_log(path, rerun) == 0
    day2 = day1.assign(issue_date=["2026-10-02"] * 2)
    assert forward.append_log(path, day2) == 2
    saved = pd.read_csv(path)
    assert len(saved) == 4
    assert saved.loc[saved["issue_date"] == "2026-10-01", "p_usable"].tolist() == [0.8, 0.3]


def test_log_forecasts_offline(monkeypatch, tmp_path, live_forecast, settings, sac):
    monkeypatch.setattr(live, "get_forecast", lambda *a, **k: live_forecast)
    monkeypatch.setattr(forward, "fetch_benchmarks", lambda c, s, site: {})
    now = pd.Timestamp("2026-09-25 01:00", tz="UTC")
    added = forward.log_forecasts(settings, (sac,), tmp_path, now_utc=now, client=object())
    assert added == 7
    saved = pd.read_csv(tmp_path / forward.LOG_FILE)
    assert "code_version" in saved and saved["site"].eq("SAC").all()


def test_log_forecasts_skips_site_without_forecast(monkeypatch, tmp_path, settings, sac):
    def down(*a, **k):
        raise live.LiveUnavailableError("down")

    monkeypatch.setattr(live, "get_forecast", down)
    assert forward.log_forecasts(settings, (sac,), tmp_path, client=object()) == 0


# ---------- verification ----------


def practice_settings(settings, start: dt.date):
    raw = copy.deepcopy(settings.raw)
    raw["forward"]["start_date"] = start
    raw["min_weeks_for_ci"] = 2
    raw["bootstrap_resamples"] = 50
    return dataclasses.replace(settings, raw=raw)


def make_log(tmp_path, n_days: int = 21):
    """Synthetic log: each issue day, SAC forecasts for tonight (lead 1) and tomorrow (lead 2)."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_days):
        issue = dt.date(2026, 8, 1) + dt.timedelta(days=i)
        for lead in (1, 2):
            p = float(rng.uniform())
            rows.append(
                {
                    "issue_date": str(issue),
                    "site": "SAC",
                    "night_date": str(issue + dt.timedelta(days=lead - 1)),
                    "lead": lead,
                    "p_usable": p,
                    "nbm_frac_clear": float(rng.uniform()),
                    "ens_ecmwf_p_usable": p,
                    "ens_gefs_p_usable": 1 - p,
                }
            )
    pd.DataFrame(rows).to_csv(tmp_path / forward.LOG_FILE, index=False)


def fake_labels(settings, sites, first, last, today, client):
    nights = pd.date_range(first, last).date
    usable = pd.array([i % 3 != 0 for i in range(len(nights))], dtype="boolean")
    return pd.DataFrame(
        {
            "night_date": nights,
            "site": "SAC",
            "usable_primary": usable,
            "usable_asos": usable,
            "usable_era5": usable,
            "exclusion_primary": pd.NA,
        }
    )


def test_verify_scores_only_old_enough_nights_after_start(monkeypatch, tmp_path, settings, sac):
    make_log(tmp_path)
    monkeypatch.setattr(forward, "observed_labels", fake_labels)
    s = practice_settings(settings, dt.date(2026, 8, 5))
    summary = forward.verify(s, (sac,), tmp_path, today=dt.date(2026, 8, 25), client=object())
    # issues from Aug 5 on (start) and nights up to Aug 16 (25 - 9 days)
    assert summary["n_logged"] == 17 * 2
    verified = pd.read_csv(tmp_path / forward.VERIFIED_FILE)
    assert pd.to_datetime(verified["night_date"]).max() <= pd.Timestamp("2026-08-16")
    assert pd.to_datetime(verified["issue_date"]).min() >= pd.Timestamp("2026-08-05")
    leads = {e["lead"]: e for e in summary["by_lead"]}
    assert set(leads) == {"all", 1, 2}
    blend = leads["all"]["methods"]["blend"]
    assert 0 <= blend["brier"] <= 1 and blend["go_calls_usable"] <= blend["go_calls"]
    assert {"climatology", "nbm_frac_clear", "ens_ecmwf", "ens_gefs"} <= set(leads[1]["methods"])
    assert "backtest" in leads[1]  # expectation from metrics.json shown alongside
    text = (tmp_path / forward.SUMMARY_FILE).read_text()
    assert "NaN" not in text and json.loads(text)["n_verified"] == summary["n_verified"]


def test_verify_before_anything_is_ready(tmp_path, settings, sac):
    make_log(tmp_path, n_days=3)
    s = practice_settings(settings, dt.date(2026, 8, 1))
    summary = forward.verify(s, (sac,), tmp_path, today=dt.date(2026, 8, 4))
    assert summary["n_verified"] == 0 and summary["by_lead"] == []


def test_confidence_intervals_only_with_enough_weeks(settings):
    y = np.array([1, 0, 1, 1, 0, 1, 0, 1], dtype=float)
    p = np.linspace(0.1, 0.9, 8)
    one_week = np.zeros(8, dtype=int)
    assert "brier_lo" not in forward._score(y, p, np.full(8, 0.5), one_week, settings)
    raw = copy.deepcopy(settings.raw)
    raw["min_weeks_for_ci"] = 2
    raw["bootstrap_resamples"] = 50
    many = dataclasses.replace(settings, raw=raw)
    out = forward._score(y, p, np.full(8, 0.5), np.arange(8) % 4, many)
    assert out["brier_lo"] <= out["brier"] <= out["brier_hi"]


def test_cli_forward_commands(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(forward, "log_forecasts", lambda *a, **k: 35)
    assert cli.main(["forward-log", "--out", str(tmp_path)]) == 0
    monkeypatch.setattr(
        forward,
        "verify",
        lambda *a, **k: {"n_logged": 35, "n_verified": 0, "forward_start": "2026-09-30"},
    )
    assert cli.main(["forward-verify", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "Logged 35" in out and "0 verified" in out
