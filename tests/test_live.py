"""Live forecast (SPEC 12.1 'Live'): lead assignment, API failure fallback, missing model,
plus the pure assembly on the recorded live payload."""

from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from skytrust import __main__ as cli
from skytrust import evaluate, live
from skytrust.config import load_sites
from skytrust.data import openmeteo
from skytrust.data.http import SourceUnavailableError

# The fixture was recorded 2026-09-25 00:00 UTC (Thu 17:00 PDT); SAC dusk that evening ~20:27 PDT.
NOW = pd.Timestamp("2026-09-25 01:00", tz="UTC")  # Thu 18:00 PDT, before dusk


@pytest.fixture(scope="module")
def sac():
    return next(s for s in load_sites() if s.id == "SAC")


@pytest.fixture(scope="module")
def payload(fixtures_dir_module):
    return json.loads((fixtures_dir_module / "openmeteo_forecast_SAC_live.json").read_text())


@pytest.fixture(scope="module")
def fixtures_dir_module():
    from conftest import FIXTURES

    return FIXTURES


@pytest.fixture(scope="module")
def metrics():
    return evaluate.load_metrics()


@pytest.fixture(scope="module")
def forecast(sac, payload, settings, metrics):
    return live.build_forecast(sac, payload, NOW, NOW, settings, metrics=metrics)


# ---------- lead assignment ----------

DUSK = pd.Timestamp("2026-09-25 03:27", tz="UTC")  # Thu 20:27 PDT


@pytest.mark.parametrize(
    ("now", "tonight_lead", "tomorrow_lead"),
    [
        ("2026-09-24 22:00", 1, 2),  # Thu 15:00 PDT: afternoon before dusk
        ("2026-09-25 06:00", 1, 1),  # Thu 23:00 PDT: inside the dark window
        ("2026-09-25 09:30", 1, 1),  # Fri 02:30 PDT: after midnight, before dawn
    ],
)
def test_lead_assignment(now, tonight_lead, tomorrow_lead):
    now = pd.Timestamp(now, tz="UTC")
    assert live.assign_lead(DUSK, now) == tonight_lead
    assert live.assign_lead(DUSK + pd.Timedelta(days=1), now) == tomorrow_lead


def test_lead_is_capped_at_7():
    assert live.assign_lead(DUSK + pd.Timedelta(days=12), DUSK) == 7


def test_tonight_is_the_current_dark_window_after_midnight(sac, settings):
    after_midnight = pd.Timestamp("2026-09-25 09:30", tz="UTC")  # Fri 02:30 PDT
    nights = live.upcoming_nights(sac, after_midnight, settings)
    assert str(nights.index[0]) == "2026-09-24"  # the night that began Thursday evening
    assert len(nights) == live.N_NIGHTS


# ---------- assembly on the recorded payload ----------


def test_forecast_has_seven_nights_with_valid_probabilities(forecast, settings):
    assert len(forecast.nights) == 7
    assert str(forecast.nights[0].night_date) == "2026-09-24"
    assert [n.lead for n in forecast.nights] == [1, 2, 3, 4, 5, 6, 7]
    for n in forecast.nights:
        assert n.p_usable is not None and 0 <= n.p_usable <= 1
        assert n.verdict == live.verdict_for(n.p_usable, settings)
        assert 4 <= n.dark_hours <= 13 and n.dusk_utc < n.dawn_utc
        assert n.models_used and not n.models_missing
    assert forecast.source == "live" and forecast.warning is None


def test_track_record_and_trust_come_from_metrics(forecast, metrics):
    n = forecast.nights[0]
    rec = pd.DataFrame(metrics["records"])
    expected = rec[(rec["method"] == "blend") & (rec["label"] == "primary") & (rec["lead"] == 1)
                   & (rec["subset_type"] == "site") & (rec["subset"] == "SAC")].iloc[0]  # fmt: skip
    assert n.track_record["site"]["false_clear_rate"] == pytest.approx(expected["false_clear_rate"])
    assert n.trust in {"High", "Medium", "Low"}


def test_one_model_missing_still_forecasts(sac, payload, settings, metrics):
    broken = copy.deepcopy(payload)
    for key in list(broken["hourly"]):
        if key.endswith("_icon_seamless"):
            broken["hourly"][key] = [None] * len(broken["hourly"]["time"])
    fc = live.build_forecast(sac, broken, NOW, NOW, settings, metrics=metrics)
    tonight = fc.nights[0]
    assert tonight.p_usable is not None and 0 <= tonight.p_usable <= 1
    assert "icon" in tonight.models_missing and "icon" not in tonight.models_used
    assert len(tonight.models_used) == 4


def test_no_models_at_all_gives_no_data(sac, payload, settings):
    empty = copy.deepcopy(payload)
    for key in list(empty["hourly"]):
        if key.startswith("cloud_cover"):
            empty["hourly"][key] = [None] * len(empty["hourly"]["time"])
    fc = live.build_forecast(sac, empty, NOW, NOW, settings)
    assert all(n.p_usable is None and n.verdict == "No data" for n in fc.nights)


def test_format_text_mentions_every_night(forecast):
    text = live.format_text(forecast)
    assert text.count("P(usable)") == 7 and "TONIGHT" in text and "Open-Meteo" in text


# ---------- pure helpers ----------


def test_best_window_is_longest_clear_run_of_median():
    hours = pd.date_range("2026-01-01 04:00", periods=7, freq="h", tz="UTC")
    median = np.array([0.1, 0.5, 0.0, 0.1, 0.2, np.nan, 0.0])
    bw = live.best_window(hours, median, 0.20)
    assert bw.hours == 3 and bw.start_utc == hours[2] and bw.end_utc == hours[4]
    assert live.best_window(hours, np.full(7, 0.9), 0.20) is None


def test_verdict_thresholds(settings):
    assert live.verdict_for(0.70, settings) == "Go"
    assert live.verdict_for(0.69, settings) == "Maybe"
    assert live.verdict_for(0.40, settings) == "Maybe"
    assert live.verdict_for(0.39, settings) == "Skip"
    assert live.verdict_for(None, settings) == "No data"


def test_agreement_and_trust_levels(settings):
    assert live._agreement(0.30, 0.20) == "Models split"
    assert live._agreement(0.10, 0.20) == "Models agree"
    assert live._agreement(np.nan, 0.20) == "Not enough models"
    assert live.trust_level(0.6, settings) == "High"
    assert live.trust_level(0.35, settings) == "Medium"
    assert live.trust_level(0.1, settings) == "Low"
    assert live.trust_level(None, settings) == "Unknown"


def test_track_record_without_metrics_is_none():
    assert live.track_record(None, "SAC", 1) is None


# ---------- network failure and last-good cache ----------


def test_api_failure_falls_back_to_last_good_with_timestamp(
    sac, payload, settings, tmp_path, monkeypatch
):
    earlier = pd.Timestamp("2026-09-24 20:00", tz="UTC")
    monkeypatch.setattr(openmeteo, "fetch_live", lambda c, s, site: payload)
    fc = live.get_forecast(sac, settings, now_utc=earlier, client=object(), cache_dir=tmp_path)
    assert fc.source == "live" and (tmp_path / "SAC.json").exists()

    def down(*a, **k):
        raise SourceUnavailableError("HTTP 503")

    monkeypatch.setattr(openmeteo, "fetch_live", down)
    fc2 = live.get_forecast(sac, settings, now_utc=NOW, client=object(), cache_dir=tmp_path)
    assert fc2.source == "cache"
    assert fc2.fetched_at_utc == earlier
    assert "2026-09-24 20:00 UTC" in fc2.warning
    assert len(fc2.nights) == 7


def test_api_failure_without_cache_raises_clear_error(sac, settings, tmp_path, monkeypatch):
    def down(*a, **k):
        raise SourceUnavailableError("HTTP 503")

    monkeypatch.setattr(openmeteo, "fetch_live", down)
    with pytest.raises(live.LiveUnavailableError, match="no saved copy"):
        live.get_forecast(sac, settings, now_utc=NOW, client=object(), cache_dir=tmp_path)


def test_cli_tonight(monkeypatch, forecast, capsys):
    monkeypatch.setattr(live, "get_forecast", lambda *a, **k: forecast)
    assert cli.main(["tonight", "--site", "sac"]) == 0
    assert "TONIGHT" in capsys.readouterr().out
    assert cli.main(["tonight", "--site", "XYZ"]) == 2


def test_cli_tonight_when_unavailable(monkeypatch, capsys):
    def boom(*a, **k):
        raise live.LiveUnavailableError("down and no copy")

    monkeypatch.setattr(live, "get_forecast", boom)
    assert cli.main(["tonight"]) == 1
    assert "Can't forecast right now" in capsys.readouterr().out
