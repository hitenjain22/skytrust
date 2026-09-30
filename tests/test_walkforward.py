from __future__ import annotations

import copy
import dataclasses
import datetime as dt

import pandas as pd
import pytest

from skytrust import walkforward


@pytest.fixture(scope="module")
def wf_settings(fast_settings):
    raw = copy.deepcopy(fast_settings.raw)
    raw["walkforward"]["start"] = dt.date(2026, 1, 1)
    raw["split"]["test_end"] = dt.date(2026, 1, 31)
    raw["bootstrap_resamples"] = 50
    return dataclasses.replace(fast_settings, raw=raw)


@pytest.fixture(scope="module")
def preds(synthetic_built, wf_settings):
    return walkforward.predictions(synthetic_built[0], wf_settings, leads=[1])


def test_predictions_are_out_of_sample(preds):
    assert len(preds) > 0 and set(preds["period"]) == {"2026-01"}
    assert (pd.to_datetime(preds["night_date"]) >= "2026-01-01").all()
    for m in walkforward.METHODS:
        assert preds[m].between(0, 1).all(), m


def test_training_never_sees_the_forecast_month(synthetic_built, wf_settings, monkeypatch):
    seen = []
    real = walkforward.fit_tuned_logistic

    def spy(X, y, dates, settings, impute):
        seen.append(pd.to_datetime(dates).max())
        return real(X, y, dates, settings, impute)

    monkeypatch.setattr(walkforward, "fit_tuned_logistic", spy)
    walkforward.month_predictions(
        synthetic_built[0],
        pd.Period("2026-01", "M"),
        1,
        wf_settings,
        walkforward.baselines.climatology_table(synthetic_built[0], "usable_primary"),
    )
    assert seen and max(seen) < pd.Timestamp("2026-01-01")


def test_summary_has_pooled_cis_differences_and_monthly(preds, wf_settings, tmp_path):
    result = walkforward.summarize(preds, wf_settings)
    pooled = result["pooled"][0]
    blend = pooled["methods"]["blend"]
    assert blend["brier_lo"] <= blend["brier"] <= blend["brier_hi"]
    assert {(d["a"], d["b"]) for d in pooled["differences"]} >= {("blend", "nbm_lr")}
    assert result["monthly"][0]["period"] == "2026-01" and "bss_blend" in result["monthly"][0]
    path = walkforward.save(result, tmp_path / "wf.json")
    assert walkforward.load(path)["pooled"]
