"""Live API contract tests (SPEC 12: opt-in with `pytest -m network`; run weekly in CI).

Each test pins down an assumption recorded in docs/DATA_NOTES.md against the live service, so a
silent upstream change (a renamed model, a new variable name, a longer ERA5 lag) fails loudly.
Requests are small and few, to stay polite to free services.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from skytrust import forward
from skytrust.config import load_settings, load_sites
from skytrust.data import iem, openmeteo
from skytrust.data.http import HttpClient

pytestmark = pytest.mark.network

SETTINGS = load_settings()
SITES = load_sites()
SAC = next(s for s in SITES if s.id == "SAC")


@pytest.fixture(scope="module")
def client():
    return HttpClient(SETTINGS.http)


def test_iem_network_lists_every_site(client):
    src = SETTINGS.sources
    meta = iem.parse_network_geojson(
        iem.fetch_network(client, src["iem_network_url"], src["iem_network"]), [s.id for s in SITES]
    )
    assert set(meta) == {s.id for s in SITES}


def test_iem_routine_and_special_reports(client):
    src = SETTINGS.sources
    end = dt.datetime.now(dt.UTC).replace(minute=0, second=0, microsecond=0)
    params = [("station", "SAC"), *[("data", c) for c in iem.ASOS_DATA_COLS]]
    params += [("report_type", r) for r in src["iem_report_types"]]
    params += [
        ("sts", f"{end - dt.timedelta(days=2):%Y-%m-%dT%H:%MZ}"),
        ("ets", f"{end:%Y-%m-%dT%H:%MZ}"),
        ("tz", "UTC"),
        ("format", "onlycomma"),
    ]
    df = iem.parse_asos_csv(client.get(src["iem_asos_url"], params).text)
    assert len(df) >= 40  # ~24 routine reports a day, plus specials
    assert df["valid"].dt.minute.mode().iloc[0] == 53  # SAC's routine minute, not 5-min data


@pytest.mark.parametrize(
    "model", [*SETTINGS.models, "ncep_nbm_conus"], ids=lambda m: getattr(m, "id", m)
)
def test_previous_runs_model_and_leads(client, model):
    model_id = getattr(model, "id", model)
    leads = getattr(model, "leads", tuple(range(1, 8)))
    end = dt.datetime.now(dt.UTC).date() - dt.timedelta(days=2)
    payload = client.get_json(
        SETTINGS.sources["prevruns_url"],
        {
            **openmeteo.location_params(SAC),
            "models": model_id,
            "hourly": ",".join(openmeteo.prevruns_var(d) for d in leads),
            "start_date": f"{end - dt.timedelta(days=6):%Y-%m-%d}",
            "end_date": f"{end:%Y-%m-%d}",
        },
    )
    df = openmeteo.parse_hourly(payload, [openmeteo.prevruns_var(d) for d in leads])
    assert df.notna().mean().min() > 0.9, df.notna().mean().to_dict()


def test_era5_is_explicit_and_lag_is_within_config(client):
    today = dt.datetime.now(dt.UTC).date()
    payload = client.get_json(
        SETTINGS.sources["era5_url"],
        {
            **openmeteo.location_params(SAC),
            "models": SETTINGS.sources["era5_model"],
            "hourly": "cloud_cover",
            "start_date": f"{today - dt.timedelta(days=20):%Y-%m-%d}",
            "end_date": f"{today:%Y-%m-%d}",
        },
    )
    series = openmeteo.parse_hourly(payload, ["cloud_cover"])["cloud_cover"].dropna()
    lag_days = (pd.Timestamp(today, tz="UTC") - series.index.max()).days
    assert lag_days <= SETTINGS.sources["era5_lag_days"], f"ERA5 lag is now {lag_days} days"


def test_live_forecast_has_every_model_and_layer(client):
    payload = openmeteo.fetch_live(client, SETTINGS, SAC)
    for m in SETTINGS.models:
        df = openmeteo.parse_hourly(payload, openmeteo.LIVE_CLOUD_VARS, m.id)
        horizon_h = df["cloud_cover"].notna().sum()
        assert horizon_h >= (24 if m.short == "hrrr" else 6 * 24), (m.id, horizon_h)
        assert df["cloud_cover_high"].notna().any(), m.id


def test_ensemble_member_counts(client, settings=SETTINGS):
    fwd = settings.raw["forward"]
    payload = client.get_json(
        fwd["ensemble_url"],
        {
            **openmeteo.location_params(SAC),
            "models": ",".join(fwd["ensemble_models"]),
            "hourly": "cloud_cover",
            "forecast_days": 8,
        },
    )
    assert len(forward.ensemble_members(payload, "ecmwf_ifs025_ensemble")) == 51
    assert len(forward.ensemble_members(payload, "ncep_gefs025")) == 31
