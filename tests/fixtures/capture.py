"""Re-capture the small real API responses used as offline test fixtures.

Run manually (hits the network): `uv run python tests/fixtures/capture.py`
Fixtures are frozen snapshots; tests must not depend on them being re-captured.
"""

from __future__ import annotations

import json
from pathlib import Path

from skytrust.config import load_settings
from skytrust.data.http import HttpClient

HERE = Path(__file__).parent
SAC = {"latitude": 38.5069, "longitude": -121.495, "elevation": 8}
MODELS = "gfs_global,ncep_hrrr_conus,ecmwf_ifs025,gem_seamless,icon_seamless"
PREV_VARS = ",".join(["cloud_cover"] + [f"cloud_cover_previous_day{d}" for d in range(1, 8)])
LAYERS = "cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high"


def main() -> None:
    settings = load_settings()
    client = HttpClient(settings.http)
    src = settings.sources

    def save_json(name: str, url: str, params: dict) -> None:
        (HERE / name).write_text(json.dumps(client.get_json(url, params), indent=1))

    # Previous Runs: 3 days incl. the 2025-03-09 DST night, all proposed models in one call.
    save_json(
        "openmeteo_prevruns_SAC_20250308_20250310.json",
        src["prevruns_url"],
        {
            **SAC,
            "hourly": PREV_VARS,
            "models": MODELS,
            "timezone": "UTC",
            "start_date": "2025-03-08",
            "end_date": "2025-03-10",
        },
    )
    save_json(
        "openmeteo_era5_SAC_20250308_20250310.json",
        src["era5_url"],
        {
            **SAC,
            "hourly": LAYERS,
            "models": src["era5_model"],
            "timezone": "UTC",
            "start_date": "2025-03-08",
            "end_date": "2025-03-10",
        },
    )
    save_json(
        "openmeteo_forecast_SAC_live.json",
        src["forecast_url"],
        {
            **SAC,
            "models": MODELS,
            "forecast_days": 8,
            "timezone": "UTC",
            "hourly": LAYERS + ",temperature_2m,dew_point_2m,wind_speed_10m,wind_gusts_10m",
        },
    )
    # IEM: two nights of routine + SPECI reports.
    params = [("station", "SAC")] + [
        ("data", c)
        for c in ["skyc1", "skyc2", "skyc3", "skyc4", "skyl1", "skyl2", "skyl3", "skyl4", "metar"]
    ]
    params += [("report_type", r) for r in src["iem_report_types"]]
    params += [
        ("sts", "2025-03-09T00:00Z"),
        ("ets", "2025-03-11T00:00Z"),
        ("tz", "UTC"),
        ("format", "onlycomma"),
        ("elev", "yes"),
    ]
    (HERE / "iem_asos_SAC_20250309_20250311.csv").write_text(
        client.get(src["iem_asos_url"], params).text
    )
    # IEM network metadata, trimmed to the candidate stations to keep the fixture small.
    net = client.get_json(src["iem_network_url"].format(network=src["iem_network"]), {})
    keep = {c["id"] for c in settings.raw["site_validation"]["candidates"]}
    net["features"] = [f for f in net["features"] if f["properties"]["sid"] in keep]
    (HERE / "iem_network_CA_ASOS_subset.geojson").write_text(json.dumps(net, indent=1))


if __name__ == "__main__":
    main()
