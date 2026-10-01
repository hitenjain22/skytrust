"""Statewide verification network: airport weather stations across all of California.

Why: the five evaluated airports are all inland Northern California (the Sacramento and San
Joaquin valleys, the Sierra foothills, Truckee, Bishop). Nothing measured how the forecasts do on
the coast (the marine layer), in Southern California or in the low deserts, yet the app forecasts
for all of them. This module picks a statewide network of ASOS stations by a fixed rule, so the
choice can't be tuned to flatter the results:

1. **Candidates:** IEM `CA_ASOS` stations that are online, have archived since before the
   history start (so the whole 2024- record exists), and are on the mainland.
2. **Regions:** the National Weather Service forecast office (WFO) responsible for the station.
   WFO areas follow climate and terrain (the North Coast, the Bay Area and Central Coast, the
   Sacramento and San Joaquin valleys, the Southern California coast and inland valleys, the
   Mojave and Colorado deserts, the east side of the Sierra) and IEM lists the office for every
   station, so no map work is needed.
3. **Quota per region:** max(min_per_region, round(online stations / stations_per_quota)). Regions
   with more airports (more people) get more stations, and every region gets some.
4. **Choice within the quotas:** a greedy maximin ("farthest point") design. Start from the five
   evaluated airports, then keep adding the candidate (from any region with quota left) that is
   farthest from every station chosen so far. Distance mixes ground distance and height
   (`km_per_m`: 1,000 m of elevation counts as 100 km), because a mountain station 30 km from a
   valley station sees very different weather.
5. **Coverage:** each pick must pass the same check as the original airports (>= 85 % of night
   hours with a valid sky report since the history start); one that fails is dropped and the next
   farthest candidate is taken.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from skytrust.config import CONFIG_DIR, Settings, Site

log = logging.getLogger(__name__)

NETWORK_PATH = CONFIG_DIR / "network.yaml"

# What each NWS forecast office covers in California, for people reading the results.
REGION_NAMES = {
    "EKA": "North Coast",
    "MFR": "Far north interior",
    "STO": "Sacramento Valley and foothills",
    "REV": "Tahoe, Mono and the northeast",
    "MTR": "Bay Area and Central Coast",
    "HNX": "San Joaquin Valley and Kern",
    "LOX": "Los Angeles to San Luis Obispo",
    "SGX": "San Diego, Orange County and the Inland Empire",
    "VEF": "Mojave Desert and Inyo",
    "PSR": "Imperial Valley and the Colorado River",
}


@dataclass(frozen=True)
class Station:
    id: str
    name: str
    lat: float
    lon: float
    elevation_m: float
    region: str  # NWS forecast office (WFO) id
    county: str
    timezone: str
    archive_begin: str | None = None
    online: bool = True


def stations_from_geojson(payload: dict[str, Any]) -> list[Station]:
    """Every station in IEM's network GeoJSON (coordinates are [lon, lat])."""
    out = []
    for feat in payload.get("features", []):
        p = feat["properties"]
        lon, lat = feat["geometry"]["coordinates"][:2]
        out.append(
            Station(
                id=p["sid"],
                name=p["sname"],
                lat=float(lat),
                lon=float(lon),
                elevation_m=float(p["elevation"]),
                region=p.get("wfo") or "",
                county=p.get("county") or "",
                timezone=p.get("tzname") or "America/Los_Angeles",
                archive_begin=p.get("archive_begin"),
                online=bool(p.get("online")),
            )
        )
    return out


def candidates(stations: list[Station], cfg: dict, history_start: dt.date) -> list[Station]:
    """Online mainland stations whose archive covers the whole history period."""
    exclude = set(cfg.get("exclude", []))
    out = []
    for s in stations:
        if not s.online or s.id in exclude or not s.region:
            continue
        if s.archive_begin is None or dt.date.fromisoformat(s.archive_begin) > history_start:
            continue
        out.append(s)
    return out


def quotas(stations: list[Station], cfg: dict) -> dict[str, int]:
    """Stations to pick per region, from the number of online stations there. Rounds halves up
    (Python's round() would send 4.5 to 4)."""
    counts: dict[str, int] = {}
    for s in stations:
        if s.online and s.region:
            counts[s.region] = counts.get(s.region, 0) + 1
    per, floor_ = cfg["stations_per_quota"], cfg["min_per_region"]
    return {r: max(floor_, math.floor(n / per + 0.5)) for r, n in sorted(counts.items())}


def design_distance_km(a: Station, b: Station, km_per_m: float) -> float:
    """Ground distance (haversine) combined with height difference scaled to km."""
    la1, la2 = math.radians(a.lat), math.radians(b.lat)
    dla, dlo = la2 - la1, math.radians(b.lon - a.lon)
    h = math.sin(dla / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dlo / 2) ** 2
    ground = 2 * 6371.0 * math.asin(math.sqrt(min(1.0, h)))
    return math.hypot(ground, (a.elevation_m - b.elevation_m) * km_per_m)


def select(
    pool: list[Station],
    seeds: list[Station],
    quota: dict[str, int],
    km_per_m: float,
    passes: Callable[[Station], bool] = lambda s: True,
) -> list[Station]:
    """Greedy maximin: the seeds, then repeatedly the pool station farthest from all chosen
    stations, among regions with quota left. `passes` is asked only about stations that would be
    picked (the coverage check downloads data, so it isn't run on all of them). Ties go to the
    alphabetically first id, so the result is deterministic."""
    chosen = list(seeds)
    left = dict(quota)
    for s in seeds:
        left[s.region] = left.get(s.region, 0) - 1
    seed_ids = {s.id for s in seeds}
    remaining = sorted((s for s in pool if s.id not in seed_ids), key=lambda s: s.id)
    while True:
        open_ = [s for s in remaining if left.get(s.region, 0) > 0]
        if not open_:
            return chosen
        if chosen:
            score = [min(design_distance_km(s, c, km_per_m) for c in chosen) for s in open_]
        else:  # no seeds: start from the most central station
            lat, lon = np.mean([s.lat for s in open_]), np.mean([s.lon for s in open_])
            centre = Station("", "", float(lat), float(lon), 0.0, "", "", "")
            score = [-design_distance_km(s, centre, 0.0) for s in open_]
        best = open_[int(np.argmax(score))]
        remaining.remove(best)
        if passes(best):
            chosen.append(best)
            left[best.region] -= 1
        else:
            log.info("network: %s fails the coverage check; trying the next station", best.id)


class CoverageCheck:
    """The site-validation rule (SPEC 5) applied to a candidate, remembering each station's
    coverage for network.yaml."""

    def __init__(self, client, settings: Settings, end: dt.date):
        self.client, self.settings, self.end = client, settings, end
        self.results: dict[str, float] = {}

    def __call__(self, s: Station) -> bool:
        from skytrust import sites

        meta = {"name": s.name, "timezone": s.timezone, "is_awos": False}
        check = sites.check_site(self.client, self.settings, s.id, s.region, meta, self.end)
        self.results[s.id] = check.coverage_total
        log.info("network: %s %s coverage %.1f%%", s.id, s.name, 100 * check.coverage_total)
        return check.passed


def write_network_yaml(
    chosen: list[Station],
    seeds: set[str],
    coverage: dict[str, float],
    quota: dict[str, int],
    path: Path = NETWORK_PATH,
) -> Path:
    rows = [
        {
            "id": s.id,
            "name": s.name,
            "lat": s.lat,
            "lon": s.lon,
            "elevation_m": s.elevation_m,
            "region": s.region,
            "county": s.county,
            "timezone": s.timezone,
            "evaluated_airport": s.id in seeds,
            "night_coverage": None if s.id not in coverage else round(coverage[s.id], 4),
        }
        for s in chosen
    ]
    header = (
        "# Statewide verification network, generated by `python -m skytrust network-select`\n"
        "# from IEM CA_ASOS metadata (see src/skytrust/network.py for the selection rule).\n"
    )
    path.write_text(header + yaml.safe_dump({"quotas": quota, "stations": rows}, sort_keys=False))
    return path


def load_network(path: Path = NETWORK_PATH) -> tuple[Site, ...]:
    """The network as Sites (`terrain_class` carries the region id)."""
    raw = yaml.safe_load(path.read_text())
    return tuple(
        Site(
            s["id"],
            s["name"],
            float(s["lat"]),
            float(s["lon"]),
            float(s["elevation_m"]),
            s["region"],
            s["timezone"],
        )  # fmt: skip
        for s in raw["stations"]
    )


def load_network_table(path: Path = NETWORK_PATH) -> list[dict]:
    return list(yaml.safe_load(path.read_text())["stations"])
