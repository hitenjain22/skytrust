"""Orchestrates `skytrust fetch`: which dates each source needs, and per-site error handling.

Dates are in terms of *nights*. A night's dark hours run into the next UTC day, so every
source is fetched through last_night + 1 day (ERA5 is additionally capped by its lag).
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import logging
from dataclasses import dataclass, field

from skytrust.config import Settings, Site
from skytrust.data import iem, openmeteo
from skytrust.data.http import BadResponseError, HttpClient, SourceUnavailableError

log = logging.getLogger(__name__)

SOURCES = ("asos", "era5", "prevruns", "goes")


@dataclass
class FetchSummary:
    source: str
    first_night: dt.date
    last_night: dt.date
    network_requests: int = 0
    failures: list[str] = field(default_factory=list)


def default_last_night(settings: Settings, today: dt.date) -> dt.date:
    """The most recent night whose data is fully in the past for every source but ERA5."""
    return today - dt.timedelta(days=settings.sources["complete_lag_days"])


def era5_end(settings: Settings, last_night: dt.date, today: dt.date) -> dt.date:
    """ERA5 is published ~6 days late; never ask for days that can't exist yet."""
    return min(
        last_night + dt.timedelta(days=1),
        today - dt.timedelta(days=settings.sources["era5_lag_days"]),
    )


def fetch_source(
    client: HttpClient,
    settings: Settings,
    sites: tuple[Site, ...],
    source: str,
    first_night: dt.date,
    last_night: dt.date,
    today: dt.date,
    refresh: bool = False,
    members_only: bool = False,
) -> FetchSummary:
    """`members_only` skips the benchmark models (NBM) for Previous Runs: the statewide network
    only needs the blend's members."""
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}")
    summary = FetchSummary(source, first_night, last_night)
    before = client.n_requests
    src = settings.sources
    next_day = last_night + dt.timedelta(days=1)
    for site in sites:
        try:
            if source == "asos":
                iem.fetch_asos_range(
                    client, src["iem_asos_url"], site.id, first_night, last_night,
                    src["iem_report_types"], refresh,
                )  # fmt: skip
            elif source == "goes":
                continue  # handled once for all sites below (one scan serves every site)
            elif source == "era5":
                openmeteo.fetch_era5(
                    client,
                    settings,
                    site,
                    first_night,
                    era5_end(settings, last_night, today),
                    refresh,
                    today=today,
                )
            else:
                for model in settings.models if members_only else settings.forecast_models:
                    openmeteo.fetch_prevruns(
                        client, settings, site, model, first_night, next_day, refresh, today=today
                    )
            log.info("%s %s done", source, site.id)
        except (SourceUnavailableError, BadResponseError) as exc:
            # One site failing shouldn't throw away the others; re-running resumes from cache.
            log.error("%s %s failed: %s", source, site.id, exc)
            summary.failures.append(f"{site.id}: {exc}")
    if source == "goes":
        from skytrust.data import goes

        hours = goes.dark_hours_union(
            sites, first_night, last_night, settings.raw["definitions"]["sun_altitude_deg"]
        )
        # AWS S3, not a small free API: short polite delay and 4 parallel downloads are fine.
        s3 = HttpClient(dataclasses.replace(settings.http, polite_delay_s=0.05))
        goes.fetch_hours(s3, hours, sites, workers=4)
        summary.network_requests = s3.n_requests
        return summary
    summary.network_requests = client.n_requests - before
    return summary
