"""Load YAML config into typed, frozen dataclasses.

Why: thresholds and definitions must live in one place (config/settings.yaml) so that
nothing is hard-coded, and typed access catches typos at load time instead of deep
inside a pipeline run.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


@dataclass(frozen=True)
class ModelSpec:
    short: str
    id: str
    name: str
    leads: tuple[int, ...]


@dataclass(frozen=True)
class HttpSettings:
    timeout_s: float
    max_attempts: int
    backoff_base_s: float
    polite_delay_s: float
    user_agent: str


@dataclass(frozen=True)
class Site:
    id: str
    name: str
    lat: float
    lon: float
    elevation_m: float
    terrain_class: str
    timezone: str


@dataclass(frozen=True)
class Settings:
    """Typed view of settings.yaml. `raw` keeps the full dict for sections not yet typed."""

    clear_threshold: float
    min_run_hours: int
    max_missing_frac: float
    sky_cover_mapping: dict[str, float]
    asos_hour_aggregation: str
    models: tuple[ModelSpec, ...]
    http: HttpSettings
    history_start: dt.date
    raw: dict[str, Any]

    @property
    def sources(self) -> dict[str, Any]:
        return self.raw["sources"]


def load_settings(path: Path | None = None) -> Settings:
    path = path or CONFIG_DIR / "settings.yaml"
    raw = yaml.safe_load(path.read_text())
    defs = raw["definitions"]
    mode = defs["sky_cover_mode"]
    if mode not in defs["sky_cover_mapping"]:
        raise ValueError(f"sky_cover_mode {mode!r} not in sky_cover_mapping")
    clear = float(defs["clear_threshold"])
    if not 0.0 <= clear <= 1.0:
        raise ValueError(f"clear_threshold must be on a 0-1 scale, got {clear}")
    return Settings(
        clear_threshold=clear,
        min_run_hours=int(defs["min_run_hours"]),
        max_missing_frac=float(defs["max_missing_frac"]),
        sky_cover_mapping={k: float(v) for k, v in defs["sky_cover_mapping"][mode].items()},
        asos_hour_aggregation=defs["asos_hour_aggregation"],
        models=tuple(
            ModelSpec(m["short"], m["id"], m["name"], tuple(m["leads"])) for m in raw["models"]
        ),
        http=HttpSettings(**raw["http"]),
        history_start=raw["sources"]["history_start"],
        raw=raw,
    )


def load_sites(path: Path | None = None) -> tuple[Site, ...]:
    path = path or CONFIG_DIR / "sites.yaml"
    raw = yaml.safe_load(path.read_text())
    return tuple(Site(**s) for s in raw["sites"])
