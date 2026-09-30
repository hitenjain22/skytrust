"""Robustness of the conclusions to the cloud definitions (clear threshold x minimum run).

For each definition in the grid, the whole pipeline is re-run in memory: relabel every night,
recompute the forecast features, retrain the blend on the training years, and score everything
on the test period with the usual paired week-block bootstrap. If the conclusions (the blend
beats the best single model and NOAA's NBM) only held for one hand-picked threshold, this is
where it would show.

The long-term climatology was built with the default definition, so each grid cell uses its own
training-years climatology as the skill reference, keeping every cell internally consistent.
"""

from __future__ import annotations

import copy
import dataclasses
import itertools
import json
import logging
from pathlib import Path

import pandas as pd

from skytrust import astro, baselines, blend, dataset, evaluate, features, inference, labels
from skytrust.config import REPO_ROOT, Settings, Site
from skytrust.data.cache import RAW_DIR
from skytrust.modeling import split_train_test

log = logging.getLogger(__name__)

SENSITIVITY_PATH = REPO_ROOT / "artifacts" / "sensitivity.json"
CLEAR_THRESHOLDS = (0.10, 0.20, 0.30)
MIN_RUNS = (2, 3, 4)
COMPARISONS = {"best_single": None, "nbm_lr": "nbm_lr", "equal_weight": "equal_weight"}


def with_definition(settings: Settings, clear: float, run: int) -> Settings:
    raw = copy.deepcopy(settings.raw)
    raw["definitions"]["clear_threshold"] = clear
    raw["definitions"]["min_run_hours"] = run
    raw["climatology"]["use_long_term"] = False
    return dataclasses.replace(settings, clear_threshold=clear, min_run_hours=run, raw=raw)


def site_inputs(settings: Settings, sites: tuple[Site, ...], first, last, root: Path) -> dict:
    """Everything that doesn't depend on the cloud definition, computed once per site."""
    out = {}
    for site in sites:
        out[site.id] = {
            "astro": astro.night_table(
                site,
                first,
                last,
                settings.raw["definitions"]["sun_altitude_deg"],
                settings.raw["astro"]["moon_up_altitude_deg"],
            ),
            "observed": labels.load_observed_hourly(site.id, settings, root),
            "forecasts": features.load_forecasts(site.id, settings.forecast_models, root),
        }
    return out


def score_cell(df: pd.DataFrame, cfg: Settings, leads: list[int], seed: int) -> list[dict]:
    rows = []
    train, _ = split_train_test(df)
    for lead in leads:
        result = baselines.run_baselines(df, "primary", lead, cfg)
        artifact = blend.fit_blend(train, "primary", lead, cfg)
        p = inference.predict_proba(artifact, inference.raw_inputs(result.features, artifact))
        result.methods["blend"] = baselines.MethodPrediction("blend", "prob", "blend", p)
        ev = evaluate.evaluate_lead(result, cfg, seed=seed + lead)
        rec = next(
            r for r in ev["records"] if r["method"] == "blend" and r["subset_type"] == "overall"
        )
        entry = {
            "lead": lead,
            "n_eval": ev["n_eval"],
            "base_rate": rec["base_rate"],
            **{k: rec.get(k) for k in ["bss", "bss_lo", "bss_hi", "false_clear_rate", "brier"]},
        }
        for name, target in COMPARISONS.items():
            b = result.best_single if target is None else target
            d = next((x for x in ev["differences"] if x["a"] == "blend" and x["b"] == b), None)
            if d:
                entry[f"vs_{name}"] = {k: d[k] for k in ["brier_diff", "lo", "hi", "significant"]}
        rows.append(entry)
    return rows


def run(
    settings: Settings,
    sites: tuple[Site, ...],
    leads: list[int] | None = None,
    root: Path = RAW_DIR,
    first=None,
) -> dict:
    leads = leads or settings.raw["leads"]
    first = first or settings.history_start
    last = dataset.default_last_night(sites, settings, root)
    inputs = site_inputs(settings, sites, first, last, root)
    cells = []
    for clear, run_h in itertools.product(CLEAR_THRESHOLDS, MIN_RUNS):
        cfg = with_definition(settings, clear, run_h)
        log.info("sensitivity: clear <= %.2f, run >= %d h", clear, run_h)
        df = pd.concat(
            [
                dataset.build_site_dataset(
                    s,
                    cfg,
                    first,
                    last,
                    inputs[s.id]["observed"],
                    inputs[s.id]["forecasts"],
                    inputs[s.id]["astro"],
                )
                for s in sites
            ],
            ignore_index=True,
        )
        is_default = (clear == settings.clear_threshold) and (run_h == settings.min_run_hours)
        cells.append(
            {
                "clear_threshold": clear,
                "min_run_hours": run_h,
                "is_default": is_default,
                "leads": score_cell(df, cfg, leads, int(settings.raw["seed"])),
            }
        )
    return {
        "grid": {"clear_threshold": list(CLEAR_THRESHOLDS), "min_run_hours": list(MIN_RUNS)},
        "label": "primary",
        "cells": cells,
    }


def save(result: dict, path: Path = SENSITIVITY_PATH) -> Path:
    path.write_text(json.dumps(evaluate._clean(result), indent=1))
    return path


def load(path: Path = SENSITIVITY_PATH) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None
