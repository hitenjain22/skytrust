"""Accuracy across California: the statewide verification network's evaluation.

The plan (DECISIONS 2026-10-01) was written before any result was seen:

1. **Leave-one-region-out (LORO):** for each of the ten NWS forecast regions, train the
   site-agnostic blend on the *other* regions' stations (training years only) and forecast the
   held-out region's test nights. Every station is scored as a place the model never saw.
2. **Leave-one-station-out (LOSO)** as a cross-check: neighbours in the same region can share
   weather, so if LORO and LOSO agree, the transfer to new places is robust.
3. On identical nights: the five-airport geo blend that shipped until now, the equal-weight
   average (no training at all), each member model calibrated the same leave-region-out way, and
   climatology (each station's training-years rate per month, the skill reference).
4. Week-block bootstrap CIs (all stations' nights in a week are resampled together, which also
   respects that one storm covers many stations) and paired differences.
5. The same with the ASOS-only and ERA5-only labels (sensitivity).

The shipped statewide blend is then fit on every station's training years (`train_final`).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from skytrust import baselines, blend, dataset, evaluate, inference, network
from skytrust.config import REPO_ROOT, Settings, load_sites
from skytrust.modeling import design_matrix, split_train_test

log = logging.getLogger(__name__)

NETWORK_DATASET = REPO_ROOT / "data" / "processed" / "network_dataset.parquet"
STATEWIDE_PATH = inference.ARTIFACTS / "statewide.json"
AIRPORT_GEO_DIR = inference.ARTIFACTS / "geo_airports"  # the five-airport geo blend, kept
# The statewide blend is written here first; it replaces artifacts/geo only after the shipping
# rule in DECISIONS (2026-10-01) has been checked against the results.
CANDIDATE_DIR = inference.ARTIFACTS / "geo_statewide"
LABEL_COLS = {"primary": "usable_primary", "asos": "usable_asos", "era5": "usable_era5"}
METHODS = ["statewide_loro", "statewide_loso", "geo_airports", "equal_weight", "climatology"]


# ---------- data ----------


def build(settings: Settings) -> pd.DataFrame:
    """The modelling dataset for every network station (same code as the five airports)."""
    sites = network.load_network()
    return dataset.build_dataset(
        settings, sites, settings.history_start, settings.raw["split"]["test_end"]
    )


def regions(path: Path = network.NETWORK_PATH) -> dict[str, str]:
    return {s["id"]: s["region"] for s in network.load_network_table(path)}


def evaluation_rows(test: pd.DataFrame, label_col: str, lead: int, settings: Settings):
    """Labelled test rows with every member model present, so every method is scored on
    exactly the same nights (NBM, a benchmark at the original airports only, is not needed)."""
    rows = test[(test["lead"] == lead) & test[label_col].notna()]
    keep = np.ones(len(rows), dtype=bool)
    for m in baselines.models_at_lead(settings, lead):
        keep &= rows[baselines.single_model_columns(m)].notna().all(axis=1).to_numpy()
    return rows[keep].sort_values(["night_date", "site"]).reset_index(drop=True)


# ---------- predictions ----------


def _geo(train: pd.DataFrame, label: str, lead: int, settings: Settings) -> dict:
    return blend.fit_blend(train, label, lead, settings, use_site=False)


def _predict(artifact: dict, rows: pd.DataFrame) -> np.ndarray:
    return inference.predict_proba(artifact, inference.raw_inputs(rows, artifact))


def held_out(
    train: pd.DataFrame,
    rows: pd.DataFrame,
    groups_train: pd.Series,
    groups_rows: pd.Series,
    fit,
    predict,
) -> np.ndarray:
    """For each group: fit on training rows of the *other* groups, predict this group's rows.
    The spy test in tests/test_statewide.py checks a group's rows never reach its own fit."""
    p = np.full(len(rows), np.nan)
    for g in sorted(groups_rows.unique()):
        model = fit(train[(groups_train != g).to_numpy()])
        mask = (groups_rows == g).to_numpy()
        p[mask] = predict(model, rows[mask])
    return p


def _single_model_fit(model_spec, label_col: str, lead: int, settings: Settings):
    def fit(tr: pd.DataFrame):
        return baselines.fit_single_model(tr, model_spec, label_col, lead, settings, [])

    def predict(tuned, rows: pd.DataFrame) -> np.ndarray:
        X = design_matrix(rows, baselines.single_model_columns(model_spec), [])
        return tuned.pipeline.predict_proba(X)[:, 1]

    return fit, predict


def predictions(
    df: pd.DataFrame,
    settings: Settings,
    label: str,
    lead: int,
    region_of: dict[str, str],
    airport_geo: dict | None = None,
    loso: bool = True,
) -> pd.DataFrame:
    """Every method's forecast for one (label, lead) on the common evaluation nights."""
    train, test = split_train_test(df)
    label_col = LABEL_COLS[label]
    rows = evaluation_rows(test, label_col, lead, settings)
    out = rows[["site", "night_date", "lead", "month"]].copy()
    out["region"] = rows["site"].map(region_of)
    out["y"] = rows[label_col].astype(float).to_numpy()
    g_train, g_rows = train["site"].map(region_of), out["region"]

    def fit_geo(tr: pd.DataFrame) -> dict:
        return _geo(tr, label, lead, settings)

    out["statewide_loro"] = held_out(train, rows, g_train, g_rows, fit_geo, _predict)
    log.info("statewide %s lead %d: leave-one-region-out done (%d nights)", label, lead, len(rows))
    if loso:
        out["statewide_loso"] = held_out(
            train, rows, train["site"], rows["site"], fit_geo, _predict
        )
        log.info("statewide %s lead %d: leave-one-station-out done", label, lead)
    if airport_geo is not None:
        out["geo_airports"] = _predict(airport_geo, rows)
    members = [f"{m.short}_frac_clear" for m in baselines.models_at_lead(settings, lead)]
    out["equal_weight"] = rows[members].mean(axis=1).to_numpy()
    for m in baselines.models_at_lead(settings, lead):
        fit, predict = _single_model_fit(m, label_col, lead, settings)
        out[f"single_{m.short}"] = held_out(train, rows, g_train, g_rows, fit, predict)
    clim = baselines.climatology_table(train, label_col)
    out["climatology"] = baselines.predict_climatology(clim, rows)
    return out


# ---------- scoring ----------


def _weights(rows: pd.DataFrame, settings: Settings, seed_offset: int = 0) -> np.ndarray:
    rng = np.random.default_rng(int(settings.raw["seed"]) + seed_offset)
    codes = evaluate.week_codes(rows["night_date"])
    return evaluate.bootstrap_weights(codes, settings.raw["bootstrap_resamples"], rng)


def _paired(y, a, b, W) -> dict[str, Any]:
    diff = evaluate.brier_matrix(y, a, W) - evaluate.brier_matrix(y, b, W)
    lo, hi = evaluate._ci(diff)
    point = float(np.mean((a - y) ** 2) - np.mean((b - y) ** 2))
    return {"brier_diff": point, "lo": lo, "hi": hi, "significant": bool(hi < 0 or lo > 0)}


def score(preds: pd.DataFrame, settings: Settings, airports: set[str]) -> dict:
    """Metrics with CIs for every method: statewide, on the new stations only, per region and
    per station; paired differences that answer the plan's questions."""
    threshold = settings.raw["decision_threshold"]
    methods = [c for c in preds.columns if c in METHODS or c.startswith("single_")]
    singles = [c for c in methods if c.startswith("single_")]
    subsets = {
        "all": np.ones(len(preds), dtype=bool),
        "new_stations": ~preds["site"].isin(airports).to_numpy(),
    }
    out: dict[str, Any] = {"n": int(len(preds)), "subsets": {}, "regions": {}, "stations": {}}
    for name, mask in subsets.items():
        sub = preds[mask]
        y, W = sub["y"].to_numpy(), _weights(sub, settings)
        p_clim = sub["climatology"].to_numpy()
        entry: dict[str, Any] = {"n": int(len(sub)), "n_stations": int(sub["site"].nunique())}
        entry["methods"] = {
            m: evaluate.point_and_ci(y, sub[m].to_numpy(), p_clim, W, m, threshold) for m in methods
        }
        loro = sub["statewide_loro"].to_numpy()
        best_single = min(singles, key=lambda m: float(np.mean((sub[m] - sub["y"]) ** 2)))
        entry["best_single"] = best_single
        entry["paired"] = {
            "statewide_minus_equal_weight": _paired(y, loro, sub["equal_weight"].to_numpy(), W),
            "statewide_minus_best_single": _paired(y, loro, sub[best_single].to_numpy(), W),
        }
        if "geo_airports" in sub:
            entry["paired"]["statewide_minus_airport_geo"] = _paired(
                y, loro, sub["geo_airports"].to_numpy(), W
            )
        if "statewide_loso" in sub:
            entry["paired"]["loro_minus_loso"] = _paired(
                y, loro, sub["statewide_loso"].to_numpy(), W
            )
        out["subsets"][name] = entry
    for group_col, store in [("region", out["regions"]), ("site", out["stations"])]:
        for i, (g, sub) in enumerate(preds.groupby(group_col)):
            y, W = sub["y"].to_numpy(), _weights(sub, settings, seed_offset=1 + i)
            p_clim = sub["climatology"].to_numpy()
            store[g] = {
                "n": int(len(sub)),
                "base_rate": float(y.mean()),
                "methods": {
                    m: evaluate.point_and_ci(y, sub[m].to_numpy(), p_clim, W, m, threshold)
                    for m in ["statewide_loro", "equal_weight", "climatology"]
                    + (["geo_airports"] if "geo_airports" in sub else [])
                },
            }
    return out


# ---------- the whole evaluation ----------


def load_airport_geo(lead: int, root: Path = AIRPORT_GEO_DIR) -> dict | None:
    path = root / f"model_geo_lead{lead}.json"
    return inference.load_artifact(path) if path.exists() else None


def run(df: pd.DataFrame, settings: Settings, loso: bool = True) -> dict:
    region_of = regions()
    airports = {s.id for s in load_sites()}
    result: dict[str, Any] = {
        "stations": network.load_network_table(),
        "region_names": network.REGION_NAMES,
        "period": {
            "train": [
                str(settings.raw["split"]["train_start"]),
                str(settings.raw["split"]["train_end"]),
            ],  # fmt: skip
            "test": [
                str(settings.raw["split"]["test_start"]),
                str(settings.raw["split"]["test_end"]),
            ],  # fmt: skip
        },
        "labels": {},
        "git_commit": evaluate.git_commit(),
    }
    for label in LABEL_COLS:
        leads = []
        for lead in settings.raw["leads"]:
            geo5 = load_airport_geo(lead) if label == "primary" else None
            preds = predictions(df, settings, label, lead, region_of, geo5,
                                loso=loso and label == "primary")  # fmt: skip
            leads.append({"lead": lead, **score(preds, settings, airports)})
        result["labels"][label] = leads
    return result


def train_final(df: pd.DataFrame, settings: Settings, root: Path = CANDIDATE_DIR) -> list[Path]:
    """The shipped statewide blend: every station's training years, primary label."""
    train, _ = split_train_test(df)
    paths = []
    for lead in settings.raw["leads"]:
        artifact = _geo(train, "primary", lead, settings)
        artifact["trained_on"] = sorted(train["site"].unique())
        path = root / f"model_geo_lead{lead}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(artifact, indent=1))
        paths.append(path)
    return paths


def shipping_decision(result: dict) -> dict:
    """The shipping rule fixed before any result (DECISIONS 2026-10-01): the statewide blend
    replaces the five-airport geo blend for places without their own record, unless it scores
    significantly worse than it at the new stations (paired week-block Brier difference, 95% CI
    above zero) at any lead."""
    worse = []
    for e in result["labels"]["primary"]:
        d = e["subsets"]["new_stations"]["paired"].get("statewide_minus_airport_geo")
        if d is not None and d["lo"] > 0:
            worse.append(int(e["lead"]))
    return {"method": "geo_airports" if worse else "statewide_loro", "worse_at_leads": worse}


def ship(path: Path = STATEWIDE_PATH, candidates: Path = CANDIDATE_DIR,
         live: Path | None = None) -> dict:  # fmt: skip
    """Apply the shipping rule: copy the statewide blend into the live geo directory when it
    wins (else keep the five-airport one) and record the choice in statewide.json, which the
    app reads to show the matching track record."""
    import shutil

    live = live or inference.ARTIFACTS / "geo"
    result = json.loads(path.read_text())
    decision = shipping_decision(result)
    if decision["method"] == "statewide_loro":
        for f in sorted(candidates.glob("model_geo_lead*.json")):
            shutil.copyfile(f, live / f.name)
    result["shipped_method"] = decision["method"]
    result["shipping"] = decision
    path.write_text(json.dumps(result, indent=1))
    return decision


def save(result: dict, path: Path = STATEWIDE_PATH) -> Path:
    path.write_text(json.dumps(evaluate._clean(result), indent=1))
    return path


def load(path: Path = STATEWIDE_PATH) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None
