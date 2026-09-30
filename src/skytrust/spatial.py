"""Leave-one-site-out (LOSO) test: does the blend work at a place it has never seen?

The shipped blend knows which site it's forecasting (site one-hot). A site-agnostic ("geo")
blend drops that, so it can be applied anywhere. To measure how well it transfers, each site
in turn is held out: the geo blend is trained on the *other* sites' training years only and
scored on the held-out site's test period. It's compared, on the same nights, with the shipped
site-aware blend (which did see that site's training years) and with the equal-weight average,
which needs no training at all.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from skytrust import baselines, blend, evaluate, inference
from skytrust.config import REPO_ROOT, Settings
from skytrust.modeling import split_train_test

log = logging.getLogger(__name__)

SPATIAL_PATH = REPO_ROOT / "artifacts" / "spatial.json"


def loso_predictions(
    df: pd.DataFrame, settings: Settings, artifacts_dir: Path, leads: list[int] | None = None
) -> pd.DataFrame:
    leads = leads or settings.raw["leads"]
    train, test = split_train_test(df)
    frames = []
    for lead in leads:
        shipped = inference.load_artifact(inference.artifact_path("primary", lead, artifacts_dir))
        rows, _ = baselines.evaluation_rows(df, test, "usable_primary", lead, settings)
        members = [f"{m.short}_frac_clear" for m in baselines.models_at_lead(settings, lead)]
        for site in sorted(rows["site"].unique()):
            held = rows[rows["site"] == site]
            geo = blend.fit_blend(
                train[train["site"] != site], "primary", lead, settings, use_site=False
            )
            out = held[["site", "night_date", "lead"]].copy()
            out["y"] = held["usable_primary"].astype(float).to_numpy()
            out["geo_unseen"] = inference.predict_proba(geo, inference.raw_inputs(held, geo))
            out["blend_site"] = inference.predict_proba(
                shipped, inference.raw_inputs(held, shipped)
            )
            out["equal_weight"] = held[members].mean(axis=1).to_numpy()
            frames.append(out)
            log.info("LOSO lead %d: held out %s (%d nights)", lead, site, len(held))
    return pd.concat(frames, ignore_index=True)


def summarize(preds: pd.DataFrame, settings: Settings) -> dict:
    from skytrust import climatology

    clim = climatology.load_table("primary")
    rng = np.random.default_rng(int(settings.raw["seed"]))
    threshold = settings.raw["decision_threshold"]
    out: dict[str, Any] = {"leads": []}
    for lead, g in preds.groupby("lead"):
        y = g["y"].to_numpy()
        p_clim = (
            baselines.predict_climatology(
                clim, g.assign(month=pd.to_datetime(g["night_date"]).dt.month)
            )
            if clim is not None
            else np.full(len(g), y.mean())
        )
        codes = evaluate.week_codes(g["night_date"])
        W = evaluate.bootstrap_weights(codes, settings.raw["bootstrap_resamples"], rng)
        entry: dict[str, Any] = {"lead": int(lead), "n": int(len(g)), "methods": {}, "per_site": {}}
        for m in ["geo_unseen", "blend_site", "equal_weight"]:
            p = g[m].to_numpy(float)
            entry["methods"][m] = evaluate.point_and_ci(y, p, p_clim, W, m, threshold)
        diff = evaluate.brier_matrix(y, g["geo_unseen"].to_numpy(float), W) - evaluate.brier_matrix(
            y, g["blend_site"].to_numpy(float), W
        )
        lo, hi = evaluate._ci(diff)
        entry["geo_minus_site"] = {
            "brier_diff": float(
                np.mean((g["geo_unseen"] - g["y"]) ** 2) - np.mean((g["blend_site"] - g["y"]) ** 2)
            ),
            "lo": lo,
            "hi": hi,
            "significant": bool(hi < 0 or lo > 0),
        }
        for site, sg in g.groupby("site"):
            ref = np.mean((sg["equal_weight"] - sg["y"]) ** 2)
            entry["per_site"][site] = {
                "n": int(len(sg)),
                **{
                    f"brier_{m}": float(np.mean((sg[m] - sg["y"]) ** 2))
                    for m in ["geo_unseen", "blend_site", "equal_weight"]
                },
                "ew_reference_brier": float(ref),
            }
        out["leads"].append(entry)
    return out


def run(df: pd.DataFrame, settings: Settings, artifacts_dir: Path = inference.ARTIFACTS) -> dict:
    return summarize(loso_predictions(df, settings, artifacts_dir), settings)


def save(result: dict, path: Path = SPATIAL_PATH) -> Path:
    path.write_text(json.dumps(evaluate._clean(result), indent=1))
    return path


def load(path: Path = SPATIAL_PATH) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None
