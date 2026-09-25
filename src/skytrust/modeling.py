"""Shared modeling machinery: the leakage-safe split, date-based CV folds, and a tuned
logistic regression. Used by the single-model baseline (B4) and the blend (Phase 4).

Leakage rules enforced here (SPEC 8.1):
- train and test are separated by date, and `assert_no_date_overlap` checks it;
- CV folds are split on *unique night dates*, so the 5 sites' rows for one night never
  land on both sides of a fold boundary (same-night weather is strongly correlated);
- every fit sees training rows only; tests spy on `fit` to prove it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from skytrust.config import Settings

log = logging.getLogger(__name__)

PROB_CLIP = 1e-3  # log loss is computed on probabilities clipped to [0.001, 0.999]


MODEL_FEATURES = ("frac_clear", "longest_clear_run_frac", "mean_cover")


def model_columns(model_short: str) -> list[str]:
    """The three per-model forecast features used by B4 and the blend."""
    return [f"{model_short}_{f}" for f in MODEL_FEATURES]


def context_features(rows: pd.DataFrame, site_ids: list[str]) -> pd.DataFrame:
    """Context shared by B4 and the blend (SPEC 8.3): site one-hot, month as a point on a
    circle (so December and January are neighbours), and the night's dark-hour count.

    Giving B4 exactly the same context as the blend means the only difference between
    "best single model" and "blend" is how many weather models they see.
    """
    out = pd.DataFrame(index=rows.index)
    for site in site_ids:
        out[f"site_{site}"] = (rows["site"] == site).astype(float)
    angle = 2 * np.pi * rows["month"].astype(float) / 12
    out["month_sin"] = np.sin(angle)
    out["month_cos"] = np.cos(angle)
    out["dark_hours"] = rows["dark_hours"].astype(float)
    return out


def design_matrix(rows: pd.DataFrame, feature_cols: list[str], site_ids: list[str]) -> pd.DataFrame:
    """Forecast feature columns (as given) followed by the context features."""
    return pd.concat([rows[feature_cols].astype(float), context_features(rows, site_ids)], axis=1)


class LeakageError(AssertionError):
    """Train and test data overlap in time."""


def assert_no_date_overlap(train_dates: pd.Series, test_dates: pd.Series) -> None:
    """Every training night must be strictly earlier than every test night."""
    if len(train_dates) and len(test_dates):
        if pd.to_datetime(train_dates).max() >= pd.to_datetime(test_dates).min():
            raise LeakageError(
                f"train ends {pd.to_datetime(train_dates).max():%Y-%m-%d} but test starts "
                f"{pd.to_datetime(test_dates).min():%Y-%m-%d}"
            )


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rows by the dataset's `split` column (set from config), with the overlap check."""
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    assert_no_date_overlap(train["night_date"], test["night_date"])
    return train, test


def date_folds(dates: pd.Series, n_splits: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window CV folds over unique night dates (TimeSeriesSplit on dates, then
    mapped back to row positions). Always trains on the past, validates on the future."""
    d = pd.to_datetime(pd.Series(dates)).to_numpy()
    unique = np.unique(d)
    for tr_dates, va_dates in TimeSeriesSplit(n_splits=n_splits).split(unique):
        tr = np.flatnonzero(np.isin(d, unique[tr_dates]))
        va = np.flatnonzero(np.isin(d, unique[va_dates]))
        yield tr, va


def make_pipeline(C: float, impute: bool) -> Pipeline:
    """[median imputer + missing indicators] -> standardize -> L2 logistic regression.

    No class weighting: we want calibrated probabilities, not balanced accuracy.
    """
    steps = []
    if impute:
        steps.append(("impute", SimpleImputer(strategy="median", add_indicator=True)))
    steps += [
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(C=C, l1_ratio=0.0, max_iter=2000)),
    ]
    return Pipeline(steps)


def log_loss(y: np.ndarray, p: np.ndarray, weights: np.ndarray | None = None) -> float:
    p = np.clip(np.asarray(p, float), PROB_CLIP, 1 - PROB_CLIP)
    y = np.asarray(y, float)
    losses = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(np.average(losses, weights=weights))


@dataclass
class TunedModel:
    pipeline: Pipeline
    C: float
    cv_log_loss: float
    cv_scores: dict[float, float]
    features: list[str]
    n_train: int


def c_grid(settings: Settings) -> tuple[float, ...]:
    start, stop, num = settings.raw["modeling"]["c_grid_log10"]
    return tuple(float(c) for c in np.logspace(start, stop, int(num)))


def choose_C(scores: dict[float, float], tolerance: float) -> float:
    """Smallest C (strongest regularization) whose CV loss is within `tolerance` of the best.

    Above C ~ 1 the CV curve is flat to the 5th decimal; without a tolerance, noise-level
    differences would push the choice to the edge of the grid. Among ties, the simpler
    (more regularized) model is preferred.
    """
    best = min(scores.values())
    return min(c for c, v in scores.items() if v <= best + tolerance)


def fit_tuned_logistic(
    X: pd.DataFrame, y: pd.Series, dates: pd.Series, settings: Settings, impute: bool
) -> TunedModel:
    """Pick C by mean validation log loss over date-based TimeSeriesSplit folds (train data
    only), then refit on all of X with the chosen C."""
    n_splits = settings.raw["split"]["cv_folds"]
    y_arr = np.asarray(y, dtype=int)
    folds = list(date_folds(dates, n_splits))
    grid = c_grid(settings)
    scores = {}
    for C in grid:
        fold_losses = []
        for tr, va in folds:
            if len(np.unique(y_arr[tr])) < 2:
                continue  # can't fit a classifier on one class
            model = make_pipeline(C, impute).fit(X.iloc[tr], y_arr[tr])
            fold_losses.append(log_loss(y_arr[va], model.predict_proba(X.iloc[va])[:, 1]))
        scores[C] = float(np.mean(fold_losses)) if fold_losses else np.inf
    best_C = choose_C(scores, settings.raw["modeling"]["cv_tie_tolerance"])
    if len(grid) > 1 and best_C in (grid[0], grid[-1]):
        log.warning("chosen C=%g is at the edge of the grid; CV curve may still be falling", best_C)
    final = make_pipeline(best_C, impute).fit(X, y_arr)
    return TunedModel(final, best_C, scores[best_C], scores, list(X.columns), len(X))
