"""Prospective (forward) verification: the only evaluation that cannot be influenced by development.

Every day a scheduled job calls `log_forecasts`: it records what SkyTrust (and NOAA's NBM, and
the raw ECMWF / GEFS ensembles) predicted for the next 7 nights at every site, *before* the
outcome exists. Days later, once airport and ERA5 observations are published, `verify` labels
those nights with exactly the backtest's label code and scores the logged probabilities.

The log is append-only and each row records the model and code version that produced it, so
the record can't be rewritten after the fact. Nights issued before `forward.start_date` are
excluded: that's where development ended.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from skytrust import astro, evaluate, labels, live
from skytrust.baselines import climatology_table, predict_climatology
from skytrust.config import Settings, Site
from skytrust.data import iem, openmeteo
from skytrust.data.http import BadResponseError, HttpClient, SourceUnavailableError
from skytrust.dataset import load_dataset, night_rules
from skytrust.inference import load_metrics
from skytrust.nightly import NightRules, summarize_nights

log = logging.getLogger(__name__)

LOG_FILE = "forecasts.csv"
VERIFIED_FILE = "verified.csv"
SUMMARY_FILE = "summary.json"
KEY = ["issue_date", "site", "night_date"]
FEATURES = ["frac_clear", "longest_clear_run_frac", "mean_cover"]
ENSEMBLE_NAMES = {"ecmwf_ifs025": "ens_ecmwf", "gfs025": "ens_gefs"}


# ---------- benchmark forecasts (NBM + raw ensembles) ----------


def ensemble_members(payload: dict, model_suffix: str) -> dict[str, pd.Series]:
    """Every member (including the control run) of one ensemble, as hourly fractions.

    Keys look like `cloud_cover_member07_ecmwf_ifs025_ensemble`; the control run is
    `cloud_cover_ecmwf_ifs025_ensemble`.
    """
    hourly = payload.get("hourly", {})
    times = pd.DatetimeIndex(pd.to_datetime(hourly.get("time", []), utc=True))
    out = {}
    for key, values in hourly.items():
        if key == "time" or not key.endswith(model_suffix):
            continue
        member = key[len("cloud_cover_") : -len(model_suffix) - 1] or "control"
        out[member] = pd.Series(values, index=times, dtype="float64") / 100.0
    return out


def ensemble_p_usable(
    members: dict[str, pd.Series], night_hours: pd.DataFrame, rules: NightRules
) -> tuple[pd.Series, pd.Series]:
    """P(usable) per night = share of members whose own forecast meets the usable rule.

    This is the classic 'raw ensemble probability': no statistics learned, just counting
    members. Members with too much missing data for a night don't vote on that night.
    """
    votes = pd.DataFrame(
        {m: summarize_nights(night_hours, s, rules)["usable"] for m, s in members.items()}
    )
    n_voting = votes.notna().sum(axis=1)
    p = votes.astype("float").mean(axis=1, skipna=True).where(n_voting > 0)
    return p, n_voting


def fetch_benchmarks(client: HttpClient, settings: Settings, site: Site) -> dict:
    """Live NBM + ensemble payloads. Failures return None so the main log is never lost."""
    fwd = settings.raw["forward"]
    loc = openmeteo.location_params(site)
    out: dict = {"nbm": None, "ensemble": None}
    try:
        out["nbm"] = client.get_json(
            settings.sources["forecast_url"],
            {
                **loc,
                "models": ",".join(fwd["benchmark_models"]),
                "hourly": "cloud_cover",
                "forecast_days": 8,
            },
        )
    except (SourceUnavailableError, BadResponseError) as exc:
        log.warning("NBM fetch failed for %s: %s", site.id, exc)
    try:
        out["ensemble"] = client.get_json(
            fwd["ensemble_url"],
            {
                **loc,
                "models": ",".join(fwd["ensemble_models"]),
                "hourly": "cloud_cover",
                "forecast_days": 8,
            },
        )
    except (SourceUnavailableError, BadResponseError) as exc:
        log.warning("ensemble fetch failed for %s: %s", site.id, exc)
    return out


def benchmark_columns(
    site: Site, forecast: live.LiveForecast, bench: dict, settings: Settings
) -> pd.DataFrame:
    """Per-night NBM features and raw-ensemble P(usable) for the forecast's nights."""
    rules = night_rules(settings)
    nights = [n.night_date for n in forecast.nights]
    windows = pd.DataFrame(
        {
            "dusk_utc": [n.dusk_utc for n in forecast.nights],
            "dawn_utc": [n.dawn_utc for n in forecast.nights],
        },
        index=pd.Index(nights, name="night_date"),
    )
    night_hours = astro.night_hours(windows)
    out = pd.DataFrame(index=pd.Index(nights, name="night_date"))
    if bench.get("nbm"):
        nbm_id = settings.raw["forward"]["benchmark_models"][0]
        series = openmeteo.parse_hourly(bench["nbm"], ["cloud_cover"], nbm_id)["cloud_cover"]
        summary = summarize_nights(night_hours, series, rules).reindex(out.index)
        for f in FEATURES:
            out[f"nbm_{f}"] = summary[f]
        out["nbm_pred_usable"] = summary["usable"].astype("float")
    if bench.get("ensemble"):
        suffixes = {"ecmwf_ifs025": "ecmwf_ifs025_ensemble", "gfs025": "ncep_gefs025"}
        for model, name in ENSEMBLE_NAMES.items():
            members = ensemble_members(bench["ensemble"], suffixes[model])
            if members:
                p, n = ensemble_p_usable(members, night_hours, rules)
                out[f"{name}_p_usable"] = p.reindex(out.index)
                out[f"{name}_members"] = n.reindex(out.index)
    return out


# ---------- logging ----------


def code_version() -> str | None:
    return os.environ.get("GITHUB_SHA", "")[:7] or evaluate.git_commit()


def forecast_rows(forecast: live.LiveForecast, issued_at: pd.Timestamp) -> pd.DataFrame:
    """One row per (site, night) exactly as the app showed it at `issued_at`."""
    tz = forecast.site.timezone
    rows = []
    for n in forecast.nights:
        row = {
            "issued_at_utc": issued_at.isoformat(),
            "issue_date": issued_at.tz_convert(tz).date(),
            "site": forecast.site.id,
            "night_date": n.night_date,
            "lead": n.lead,
            "p_usable": n.p_usable,
            "verdict": n.verdict,
            "dark_hours": n.dark_hours,
            "n_models_used": len(n.models_used),
            "models_missing": ",".join(n.models_missing),
            "spread": n.spread,
            "agreement": n.agreement,
            "best_window_hours": n.best_window.hours if n.best_window else 0,
            "blend_version": n.blend_version,
            "forecast_source": forecast.source,
        }
        for model, feats in n.model_features.items():
            for f in FEATURES:
                row[f"{model}_{f}"] = feats[f]
        rows.append(row)
    return pd.DataFrame(rows)


def append_log(path: Path, new_rows: pd.DataFrame) -> int:
    """Append-only with de-duplication on (issue_date, site, night_date): the first forecast
    issued for a night on a given day is the one that counts; later re-runs can't replace it."""
    if path.exists():
        existing = pd.read_csv(path, dtype={"issue_date": str, "night_date": str})
        before = len(existing)
    else:
        existing, before = pd.DataFrame(), 0
    new_rows = new_rows.astype({"issue_date": str, "night_date": str})
    combined = pd.concat([existing, new_rows], ignore_index=True)
    combined = combined.drop_duplicates(KEY, keep="first").sort_values(KEY)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    return len(combined) - before


def log_forecasts(
    settings: Settings,
    sites: tuple[Site, ...],
    out_dir: Path,
    now_utc: pd.Timestamp | None = None,
    client: HttpClient | None = None,
) -> int:
    """Record today's forecasts for every site. Returns the number of new rows."""
    now_utc = now_utc or live.utcnow()
    client = client or HttpClient(settings.http)
    try:
        metrics = load_metrics()
    except FileNotFoundError:
        metrics = None
    frames = []
    for site in sites:
        try:
            forecast = live.get_forecast(site, settings, metrics, now_utc=now_utc, client=client)
        except live.LiveUnavailableError as exc:
            log.error("no forecast for %s today: %s", site.id, exc)
            continue
        rows = forecast_rows(forecast, now_utc)
        bench = benchmark_columns(
            site, forecast, fetch_benchmarks(client, settings, site), settings
        )
        frames.append(rows.merge(bench.reset_index(), on="night_date", how="left"))
    if not frames:
        return 0
    rows = pd.concat(frames, ignore_index=True)
    rows["code_version"] = code_version()
    return append_log(out_dir / LOG_FILE, rows)


# ---------- verification ----------


def read_log(out_dir: Path) -> pd.DataFrame:
    path = out_dir / LOG_FILE
    if not path.exists():
        return pd.DataFrame(columns=[*KEY, "lead", "p_usable"])
    df = pd.read_csv(path)
    for col in ["issue_date", "night_date"]:
        df[col] = pd.to_datetime(df[col]).dt.date
    return df


def observed_labels(
    settings: Settings,
    sites: tuple[Site, ...],
    first: dt.date,
    last: dt.date,
    today: dt.date,
    client: HttpClient,
) -> pd.DataFrame:
    """Labels for [first, last] nights with exactly the backtest's code (fetching as needed)."""
    from skytrust.data.backfill import era5_end

    src, rules, frames = settings.sources, night_rules(settings), []
    for site in sites:
        iem.fetch_asos_range(
            client, src["iem_asos_url"], site.id, first, last, src["iem_report_types"]
        )
        openmeteo.fetch_era5(client, settings, site, first, era5_end(settings, last, today))
        windows = astro.dark_windows(
            site, first, last, settings.raw["definitions"]["sun_altitude_deg"]
        )
        observed = labels.load_observed_hourly(site.id, settings)
        lab = labels.build_labels(windows, astro.night_hours(windows), observed, rules)
        lab = lab[["usable_primary", "usable_asos", "usable_era5", "exclusion_primary"]]
        frames.append(lab.reset_index().assign(site=site.id))
    return pd.concat(frames, ignore_index=True)


def _score(
    y: np.ndarray, p: np.ndarray, p_clim: np.ndarray, weeks: np.ndarray, settings: Settings
) -> dict:
    """Same metric code as the backtest. CIs only once there are enough weekly blocks."""
    kind = "hard" if set(np.unique(p[~np.isnan(p)])) <= {0.0, 1.0} else "prob"
    ones = np.ones((1, len(y)))
    point = evaluate.all_metrics(
        y, p, p_clim, ones, kind, "forward", settings.raw["decision_threshold"]
    )
    out = {"n": int(len(y)), "base_rate": float(y.mean())}
    n_weeks = int(len(np.unique(weeks)))
    enough = n_weeks >= settings.raw["min_weeks_for_ci"]
    if enough:
        W = evaluate.bootstrap_weights(
            weeks, settings.raw["bootstrap_resamples"], np.random.default_rng(settings.raw["seed"])
        )
        boot = evaluate.all_metrics(
            y, p, p_clim, W, kind, "forward", settings.raw["decision_threshold"]
        )
    for name, value in point.items():
        out[name] = float(value[0])
        if enough:
            out[f"{name}_lo"], out[f"{name}_hi"] = evaluate._ci(boot[name])
    out["go_calls"] = int((p >= settings.raw["decision_threshold"]).sum())
    out["go_calls_usable"] = int(((p >= settings.raw["decision_threshold"]) & (y == 1)).sum())
    return out


METHODS = {
    "blend": "p_usable",
    "nbm_frac_clear": "nbm_frac_clear",
    "ens_ecmwf": "ens_ecmwf_p_usable",
    "ens_gefs": "ens_gefs_p_usable",
}


def score_forward(verified: pd.DataFrame, settings: Settings, backtest: dict | None) -> dict:
    """Metrics per lead (and pooled) for SkyTrust and each benchmark on identical rows."""
    df = verified.dropna(subset=["usable_primary", "p_usable"]).copy()
    by_lead = []
    for lead, g in [(0, df)] + list(df.groupby("lead")):
        if g.empty:
            continue
        y = g["usable_primary"].astype(float).to_numpy()
        weeks = evaluate.week_codes(g["night_date"]) if len(g) else np.array([])
        entry = {
            "lead": int(lead) if lead else "all",
            "n": int(len(g)),
            "n_weeks": int(len(np.unique(weeks))),
            "methods": {},
        }
        p_clim = g["p_climatology"].to_numpy()
        entry["methods"]["climatology"] = _score(y, p_clim, p_clim, weeks, settings)
        for name, col in METHODS.items():
            if col in g and g[col].notna().all():
                entry["methods"][name] = _score(y, g[col].to_numpy(float), p_clim, weeks, settings)
        if backtest and lead:
            rec = pd.DataFrame(backtest["records"])
            hit = rec[
                (rec["method"] == "blend")
                & (rec["label"] == "primary")
                & (rec["lead"] == lead)
                & (rec["subset_type"] == "overall")
            ]
            if len(hit):
                entry["backtest"] = {k: hit.iloc[0][k] for k in ["bss", "false_clear_rate"]}
        by_lead.append(entry)
    return {"by_lead": by_lead}


def verify(
    settings: Settings,
    sites: tuple[Site, ...],
    out_dir: Path,
    today: dt.date | None = None,
    client: HttpClient | None = None,
) -> dict:
    """Label and score every logged forecast whose night is old enough to have observations."""
    fwd = settings.raw["forward"]
    today = today or live.utcnow().date()
    ready_before = today - dt.timedelta(days=fwd["verify_after_days"])
    logged = read_log(out_dir)
    logged = logged[logged["issue_date"] >= fwd["start_date"]] if len(logged) else logged
    ready = logged[logged["night_date"] <= ready_before] if len(logged) else logged
    summary = {
        "generated_utc": live.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "forward_start": str(fwd["start_date"]),
        "n_logged": int(len(logged)),
        "first_issue": str(min(logged["issue_date"])) if len(logged) else None,
        "last_issue": str(max(logged["issue_date"])) if len(logged) else None,
        "verify_after_days": fwd["verify_after_days"],
    }
    if ready.empty:
        summary.update(n_verified=0, by_lead=[])
    else:
        client = client or HttpClient(settings.http)
        first, last = min(ready["night_date"]), max(ready["night_date"])
        lab = observed_labels(settings, sites, first, last, today, client)
        verified = ready.merge(lab, on=["site", "night_date"], how="left")
        train = load_dataset().query("split == 'train'")
        verified["month"] = pd.to_datetime(verified["night_date"]).dt.month
        verified["p_climatology"] = predict_climatology(
            climatology_table(train, "usable_primary"), verified
        )
        verified.to_csv(out_dir / VERIFIED_FILE, index=False)
        try:
            backtest = load_metrics()
        except FileNotFoundError:
            backtest = None
        scored = score_forward(verified, settings, backtest)
        labeled = verified["usable_primary"].notna()
        summary.update(
            n_verified=int(labeled.sum()),
            last_verified_night=str(max(verified.loc[labeled, "night_date"]))
            if labeled.any()
            else None,
            **scored,
        )
    (out_dir / SUMMARY_FILE).write_text(json.dumps(evaluate._clean(summary), indent=1, default=str))
    return summary
