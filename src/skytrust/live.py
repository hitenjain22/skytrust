"""Tonight + the 7-night outlook (SPEC 9), shared by the CLI and the Streamlit app.

Flow for one site:
1. fetch the live Open-Meteo forecast for every model (or fall back to the last good copy);
2. find the next 7 nights (tonight = the dark window we're in, or the next one);
3. assign each night a lead d = max(1, ceil((dusk - now) / 24 h)), capped at 7;
4. compute exactly the training features over each night's dark hours and apply the lead-d
   blend (JSON artifact, numpy loader) -> P(usable);
5. add verdict, best window, model agreement, the backtest track record, and the Moon.

`build_forecast` is pure (payload + time in, forecast out) so it's tested offline with a
recorded payload; `get_forecast` adds the network and the last-good cache around it.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from skytrust import astro, inference
from skytrust.config import REPO_ROOT, Settings, Site
from skytrust.data import openmeteo
from skytrust.data.http import BadResponseError, HttpClient, SourceUnavailableError
from skytrust.dataset import night_rules
from skytrust.nightly import summarize_nights

log = logging.getLogger(__name__)

N_NIGHTS = 7
LAYERS = ["cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high"]


CUSTOM_ID = "CUSTOM"
# Custom locations must be in the Pacific-time West (the time zone the app displays) and near
# the region the models were validated on.
CUSTOM_BOUNDS = {"lat": (32.0, 49.0), "lon": (-125.0, -114.0)}
ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"


class LiveUnavailableError(Exception):
    """No live forecast and no saved copy to fall back on."""


def custom_site(
    lat: float, lon: float, name: str, settings: Settings, client: HttpClient | None = None
) -> Site:
    """A user-chosen location. Elevation comes from Open-Meteo's terrain model (90 m DEM) so
    forecasts are downscaled the same way as for the airports."""
    (la0, la1), (lo0, lo1) = CUSTOM_BOUNDS["lat"], CUSTOM_BOUNDS["lon"]
    if not (la0 <= lat <= la1 and lo0 <= lon <= lo1):
        raise ValueError(f"custom locations must be within lat {la0}-{la1}, lon {lo0}-{lo1}")
    client = client or HttpClient(settings.http)
    try:
        elevation = float(
            client.get_json(ELEVATION_URL, {"latitude": lat, "longitude": lon})["elevation"][0]
        )
    except (SourceUnavailableError, BadResponseError, KeyError, IndexError):
        elevation = float("nan")  # the forecast API then uses its own terrain lookup
    return Site(
        f"{CUSTOM_ID}_{lat:.3f}_{lon:.3f}",
        name or "Custom location",
        lat,
        lon,
        elevation,
        "custom",
        "America/Los_Angeles",
    )


def is_custom(site: Site) -> bool:
    return site.id.startswith(CUSTOM_ID)


@lru_cache(maxsize=1)
def evaluated_site_ids() -> frozenset[str]:
    from skytrust.config import load_sites

    return frozenset(s.id for s in load_sites())


def uses_site_blend(site: Site) -> bool:
    """Only the evaluated airports have a blend and a track record of their own; every other
    place (the app's California places, custom spots) gets the site-agnostic blend, whose
    accuracy was measured at airports it had never seen."""
    return site.id in evaluated_site_ids()


def utcnow() -> pd.Timestamp:
    """Current time (a function so tests can freeze it)."""
    return pd.Timestamp.now(tz="UTC")


def default_cache_dir() -> Path:
    return Path(os.environ.get("SKYTRUST_LIVE_CACHE", REPO_ROOT / "data" / "live_cache"))


# ---------- data classes ----------


@dataclass
class BestWindow:
    start_utc: pd.Timestamp
    end_utc: pd.Timestamp  # the last clear hour (inclusive)
    hours: int  # number of clear hourly readings in the run
    until_utc: (
        pd.Timestamp
    )  # when the window ends: an hour after the last clear reading, capped at dawn


@dataclass
class NightForecast:
    night_date: dt.date
    lead: int
    dusk_utc: pd.Timestamp
    dawn_utc: pd.Timestamp
    dark_hours: int
    p_usable: float | None
    verdict: str
    best_window: BestWindow | None
    spread: float | None
    agreement: str
    models_used: list[str]
    models_missing: list[str]
    track_record: dict | None
    trust: str
    moon_illum: float | None
    moon_free_hours: int
    moon_events: list[dict] = field(default_factory=list)
    # {model: {frac_clear, longest_clear_run_frac, mean_cover}} exactly as fed to the blend
    model_features: dict = field(default_factory=dict)
    blend_version: str | None = None  # git commit recorded in the artifact that produced p_usable
    hourly_clear: pd.Series | None = None  # P(clear) per dark hour from the hourly model
    moon_phase_deg: float | None = None  # at mid-darkness: 0 new, 180 full; < 180 waxing
    # When the Moon is up, from an hour before dusk to an hour after dawn (10-minute steps)
    moon_up: list[tuple[pd.Timestamp, pd.Timestamp]] = field(default_factory=list)


@dataclass
class LiveForecast:
    site: Site
    fetched_at_utc: pd.Timestamp
    source: str  # "live" or "cache"
    warning: str | None
    nights: list[NightForecast]
    hourly: pd.DataFrame  # long: time, model, cloud_cover, cloud_cover_low/mid/high (0-1)


# ---------- small pure helpers ----------


def assign_lead(dusk_utc: pd.Timestamp, now_utc: pd.Timestamp, max_lead: int = 7) -> int:
    """SPEC 9.3: how many days ahead this night is, as the training data defines lead."""
    days = (dusk_utc - now_utc) / pd.Timedelta(hours=24)
    return int(min(max_lead, max(1, math.ceil(days))))


def verdict_for(p: float | None, settings: Settings) -> str:
    if p is None or np.isnan(p):
        return "No data"
    v = settings.raw["verdict"]
    return "Go" if p >= v["go"] else "Maybe" if p >= v["maybe"] else "Skip"


def best_window(
    hours: pd.DatetimeIndex,
    median_cover: np.ndarray,
    threshold: float,
    dawn_utc: pd.Timestamp | None = None,
) -> BestWindow | None:
    """Longest run of consecutive dark hours whose cross-model median cover is clear. The
    window is shown as lasting until an hour after its last clear reading, but never past dawn."""
    best_len, best_end, run = 0, -1, 0
    for i, value in enumerate(median_cover):
        run = run + 1 if (not np.isnan(value) and value <= threshold + 1e-9) else 0
        if run > best_len:
            best_len, best_end = run, i
    if best_len == 0:
        return None
    until = hours[best_end] + pd.Timedelta(hours=1)
    if dawn_utc is not None:
        until = min(until, dawn_utc)
    return BestWindow(hours[best_end - best_len + 1], hours[best_end], best_len, until)


def trust_level(bss: float | None, settings: Settings) -> str:
    """Outlook trust indicator from the backtest skill at that lead (SPEC 10, page 2)."""
    if bss is None or np.isnan(bss):
        return "Unknown"
    t = settings.raw["live"]["trust_bss"]
    return "High" if bss >= t["high"] else "Medium" if bss >= t["medium"] else "Low"


def blend_records(metrics: dict | None) -> pd.DataFrame | None:
    """The shipped blend's primary-label records from metrics.json (built once per forecast:
    the full table has thousands of rows and every night needs a lookup)."""
    if not metrics:
        return None
    rec = pd.DataFrame(metrics["records"])
    return rec[(rec["method"] == "blend") & (rec["label"] == "primary")]


def track_record(
    metrics: dict | None, site_id: str, lead: int, records: pd.DataFrame | None = None
) -> dict | None:
    """Blend's test-set skill at this site and lead, plus the all-site figure, from metrics.json.
    Pass `records` (from blend_records) to skip rebuilding the table."""
    rec = records if records is not None else blend_records(metrics)
    if rec is None:
        return None
    base = rec[rec["lead"] == lead]
    site_row = base[(base["subset_type"] == "site") & (base["subset"] == site_id)]
    overall = base[base["subset_type"] == "overall"]
    if overall.empty:
        return None
    keys = [
        "bss",
        "bss_lo",
        "bss_hi",
        "false_clear_rate",
        "false_clear_rate_lo",
        "false_clear_rate_hi",
        "n",
        "n_weeks",
    ]
    out: dict[str, dict | None] = {"overall": {k: overall.iloc[0].get(k) for k in keys}}
    out["site"] = {k: site_row.iloc[0].get(k) for k in keys} if len(site_row) else None
    return out


# ---------- parsing the live payload ----------


def parse_live(payload: dict, settings: Settings) -> pd.DataFrame:
    """Long table (time, model, cloud layers as fractions) for every model in the payload."""
    frames = []
    for m in settings.models:
        df = openmeteo.parse_hourly(payload, LAYERS, m.id)
        df = df.reset_index().assign(model=m.short)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def upcoming_nights(site: Site, now_utc: pd.Timestamp, settings: Settings) -> pd.DataFrame:
    """The dark window we're in (or the next one) plus the following nights, N_NIGHTS total."""
    today = now_utc.tz_convert(site.timezone).date()
    windows = astro.dark_windows(
        site,
        today - dt.timedelta(days=1),
        today + dt.timedelta(days=N_NIGHTS + 1),
        settings.raw["definitions"]["sun_altitude_deg"],
    )
    windows = windows.dropna()
    return windows[windows["dawn_utc"] > now_utc].head(N_NIGHTS)


# ---------- assembling the forecast ----------

FEATURES = ["frac_clear", "longest_clear_run_frac", "mean_cover"]


def _feature_row(
    site: Site, night: dt.date, dark_hours: int, lead: int, per_model: dict
) -> pd.DataFrame:
    """One dataset-shaped row, so the exact training design matrix can be built from it."""
    row = {
        "site": site.id,
        "night_date": night,
        "lead": lead,
        "month": pd.Timestamp(night).month,
        "dark_hours": dark_hours,
    }
    frac = []
    for short, feats in per_model.items():
        for name in FEATURES:
            row[f"{short}_{name}"] = feats[name]
        if not np.isnan(feats["frac_clear"]):
            frac.append(feats["frac_clear"])
    # Same definition as training: population std across the models that have a value.
    row["spread_frac_clear"] = float(np.std(frac)) if len(frac) >= 2 else np.nan
    return pd.DataFrame([row])


def _model_features(summaries: dict, models: list[str], night: dt.date) -> dict:
    """{model: {feature: value}} for one night; NaN where a model has no data."""
    out = {}
    for m in models:
        table = summaries.get(m)
        if table is None or night not in table.index:
            out[m] = {k: np.nan for k in FEATURES}
        else:
            out[m] = {k: table.loc[night, k] for k in FEATURES}
    return out


def hourly_probabilities(
    ctx: _Context,
    night: dt.date,
    hours: pd.DatetimeIndex,
    dusk: pd.Timestamp,
    dawn: pd.Timestamp,
    lead: int,
) -> pd.Series | None:
    """P(clear) for each dark hour from the hourly model (same features as its training)."""
    path = ctx.artifacts_dir / "hourly" / f"model_hourly_lead{lead}.json"
    if not path.exists() or len(hours) == 0 or ctx.hourly_covers is None:
        return None
    artifact = inference.load_artifact(path)
    covers = ctx.hourly_covers.reindex(hours)
    rows = pd.DataFrame(index=hours)
    for m in artifact["models"]:
        rows[f"{m}_cover"] = covers[m] if m in covers else np.nan
    cols = [f"{m}_cover" for m in artifact["models"]]
    rows["mean_cover"] = rows[cols].mean(axis=1)
    rows["spread_cover"] = rows[cols].std(axis=1, ddof=0).where(rows[cols].notna().sum(axis=1) >= 2)
    span = (dawn - dusk).total_seconds()
    elapsed = (hours - dusk).total_seconds().to_numpy()
    rows["night_position"] = np.clip(elapsed / span, 0, 1)
    rows = rows.assign(site=ctx.site.id, month=pd.Timestamp(night).month, dark_hours=0.0)
    if rows["mean_cover"].isna().all():
        return None
    return pd.Series(
        inference.predict_proba(artifact, inference.raw_inputs(rows, artifact)), index=hours
    )


def _agreement(spread: float, threshold: float | None) -> str:
    if np.isnan(spread) or threshold is None:
        return "Not enough models"
    return "Models split" if spread > threshold else "Models agree"


@dataclass
class _Context:
    """Everything shared by all nights of one forecast."""

    site: Site
    settings: Settings
    now_utc: pd.Timestamp
    artifacts_dir: Path
    metrics: dict | None
    summaries: dict
    median: pd.Series
    night_hours: pd.DataFrame
    nights_astro: pd.DataFrame
    moon_events: pd.DataFrame
    variant: str = "primary"  # "geo" = site-agnostic blend for custom locations
    hourly_covers: pd.DataFrame | None = None  # time x model cloud cover (for the hourly model)
    records: pd.DataFrame | None = None  # blend records from metrics.json (see blend_records)


def _forecast_night(
    ctx: _Context, night: dt.date, dusk: pd.Timestamp, dawn: pd.Timestamp
) -> NightForecast:
    lead = assign_lead(dusk, ctx.now_utc)
    artifact = inference.load_artifact(
        inference.artifact_path(ctx.variant, lead, ctx.artifacts_dir)
    )
    per_model = _model_features(ctx.summaries, artifact["models"], night)
    used = [m for m, f in per_model.items() if not np.isnan(f["frac_clear"])]
    dark = int(ctx.nights_astro.loc[night, "dark_hours"])
    row = _feature_row(ctx.site, night, dark, lead, per_model)
    p = (
        float(inference.predict_proba(artifact, inference.raw_inputs(row, artifact))[0])
        if used
        else None
    )

    hours = pd.DatetimeIndex(ctx.night_hours.loc[ctx.night_hours["night_date"] == night, "hour"])
    window = best_window(
        hours, ctx.median.reindex(hours).to_numpy(), ctx.settings.clear_threshold, dawn
    )
    spread = float(row["spread_frac_clear"].iloc[0])
    record = (
        track_record(ctx.metrics, ctx.site.id, lead, ctx.records)
        if ctx.variant == "primary"
        else inference.place_record(ctx.site.lat, ctx.site.lon, lead)
    )
    ev = ctx.moon_events
    near = (ev["time_utc"] >= dusk - pd.Timedelta(hours=6)) & (
        ev["time_utc"] <= dawn + pd.Timedelta(hours=6)
    )
    events = ev[near]
    return NightForecast(
        night_date=night,
        lead=lead,
        dusk_utc=dusk,
        dawn_utc=dawn,
        dark_hours=dark,
        p_usable=p,
        verdict=verdict_for(p, ctx.settings),
        best_window=window,
        spread=None if np.isnan(spread) else spread,
        agreement=_agreement(spread, artifact["training"].get("spread_threshold")),
        models_used=used,
        models_missing=[m for m in artifact["models"] if m not in used],
        track_record=record,
        trust=trust_level(record["overall"]["bss"] if record else None, ctx.settings),
        moon_illum=_nan_to_none(ctx.nights_astro.loc[night, "moon_illum_mean"]),
        moon_free_hours=int(ctx.nights_astro.loc[night, "moon_free_dark_hours"]),
        moon_events=[
            {"time_utc": t, "event": e}
            for t, e in zip(events["time_utc"], events["event"], strict=True)
        ],
        model_features=per_model,
        blend_version=artifact.get("git_commit"),
        hourly_clear=hourly_probabilities(ctx, night, hours, dusk, dawn, lead),
        moon_phase_deg=float(astro.moon_phase_deg(pd.DatetimeIndex([dusk + (dawn - dusk) / 2]))[0]),
        moon_up=astro.moon_up_intervals(
            ctx.site,
            dusk - pd.Timedelta(hours=1),
            dawn + pd.Timedelta(hours=1),
            ctx.settings.raw["astro"]["moon_up_altitude_deg"],
        ),
    )


def build_forecast(
    site: Site,
    payload: dict,
    fetched_at: pd.Timestamp,
    now_utc: pd.Timestamp,
    settings: Settings,
    artifacts_dir: Path = inference.ARTIFACTS,
    metrics: dict | None = None,
    source: str = "live",
    warning: str | None = None,
) -> LiveForecast:
    """Pure: payload + time in, 7-night forecast out."""
    rules = night_rules(settings)
    hourly = parse_live(payload, settings)
    windows = upcoming_nights(site, now_utc, settings)
    nights_astro, night_hours = astro.night_table(
        site,
        windows.index.min(),
        windows.index.max(),
        settings.raw["definitions"]["sun_altitude_deg"],
        settings.raw["astro"]["moon_up_altitude_deg"],
        windows=windows[["dusk_utc", "dawn_utc"]],
    )
    # One moonrise/moonset search for the whole week (much faster than one per night).
    moon_events = astro.moon_events(
        site,
        windows["dusk_utc"].min() - pd.Timedelta(hours=6),
        windows["dawn_utc"].max() + pd.Timedelta(hours=6),
    )
    # Per-model night summaries with exactly the training rules.
    summaries = {
        short: summarize_nights(night_hours, g.set_index("time")["cloud_cover"], rules)
        for short, g in hourly.groupby("model", sort=False)
    }
    median = hourly.pivot_table(index="time", columns="model", values="cloud_cover").median(axis=1)
    variant = "primary" if uses_site_blend(site) else "geo"
    covers = hourly.pivot_table(index="time", columns="model", values="cloud_cover")
    ctx = _Context(
        site,
        settings,
        now_utc,
        artifacts_dir,
        metrics,
        summaries,
        median,
        night_hours,
        nights_astro.loc[windows.index],
        moon_events,
        variant,
        covers,
        blend_records(metrics),
    )
    nights = [_forecast_night(ctx, n, w["dusk_utc"], w["dawn_utc"]) for n, w in windows.iterrows()]
    return LiveForecast(site, fetched_at, source, warning, nights, hourly)


def _nan_to_none(x) -> float | None:
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


# ---------- network + last-good cache ----------


def _cache_path(cache_dir: Path, site_id: str) -> Path:
    return cache_dir / f"{site_id}.json"


def fetch_payload(
    client: HttpClient, settings: Settings, site: Site, cache_dir: Path, now_utc: pd.Timestamp
) -> tuple[dict, pd.Timestamp, str, str | None]:
    """Live payload, saved as the last good copy; on failure, the last good copy plus a
    warning. Raises LiveUnavailableError only when there's nothing to fall back to."""
    path = _cache_path(cache_dir, site.id)
    try:
        payload = openmeteo.fetch_live(client, settings, site)
        openmeteo.parse_hourly(payload, ["cloud_cover"], settings.models[0].id)  # validate
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"fetched_at_utc": now_utc.isoformat(), "payload": payload}))
        return payload, now_utc, "live", None
    except (SourceUnavailableError, BadResponseError) as exc:
        log.warning("live forecast for %s failed: %s", site.id, exc)
        if not path.exists():
            raise LiveUnavailableError(
                f"Open-Meteo is unavailable ({exc}) and no saved copy exists yet"
            ) from exc
        saved = json.loads(path.read_text())
        fetched = pd.Timestamp(saved["fetched_at_utc"])
        warning = (
            f"Live forecast unavailable; showing the saved forecast from "
            f"{fetched:%Y-%m-%d %H:%M} UTC."
        )
        return saved["payload"], fetched, "cache", warning


def get_forecast(
    site: Site,
    settings: Settings,
    metrics: dict | None = None,
    now_utc: pd.Timestamp | None = None,
    client: HttpClient | None = None,
    cache_dir: Path | None = None,
    artifacts_dir: Path = inference.ARTIFACTS,
) -> LiveForecast:
    now_utc = now_utc or utcnow()
    client = client or HttpClient(settings.http)
    payload, fetched, source, warning = fetch_payload(
        client, settings, site, cache_dir or default_cache_dir(), now_utc
    )
    return build_forecast(
        site, payload, fetched, now_utc, settings, artifacts_dir, metrics, source, warning
    )


# ---------- text output for the CLI ----------


def _local(ts: pd.Timestamp, tz: str) -> str:
    return ts.tz_convert(tz).strftime("%a %H:%M")


def _night_lines(i: int, n: NightForecast, tz: str) -> list[str]:
    title = "TONIGHT" if i == 0 else f"{pd.Timestamp(n.night_date):%a %b %d}"
    p = "no data" if n.p_usable is None else f"{n.p_usable:.0%}"
    lines = [
        "",
        f"{title}: P(usable) = {p} -> {n.verdict.upper()}   (lead {n.lead}, trust: {n.trust})",
        f"  Dark {_local(n.dusk_utc, tz)} -> {_local(n.dawn_utc, tz)} ({n.dark_hours} h)",
    ]
    if n.best_window:
        bw = n.best_window
        lines.append(
            f"  Best window: {_local(bw.start_utc, tz)} -> {_local(bw.until_utc, tz)} "
            f"({bw.hours} h clear by the model median)"
        )
    else:
        lines.append("  Best window: none (no dark hour is clear by the model median)")
    agree = n.agreement + (f" (spread {n.spread:.2f})" if n.spread is not None else "")
    used = ", ".join(m.upper() for m in n.models_used) or "none"
    missing = (
        f"; missing: {', '.join(m.upper() for m in n.models_missing)}" if n.models_missing else ""
    )
    lines.append(f"  {agree}; models used: {used}{missing}")
    site_record = (n.track_record or {}).get("site")
    if site_record:
        lines.append(
            f"  Track record here at lead {n.lead} (backtest): false-clear "
            f"{site_record['false_clear_rate']:.0%}, skill {site_record['bss']:.2f} vs climatology"
        )
    moon = "–" if n.moon_illum is None else f"{n.moon_illum:.0%} lit"
    lines.append(f"  Moon: {moon}, {n.moon_free_hours} moon-free dark h")
    if i == 0:
        lines.append(
            "  (Tonight's live forecast is fresher than the lead-1 backtest data, "
            "so P is slightly conservative.)"
        )
    return lines


def format_text(fc: LiveForecast) -> str:
    tz = fc.site.timezone
    lines = [
        f"SkyTrust: {fc.site.name} ({fc.site.id}), forecast as of "
        f"{fc.fetched_at_utc.tz_convert(tz):%Y-%m-%d %H:%M %Z}"
    ]
    if fc.warning:
        lines.append(f"!! {fc.warning}")
    for i, n in enumerate(fc.nights):
        lines += _night_lines(i, n, tz)
    lines += ["", ATTRIBUTION]
    return "\n".join(lines)


ATTRIBUTION = (
    "Weather data by Open-Meteo.com (CC BY 4.0). ASOS observations: Iowa Environmental Mesonet."
)
