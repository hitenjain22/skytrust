"""CLI entry point: `python -m skytrust <command>`."""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

import pandas as pd

from skytrust.config import load_settings, load_sites
from skytrust.data.http import HttpClient


def cmd_validate_sites(args: argparse.Namespace) -> int:
    from skytrust.sites import validate_sites, write_sites_yaml

    settings = load_settings()
    client = HttpClient(settings.http)
    # Two days back so the last night (and its morning hours) is fully in the past.
    end = dt.datetime.now(dt.UTC).date() - dt.timedelta(days=2)
    checks = validate_sites(client, settings, end, refresh=args.refresh)
    threshold = settings.raw["site_validation"]["min_night_coverage"]

    period = f"{settings.history_start} to {end}"
    print(f"\nNight-hour sky-report coverage, {period} (need >= {threshold:.0%})")
    for c in checks:
        years = "  ".join(f"{y}: {v:.1%}" for y, v in c.coverage_by_year.items())
        status = "PASS" if c.passed else "FAIL"
        extra = f"  [{c.note}]" if c.note else ""
        print(f"  {c.id:<4} {status}  total {c.coverage_total:.1%}   {years}{extra}")
    path = write_sites_yaml(checks)
    print(f"\nWrote passing sites to {path}")
    failed = [c.id for c in checks if not c.passed]
    if failed:
        print(f"Failed: {failed}. Propose a same-terrain replacement and ask before substituting.")
        return 1
    return 0


def cmd_network_select(args: argparse.Namespace) -> int:
    from skytrust import network
    from skytrust.data import iem

    settings = load_settings()
    cfg = settings.raw["network"]
    client = HttpClient(settings.http)
    src = settings.sources
    payload = iem.fetch_network(client, src["iem_network_url"], src["iem_network"])
    stations = network.stations_from_geojson(payload)
    by_id = {s.id: s for s in stations}
    airports_ids = {s.id for s in load_sites()}
    seeds = [by_id[sid] for sid in sorted(airports_ids)]
    seeds += [by_id[sid] for sid in cfg.get("metro_seeds", []) if sid not in {s.id for s in seeds}]
    pool = network.candidates(stations, cfg, settings.history_start)
    quota = network.quotas(stations, cfg)
    passes = network.CoverageCheck(client, settings, settings.raw["split"]["test_end"])
    seeds = [s for s in seeds if s.id in airports_ids or passes(s)]
    chosen = network.select(pool, seeds, quota, cfg["km_per_m"], passes)
    airports = {s.id for s in load_sites()}
    path = network.write_network_yaml(
        chosen,
        airports,
        passes.results,
        quota,
    )
    print(f"Quotas per NWS region: {quota}")
    for s in chosen:
        cov = passes.results.get(s.id)
        tag = "evaluated airport" if s.id in airports else f"coverage {cov:.1%}"
        print(f"  {s.region}  {s.id:<4} {s.name:<32} {s.elevation_m:>6.0f} m  {tag}")
    print(f"Wrote {path} ({len(chosen)} stations)")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    import dataclasses

    from skytrust.data.backfill import default_last_night, fetch_source

    settings = load_settings()
    http = settings.http
    if args.network:
        from skytrust import network

        sites = network.load_network()
        http = dataclasses.replace(http, polite_delay_s=settings.raw["network"]["polite_delay_s"])
    else:
        sites = load_sites()
    if args.site:
        sites = tuple(s for s in sites if s.id == args.site.upper())
        if not sites:
            print(f"Unknown site {args.site!r}; see config/sites.yaml (or network.yaml)")
            return 2
    today = dt.datetime.now(dt.UTC).date()
    first = args.start or settings.history_start
    last = args.end or default_last_night(settings, today)
    client = HttpClient(http)
    summary = fetch_source(
        client, settings, sites, args.source, first, last, today, args.refresh,
        members_only=args.network,
    )  # fmt: skip
    print(
        f"{args.source}: nights {first}..{last}, sites {[s.id for s in sites]}, "
        f"network requests: {summary.network_requests}"
    )
    for failure in summary.failures:
        print(f"  FAILED {failure}")
    return 1 if summary.failures else 0


def cmd_build_dataset(args: argparse.Namespace) -> int:
    from skytrust import dataset, report

    settings = load_settings()
    df = dataset.build_dataset(settings, load_sites(), args.start, args.end)
    path = dataset.save_dataset(df)  # validates first; refuses to save a broken dataset
    quality = report.write_data_quality(df, settings)
    nights = report.one_row_per_night(df)
    n_test, verdict = report.test_sufficiency(df)
    print(f"Wrote {path} ({len(df):,} rows, {len(nights):,} site-nights)")
    print(f"Wrote {quality}")
    print(f"Test nights at lead 1: {verdict}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from skytrust import dataset, evaluate

    settings = load_settings()
    from skytrust.blend import ARTIFACTS

    metrics = evaluate.run_evaluation(dataset.load_dataset(), settings, artifacts_dir=ARTIFACTS)
    path = evaluate.save_metrics(metrics)
    print(f"Wrote {path} ({len(metrics['records'])} metric records)")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from skytrust import blend, dataset

    settings = load_settings()
    paths = blend.train_all(dataset.load_dataset(), settings)
    for path in paths:
        a = blend.load_artifact(path)
        cal = a["calibration"]
        print(
            f"{a['label']:>7} lead {a['lead']}: C={a['logistic']['C']:g}  "
            f"CV log loss {a['training']['cv_log_loss']:.4f}  OOF ECE {cal['oof_ece']:.3f} "
            f"(calibration {'applied' if cal['applied'] else 'not needed'})  -> {path.name}"
        )
    return 0


def cmd_tonight(args: argparse.Namespace) -> int:
    from skytrust import inference, live

    settings = load_settings()
    sites = {s.id: s for s in load_sites()}
    site = sites.get(args.site.upper())
    if site is None:
        print(f"Unknown site {args.site!r}. Choose from: {', '.join(sites)}")
        return 2
    try:
        metrics = inference.load_metrics()
    except FileNotFoundError:
        metrics = None  # still forecast; just no track record
    try:
        forecast = live.get_forecast(site, settings, metrics)
    except live.LiveUnavailableError as exc:
        print(f"Can't forecast right now: {exc}")
        return 1
    print(live.format_text(forecast))
    return 0


def cmd_build_climatology(args: argparse.Namespace) -> int:
    from skytrust import climatology

    table, labels_df = climatology.build(load_settings(), load_sites())
    climatology.save(table, labels_df)
    years = {site: f"{len(y)} yrs" for site, y in table["years_used"].items()}
    print(f"Wrote {climatology.TABLE_PATH} ({table['period'][0]}..{table['period'][1]}; {years})")
    return 0


def cmd_build_light_pollution(args: argparse.Namespace) -> int:
    from skytrust import lightpollution, skyglow

    settings = load_settings()
    cfg = settings.raw["light_pollution"]
    tif = Path(args.atlas) if args.atlas else lightpollution.ATLAS_TIF
    missing = [p for p in [tif, *(skyglow.viirs_path(y) for y in (2015, cfg["year"]))]
               if not p.exists()]  # fmt: skip
    if missing:
        print(
            "Missing inputs: " + ", ".join(str(p) for p in missing) + ". Download the atlas "
            f"({cfg['atlas_url']}) and the VIIRS night lights ({cfg['viirs_url']}), unzip them "
            "into data/raw/light_pollution/ (see docs/DATA_NOTES.md §11-12).",
            file=sys.stderr,
        )
        return 2
    atlas = lightpollution.build(settings, tif)
    lightpollution.save(atlas, lightpollution.BASE_PATH)
    updated, card = skyglow.build(settings, atlas, base_year=2015, year=cfg["year"])
    lightpollution.save(updated)
    lightpollution.save_overlay(updated)
    skyglow.save_sources(card["_sources"])
    skyglow.save_cities(cfg["bounds"])
    card["validation_2025_atlas"] = skyglow.validate_against_lorenz(
        atlas, updated, skyglow.LP_RAW / "NorthAmerica2025.png"
    )
    skyglow.save_model(card)
    cv = card["spatial_cv"]
    print(
        f"Wrote {lightpollution.GRID_PATH.name} (atlas updated to {cfg['year']} lights), "
        f"{lightpollution.BASE_PATH.name}, the overlay, light sources, cities and "
        f"{skyglow.MODEL_PATH.name}. Kernel spatial-CV error {cv['rmse_mag']:.3f} mag; median "
        f"change in lit areas x{card['ratio']['median_lit']:.2f}."
    )
    return 0


def cmd_build_places(args: argparse.Namespace) -> int:
    import json

    from skytrust import gazetteer

    settings = load_settings()
    client = HttpClient(settings.http)
    got = gazetteer.download(client)
    if got:
        print(f"Downloaded {', '.join(got)} into {gazetteer.RAW}")
    table, _ = gazetteer.build()
    elevations = gazetteer.fetch_elevations(settings.http, table["lat"], table["lon"])
    table, report = gazetteer.build(elevations=elevations, om_sample=gazetteer.open_meteo_sample())
    path = gazetteer.save(table)
    (gazetteer.RAW / "crosscheck_report.json").write_text(json.dumps(report, indent=1))
    print(
        f"Wrote {path} ({report['n_total']:,} places: {report['n_incorporated']} cities and "
        f"towns, {report['n_communities']:,} communities, {report['n_zip']:,} ZIP codes); "
        f"cross-checks in {gazetteer.RAW / 'crosscheck_report.json'}"
    )
    return 0


SKY_SOURCES = {
    "hip_main.dat": "https://cdsarc.cds.unistra.fr/ftp/cats/I/239/hip_main.dat",
    "constellations.lines.json": "d3-celestial data/ (github.com/ofrohn/d3-celestial)",
    "constellations.json": "d3-celestial data/",
    "mw.json": "d3-celestial data/",
    "messier.json": "d3-celestial data/",
    "dsos.bright.json": "d3-celestial data/",
}


def cmd_build_sky(args: argparse.Namespace) -> int:
    from skytrust import skycatalog

    missing = [name for name in SKY_SOURCES if not (skycatalog.RAW / name).exists()]
    if missing:
        where = "; ".join(f"{n} <- {SKY_SOURCES[n]}" for n in missing)
        print(f"Missing inputs in {skycatalog.RAW}: {where} (see docs/DATA_NOTES.md §13).",
              file=sys.stderr)  # fmt: skip
        return 2
    skycatalog.build_all()
    print(
        f"Wrote the star, constellation, Milky Way and deep-sky catalogues to {skycatalog.SKY_DIR}"
    )
    return 0


def cmd_sensitivity(args: argparse.Namespace) -> int:
    from skytrust import sensitivity

    path = sensitivity.save(sensitivity.run(load_settings(), load_sites()))
    print(f"Wrote {path}")
    return 0


def cmd_walkforward(args: argparse.Namespace) -> int:
    from skytrust import dataset, walkforward

    path = walkforward.save(walkforward.run(dataset.load_dataset(), load_settings()))
    print(f"Wrote {path}")
    return 0


def cmd_spatial(args: argparse.Namespace) -> int:
    from skytrust import dataset, spatial

    path = spatial.save(spatial.run(dataset.load_dataset(), load_settings()))
    print(f"Wrote {path}")
    return 0


def cmd_statewide(args: argparse.Namespace) -> int:
    from skytrust import dataset, statewide

    settings = load_settings()
    if args.rebuild or not statewide.NETWORK_DATASET.exists():
        df = statewide.build(settings)
        dataset.validate_dataset(df)
        statewide.NETWORK_DATASET.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(statewide.NETWORK_DATASET, index=False)
        print(
            f"Wrote {statewide.NETWORK_DATASET} ({len(df):,} rows, {df['site'].nunique()} stations)"
        )
    df = pd.read_parquet(statewide.NETWORK_DATASET)
    path = statewide.save(statewide.run(df, settings))
    models = statewide.train_final(df, settings, root=statewide.CANDIDATE_DIR)
    print(f"Wrote {path} and the statewide blend candidates in {models[0].parent}")
    return 0


def cmd_hourly(args: argparse.Namespace) -> int:
    from skytrust import hourly

    settings = load_settings()
    df = hourly.build_hourly(settings, load_sites())
    df.to_parquet(hourly.HOURLY_DATASET, index=False)
    path = hourly.save(hourly.train_and_evaluate(df, settings))
    print(f"Wrote {hourly.HOURLY_DATASET} ({len(df):,} hour rows) and {path}")
    return 0


def cmd_forward_log(args: argparse.Namespace) -> int:
    from pathlib import Path

    from skytrust import forward

    added = forward.log_forecasts(load_settings(), load_sites(), Path(args.out))
    print(f"Logged {added} new forecast rows to {Path(args.out) / forward.LOG_FILE}")
    return 0 if added else 1


def cmd_forward_verify(args: argparse.Namespace) -> int:
    from pathlib import Path

    from skytrust import forward

    summary = forward.verify(load_settings(), load_sites(), Path(args.out))
    print(
        f"Forward test: {summary['n_logged']} forecasts logged since {summary['forward_start']}, "
        f"{summary['n_verified']} verified so far -> {Path(args.out) / forward.SUMMARY_FILE}"
    )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from skytrust import evaluate, report

    metrics = evaluate.load_metrics()
    path = report.write_results(metrics)
    print(f"Wrote {path} and figures in {path.parent / 'figures'}")
    if report.update_readme(metrics):
        print("Updated the README results block")
    if report.resume_bullets(metrics):
        print(f"Wrote {report.write_resume_bullets(metrics)}")
    return 0


def _date(text: str) -> dt.date:
    return dt.date.fromisoformat(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="skytrust", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    vs = sub.add_parser("validate-sites", help="Check candidate stations in IEM (SPEC 5)")
    vs.add_argument("--refresh", action="store_true", help="bypass the disk cache")
    vs.set_defaults(func=cmd_validate_sites)
    fe = sub.add_parser("fetch", help="Download raw data into the disk cache (idempotent)")
    fe.add_argument("--source", required=True, choices=["asos", "era5", "prevruns", "goes"])
    fe.add_argument("--site", help="one site ID (default: all in config/sites.yaml)")
    fe.add_argument("--start", type=_date, help="first night, YYYY-MM-DD (default: history_start)")
    fe.add_argument("--end", type=_date, help="last night, YYYY-MM-DD (default: today - 2 days)")
    fe.add_argument("--refresh", action="store_true", help="bypass the disk cache")
    fe.add_argument(
        "--network",
        action="store_true",
        help="the statewide network (config/network.yaml), member models only, slower pace",
    )
    fe.set_defaults(func=cmd_fetch)
    ns = sub.add_parser(
        "network-select", help="Choose the statewide verification network -> network.yaml"
    )
    ns.set_defaults(func=cmd_network_select)
    bd = sub.add_parser(
        "build-dataset", help="Labels + features -> dataset.parquet, DATA_QUALITY.md"
    )
    bd.add_argument("--start", type=_date, help="first night (default: history_start)")
    bd.add_argument("--end", type=_date, help="last night (default: last fully cached night)")
    bd.set_defaults(func=cmd_build_dataset)
    ev = sub.add_parser("evaluate", help="Score every method on the test set -> metrics.json")
    ev.set_defaults(func=cmd_evaluate)
    cl = sub.add_parser("build-climatology", help="20-year reference climatology (2004-2023)")
    cl.set_defaults(func=cmd_build_climatology)
    lp = sub.add_parser(
        "build-light-pollution", help="Crop the light-pollution atlas to the mapped region"
    )
    lp.add_argument("--atlas", help="path to World_Atlas_2015.tif (default: data/raw/...)")
    lp.set_defaults(func=cmd_build_light_pollution)
    sk = sub.add_parser(
        "build-sky", help="Star, constellation and Milky Way catalogues for the app"
    )
    sk.set_defaults(func=cmd_build_sky)
    pl = sub.add_parser(
        "build-places", help="Every California place and ZIP code for the location menu"
    )
    pl.set_defaults(func=cmd_build_places)
    se = sub.add_parser("sensitivity", help="Re-run everything for 9 cloud definitions")
    se.set_defaults(func=cmd_sensitivity)
    wf = sub.add_parser("walkforward", help="Monthly refit-and-forecast evaluation from 2025")
    wf.set_defaults(func=cmd_walkforward)
    sp = sub.add_parser("spatial", help="Leave-one-site-out test of the site-agnostic blend")
    sp.set_defaults(func=cmd_spatial)
    sw = sub.add_parser(
        "statewide", help="Accuracy at the statewide network (leave-one-region-out) + candidates"
    )
    sw.add_argument("--rebuild", action="store_true", help="rebuild the network dataset first")
    sw.set_defaults(func=cmd_statewide)
    hr = sub.add_parser("hourly", help="Hourly P(clear) model: build, train, evaluate")
    hr.set_defaults(func=cmd_hourly)
    tr = sub.add_parser("train", help="Fit the blend per label x lead (train years only) -> JSON")
    tr.set_defaults(func=cmd_train)
    tn = sub.add_parser("tonight", help="Tonight + 7-night outlook for one site (live)")
    tn.add_argument("--site", default="SAC", help="site ID (default SAC)")
    tn.set_defaults(func=cmd_tonight)
    fl = sub.add_parser("forward-log", help="Record today's forecasts for the prospective test")
    fl.add_argument("--out", default="forward", help="folder holding the forward-test log")
    fl.set_defaults(func=cmd_forward_log)
    fv = sub.add_parser(
        "forward-verify", help="Score logged forecasts whose nights have observations"
    )
    fv.add_argument("--out", default="forward", help="folder holding the forward-test log")
    fv.set_defaults(func=cmd_forward_verify)
    rp = sub.add_parser("report", help="metrics.json -> docs/RESULTS.md + figures")
    rp.set_defaults(func=cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
