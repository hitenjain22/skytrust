"""CLI entry point: `python -m skytrust <command>`."""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys

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


def cmd_fetch(args: argparse.Namespace) -> int:
    from skytrust.data.backfill import default_last_night, fetch_source

    settings = load_settings()
    sites = load_sites()
    if args.site:
        sites = tuple(s for s in sites if s.id == args.site.upper())
        if not sites:
            print(f"Unknown site {args.site!r}; see config/sites.yaml")
            return 2
    today = dt.datetime.now(dt.UTC).date()
    first = args.start or settings.history_start
    last = args.end or default_last_night(settings, today)
    client = HttpClient(settings.http)
    summary = fetch_source(client, settings, sites, args.source, first, last, today, args.refresh)
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
    fe.add_argument("--source", required=True, choices=["asos", "era5", "prevruns"])
    fe.add_argument("--site", help="one site ID (default: all in config/sites.yaml)")
    fe.add_argument("--start", type=_date, help="first night, YYYY-MM-DD (default: history_start)")
    fe.add_argument("--end", type=_date, help="last night, YYYY-MM-DD (default: today - 2 days)")
    fe.add_argument("--refresh", action="store_true", help="bypass the disk cache")
    fe.set_defaults(func=cmd_fetch)
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
    se = sub.add_parser("sensitivity", help="Re-run everything for 9 cloud definitions")
    se.set_defaults(func=cmd_sensitivity)
    wf = sub.add_parser("walkforward", help="Monthly refit-and-forecast evaluation from 2025")
    wf.set_defaults(func=cmd_walkforward)
    sp = sub.add_parser("spatial", help="Leave-one-site-out test of the site-agnostic blend")
    sp.set_defaults(func=cmd_spatial)
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
