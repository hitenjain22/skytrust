"""CLI entry point: `python -m skytrust <command>`."""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys

from skytrust.config import load_settings, load_sites
from skytrust.data.http import HttpClient

LATER_PHASE = {
    "build-dataset": 2,
    "train": 4,
    "evaluate": 3,
    "report": 3,
    "tonight": 5,
}


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
    for name in LATER_PHASE:
        sub.add_parser(name, help=f"(Phase {LATER_PHASE[name]})").set_defaults(func=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.func is None:
        print(f"`{args.command}` is not implemented yet (Phase {LATER_PHASE[args.command]}).")
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
