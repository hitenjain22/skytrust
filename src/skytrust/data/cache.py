"""Disk cache for raw API responses.

Why: backfilling ~3 years x 5 sites x several sources is slow and puts load on free
services. Every raw response is saved once under a predictable path, and re-running a
fetch reads from disk instead of the network (unless --refresh is passed).
"""

from __future__ import annotations

from pathlib import Path

from skytrust.config import REPO_ROOT

RAW_DIR = REPO_ROOT / "data" / "raw"


def raw_path(
    source: str, site: str, model: str | None, start: str, end: str, ext: str, root: Path = RAW_DIR
) -> Path:
    """data/raw/{source}/{site}/{model_or_na}/{start}_{end}.{ext}"""
    return root / source / site / (model or "na") / f"{start}_{end}.{ext}"


def read_cached(path: Path) -> str | None:
    return path.read_text() if path.exists() and path.stat().st_size > 0 else None


def write_cached(path: Path, text: str) -> None:
    """Write atomically (temp file + rename) so an interrupted run never leaves a
    half-written file that later looks like a complete cache entry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)
