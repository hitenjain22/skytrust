from __future__ import annotations

import datetime as dt

from skytrust import sensitivity
from synthetic import SYNTH_SITES


def test_with_definition_changes_rules_and_disables_long_term_climatology(settings):
    cfg = sensitivity.with_definition(settings, 0.3, 2)
    assert cfg.clear_threshold == 0.3 and cfg.min_run_hours == 2
    assert cfg.raw["definitions"]["clear_threshold"] == 0.3
    assert cfg.raw["climatology"]["use_long_term"] is False
    assert settings.clear_threshold == 0.20  # original untouched


def test_grid_runs_end_to_end_on_synthetic_cache(
    fast_settings, synthetic_built, monkeypatch, tmp_path
):
    _, root = synthetic_built
    monkeypatch.setattr(sensitivity, "CLEAR_THRESHOLDS", (0.2, 0.3))
    monkeypatch.setattr(sensitivity, "MIN_RUNS", (3,))
    result = sensitivity.run(
        fast_settings, SYNTH_SITES, leads=[1], root=root, first=dt.date(2025, 12, 1)
    )
    assert len(result["cells"]) == 2
    default = next(c for c in result["cells"] if c["is_default"])
    lead1 = default["leads"][0]
    assert lead1["lead"] == 1 and 0 <= lead1["base_rate"] <= 1
    assert {"vs_best_single", "vs_equal_weight"} <= set(lead1)
    looser = next(c for c in result["cells"] if c["clear_threshold"] == 0.3)
    assert looser["leads"][0]["base_rate"] >= lead1["base_rate"]  # looser "clear" -> more usable
    path = sensitivity.save(result, tmp_path / "s.json")
    assert sensitivity.load(path)["cells"][0]["leads"]
