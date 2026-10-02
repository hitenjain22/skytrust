"""The live chart's JavaScript, loaded in a real browser engine (opt-in: `pytest -m browser`
with Playwright and Chromium available, e.g. `uv run --with playwright pytest -m browser`).
Python can't parse JavaScript, so a syntax error would otherwise only show up in the app."""

from __future__ import annotations

import pytest

from skytrust.config import REPO_ROOT

pytestmark = pytest.mark.browser


def test_live_chart_script_parses_and_exports_a_component():
    sync_api = pytest.importorskip("playwright.sync_api")
    source = (REPO_ROOT / "app" / "views" / "skylive.js").read_text()
    with sync_api.sync_playwright() as p:
        browser = None
        for kwargs in ({}, {"channel": "chrome"}):  # Playwright's Chromium, else installed Chrome
            try:
                browser = p.chromium.launch(**kwargs)
                break
            except Exception:
                continue
        if browser is None:
            pytest.skip("no Chromium or Chrome for Playwright")
        page = browser.new_page()
        result = page.evaluate(
            """async (src) => {
                const url = URL.createObjectURL(new Blob([src], {type: "text/javascript"}));
                try { const m = await import(url); return typeof m.default; }
                catch (e) { return "error: " + e.message; }
            }""",
            source,
        )
        browser.close()
    assert result == "function", result


def test_the_browser_computes_the_same_visibility_as_python():
    """Runs skylive.js on a real night's data (Los Angeles, Moon up, twilight at the ends) and
    compares its faintest-visible magnitudes with the exact Python model at the same moments:
    the browser's sum of brightness parts must agree with sky.faintest_visible."""
    import json

    import numpy as np
    import pandas as pd

    sync_api = pytest.importorskip("playwright.sync_api")
    import sys

    sys.path.insert(0, str(REPO_ROOT / "app"))
    from skytrust import sky
    from skytrust.config import load_sites
    from views import skylive  # noqa: E402

    la = next(s for s in load_sites() if s.id == "LAX") if any(
        s.id == "LAX" for s in load_sites()) else load_sites()[0]  # fmt: skip
    dusk = pd.Timestamp("2026-10-02 02:15", tz="UTC")  # civil dusk (Sun 6° down)
    dawn = pd.Timestamp("2026-10-02 13:20", tz="UTC")  # before civil dawn
    cond = sky.Conditions(
        18.2, k=0.45, k_ref=0.25, glow=sky._glow(round(la.lat, 3), round(la.lon, 3))
    )
    guide = sky.tonight(la, dusk, dawn, cond)
    data = skylive.payload(la, guide, cond, dusk)
    rng = np.random.default_rng(5)
    alt, az = rng.uniform(1, 89, 400), rng.uniform(0, 360, 400)
    obs = sky.Observer(la)
    times = pd.DatetimeIndex(guide["times"])
    sun = obs.body("sun", times)
    steps = [0, len(times) // 3, len(times) // 2, len(times) - 1]
    exact = []
    for i in steps:
        moon = sky.moon_at(guide["moon"], i, alt, az)
        ssep = sky.separation_deg(az, alt, sun["az"][i], sun["alt"][i])
        exact.append(sky.faintest_visible(alt, cond, moon, sun_alt=float(sun["alt"][i]), az=az,
                                          sun_sep=ssep).tolist())  # fmt: skip
    source = (REPO_ROOT / "app" / "views" / "skylive.js").read_text()
    source += "\nexport { prepare, frame, limitAt, seps, maxLimit };\n"
    with sync_api.sync_playwright() as p:
        browser = None
        for kwargs in ({}, {"channel": "chrome"}):
            try:
                browser = p.chromium.launch(**kwargs)
                break
            except Exception:
                continue
        if browser is None:
            pytest.skip("no Chromium or Chrome for Playwright")
        page = browser.new_page()
        got = page.evaluate(
            """async ([src, data, steps, alt, az]) => {
                const url = URL.createObjectURL(new Blob([src], {type: "text/javascript"}));
                const m = await import(url);
                const P = m.prepare(data);
                P.maxLimit = m.maxLimit(P);
                return steps.map((i) => {
                    const F = m.frame(P, data.steps[i], "here");
                    return alt.map((a, k) => m.limitAt(P, F, a, az[k], ...m.seps(F, a, az[k])));
                });
            }""",
            [source, json.loads(json.dumps(data)), steps, alt.tolist(), az.tolist()],
        )
        browser.close()
    for e, g in zip(exact, got, strict=True):
        e, g = np.array(e), np.array(g)
        err = np.abs(e - g)[e > -1.5]
        assert len(err) > 100
        assert np.percentile(err, 99) < 0.06 and err.max() < 0.25
