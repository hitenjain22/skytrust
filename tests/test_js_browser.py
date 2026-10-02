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
