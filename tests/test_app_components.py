"""Fast unit tests for the app's building blocks (no Streamlit runtime needed)."""

from __future__ import annotations

import re
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from skytrust.config import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "app"))

from views import common, theme  # noqa: E402
from views import components as ui  # noqa: E402

DUSK = pd.Timestamp("2026-10-01 03:10", tz="UTC")  # 8:10 PM PDT
DAWN = pd.Timestamp("2026-10-01 12:30", tz="UTC")  # 5:30 AM PDT
TZ = "America/Los_Angeles"


def night(**kw):
    base = {"dusk_utc": DUSK, "dawn_utc": DAWN, "moon_illum": 0.75, "moon_up": [],
            "best_window": None}  # fmt: skip
    return SimpleNamespace(**(base | kw))


@pytest.mark.parametrize(
    ("deg", "name"),
    [(0, "New Moon"), (45, "Waxing crescent"), (90, "First quarter"), (135, "Waxing gibbous"),
     (180, "Full Moon"), (225, "Waning gibbous"), (270, "Last quarter"), (315, "Waning crescent"),
     (359, "New Moon"), (None, "Moon")],
)  # fmt: skip
def test_phase_names(deg, name):
    assert ui.phase_name(deg) == name


def _arcs(svg: str) -> list[tuple[float, int]]:
    """(rx, sweep flag) of the two arcs in the lit path."""
    path = re.search(r'd="([^"]+)"', svg).group(1)
    return [(float(rx), int(sweep)) for rx, sweep in re.findall(r"A([\d.]+),[\d.]+ 0 0 (\d)", path)]


def test_moon_svg_geometry():
    # full and new: the terminator ellipse is as wide as the disk; quarters: a straight line
    for deg in (0, 180):
        assert _arcs(ui.moon_svg(deg, 44))[1][0] == pytest.approx(21, abs=0.01)
    for deg in (90, 270):
        assert _arcs(ui.moon_svg(deg, 44))[1][0] == pytest.approx(0, abs=0.01)
    # waxing is lit on the right (edge arc drawn clockwise, sweep 1); waning mirrored
    assert _arcs(ui.moon_svg(60))[0][1] == 1 and _arcs(ui.moon_svg(300))[0][1] == 0
    # crescent vs gibbous: the terminator bulges the opposite way
    assert _arcs(ui.moon_svg(45))[1][1] != _arcs(ui.moon_svg(135))[1][1]
    assert 'aria-label="Waning gibbous"' in ui.moon_svg(225)


def test_block_keeps_meaningful_spaces_and_drops_indentation():
    assert ui.block("  <b>Auburn ", "<i>GO</i>\n    </b>") == "<b>Auburn <i>GO</i></b>"


def test_user_text_is_escaped():
    assert "<script>" not in ui.pill("<script>x</script>", "#fff")
    assert ui.esc('a"b<') == "a&quot;b&lt;"


def test_meter_and_bar_clamp_and_handle_missing_probability():
    assert "width:0.0%" in ui.meter(None) and "width:42.0%" in ui.meter(0.42)
    assert "width:100.0%" in ui.bar(1.7, "#fff") and "width:0.0%" in ui.bar(-0.2, "#fff")


def test_status_pill_carries_meaning_in_the_dot_not_the_text_colour():
    html = ui.status("GO", "#5E9E80")
    assert "--c:#5E9E80" in html and "sk-dot" in html and ">GO<" in html


def test_duration_and_short_time():
    assert common.duration(DUSK, DAWN) == "9 h 20 min"
    assert common.duration(DUSK, DUSK + pd.Timedelta(hours=2)) == "2 h"
    assert common.duration(DUSK, DUSK + pd.Timedelta(minutes=40)) == "40 min"
    assert common.short_time(pd.Timestamp("2026-10-01 04:00", tz="UTC"), TZ) == "9 PM"
    assert common.short_time(DUSK, TZ) == "8:10 PM"


def test_moon_sentence_describes_when_it_is_up():
    assert "below the horizon all night" in common.moon_sentence(night(), TZ)
    up_all = night(moon_up=[(DUSK - pd.Timedelta(hours=1), DAWN + pd.Timedelta(hours=1))])
    assert "up all night" in common.moon_sentence(up_all, TZ)
    rising = night(moon_up=[(pd.Timestamp("2026-10-01 06:00", tz="UTC"), DAWN)])
    assert "rising at 11 PM" in common.moon_sentence(rising, TZ)
    setting = night(
        moon_up=[(DUSK - pd.Timedelta(hours=2), pd.Timestamp("2026-10-01 07:30", tz="UTC"))]
    )
    assert "up until 12:30 AM" in common.moon_sentence(setting, TZ)


def test_moon_advice_only_when_it_matters():
    bright = night(moon_up=[(DUSK, DAWN)], moon_illum=0.9)
    assert common.moon_advice(bright)[0] == "bright"
    assert common.moon_advice(night(moon_illum=0.6))[0] == "dark"  # below the horizon all night
    brief = night(moon_up=[(DUSK, DUSK + pd.Timedelta(hours=1))], moon_illum=0.5)
    assert common.moon_advice(brief) is None


def test_nights_that_have_ended_are_dropped():
    """A forecast cached before dawn must not keep showing last night as 'tonight'."""
    past = SimpleNamespace(dawn_utc=DAWN)
    nxt = SimpleNamespace(dawn_utc=DAWN + pd.Timedelta(days=1))
    fc = SimpleNamespace(nights=[past, nxt])
    assert common.upcoming(fc, DAWN - pd.Timedelta(minutes=1)) == [past, nxt]
    assert common.upcoming(fc, DAWN + pd.Timedelta(minutes=1)) == [nxt]
    assert common.upcoming(None, DAWN) == []


def test_stylesheet_follows_the_active_theme_and_respects_reduced_motion():
    """Colours derive from currentColor (so light, dark and night vision all work) and motion
    has a reduced-motion fallback."""
    css = theme.css(theme.MONO)
    assert css.count("currentColor") > 20
    assert "prefers-reduced-motion: reduce" in css and "cubic-bezier(0.23, 1, 0.32, 1)" in css
    assert "@media (hover: hover) and (pointer: fine)" in css  # hover effects only with a mouse
    assert "animation-timeline: view()" in css and "@supports (animation-timeline: view())" in css


def test_place_names_are_friendly_and_details_come_from_config():
    from skytrust.config import load_sites

    sites = {s.id: s for s in load_sites()}
    assert common.place_name(sites["AUN"]) == "Auburn"  # not IEM's "AURBURN MUNICIPAL AIRPORT"
    assert common.place_detail(sites["TRK"]) == "High Sierra · 1,798 m"


def test_map_labels_for_neighbouring_sites_do_not_collide():
    """Regression: with one shared label position, Truckee's label was hidden behind Auburn's."""
    from views import charts

    table = pd.DataFrame(
        {"site": ["AUN", "TRK", "SAC"], "place": ["Auburn", "Truckee", "Sacramento"],
         "p": [0.99, 0.98, None], "lat": [38.95, 39.32, 38.51], "lon": [-121.08, -120.14, -121.5]}
    )  # fmt: skip
    fig = charts.site_map(table, theme.MOON, 0.7, 0.4)
    positions = {t.text[0].split()[0]: t.textposition for t in fig.data}
    assert len(fig.data) == 3 and positions["Auburn"] != positions["Truckee"]
    assert fig.data[2].text[0].endswith("–")  # missing probability shown as a dash, not "nan%"
