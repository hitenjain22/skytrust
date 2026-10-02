"""City glow by direction (domes) and smoke or haze (haze): the physics against its sources and
limiting cases, the site grid against the exact sum, and the air-quality parsing offline."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from skytrust import domes, haze, sky
from skytrust.config import REPO_ROOT

FIXTURES = REPO_ROOT / "tests" / "fixtures"


# ---------- domes ----------


def test_the_scattering_phase_functions_are_normalised_and_continuous():
    """Garstang's aerosol fit (Cinzano et al. 2000, eq. 14) and Rayleigh scattering each
    integrate to 1 over the sphere, and the aerosol fit's pieces join at 10° and 124°."""
    w = np.linspace(0, 180, 180_001)
    for f in (domes.aerosol_phase, domes.rayleigh_phase):
        total = np.trapezoid(f(w) * 2 * math.pi * np.sin(np.radians(w)), np.radians(w))
        assert total == pytest.approx(1.0, abs=0.005)
    for edge in (10.0, 124.0):
        assert float(domes.aerosol_phase(edge - 1e-6)) == pytest.approx(
            float(domes.aerosol_phase(edge + 1e-6)), rel=0.03)  # fmt: skip


def test_the_emission_function_sends_all_light_upwards():
    """Garstang's city emission function integrates to 1 over the upper hemisphere."""
    psi = np.linspace(0, math.pi / 2, 20_001)
    total = np.trapezoid(domes.emission(psi) * 2 * math.pi * np.sin(psi), psi)
    assert total == pytest.approx(1.0, abs=0.002)
    assert float(domes.emission(math.pi / 2 + 0.01)) == 0.0


def test_the_atmosphere_matches_the_papers_stated_extinction():
    """K = 1 is quoted as 0.33 mag of vertical extinction in V, optical depth 0.3."""
    tau = domes.BETA_MOL / domes.C_MOL + domes.BETA_AER / domes.A_AER
    assert tau == pytest.approx(0.30, abs=0.01)
    assert haze.MAG_PER_TAU * tau == pytest.approx(0.33, abs=0.01)


def test_a_dome_is_brightest_low_towards_its_city():
    alts = [5.0, 20.0, 45.0, 90.0]
    toward = domes.brightness(30.0, alts, [0.0] * 4)
    away = domes.brightness(30.0, alts, [180.0] * 4)
    assert np.all(np.diff(toward) < 0)  # brighter the lower you look towards the city
    assert toward[0] > 20 * away[0]
    assert toward[-1] == pytest.approx(away[-1])  # the zenith has no direction


@pytest.mark.slow
def test_with_air_molecules_alone_a_lit_plain_brightens_like_the_air_path():
    """Limiting case: a uniform lit plain and symmetric (Rayleigh) scattering give a sky that
    brightens towards the horizon about like sec z (less near the horizon, where extinction
    takes over). Checks the geometry independently of the aerosol phase function."""
    saved = domes.BETA_AER
    try:
        domes.BETA_AER = 0.0
        alts = np.array([10.0, 20.0, 45.0, 90.0])
        dist = np.geomspace(0.3, 150, 60)
        dazs = np.arange(0, 181, 6.0)
        total = np.zeros(len(alts))
        for i, d in enumerate(dist):
            a, z = np.meshgrid(alts, dazs, indexing="ij")
            b = domes.brightness(float(d), a.ravel(), z.ravel()).reshape(a.shape).mean(1)
            width = (dist[min(i + 1, len(dist) - 1)] - dist[max(i - 1, 0)]) / 2
            total += b * 2 * math.pi * d * width
        ratio = total / total[-1]
        sec_z = 1 / np.sin(np.radians(alts))
        assert ratio == pytest.approx(sec_z, rel=0.2)
    finally:
        domes.BETA_AER = saved


@pytest.mark.skipif(domes.load_shape() is None, reason="dome table not built")
def test_the_site_grid_matches_the_exact_sum_over_sources():
    """sky_shape bins sources by distance node and 5° of azimuth; against summing every
    source's own (interpolated) shape it agrees to within 15% (~0.15 mag)."""
    rng = np.random.default_rng(3)
    d = rng.uniform(2, 120, 40)
    az = rng.uniform(0, 360, 40)
    w = rng.uniform(0.1, 1, 40)
    alts = np.array([5.0, 10.0, 20.0, 40.0, 90.0])
    grid = domes.sky_shape(d, az, w, alts=alts, az=np.array([0.0, 90.0, 200.0]))
    exact = np.zeros_like(grid)
    for di, ai, wi in zip(d, az, w, strict=True):
        for c, look in enumerate((0.0, 90.0, 200.0)):
            daz = abs((look - ai + 180) % 360 - 180)
            b = domes.brightness(float(di), alts, [daz] * len(alts))
            exact[:, c] += wi * b / domes.brightness(float(di), [90.0], [0.0])[0]
    exact /= w.sum()
    assert np.all(np.abs(grid / exact - 1) < 0.15)
    assert grid[-1] == pytest.approx([1.0, 1.0, 1.0], abs=1e-6)


@pytest.mark.skipif(sky._sources() is None or domes.load_shape() is None,
                    reason="light sources or dome table not built")  # fmt: skip
def test_from_santa_barbara_the_glow_is_towards_los_angeles_not_the_ocean():
    """Los Angeles lies to the ESE of Santa Barbara (about 140 km) and the Pacific to the S:
    10° up, the sky towards LA is brighter."""
    g = sky._glow(34.42, -119.70)
    alt10 = int(np.flatnonzero(sky.GLOW_ALTS == 10)[0])
    ese = g[alt10, int(np.flatnonzero(sky.GLOW_AZ == 110)[0])]
    south = g[alt10, int(np.flatnonzero(sky.GLOW_AZ == 190)[0])]
    assert ese > 1.5 * south


# ---------- haze ----------


def test_extinction_follows_schaefers_split():
    """Rayleigh 0.1066 exp(-h/8.2 km) + ozone 0.031 + 1.086 AOD; Schaefer's standard aerosol
    (0.1 mag) is the reference, so a typical night has no extra dimming."""
    assert haze.extinction(0, haze.AOD_REF) == pytest.approx(0.1066 + 0.031 + 0.1)
    assert haze.extinction(2000, 0.0) == pytest.approx(0.1066 * math.exp(-2 / 8.2) + 0.031)
    assert haze.extinction(500, None) == haze.extinction(500, haze.AOD_REF)
    assert haze.dimming(haze.AOD_REF) == 0.0 and haze.dimming(0.01) == 0.0  # never a bonus
    assert haze.dimming(1.0) == pytest.approx(haze.MAG_PER_TAU * (1.0 - haze.AOD_REF))


@pytest.mark.parametrize(
    ("aod", "title"),
    [(0.08, None), (0.25, None), (0.30, "Some haze"), (0.6, "Smoke or haze"),
     (1.2, "Thick smoke or haze"), (None, None), (float("nan"), None)],
)  # fmt: skip
def test_haze_words_only_when_it_matters(aod, title):
    got = haze.words(aod)
    assert (got[0] if got else None) == title


def test_the_air_quality_forecast_is_parsed_and_summarised_per_night():
    """A real response (Santa Barbara, saved): hourly AOD in UTC, nulls at the end where CAMS
    stops. The night's value is the median over its dark hours; none past the forecast."""
    payload = json.loads((FIXTURES / "air_quality_santa_barbara.json").read_text())
    s = haze.parse(payload)
    assert s is not None and s.index.tz is not None and s.notna().all()
    assert 0 < s.min() <= s.max() < 3
    t0 = s.index[0]
    dusk, dawn = t0 + pd.Timedelta(hours=3), t0 + pd.Timedelta(hours=13)
    expected = s[(s.index >= dusk) & (s.index <= dawn)].median()
    assert haze.night_aod(s, dusk, dawn) == pytest.approx(expected)
    late = s.index[-1] + pd.Timedelta(days=1)
    assert haze.night_aod(s, late, late + pd.Timedelta(hours=10)) is None
    assert haze.parse({"hourly": {}}) is None


def test_a_slow_air_quality_service_never_breaks_anything():
    class Down:
        def get_json(self, url, params):
            raise TimeoutError("no answer")

    from skytrust.config import load_settings

    assert haze.fetch(34.4, -119.7, load_settings(), client=Down()) is None


@pytest.mark.network
def test_live_air_quality_api_contract():
    from skytrust.config import load_settings

    s = haze.fetch(34.42, -119.70, load_settings())
    assert s is not None and len(s) >= 48 and (s >= 0).all()
