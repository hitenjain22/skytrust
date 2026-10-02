"""Light pollution: conversions from the atlas paper, the GeoTIFF tile reader, and the
questions the app asks of the grid (how dark here, darkest nearby, area shares)."""

from __future__ import annotations

import math
import struct

import numpy as np
import pytest

from skytrust import lightpollution as lp


def test_sqm_uses_the_papers_natural_background():
    assert lp.sqm(0.0) == pytest.approx(22.0)  # no artificial light: natural 22.0 mag/arcsec²
    # artificial = natural doubles the brightness: 2.5 * log10(2) ≈ 0.753 mag brighter
    assert lp.sqm(174.0) == pytest.approx(22.0 - 2.5 * math.log10(2))
    assert lp.sqm(174.0 * 99) == pytest.approx(17.0)  # 100x natural = 5 mag brighter
    assert lp.ratio(87.0) == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("r", "index", "name"),
    [(0.0, 0, "Pristine sky"), (0.0099, 0, "Pristine sky"), (0.01, 1, "Near pristine"),
     (1.0, 7, "Heavily polluted"), (3.0, 9, "Milky Way hidden"), (100.0, 13, "No dark adaptation")],
)  # fmt: skip
def test_atlas_levels_double_each_step(r, index, name):
    got = lp.level(r)
    assert got[0] == index and got[1] == name


@pytest.mark.parametrize(
    ("m", "cls"),
    [(22.0, "1"), (21.76, "1"), (21.7, "2"), (21.5, "3"), (21.0, "4"), (20.5, "4.5"),
     (19.5, "5"), (18.9, "6"), (18.2, "7"), (17.0, "8–9")],
)  # fmt: skip
def test_bortle_classes_from_sqm(m, cls):
    assert lp.bortle(m)[0] == cls


def test_bortle_number_matches_the_class_table():
    m = np.array([22.0, 21.5, 20.5, 18.9, 17.0])
    assert lp.bortle_number(m).tolist() == [1.0, 3.0, 4.5, 6.0, 8.5]


# ---------- the GeoTIFF reader, on a tiny synthetic atlas ----------


def write_tiled_tiff(path, data: np.ndarray, tile: int, west: float, north: float, step: float):
    """Minimal little-endian tiled float32 GeoTIFF with the same tags the atlas uses."""
    h, w = data.shape
    tiles_down, tiles_across = math.ceil(h / tile), math.ceil(w / tile)
    blobs = []
    for tr in range(tiles_down):
        for tc in range(tiles_across):
            t = np.zeros((tile, tile), dtype="<f4")
            block = data[tr * tile : (tr + 1) * tile, tc * tile : (tc + 1) * tile]
            t[: block.shape[0], : block.shape[1]] = block
            blobs.append(t.tobytes())
    n = len(blobs)
    header = 8
    data_start = header + 1  # an odd offset, like the real file: tiles need not be aligned
    offsets = [data_start + i * len(blobs[0]) for i in range(n)]
    extra_start = data_start + n * len(blobs[0])
    offsets_bytes = struct.pack(f"<{n}I", *offsets)
    counts_bytes = struct.pack(f"<{n}I", *[len(b) for b in blobs])
    scale = struct.pack("<3d", step, step, 0.0)
    tie = struct.pack("<6d", 0, 0, 0, west, north, 0)
    extras = [offsets_bytes, counts_bytes, scale, tie]
    extra_offsets, pos = [], extra_start
    for e in extras:
        extra_offsets.append(pos)
        pos += len(e)
    ifd_offset = pos
    entries = [
        (256, 4, 1, w), (257, 4, 1, h), (258, 3, 1, 32), (259, 3, 1, 1), (277, 3, 1, 1),
        (322, 3, 1, tile), (323, 3, 1, tile), (324, 4, n, extra_offsets[0]),
        (325, 4, n, extra_offsets[1]), (339, 3, 1, 3), (33550, 12, 3, extra_offsets[2]),
        (33922, 12, 6, extra_offsets[3]),
    ]  # fmt: skip
    with open(path, "wb") as f:
        f.write(b"II*\x00" + struct.pack("<I", ifd_offset) + b"\x00")
        for blob in blobs:
            f.write(blob)
        for e in extras:
            f.write(e)
        f.write(struct.pack("<H", len(entries)))
        for tag, typ, count, value in entries:
            f.write(struct.pack("<HHII", tag, typ, count, value))
        f.write(struct.pack("<I", 0))


@pytest.fixture
def atlas(tmp_path):
    rng = np.random.default_rng(3)
    data = rng.uniform(0, 5, size=(21, 30)).astype("<f4")  # mcd/m²
    path = tmp_path / "atlas.tif"
    write_tiled_tiff(path, data, tile=8, west=-125.0, north=40.0, step=0.1)
    return path, data


def test_reader_parses_the_tiff_layout(atlas):
    path, data = atlas
    info = lp.read_tiff_info(path)
    assert (info.width, info.height, info.tile) == (30, 21, 8)
    assert info.west == -125.0 and info.north == 40.0 and info.step == pytest.approx(0.1)


def test_window_crosses_tile_boundaries_exactly(atlas):
    path, data = atlas
    window, north, west, step = lp.read_window(path, (38.3, 39.45), (-124.35, -122.75))
    r0, c0 = round((40.0 - north) / 0.1), round((west + 125.0) / 0.1)
    assert np.array_equal(window, data[r0 : r0 + window.shape[0], c0 : c0 + window.shape[1]])
    assert north >= 39.45 and west <= -124.35  # the requested box is covered


def test_reader_rejects_other_layouts(tmp_path):
    bad = tmp_path / "bad.tif"
    bad.write_bytes(b"MM\x00*" + b"\x00" * 16)
    with pytest.raises(ValueError):
        lp.read_tiff_info(bad)


# ---------- grid questions ----------


def city_grid() -> lp.Grid:
    """0.01° grid around (38, -121): a bright 'city' at the centre fading outwards, and a dark
    corner to the north-east."""
    n = 201
    lat = 39.0 - (np.arange(n) + 0.5) * 0.01
    lon = -122.0 + (np.arange(n) + 0.5) * 0.01
    glat, glon = np.meshgrid(lat, lon, indexing="ij")
    d = lp.haversine_km(38.0, -121.0, glat, glon)
    ucd = (5000.0 * np.exp(-d / 12.0)).astype(np.float32)
    return lp.Grid(ucd, north=39.0, west=-122.0, step=0.01)


def test_point_report_at_the_city_centre():
    g = city_grid()
    p = lp.point(g, 38.0, -121.0)
    a = p["artificial_ucd"]  # the cell holding the centre, ~27x natural
    assert p["sqm"] == pytest.approx(22.0 - 2.5 * math.log10((a + 174) / 174))
    assert p["ratio"] > 20 and p["bortle"] == "7"  # 18.0-18.5 mag/arcsec²
    assert p["sqm_hi"] - p["sqm_lo"] == pytest.approx(0.30)
    assert lp.point(g, 45.0, -121.0) is None  # outside the grid


def test_darkest_within_prefers_the_nearest_spot_that_is_dark_enough():
    g = city_grid()
    d = lp.darkest_within(g, 38.0, -121.0, 50)
    assert d["distance_km"] <= 50.5
    # anything darker than it by more than the model's 0.15 mag would have been chosen instead
    vals, *_ = g.around(38.0, -121.0, 50)
    assert d["sqm"] >= float(g.sqm(vals).max()) - 0.15
    assert d["direction"] in {"N", "NE", "E", "SE", "S", "SW", "W", "NW"}


def test_nearest_dark_finds_a_bortle_3_sky():
    g = city_grid()
    d = lp.nearest_dark(g, 38.0, -121.0, max_bortle=3, radius_km=120)
    assert d is not None and float(d["bortle"]) <= 3
    assert lp.nearest_dark(g, 38.0, -121.0, max_bortle=3, radius_km=3) is None


def test_darker_skies_can_be_limited_to_places_on_land():
    """Given candidate places (on land), the searches only ever return one of them, with its
    name: the atlas has values over the sea too, so a bare grid search from a coastal town can
    point into the ocean (it once sent Santa Barbara 8 km out into the Channel)."""
    g = city_grid()
    places = {"lat": np.array([38.1, 38.6, 38.9]), "lon": np.array([-120.9, -120.4, -120.2]),
              "name": ["Near Town", "Far Hamlet", "Dark Ridge"], "id": ["a", "b", "c"]}  # fmt: skip
    d = lp.nearest_dark(g, 38.0, -121.0, max_bortle=3, radius_km=150, places=places)
    assert d is not None and d["name"] in places["name"] and float(d["bortle"]) <= 3
    w = lp.darkest_within(g, 38.0, -121.0, 120, places=places)
    assert (w["lat"], w["lon"]) in set(zip(places["lat"], places["lon"], strict=True))
    assert w["name"] == "Far Hamlet" and w["id"] == "b"  # as dark as Dark Ridge, and nearer
    assert lp.darkest_within(g, 38.0, -121.0, 5, places=places) is None  # none that close


@pytest.mark.skipif(not (lp.GRID_PATH.exists()), reason="light-pollution grid not built")
def test_santa_barbaras_darker_sky_is_on_land():
    from skytrust import gazetteer
    from skytrust.config import load_settings

    gaz = gazetteer.load()
    if gaz is None:
        pytest.skip("gazetteer not built")
    places = gazetteer.land_points(gaz)
    report = lp.site_report(lp.load(), 34.42, -119.70, load_settings(), places=places)
    for d in [report["nearest_dark"], *report["darkest"].values()]:
        if d is not None:
            assert d.get("name"), d  # a named place, not a grid cell (maybe at sea)


def test_area_shares_add_up_to_one():
    shares = lp.area_shares(city_grid(), 38.0, -121.0, 50)
    assert sum(shares.values()) == pytest.approx(1.0)
    assert shares["7–9"] > 0  # the city core


def test_save_and_load_round_trip_within_half_a_millimag(tmp_path):
    g = city_grid()
    path = lp.save(g, tmp_path / "lp.npz")
    back = lp.load(path)
    assert np.nanmax(np.abs(back.sqm(back.ucd) - g.sqm(g.ucd))) <= 0.0005 + 1e-9
    assert (back.north, back.west, back.step) == (g.north, g.west, g.step)
    assert lp.load(tmp_path / "missing.npz") is None


def test_overlay_is_transparent_where_the_sky_is_clean():
    g = city_grid()
    img = lp.overlay_rgba(g)
    assert img.shape == (*g.ucd.shape, 4)
    clean = lp.ratio(g.ucd, g.natural_ucd) < 0.04
    assert (img[clean][:, 3] == 0).all() and img[~clean][:, 3].min() > 0


def test_committed_artifact_covers_every_site():
    from skytrust.config import load_sites

    g = lp.load()
    assert g is not None, "run `python -m skytrust build-light-pollution`"
    for site in load_sites():
        assert lp.point(g, site.lat, site.lon) is not None


def test_cli_explains_a_missing_atlas(tmp_path, capsys):
    from skytrust import __main__ as cli

    assert cli.main(["build-light-pollution", "--atlas", str(tmp_path / "none.tif")]) == 2
    assert "Download" in capsys.readouterr().err
