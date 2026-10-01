"""Updating the atlas from night lights, and city glow: the LZW strip reader, the projection,
ring convolution, the kernel fit, and the directional city-glow query."""

from __future__ import annotations

import json
import math
import struct

import numpy as np
import pytest

from skytrust import lightpollution as lp
from skytrust import skyglow as sg

# ---------- LZW + strip GeoTIFF ----------


def lzw_rows(values: np.ndarray) -> list[bytes]:
    """Each row LZW-compressed by libtiff (via Pillow), the reference TIFF implementation, so
    the decoder is checked against an independent encoder."""
    import io

    Image = pytest.importorskip("PIL.Image")
    from PIL import TiffImagePlugin

    TiffImagePlugin.WRITE_LIBTIFF = True
    try:
        buf = io.BytesIO()
        Image.fromarray(values.astype("<f4"), mode="F").save(
            buf, format="TIFF", compression="tiff_lzw", tiffinfo={278: 1}
        )
    finally:
        TiffImagePlugin.WRITE_LIBTIFF = False
    buf.seek(0)
    with Image.open(buf) as im:
        offsets, counts = im.tag_v2[273], im.tag_v2[279]
    raw = buf.getvalue()
    return [raw[o : o + c] for o, c in zip(offsets, counts, strict=True)]


@pytest.mark.parametrize("width", [7, 300, 5000])
def test_lzw_decoder_matches_libtiff(width):
    rng = np.random.default_rng(width)
    # mostly zeros (like night lights) with bright speckle: exercises table growth and resets
    vals = np.where(rng.random((6, width)) < 0.7, 0, rng.uniform(0, 900, (6, width)))
    for row, blob in zip(vals.astype("<f4"), lzw_rows(vals), strict=True):
        assert sg.lzw_decode(blob)[: width * 4] == row.tobytes()


def test_lzw_stops_early_when_asked():
    vals = np.arange(4000, dtype="<f4").reshape(1, -1)
    blob = lzw_rows(vals)[0]
    assert sg.lzw_decode(blob, max_out=400)[:400] == vals.tobytes()[:400]


def write_strip_tiff(path, values: np.ndarray, west: float, north: float, step: float,
                     lzw: bool = True, nodata: float = -999.9) -> None:  # fmt: skip
    """Float32 GeoTIFF with one row per strip, like the NASA night-lights files."""
    h, w = values.shape
    strips = lzw_rows(values) if lzw else [values[r].astype("<f4").tobytes() for r in range(h)]
    nodata_txt = f"{nodata}\x00".encode()
    body = b"".join(strips)
    data_at = 8
    extra_at = data_at + len(body)
    offs = [data_at + sum(len(s) for s in strips[:i]) for i in range(h)]
    extras = [
        struct.pack(f"<{h}I", *offs), struct.pack(f"<{h}I", *[len(s) for s in strips]),
        struct.pack("<3d", step, step, 0.0), struct.pack("<6d", 0, 0, 0, west, north, 0),
        nodata_txt,
    ]  # fmt: skip
    pos, at = extra_at, []
    for e in extras:
        at.append(pos)
        pos += len(e)
    entries = [
        (256, 4, 1, w), (257, 4, 1, h), (258, 3, 1, 32), (259, 3, 1, 5 if lzw else 1),
        (273, 4, h, at[0]), (277, 3, 1, 1), (278, 3, 1, 1), (279, 4, h, at[1]), (339, 3, 1, 3),
        (33550, 12, 3, at[2]), (33922, 12, 6, at[3]), (42113, 2, len(nodata_txt), at[4]),
    ]  # fmt: skip
    with open(path, "wb") as f:
        f.write(b"II*\x00" + struct.pack("<I", pos))
        f.write(body)
        for e in extras:
            f.write(e)
        f.write(struct.pack("<H", len(entries)))
        for tag, typ, count, value in entries:
            f.write(struct.pack("<HHII", tag, typ, count, value))
        f.write(struct.pack("<I", 0))


@pytest.mark.parametrize("lzw", [True, False])
def test_strip_window_reads_the_right_cells_and_drops_nodata(tmp_path, lzw):
    rng = np.random.default_rng(1)
    vals = rng.uniform(0, 50, size=(30, 40)).astype("<f4")
    vals[3, 5] = -999.9
    path = tmp_path / "lights.tif"
    write_strip_tiff(path, vals, west=-125.0, north=40.0, step=0.25, lzw=lzw)
    out, north, west, step = sg.read_strip_window(path, (35.1, 39.4), (-123.9, -117.2))
    r0, c0 = round((40.0 - north) / 0.25), round((west + 125.0) / 0.25)
    expect = vals[r0 : r0 + out.shape[0], c0 : c0 + out.shape[1]].copy()
    expect[expect < 0] = 0
    assert np.allclose(out, expect)
    full, *_ = sg.read_strip_window(path, (32.5, 40.0), (-125.0, -115.0))
    assert full[3, 5] == 0.0  # nodata became "no light"


# ---------- projection, rings, kernel ----------


def test_laea_preserves_distances_near_the_centre():
    lat0, lon0 = 40.5, -119.5
    for la, lo in [(41.0, -120.0), (38.6, -121.5), (36.1, -115.2)]:
        x0, y0 = sg.laea(lat0, lon0, lat0, lon0)
        x, y = sg.laea(la, lo, lat0, lon0)
        true = float(lp.haversine_km(lat0, lon0, la, lo))
        assert math.hypot(x - x0, y - y0) == pytest.approx(true, rel=0.03)


def test_ring_convolution_matches_brute_force():
    rng = np.random.default_rng(2)
    vals = np.zeros((60, 70))
    vals[rng.integers(0, 60, 25), rng.integers(0, 70, 25)] = rng.uniform(1, 10, 25)
    grid = sg.KmGrid(vals, 0.0, 0.0, 1.0)
    edges = [0.0, 1.5, 4, 9, 20]
    rings = sg.convolve_rings(grid, edges)
    r, c = 31, 22
    yy, xx = np.mgrid[0:60, 0:70]
    d = np.hypot(yy - r, xx - c)
    for k, (a, b) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        assert rings[k][r, c] == pytest.approx(vals[(d >= a) & (d < b)].sum(), abs=1e-8)


def test_kernel_fit_recovers_a_known_kernel():
    rng = np.random.default_rng(3)
    X = rng.gamma(1.0, 50.0, size=(4000, 5))
    true_w = np.array([3.0, 1.0, 0.3, 0.0, 0.01])
    y = X @ true_w
    w = sg.fit_kernel(X, y, natural=174.0)
    assert np.allclose(w, true_w, rtol=1e-6, atol=1e-8)
    assert (w >= 0).all()


def test_mag_error_sign_and_scale():
    assert sg.mag_error(174.0, 174.0, 174.0) == pytest.approx(0.0)
    # predicting double the total brightness = 0.753 mag too bright (negative)
    assert sg.mag_error(174.0 * 3, 174.0, 174.0) == pytest.approx(-2.5 * math.log10(2))


def test_zone_scale_matches_the_2025_atlas_colour_bar():
    assert sg.zone_of_ratio([0.0, 0.009, 0.01, 0.99, 1.0, 50.0]).tolist() == [0, 0, 1, 6, 7, 14]
    centers = sg.zone_center_ratio()
    assert len(centers) == 15 and (np.diff(centers) > 0).all()
    assert sg.zone_of_ratio(centers).tolist() == list(range(15))


def test_emission_weights_by_cell_area():
    rad = np.ones((4, 4), dtype=np.float32)
    e = sg.emission(rad, north=60.0, step=1.0, factor=2)
    # the northern cells are smaller (cos latitude), so emit less for the same radiance
    assert e[0, 0] < e[1, 0]


# ---------- city glow ----------


def city_sources(city_lat=38.0, city_lon=-120.0) -> sg.Sources:
    rng = np.random.default_rng(4)
    lat = city_lat + rng.normal(0, 0.03, 400)
    lon = city_lon + rng.normal(0, 0.03, 400)
    cities = [{"name": "Testville", "lat": city_lat, "lon": city_lon, "pop": 200000,
               "country": "US"}]  # fmt: skip
    return sg.Sources(lat, lon, np.full(400, 5.0), cities, np.array([3.0, 1.0, 0.3, 0.1, 0.03,
                      0.01, 0.003, 0.001, 0.0003]), np.array(sg.RING_EDGES_KM), 1.0)  # fmt: skip


def test_city_glow_points_at_the_city():
    src = city_sources()
    glow = sg.city_glow(src, 38.0, -120.6)  # ~53 km west of the city
    strongest = max(glow["sectors"], key=lambda s: s["dome"])
    assert strongest["direction"] in {"E", "ENE", "ESE"}
    assert glow["cities"][0]["name"] == "Testville" and glow["cities"][0]["direction"] == "E"
    assert glow["cities"][0]["distance_km"] == pytest.approx(52.6, abs=1.0)
    # the advice faces away from the city (west), not just anywhere without lights
    assert set(glow["darkest_quarter"]) <= {"SW", "WSW", "W", "WNW", "NW"}
    assert sum(s["overhead_share"] for s in glow["sectors"]) == pytest.approx(1.0)


def test_city_glow_is_weaker_farther_away():
    src = city_sources()
    near = sg.city_glow(src, 38.0, -120.3)["dome_total"]
    far = sg.city_glow(src, 38.0, -121.2)["dome_total"]
    assert near > far > 0


def test_city_radius_grows_with_population():
    r = sg.city_radius_km([1000, 100_000, 1_500_000])
    assert r[0] == pytest.approx(2.0) and r[0] < r[1] < r[2] <= 25.0


def test_committed_artifacts_are_consistent():
    card = json.loads(sg.MODEL_PATH.read_text())
    assert card["year"] >= 2025 and len(card["kernel"]) == len(card["ring_edges_km"]) - 1
    assert all(w >= 0 for w in card["kernel"])
    assert card["spatial_cv"]["rmse_mag"] < card["atlas_sigma_mag"]  # model error below atlas σ
    src = sg.load_sources()
    assert src is not None and len(src.lat) > 1000 and src.dome_ref > 0
    from skytrust.config import load_sites

    for site in load_sites():
        assert sg.city_glow(src, site.lat, site.lon)["sectors"]
