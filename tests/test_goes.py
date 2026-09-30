"""GOES Clear Sky Mask: scan-angle geometry, pixel extraction, and resumable fetching."""

from __future__ import annotations

import io
import math
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from skytrust.config import Site
from skytrust.data import goes

h5py = pytest.importorskip("h5py")

GOES16 = {  # projection constants from the GOES-R Product User Guide worked example
    "semi_major_axis": 6378137.0,
    "semi_minor_axis": 6356752.31414,
    "perspective_point_height": 35786023.0,
    "longitude_of_projection_origin": -75.0,
}
GOES18 = GOES16 | {"longitude_of_projection_origin": -137.0}
SAC = Site("SAC", "Sacramento", 38.5069, -121.495, 8.0, "valley", "America/Los_Angeles")


def test_scan_angles_match_product_user_guide_example():
    # PUG vol. 3 §4.2.8.1: lat 33.846162, lon -84.690932 -> x = -0.024052, y = 0.095340 rad
    x, y = goes.scan_angles(33.846162, -84.690932, GOES16)
    assert x == pytest.approx(-0.024052, abs=1e-6)
    assert y == pytest.approx(0.095340, abs=1e-6)


def test_points_on_the_far_side_of_earth_are_not_visible():
    assert goes.scan_angles(0.0, 60.0, GOES16) is None  # Indian Ocean, invisible from 75 W


def synthetic_acm(site: Site, proj: dict, cloudy_cols: int = 2, bad_pixel: bool = True) -> bytes:
    """A tiny ACM-like NetCDF4/HDF5 file with a 9 x 9 grid centred on the site."""
    x0, y0 = goes.scan_angles(site.lat, site.lon, proj)
    step = 56e-6  # ~2 km in scan angle
    n = 9
    buf = io.BytesIO()
    with h5py.File(buf, "w") as f:
        p = f.create_dataset("goes_imager_projection", data=0)
        for k, v in proj.items():
            p.attrs[k] = v
        # store x/y as scaled integers like the real product
        xs = f.create_dataset("x", data=np.arange(n, dtype="int16"))
        xs.attrs["scale_factor"], xs.attrs["add_offset"] = step, x0 - 4 * step
        ys = f.create_dataset("y", data=np.arange(n, dtype="int16"))
        ys.attrs["scale_factor"], ys.attrs["add_offset"] = step, y0 - 4 * step
        bcm = np.zeros((n, n), dtype="uint8")
        bcm[:, 4 - 2 : 4 - 2 + cloudy_cols] = 1  # some cloudy columns inside the 5x5 box
        dqf = np.zeros((n, n), dtype="uint8")
        if bad_pixel:
            dqf[4, 4] = 1  # centre pixel flagged bad -> ignored
        f.create_dataset("BCM", data=bcm)
        f.create_dataset("DQF", data=dqf)
    return buf.getvalue()


def test_extract_averages_good_pixels_in_the_box():
    values = goes.extract_sites(synthetic_acm(SAC, GOES18), (SAC,))
    cover, n_good = values["SAC"]
    assert n_good == 24  # 5 x 5 box minus the one bad-quality pixel
    assert cover == pytest.approx(10 / 24)  # 2 cloudy columns x 5 rows, none flagged bad


def test_fetch_hours_resumes_and_skips_done_hours(tmp_path):
    client = MagicMock()
    client.get.side_effect = lambda url, params: MagicMock(
        text="<Key>ABI-L2-ACMC/2025/001/06/OR_x.nc</Key>",
        content=synthetic_acm(SAC, GOES18, 0, False),
    )
    hours = pd.date_range("2025-01-01 06:00", periods=3, freq="h", tz="UTC")
    assert goes.fetch_hours(client, hours, (SAC,), tmp_path) == 3
    series = goes.goes_hourly("SAC", tmp_path)
    assert len(series) == 3 and (series == 0.0).all()
    calls = client.get.call_count
    assert goes.fetch_hours(client, hours, (SAC,), tmp_path) == 0  # all done: no network
    assert client.get.call_count == calls


def test_missing_scan_records_nan(tmp_path):
    client = MagicMock()
    client.get.return_value = MagicMock(text="<ListBucketResult></ListBucketResult>")
    goes.fetch_hours(
        client, pd.DatetimeIndex([pd.Timestamp("2025-01-01 06:00", tz="UTC")]), (SAC,), tmp_path
    )
    assert math.isnan(goes.goes_hourly("SAC", tmp_path).iloc[0])
