"""Every place in California: building rules (offline, synthetic inputs) and the shipped table."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from skytrust import gazetteer as g
from skytrust.config import load_places


def test_names_and_slugs():
    assert g.short_name("Davis city") == "Davis"
    assert g.short_name("Mountain House town") == "Mountain House"
    assert g.short_name("Acton CDP") == "Acton"
    assert g.short_name("Carmel-by-the-Sea city") == "Carmel-by-the-Sea"
    assert g.slug("La Cañada Flintridge") == "la-canada-flintridge"
    assert g.slug("St. Helena") == "st-helena"


def test_ids_get_the_county_only_when_names_clash():
    df = pd.DataFrame(
        {"name": ["Greenfield", "Greenfield", "Davis"], "county": ["Monterey", "Kern", "Yolo"]}
    )
    assert g.unique_ids(df, set()) == ["greenfield-monterey", "greenfield-kern", "davis"]
    # an id already taken (a featured place) is never reused
    assert g.unique_ids(pd.DataFrame({"name": ["Davis"], "county": ["Yolo"]}), {"davis"}) == [
        "davis-yolo"
    ]


def _places(**cols):
    base = {"kind": "city", "county": "X", "land_km2": 10.0}
    n = len(cols["name"])
    return pd.DataFrame({**{k: [v] * n for k, v in base.items()}, **cols})


def test_zip_codes_are_named_after_the_city_they_lie_in_not_the_nearest_centre():
    # a big city whose centre is 7 km away beats a small one whose centre is 5 km away
    places = _places(
        name=["Big City", "Small Town"],
        lat=[34.0, 34.0],
        lon=[-118.0, -118.0 + 12.0 / (111.32 * math.cos(math.radians(34)))],
        land_km2=[1200.0, 5.0],
    )
    zip_lon = -118.0 + 7.0 / (111.32 * math.cos(math.radians(34)))
    assert g.nearest_names(34.0, zip_lon, places) == ["Big City"]
    # far from every city: the nearest place of any kind
    rural = _places(name=["Town", "Hamlet"], lat=[36.0, 36.5], lon=[-117.0, -117.0],
                    kind=["city", "community"], land_km2=[5.0, 1.0])  # fmt: skip
    assert g.nearest_names(36.49, -117.0, rural) == ["Hamlet"]


def test_town_centre_rule():
    places = pd.DataFrame(
        {
            "geoid": ["1", "2"],
            "gnis_id": [10, 20],
            "name": ["Bigville", "Tiny"],
            "kind": ["city", "community"],
            "lat_internal": [34.0, 35.0],
            "lon_internal": [-118.0, -119.0],
            "land_km2": [500.0, 2.0],  # radii ~12.6 km and ~0.8 km
        }
    )
    gnis = pd.DataFrame(
        {
            "gnis_id": [10, 20, 30, 40],
            "name": ["Bigville", "Tiny", "Bigville", "Tiny"],
            "feature_class": ["Civil", "Census", "Populated Place", "Populated Place"],
            "county": ["A", "B", "A", "B"],
            "lat": [34.01, 35.0, 34.15, 35.05],  # Bigville's downtown 16.7 km away; Tiny's 5.6 km
            "lon": [-118.0, -119.0, -118.0, -119.0],
        }
    )
    out = g.town_centres(places, gnis).set_index("name")
    assert out.loc["Bigville", "point"] == "gnis-populated-place"  # within 2.5 radii: downtown
    assert out.loc["Bigville", "lat"] == pytest.approx(34.15)
    assert out.loc["Tiny", "point"] == "gnis-place-record"  # 5.6 km is another feature
    assert out.loc["Tiny", "county"] == "B"


# ---------- the shipped table (artifacts/places_ca.json) ----------


@pytest.fixture(scope="module")
def gaz():
    gz = g.load()
    if gz is None:
        pytest.skip("artifacts/places_ca.json not built")
    return gz


def test_every_incorporated_city_and_town_is_searchable(gaz):
    t = gaz.table
    assert (t["kind"].isin(["city", "town"])).sum() == 483  # California's 482 + Mountain House
    assert (t["kind"] == "zip").sum() > 1700
    assert t.index.is_unique
    for name in ["Los Angeles", "San Diego", "Davis", "Mountain House", "Bishop", "Lone Pine",
                 "Furnace Creek", "Hollywood", "La Jolla"]:  # fmt: skip
        assert (t["name"] == name).any(), name


def test_every_place_is_in_california_with_a_height(gaz):
    t = gaz.table
    assert t["lat"].between(32.4, 42.1).all() and t["lon"].between(-124.6, -114.1).all()
    assert t["elevation_m"].notna().all()
    assert t["elevation_m"].between(-90, 3700).all()  # Badwater -86 m ... highest towns
    # Death Valley is below sea level, Big Bear Lake above 2,000 m
    by = t.set_index("name")
    assert by.loc["Furnace Creek", "elevation_m"] < 0
    assert by.loc["Big Bear Lake", "elevation_m"] > 2000


def test_biggest_places_come_first_and_zip_codes_last(gaz):
    t = gaz.table
    assert list(t["name"][:3]) == ["Los Angeles", "San Diego", "San Jose"]
    kinds = t["kind"].to_numpy()
    first_zip = int(np.argmax(kinds == "zip"))
    assert (kinds[first_zip:] == "zip").all()


def test_labels_sites_and_descriptions(gaz):
    labels = gaz.labels()
    assert labels["davis"] == "Davis · Yolo County"
    assert labels["zip-95616"] == "95616 · ZIP code near Davis"
    assert labels["zip-90004"] == "90004 · ZIP code near Los Angeles"
    s = gaz.site("zip-95616")
    assert s.name == "ZIP 95616" and s.timezone == "America/Los_Angeles"
    assert s.terrain_class == "ZIP code near Davis"
    assert gaz.describe(34.0522, -118.2437) == "Los Angeles"
    assert gaz.describe(36.45, -117.6).startswith("Near ")


def test_featured_places_match_their_gazetteer_entries(gaz):
    """The app's featured places take over the gazetteer entry with the same id; both must be
    the same place (within 3 km), or the menu would hide a different town."""
    t = gaz.table
    for p in load_places():
        if p.id in t.index:
            d = g.haversine_km(p.lat, p.lon, t.loc[p.id, "lat"], t.loc[p.id, "lon"])
            assert float(d) < 3.0, p.id
