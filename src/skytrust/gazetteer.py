"""Every place in California a person might type: cities, towns, communities, neighbourhoods and
ZIP codes, each with the point the sky and the forecast are computed for.

Sources (public; downloaded once into data/raw/places/ by `python -m skytrust build-places`):

- **US Census Bureau, 2025 Gazetteer (places):** every incorporated city and town and every
  census-designated place (CDP) in California, with its official name and an internal point.
- **USGS GNIS Domestic Names (California):** the county of each place, joined on the GNIS id the
  Census lists as ANSICODE, and the location of the settlement itself (feature class "Populated
  Place"): the town centre. The Census internal point is the middle of the boundary, which for
  big or oddly shaped cities can be far from town: San Francisco's lies in the Pacific because
  the city includes the Farallon Islands.
- **GeoNames (CC BY 4.0):** an independent second set of town-centre coordinates (to cross-check
  the first), named neighbourhoods (Hollywood, La Jolla), small communities that aren't census
  places (Furnace Creek), and population for unincorporated places.
- **Census Bureau, Vintage 2024 population estimates:** population of incorporated places.
  Population only orders search results; nothing is computed from it.
- **Census 2025 Gazetteer (ZCTAs):** ZIP code areas 900xx-961xx with internal points.
- **Open-Meteo elevation API** (Copernicus 90 m DEM): the terrain height the live forecast uses
  for any spot, cross-checked against GeoNames' SRTM heights.
"""

from __future__ import annotations

import json
import logging
import math
import re
import unicodedata
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from skytrust.config import REPO_ROOT, Site

log = logging.getLogger(__name__)

RAW = REPO_ROOT / "data" / "raw" / "places"
PLACES_PATH = REPO_ROOT / "artifacts" / "places_ca.json"
TIMEZONE = "America/Los_Angeles"

SOURCES = {
    "2025_Gaz_place_national.zip": "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
    "2025_Gazetteer/2025_Gaz_place_national.zip",
    "2025_Gaz_zcta_national.zip": "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
    "2025_Gazetteer/2025_Gaz_zcta_national.zip",
    "DomesticNames_CA_Text.zip": "https://prd-tnm.s3.amazonaws.com/StagedProducts/"
    "GeographicNames/DomesticNames/DomesticNames_CA_Text.zip",
    "US.zip": "https://download.geonames.org/export/dump/US.zip",
    "admin2Codes.txt": "https://download.geonames.org/export/dump/admin2Codes.txt",
    "sub-est2024.csv": "https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/"
    "cities/totals/sub-est2024.csv",
}
ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"
ELEVATION_BATCH = 100  # coordinates per elevation request (the API's limit)
LSAD_KIND = {"25": "city", "43": "town", "57": "community"}  # Census legal/statistical codes
ZIP_PREFIXES = (900, 961)  # California's ZIP codes run 90001-96162
# The settlement point is believed only within this many "boundary radii" (radius of a circle
# with the place's land area) of the boundary's internal point, and at least MIN_CENTRE_KM:
# Anaheim's downtown is 2.3 radii from its internal point, while for small places a same-name
# settlement 5 km away (San Simeon, East Whittier) is a different feature.
MAX_CENTRE_RADII = 2.5
MIN_CENTRE_KM = 3.0
CROSSCHECK_KM = 5.0  # flag places where GNIS and GeoNames centres disagree by more than this


def haversine_km(lat1, lon1, lat2, lon2):
    la1, lo1, la2, lo2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def slug(text: str) -> str:
    """'San Luis Obispo' -> 'san-luis-obispo'; accents dropped ('Cañada' -> 'canada')."""
    ascii_ = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_.lower()).strip("-")


def short_name(census_name: str) -> str:
    """'Davis city' -> 'Davis', 'Acton CDP' -> 'Acton', 'Mountain House town' -> ...'."""
    return re.sub(r" (city|town|CDP)$", "", census_name)


# ---------- reading the raw files ----------


def _read_pipe(path: Path, member: str | None = None) -> pd.DataFrame:
    if member is not None:
        with zipfile.ZipFile(path) as z, z.open(member) as f:
            df = pd.read_csv(f, sep="|", dtype=str, encoding="utf-8-sig")
    else:
        df = pd.read_csv(path, sep="|", dtype=str, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    return df


def read_census_places(raw: Path = RAW) -> pd.DataFrame:
    df = _read_pipe(raw / "2025_Gaz_place_national.zip", "2025_Gaz_place_national.txt")
    df = df[df["USPS"] == "CA"]
    return pd.DataFrame(
        {
            "geoid": df["GEOID"],
            "gnis_id": df["ANSICODE"].astype(int),
            "name": df["NAME"].map(short_name),
            "kind": df["LSAD"].map(LSAD_KIND).fillna("community"),
            "lat_internal": df["INTPTLAT"].astype(float),
            "lon_internal": df["INTPTLONG"].astype(float),
            "land_km2": df["ALAND"].astype(float) / 1e6,
        }
    ).reset_index(drop=True)


def read_gnis(raw: Path = RAW) -> pd.DataFrame:
    df = _read_pipe(raw / "DomesticNames_CA_Text.zip", "Text/DomesticNames_CA.txt")
    df = df[df["state_name"] == "California"]
    return pd.DataFrame(
        {
            "gnis_id": df["feature_id"].astype(int),
            "name": df["feature_name"],
            "feature_class": df["feature_class"],
            "county": df["county_name"],
            "lat": pd.to_numeric(df["prim_lat_dec"], errors="coerce"),
            "lon": pd.to_numeric(df["prim_long_dec"], errors="coerce"),
        }
    ).dropna(subset=["lat", "lon"])


GEONAMES_COLS = ("geonameid name asciiname alternatenames lat lon fclass fcode cc cc2 admin1 "
                 "admin2 admin3 admin4 population elevation dem tz moddate").split()  # fmt: skip


def read_geonames(raw: Path = RAW) -> pd.DataFrame:
    """California populated places from the GeoNames US dump, with county names."""
    with zipfile.ZipFile(raw / "US.zip") as z, z.open("US.txt") as f:
        df = pd.read_csv(
            f, sep="\t", names=GEONAMES_COLS, dtype=str, keep_default_na=False, quoting=3,
            usecols=["geonameid", "name", "lat", "lon", "fclass", "fcode", "admin1", "admin2",
                     "population", "dem"],
        )  # fmt: skip
    df = df[(df["admin1"] == "CA") & (df["fclass"] == "P")]
    counties = {}
    for line in (raw / "admin2Codes.txt").read_text(encoding="utf-8").splitlines():
        code, name, *_ = line.split("\t")
        if code.startswith("US.CA."):
            counties[code[6:]] = re.sub(r" County$", "", name)
    return pd.DataFrame(
        {
            "geonameid": df["geonameid"].astype(int),
            "name": df["name"],
            "fcode": df["fcode"],
            "county": df["admin2"].map(counties),
            "lat": df["lat"].astype(float),
            "lon": df["lon"].astype(float),
            "population": pd.to_numeric(df["population"], errors="coerce").fillna(0).astype(int),
            "dem": pd.to_numeric(df["dem"], errors="coerce"),
        }
    ).reset_index(drop=True)


def read_population(raw: Path = RAW) -> pd.Series:
    """Vintage 2024 estimates for incorporated places, by Census GEOID."""
    df = pd.read_csv(raw / "sub-est2024.csv", dtype=str, encoding="latin-1")
    df = df[(df["SUMLEV"] == "162") & (df["STATE"] == "06")]
    return pd.Series(
        df["POPESTIMATE2024"].astype(int).to_numpy(), index=("06" + df["PLACE"]).to_numpy()
    )


def read_zctas(raw: Path = RAW) -> pd.DataFrame:
    df = _read_pipe(raw / "2025_Gaz_zcta_national.zip", "2025_Gaz_zcta_national.txt")
    prefix = df["GEOID"].str[:3].astype(int)
    df = df[(prefix >= ZIP_PREFIXES[0]) & (prefix <= ZIP_PREFIXES[1])]
    return pd.DataFrame(
        {
            "zip": df["GEOID"],
            "lat": df["INTPTLAT"].astype(float),
            "lon": df["INTPTLONG"].astype(float),
        }
    ).reset_index(drop=True)


# ---------- building ----------


def town_centres(places: pd.DataFrame, gnis: pd.DataFrame) -> pd.DataFrame:
    """County and town-centre point for each census place.

    The county comes from the place's own GNIS record. The point is the GNIS "Populated Place"
    with the same name in the same county nearest the boundary's internal point (the town
    itself), if it's within MAX_CENTRE_RADII boundary radii (at least MIN_CENTRE_KM); otherwise
    the place's own GNIS point."""
    own = gnis.set_index("gnis_id")
    out = places.copy()
    out["county"] = out["gnis_id"].map(own["county"])
    out["lat"] = out["gnis_id"].map(own["lat"])
    out["lon"] = out["gnis_id"].map(own["lon"])
    out["point"] = "gnis-place-record"
    towns = gnis[gnis["feature_class"] == "Populated Place"].copy()
    towns["key"] = towns["name"].str.lower() + "|" + towns["county"]
    by_key = {k: g for k, g in towns.groupby("key")}
    for i, row in out.iterrows():
        group = by_key.get(f"{row['name'].lower()}|{row['county']}")
        if group is None:
            continue
        d = haversine_km(row["lat_internal"], row["lon_internal"], group["lat"], group["lon"])
        j = int(np.argmin(d))
        radius = math.sqrt(max(row["land_km2"], 0.01) / math.pi)
        if d[j] <= max(MAX_CENTRE_RADII * radius, MIN_CENTRE_KM):
            out.loc[i, ["lat", "lon", "point"]] = (
                float(group["lat"].iloc[j]),
                float(group["lon"].iloc[j]),
                "gnis-populated-place",
            )
    return out


def match_geonames(points: pd.DataFrame, gn: pd.DataFrame, max_km: float = 25.0) -> pd.DataFrame:
    """For each row (name, county, lat, lon), the GeoNames populated place with the same name in
    the same county nearest to it: its id, distance (the cross-check), population and height."""
    gn = gn.assign(key=gn["name"].str.lower() + "|" + gn["county"].fillna(""))
    by_key = {k: g for k, g in gn.groupby("key")}
    rows = []
    for _, r in points.iterrows():
        g = by_key.get(f"{r['name'].lower()}|{r['county']}")
        if g is None:
            rows.append((np.nan, np.nan, 0, np.nan))
            continue
        d = haversine_km(r["lat"], r["lon"], g["lat"], g["lon"])
        j = int(np.argmin(d))
        if d[j] > max_km:
            rows.append((np.nan, np.nan, 0, np.nan))
            continue
        rows.append(
            (
                int(g["geonameid"].iloc[j]),
                float(d[j]),
                int(g["population"].iloc[j]),
                float(g["dem"].iloc[j]),
            )  # fmt: skip
        )
    return pd.DataFrame(rows, columns=["geonameid", "check_km", "gn_population", "gn_dem"],
                        index=points.index)  # fmt: skip


def extra_geonames(gn: pd.DataFrame, matched_ids: set[int], places: pd.DataFrame) -> pd.DataFrame:
    """GeoNames places that aren't census places: named neighbourhoods (feature code PPLX) and
    communities (PPL) with a population on record, skipping any within 2 km of a census place
    of the same name (the same town under a slightly different record)."""
    keep = gn[
        gn["fcode"].isin(["PPL", "PPLX"])
        & (gn["population"] > 0)
        & ~gn["geonameid"].isin(matched_ids)
        & gn["county"].notna()
    ].copy()
    names = places.groupby(places["name"].str.lower())
    drop = []
    for i, r in keep.iterrows():
        key = r["name"].lower()
        if key in names.groups:
            same = places.loc[names.groups[key]]
            if float(np.min(haversine_km(r["lat"], r["lon"], same["lat"], same["lon"]))) < 2.0:
                drop.append(i)
    keep = keep.drop(index=drop)
    keep["kind"] = np.where(keep["fcode"] == "PPLX", "neighbourhood", "community")
    return keep


NEAR_SCORE = 2.0  # prefer a city or town whose footprint plausibly contains the point


def nearest_names(lat, lon, places: pd.DataFrame) -> list[str]:
    """How to describe each point (a ZIP code's internal point): the place it most plausibly
    lies in. Distance to each town centre is divided by the radius of a circle with the place's
    land area, so a point 7 km from downtown Los Angeles counts as inside Los Angeles rather than
    "near" little West Hollywood 5 km away. A city or town within NEAR_SCORE radii wins (what
    people call the area: Davis, not the campus community next to it); otherwise the census
    place of any kind with the lowest score."""
    radius = np.sqrt(places["land_km2"].clip(lower=1.0) / math.pi).to_numpy()
    incorporated = places["kind"].isin(["city", "town"]).to_numpy()
    plat, plon = places["lat"].to_numpy(), places["lon"].to_numpy()
    out = []
    for la, lo in zip(np.atleast_1d(lat), np.atleast_1d(lon), strict=True):
        score = haversine_km(la, lo, plat, plon) / radius
        town = np.where(incorporated, score, np.inf)
        j = int(np.argmin(town)) if town.min() <= NEAR_SCORE else int(np.argmin(score))
        out.append(str(places["name"].iloc[j]))
    return out


def unique_ids(df: pd.DataFrame, taken: set[str]) -> list[str]:
    """URL ids: the name's slug; places sharing a name get their county appended."""
    base = df["name"].map(slug)
    counts = base.value_counts()
    ids = []
    for b, county in zip(base, df["county"], strict=True):
        i = b if counts[b] == 1 and b not in taken else f"{b}-{slug(county or 'ca')}"
        n = 2
        while i in taken:
            i, n = f"{b}-{slug(county or 'ca')}-{n}", n + 1
        taken.add(i)
        ids.append(i)
    return ids


def build(
    raw: Path = RAW, elevations: dict | None = None, om_sample: dict | None = None
) -> tuple[pd.DataFrame, dict]:
    """The full table (one row per searchable place) and a report of the cross-checks."""
    places = town_centres(read_census_places(raw), read_gnis(raw))
    gn = read_geonames(raw)
    m = match_geonames(places, gn)
    places = places.join(m)
    pop = read_population(raw)
    places["population"] = places["geoid"].map(pop)
    places["population"] = places["population"].fillna(places["gn_population"]).fillna(0)
    extra = extra_geonames(gn, set(m["geonameid"].dropna().astype(int)), places)
    extra = extra.rename(columns={"population": "population", "dem": "gn_dem"})
    zips = read_zctas(raw)
    zips["name"] = zips["zip"]
    zips["near"] = nearest_names(zips["lat"], zips["lon"], places)
    zips["county"] = None
    zips["kind"] = "zip"
    zips["population"] = 0

    cols = ["name", "kind", "county", "lat", "lon", "population"]
    table = pd.concat(
        [
            places[cols + ["gn_dem", "check_km", "point"]],
            extra[cols + ["gn_dem"]].assign(point="geonames"),
            zips[cols + ["near"]].assign(point="census-zcta"),
        ],
        ignore_index=True,
    )
    taken: set[str] = set()
    named = table["kind"] != "zip"
    table.loc[named, "id"] = unique_ids(table[named], taken)
    table.loc[~named, "id"] = "zip-" + table.loc[~named, "name"]
    table["population"] = table["population"].astype(int)
    if elevations is not None:
        table["elevation_m"] = [elevations.get(_coord_key(a, b)) for a, b in
                                zip(table["lat"], table["lon"], strict=True)]  # fmt: skip
    if om_sample is not None:
        table["om_elevation_m"] = [om_sample.get(f"{a:.4f},{b:.4f}") for a, b in
                                   zip(table["lat"], table["lon"], strict=True)]  # fmt: skip
    report = crosscheck_report(table, places)
    return table, report


def crosscheck_report(table: pd.DataFrame, places: pd.DataFrame) -> dict:
    """How well the independent sources agree, for DATA_NOTES."""
    chk = places["check_km"].dropna()
    # Independent of GeoNames (which copies GNIS for US towns): is each centre inside its census
    # boundary's footprint? Distance from the boundary's internal point, in units of the radius
    # of a circle with the place's land area.
    radius = np.sqrt(places["land_km2"].clip(lower=0.01) / math.pi)
    spread = haversine_km(places["lat"], places["lon"], places["lat_internal"],
                          places["lon_internal"]) / radius  # fmt: skip
    rep = {
        "n_places": int(len(places)),
        "n_incorporated": int(places["kind"].isin(["city", "town"]).sum()),
        "n_communities": int((places["kind"] == "community").sum()),
        "n_neighbourhoods": int((table["kind"] == "neighbourhood").sum()),
        "n_other_communities": int(
            ((table["kind"] == "community") & (table["point"] == "geonames")).sum()
        ),
        "n_zip": int((table["kind"] == "zip").sum()),
        "n_total": int(len(table)),
        "centre_from_populated_place": int((places["point"] == "gnis-populated-place").sum()),
        "geonames_matched": int(len(chk)),
        "gnis_vs_geonames_km": {
            "median": float(chk.median()),
            "p95": float(chk.quantile(0.95)),
            "max": float(chk.max()),
            "share_within_1km": float((chk <= 1).mean()),
            "share_within_5km": float((chk <= CROSSCHECK_KM).mean()),
        },
        "centre_vs_boundary_radii": {
            "median": float(np.median(spread)),
            "share_within_1": float((spread <= 1).mean()),
            "share_within_2": float((spread <= 2).mean()),
            "outside_2": [
                {"name": n, "county": c, "radii": round(float(s), 1)}
                for n, c, s in sorted(
                    zip(places["name"], places["county"], spread, strict=True),
                    key=lambda x: -x[2],
                )
                if s > 2
            ],
        },
        "disagree_over_5km": [
            {"name": r["name"], "county": r["county"], "km": round(float(r["check_km"]), 1)}
            for _, r in places[places["check_km"] > CROSSCHECK_KM]
            .sort_values("check_km", ascending=False)
            .iterrows()
        ],
    }
    if "elevation_m" in table:
        rep["elevation_missing"] = int(table["elevation_m"].isna().sum())
        for other, name in [
            ("gn_dem", "srtm_geonames"),
            ("om_elevation_m", "copernicus_openmeteo"),
        ]:
            if other not in table:
                continue
            both = table.dropna(subset=["elevation_m", other])
            diff = (both["elevation_m"] - both[other]).abs()
            rep[f"elevation_usgs_vs_{name}_m"] = {
                "n": int(len(both)),
                "median_abs": float(diff.median()),
                "p95_abs": float(diff.quantile(0.95)),
                "share_within_25m": float((diff <= 25).mean()),
            }
    return rep


# ---------- elevation (network) ----------

EPQS_URL = "https://epqs.nationalmap.gov/v1/json"  # USGS Elevation Point Query Service (3DEP)
ELEVATION_CACHE = RAW / "elevation_usgs.json"


def _coord_key(lat: float, lon: float) -> str:
    return f"{lat:.5f},{lon:.5f}"


def fetch_elevations(
    http_settings, lats, lons, cache: Path = ELEVATION_CACHE, workers: int = 4
) -> dict[str, float]:
    """Terrain height (m) at each point from the USGS 3D Elevation Program, the national
    elevation model (1-10 m resolution in California), one point per request. Answers are cached
    on disk as they arrive, so an interrupted run resumes where it stopped."""
    import dataclasses
    from concurrent.futures import ThreadPoolExecutor

    from skytrust.data.http import HttpClient

    done: dict[str, float] = json.loads(cache.read_text()) if cache.exists() else {}
    todo = sorted({_coord_key(a, b) for a, b in zip(lats, lons, strict=True)} - set(done))
    polite = dataclasses.replace(http_settings, polite_delay_s=0.2)
    clients = [HttpClient(polite) for _ in range(workers)]

    def one(job: tuple[int, str]) -> tuple[str, float | None]:
        i, key = job
        lat, lon = key.split(",")
        params = {"x": lon, "y": lat, "wkid": 4326, "units": "Meters", "includeDate": "false"}
        try:
            value = float(clients[i % workers].get_json(EPQS_URL, params)["value"])
        except Exception as exc:  # a failed point stays missing and is retried next run
            log.warning("elevation %s failed: %s", key, exc)
            return key, None
        return key, value if value > -1000 else None  # -1000000 = no data (offshore)

    with ThreadPoolExecutor(workers) as pool:
        for n, (key, value) in enumerate(pool.map(one, enumerate(todo)), 1):
            if value is not None:
                done[key] = value
            if n % 200 == 0 or n == len(todo):
                cache.write_text(json.dumps(done))
                log.info("elevations: %d of %d new points", n, len(todo))
    return done


def open_meteo_sample(cache: Path = RAW / "elevation") -> dict[str, float]:
    """The Open-Meteo (Copernicus 90 m) heights fetched before switching to USGS, kept as an
    independent cross-check sample (keys are "lat,lon" to 4 decimals)."""
    out: dict[str, float] = {}
    for path in sorted(cache.glob("batch_*.json")):
        payload = json.loads(path.read_text())
        out.update(zip(payload["keys"], (float(e) for e in payload["elevation"]), strict=True))
    return out


def download(client, raw: Path = RAW) -> list[str]:
    """Fetch any missing raw file. Returns the names downloaded."""
    raw.mkdir(parents=True, exist_ok=True)
    got = []
    for name, url in SOURCES.items():
        path = raw / name
        if path.exists():
            continue
        path.write_bytes(client.get(url, {}).content)
        got.append(name)
    return got


# ---------- the artifact the app reads ----------

COLUMNS = ["id", "name", "kind", "county", "near", "lat", "lon", "elevation_m", "population"]


def save(table: pd.DataFrame, path: Path = PLACES_PATH) -> Path:
    """Compact JSON (one array per row), ordered by population so the biggest matches come
    first when someone types part of a name."""
    t = table.copy()
    t["near"] = t.get("near")
    t["_zip"] = t["kind"] == "zip"
    t = t.sort_values(["_zip", "population", "name"], ascending=[True, False, True])
    rows = []
    for r in t[COLUMNS].itertuples(index=False):
        row = list(r)
        row[5], row[6] = round(float(row[5]), 5), round(float(row[6]), 5)
        row[7] = None if row[7] is None or pd.isna(row[7]) else round(float(row[7]), 1)
        row[2:5] = [None if (v is None or (isinstance(v, float) and math.isnan(v))) else v
                    for v in row[2:5]]  # fmt: skip
        row[8] = int(row[8])
        rows.append(row)
    path.write_text(json.dumps({"columns": COLUMNS, "rows": rows}, separators=(",", ":")))
    return path


@dataclass(frozen=True)
class Gazetteer:
    table: pd.DataFrame  # indexed by id

    def label(self, place_id: str) -> str:
        r = self.table.loc[place_id]
        if r["kind"] == "zip":
            return f"{r['name']} · ZIP code near {r['near']}"
        return f"{r['name']} · {r['county']} County"

    def labels(self) -> dict[str, str]:
        """Menu label for every place, in the table's order (biggest first, ZIP codes last)."""
        t = self.table
        named = t["name"] + " · " + t["county"].fillna("") + " County"
        zips = t["name"] + " · ZIP code near " + t["near"].fillna("")
        return dict(zip(t.index, zips.where(t["kind"] == "zip", named), strict=True))

    def describe(self, lat: float, lon: float) -> str:
        """A name for a spot given by coordinates: the nearest named place ("Near Lone Pine"),
        or the place itself when the spot is within a kilometre of it."""
        named = self.table[self.table["kind"] != "zip"]
        d = haversine_km(lat, lon, named["lat"], named["lon"])
        j = int(np.argmin(d))
        name = str(named["name"].iloc[j])
        return name if d[j] < 1.0 else f"Near {name}"

    def site(self, place_id: str) -> Site:
        r = self.table.loc[place_id]
        elev = float(r["elevation_m"]) if pd.notna(r["elevation_m"]) else float("nan")
        where = f"ZIP code near {r['near']}" if r["kind"] == "zip" else f"{r['county']} County"
        name = f"ZIP {r['name']}" if r["kind"] == "zip" else str(r["name"])
        return Site(place_id, name, float(r["lat"]), float(r["lon"]), elev, where, TIMEZONE)


@lru_cache(maxsize=2)
def load(path: Path = PLACES_PATH) -> Gazetteer | None:
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    table = pd.DataFrame(raw["rows"], columns=raw["columns"]).set_index("id", drop=False)
    return Gazetteer(table)
