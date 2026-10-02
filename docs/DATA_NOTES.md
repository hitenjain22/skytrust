# Data Notes: what the sources actually do

Probed on **2026-09-24/25** (local 17:00 PDT on 09-24). Everything below comes from live requests
or the official docs pages cited. Where reality differs from SPEC.md, it is marked **⚠ differs from spec**.
Raw probe responses live in `tests/fixtures/` (re-capture with `uv run python tests/fixtures/capture.py`).

---

## 1. Open-Meteo Previous Runs API (historical forecasts at fixed lead)

| Item | Finding |
|---|---|
| Endpoint | `https://previous-runs-api.open-meteo.com/v1/forecast` ✅ as spec |
| Variables | `cloud_cover_previous_day1` … `cloud_cover_previous_day7` exist ✅. `cloud_cover` (= previous_day0, the latest run) also available. Values are integer percent (0–100). |
| Definition (docs) | "`_previous_day1` is the value that was predicted 24 hours before valid time, `_previous_day2` 48 hours before, and so on up to day 7." |
| Date params | `start_date` / `end_date` (YYYY-MM-DD) with `timezone=UTC` work. |
| Max range per request | **No practical limit found.** One request for 2024-01-01 → 2026-09-23 (23,928 hourly rows × 8 variables) returned 200. We will still chunk (see §6) because of call weighting and cache granularity. |
| Multiple models | `models=a,b,c` in one request → keys are suffixed: `cloud_cover_previous_day1_gfs_global`. With a single model, keys are **unsuffixed**. Parsers must handle both. |
| Elevation | `elevation=<m>` is accepted and echoed back in the response (`"elevation": 8.0`). |
| Grid cell | Response `latitude/longitude` = the selected model grid point, which differs per model (e.g. SAC request 38.5069,-121.495 → GFS 38.4836,-121.5234; HRRR 38.5177,-121.5008; ECMWF 0.25° 38.5,-121.5). Historical API docs describe `cell_selection` = `land` (default: nearby land cell with similar elevation), `sea`, `nearest`; we use the default and pass the station elevation for downscaling. |

### Model identifiers (verified by probing)

| Spec model | `models=` id | Result |
|---|---|---|
| GFS | `gfs_global` | ✅ Use this. **⚠ Do not use `gfs_seamless`:** at lead 1 its values are **100 % identical to HRRR** (seamless = HRRR near-term + GFS after). Using both would double-count HRRR. At leads 2–7 `gfs_seamless` = `gfs_global`. |
| HRRR | `ncep_hrrr_conus` (legacy alias `gfs_hrrr` also works) | ✅ lead 1 only |
| NAM | `ncep_nam_conus` (`nam_conus` → HTTP 400) | **⚠ Unusable for backtest.** No data before 2025-08-28; lead 1 only from 2025-09-07 at 74 % fill; leads 2–7 never populated. No training-period data at all. |
| ECMWF IFS HRES 9 km | `ecmwf_ifs` | **⚠ Unusable for backtest.** Leads 1–3 only from 2025-01-02 at ~57 % fill; leads 4–7 from 2025-10. |
| ECMWF IFS 0.25° | `ecmwf_ifs025` | ✅ Use this (spec's fallback). Archive starts **2024-02-04** (lead 1). |
| GEM (optional) | `gem_seamless` | ✅ full leads |
| ICON (optional) | `icon_seamless` | ✅ leads 1–6 (lead 7 never populated) |
| — | `gfs025` | returns all nulls in this API |

### Leads populated per model (SAC, 2024-01-01 → 2026-09-23)

"fill" = share of non-null hours between first and last non-null value.

| Model | L1 | L2 | L3 | L4 | L5 | L6 | L7 | First lead-1 date |
|---|---|---|---|---|---|---|---|---|
| gfs_global | ✅ 1.000 | ✅ 1.000 | ✅ .999 | ✅ 1.000 | ✅ .999 | ✅ 1.000 | ✅ .999 | 2024-01-19 |
| ncep_hrrr_conus | ✅ .999 | — | — | — | — | — | — | 2024-01-19 |
| ecmwf_ifs025 | ✅ 1.000 | ✅ .994 | ✅ .994 | ✅ 1.000 | ✅ 1.000 | ✅ 1.000 | ✅ 1.000 | 2024-02-04 |
| gem_seamless | ✅ 1.000 | ✅ 1.000 | ✅ .993 | ✅ .995 | ✅ .995 | ✅ 1.000 | ✅ .995 | 2024-01-19 |
| icon_seamless | ✅ .997 | ✅ .997 | ✅ 1.000 | ✅ .997 | ✅ 1.000 | ✅ .997 | — | 2024-01-19 |
| ncep_nam_conus | 0.744 (from 2025-09-07) | — | — | — | — | — | — | 2025-09-07 |
| ecmwf_ifs (9 km) | .576 (from 2025-01-02) | .574 | .572 | from 2025-10 | … | … | … | 2025-01-02 |

Lead *d* first appears on 2024-01-18 + *d* days: the archive begins ~**2024-01-18**, not
2024-01-01 (**⚠ differs slightly from spec's "January 2024"**). The first ~18 nights of 2024 have no
forecasts. Test period (2026) fill is ≥ 0.976 for every model/lead in the proposed list.

### Temporal resolution / instantaneous?
- Open-Meteo docs (Historical API): cloud cover is "Cloud cover as an area fraction" at the hour,
  i.e. an **instantaneous** value at the timestamp, not an hourly mean. The Previous Runs page does
  not say either way; we assume the same convention (same backend/variable).
- **Interpolation:** ECMWF 0.25° is output 3-hourly, but the API returns every hour with smoothly
  varying values (e.g. 60, 63, 67, 72, 79, 86, 92, 97, 100 …). So 3-hourly models are
  **interpolated to hourly** by Open-Meteo. GFS beyond ~5 days is also 3-hourly upstream. This makes
  "consecutive clear hours" partly an interpolation artifact for those models. Worth a caveat.

---

## 2. Open-Meteo Historical Weather API (ERA5 label)

| Item | Finding |
|---|---|
| Endpoint | `https://archive-api.open-meteo.com/v1/archive` ✅ |
| Model id | `models=era5` ✅. **⚠ Must be explicit:** docs say the default "Best Match combines IFS HRES, ERA5 and ERA5-Land seamlessly", which would mix ECMWF *forecast-model* analysis into our label. |
| Variables | `cloud_cover`, `cloud_cover_low`, `cloud_cover_mid`, `cloud_cover_high`, all in `%`, all 100 % filled. |
| Grid | 0.25° (SAC → 38.5, -121.5). |
| Publication lag | On 2026-09-25 00:20 UTC the last non-null hour was **2026-09-18T23:00** → ~**6 days** (docs: "Daily with 5 days delay"). Trailing days return `null`; code must treat them as missing, not zero. |
| Range | Full 2024-01-01 → today in one request worked. |

---

## 3. Open-Meteo Forecast API (live)

| Item | Finding |
|---|---|
| Endpoint | `https://api.open-meteo.com/v1/forecast` ✅ |
| Multi-model | Same suffix convention (`cloud_cover_low_ecmwf_ifs025`). |
| `forecast_days=8, timezone=UTC` | 192 hourly rows, starting at 00:00 UTC today. |
| Layers | low/mid/high available for **all five** proposed models. |
| Horizons (requested 2026-09-25 00Z) | GFS, ECMWF 0.25°, GEM: full 8 days · ICON: ~7 days · **HRRR: ~42 h** (last value 09-26T18Z). |

---

## 4. IEM ASOS/METAR archive (ASOS label)

| Item | Finding |
|---|---|
| Endpoint | `https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py` ✅ |
| Params | As spec (`station`, repeated `data=`, `sts`/`ets` ISO UTC, `tz=UTC`, `format=onlycomma`, `elev=yes`). Output columns: `station,valid,elevation,skyc1..4,skyl1..4,metar`; `valid` is `YYYY-MM-DD HH:MM` in UTC. Missing = `M`. |
| **⚠ report_type** | **Default returns 5-minute MADIS "HFMETAR" rows** (e.g. SAC 00:00, 00:05, 00:10 … `MADISHF`). Taking a max over 12 five-minute obs per hour is far more conservative than the spec intends ("routine + SPECI"). We pass **`report_type=3&report_type=4`** (3 = routine, 4 = specials, per the endpoint's help page). |
| **⚠ VV code** | IEM sends `"VV "` with a trailing space. The parser strips whitespace (tested). |
| Rate limit | Back-to-back yearly requests triggered **HTTP 429** several times; retries with backoff recovered each time. Polite delay raised to 1 s and backoff base to 2 s. |
| Station metadata | `https://mesonet.agron.iastate.edu/geojson/network/CA_ASOS.geojson` (161 stations). Fields used: `sid`, `sname`, `elevation` (m), `tzname`, geometry `[lon, lat]`, `attributes.IS_AWOS`. |

### Station findings that affect the ASOS label

| Station | Reports per hour window (median) | Report minutes | Layers reported > 12,000 ft? | Notes |
|---|---|---|---|---|
| SAC | 1 | :53 | never (max 12,000) | fully automated |
| FAT | ~1 | :53 | **yes: 54 % of reports** (up to 32,000 ft) | **0 % AUTO**, human-augmented. FAT's ASOS label *can* see cirrus |
| AUN | **3** | :15, :35, :55 | never | **AWOS, not ASOS** (IS_AWOS=1). 20-minute routine cycle |
| TRK | **2** | :15/:35/:55 + :47–:50 | yes: 7 % of reports | 72 % AUTO; mixed report schedule |
| BIH | 1 | :56 | never | fully automated |

Implications, raised at the Phase 0 checkpoint (resolved 2026-09-25: hourly value now = report nearest H; see DECISIONS):
1. The "max over all obs in the window" rule is **more conservative at AUN (3 obs) and TRK (2 obs)**
   than at SAC/FAT/BIH (1 obs), so ASOS-label base rates aren't strictly comparable across sites.
2. The 12,000 ft blind spot applies fully at SAC, AUN, BIH, and at TRK when AUTO, but **not at FAT**.

---

## 5. Site validation (Section 5)

Coverage = share of **proxy night hours** (local 23:00–03:00, which is inside astronomical darkness
year-round at these latitudes) that have ≥1 valid sky report in the (H−30, H+30] window, using
routine + SPECI reports. Real dark windows replace the proxy in Phase 1. Command:
`uv run python -m skytrust validate-sites`.

| Station | IEM name | lat | lon | elev (m) | Coverage 2024 | 2025 | 2026 | Total | Result |
|---|---|---|---|---|---|---|---|---|---|
| SAC | SACRAMENTO/EXECUTIV | 38.5069 | -121.4950 | 8 | 99.9 % | 97.6 % | 99.0 % | 98.8 % | PASS |
| FAT | FRESNO AIR TERMINAL | 36.7800 | -119.7194 | 100 | 100.0 % | 100.0 % | 99.6 % | 99.9 % | PASS |
| AUN | AURBURN MUNICIPAL AIRPORT (sic) | 38.9548 | -121.0817 | 467 | 98.7 % | 99.2 % | 98.4 % | 98.8 % | PASS (AWOS) |
| TRK | TRUCKEE-TAHOE | 39.3200 | -120.1396 | 1798 | 98.5 % | 99.9 % | 97.7 % | 98.8 % | PASS |
| BIH | BISHOP AIRPORT | 37.3731 | -118.3636 | 1263 | 99.5 % | 97.9 % | 98.4 % | 98.6 % | PASS |

(Run of 2026-09-24 covering nights 2024-01-01 → 2026-09-24. The CLI now stops two days before
today so the last night is fully in the past.) All five pass the 85 % bar; no substitutions needed.

---

## 6. Rate limits, terms, attribution

- **Open-Meteo free tier** (terms page): < 10,000 calls/day, 5,000/hour, 600/minute; non-commercial;
  data under **CC BY 4.0** (attribution required).
- **Call weighting** (pricing page): "Requests for data covering more than 10 weather variables or
  extending over a period of more than 2 weeks for a single location are considered multiple API
  calls." Fractional counting is used.
- **Backfill budget estimate:** ~1,000 days ÷ 14 ≈ 71 weighted calls per site × model (8 vars ≤ 10)
  → 5 sites × 5 models ≈ 1,800, plus ERA5 ≈ 360. About 2,200 total: under the daily cap, but we'll
  fetch in monthly chunks with the polite delay and can split the backfill across two days if needed.
- **IEM:** no published numeric limit; returns 429/503 under load and 422 for > ~1,000
  station-years. We request one station × one year.
- **Attribution text** (README + app footer): "Weather data by Open-Meteo.com (CC BY 4.0). ASOS
  observations courtesy of the Iowa Environmental Mesonet, Iowa State University."

---

## 7. Phase 1 backfill (2026-09-24/25)

- Nights 2024-01-01 → 2026-09-23, 5 sites. Requests: ERA5 165, ASOS 14 (most years already cached
  by `validate-sites`), Previous Runs 827 + 5. **Second/third runs: 0 network requests.**
  Raw cache: 1,010 files, 57 MB (gitignored).
- Transient issues, all recovered by retries: IEM HTTP 429 (×9), one Open-Meteo read timeout, one
  dropped connection.
- **⚠ ECMWF January 2024 chunk is permanently empty** (archive starts 2024-02-04). Chunks older than
  `settled_after_days` (30) are now cached even with nulls so they never re-download.
- **⚠ GFS out-of-range values:** exactly **−1** (132×) and **101** (157×) appear in `gfs_global`
  Previous Runs data, only 0.008 % of 3.7 M cached values, concentrated at GFS output hours
  (00/06/12/18 UTC). Looks like a rounding/packing artifact. Values within 1 point of [0, 100] are
  clipped; anything further out is rejected as a corrupt payload. Payloads are now fully parsed
  before being cached.
- Loaded coverage (from 2024-02-10) matches the §1 lead table at all 5 sites; ERA5 has 0 nulls
  through 2026-09-17 23:00 UTC.

---

## 8. Phase 2 findings (from `docs/DATA_QUALITY.md`)

- **ASOS outages** are real, not a parsing issue: SAC has an 8-night gap 2025-03-26 → 04-02, and
  several dates are missing at multiple stations at once (2025-03-04, 2026-01-20, 2026-04-02,
  2026-04-23), which points to IEM/feed gaps. Those nights are excluded with reason `asos_missing`.
- **FAT's disagreement goes the other way.** At SAC/AUN/TRK/BIH, 20–24 % of nights are "ASOS usable,
  ERA5 not" (the cirrus blind spot). At FAT only 7 %, while 12 % are "ERA5 usable, ASOS not". On those
  FAT nights ASOS mean cover is 62 % but ERA5's is 18 % (high cloud 9 %): FAT's human observers
  report thin cirrus (e.g. `SCT200 BKN250`) that ERA5 barely registers. So at FAT ASOS sees
  cirrus ERA5 misses, and the primary label (max of both) catches it either way. It does make
  FAT's primary label a little stricter than other sites'.
- **ASOS nearest vs max rule:** base rates differ by ≤ 3.1 points (largest at TRK and AUN, the
  multi-report stations; identical at FAT and BIH), consistent with the Phase 0 reasoning.

---

## 9. NOAA National Blend of Models (NBM), added 2026-09-30

- `models=ncep_nbm_conus` (2.5 km; hourly to 36 h, 3-hourly after). Works in both the Previous
  Runs and live Forecast APIs. Previous Runs archive: **lead 1 from 2024-10-09 (98.6 % fill), leads
  2–7 from 2024-10-10…15 (100 %)**, so NBM covers the whole 2026 test period and 15 of the 24
  training months.
- NBM is NOAA's operational statistically post-processed blend of many models, i.e. the
  professional version of what SkyTrust's blend does, which makes it the natural benchmark.
- Open-Meteo's docs page lists GFS as `gfs_global_011` / `gfs_global_025`, but the API rejects
  both; `gfs_global` still works (checked 2026-09-29; guarded by the weekly contract tests).

---

## 10. GOES-18 Clear Sky Mask (satellite truth), added 2026-09-30

Checked against real files (2025-01-10, 2025-06-15, 2026-02-03) before any code relied on them:

- **Where:** public AWS bucket `noaa-goes18`, product `ABI-L2-ACMC` (CONUS sector, every 5 min),
  keys `ABI-L2-ACMC/{year}/{day-of-year}/{hour}/OR_ABI-L2-ACMC-M6_G18_s{start}_e{end}_c{created}.nc`.
  Listed with the S3 REST API (`list-type=2&prefix=…`); the first key of an hour starts at ≈ H:01.
  No account or key needed. One NetCDF4 (HDF5) file is ≈ 3.7 MB and covers every site.
- **Variables used:** `BCM` (binary cloud mask, uint8, 0 = clear or probably clear,
  1 = cloudy or probably cloudy, fill 255), `DQF` (uint8, 0 = good, 1 = bad, 2 = space,
  6 = degraded; only 0 is used), `x`/`y` (int16 scan angles, `scale_factor` 5.6e-05 rad, `y`'s is
  negative, i.e. rows run north → south), grid 1500 × 2500, and `goes_imager_projection`
  (perspective height 35,786,023 m, GRS80 axes, sub-satellite longitude −137°).
- **Geometry:** latitude/longitude → scan angle with the GOES-R Product User Guide (vol. 3,
  §4.2.8.1) formulas; the implementation reproduces the guide's worked example to 1e-6 rad.
- **Viewing angle:** GOES-18 sees the five sites at a satellite zenith angle of 46–49°, so a
  cloud at 10 km altitude appears ≈ 11 km away from where it really is (parallax), about one
  5 × 5-pixel box. Negligible for widespread cloud; it blurs isolated clouds. Not corrected
  (that would need each cloud's height, a separate product).
- **Night-time:** at night the mask uses infrared channels only, so very thin cirrus is harder
  to detect than by day. The satellite label is therefore not "perfect truth", just a truth with
  different blind spots from ASOS (can't see above 12,000 ft) and ERA5 (a model).
- **In SkyTrust:** per dark hour, the scan nearest the top of the hour; cloud fraction = mean BCM
  over the good-quality pixels of a 5 × 5 box (≈ 12–15 km at this viewing angle) centred on the
  airport. Only these per-site numbers are stored (`data/raw/goes/{site}/{month}.csv`).
- **Coverage (fetched 2026-09-30):** 8,547 dark hours (union over the five sites, nights
  2024-01-01 → 2026-09-28), zero download errors. 116 hours (1.4 %) have no Clear Sky Mask file:
  almost all in 2025-11-07 → 11-15 plus 2025-02-22. For 2025-11-10 the bucket has radiance files
  (`ABI-L1b-RadC`) but almost no mask files all day, so it's a gap in the derived product, not a
  satellite outage. Those nights get the `goes_missing` exclusion like any other gap.
- **First look (hourly, all 2024–26 dark hours):** GOES cover correlates 0.72–0.83 with ERA5 and
  0.44–0.72 with ASOS; FAT is the exception where ASOS matches GOES about as well as ERA5 does
  (FAT's human observers report cirrus, §8). The nightly comparison is in DATA_QUALITY §7b.

---

## 11. Light pollution: World Atlas of Artificial Night Sky Brightness, added 2026-09-30

- **Source:** Falchi F., Cinzano P., Duriscoe D., Kyba C. C. M., Elvidge C. D., Baugh K.,
  Portnov B. A., Rybnikova N. A., Furgoni R. (2016), *The new world atlas of artificial night sky
  brightness*, Science Advances 2(6):e1600377; dataset doi:10.5880/GFZ.1.4.2016.001. Both must be
  cited (dataset README).
- **Download:** `World_Atlas_2015.zip` (653 MB) from the GFZ data page, containing
  `World_Atlas_2015.tif` (3.01 GB). Checked 2026-09-30 by reading the TIFF header: little-endian
  classic TIFF, 43200 × 17406, uncompressed float32 in 128 × 128 tiles, tie point (−180°,
  85.054°), pixel 0.00833333° (30″), values = **artificial** zenith brightness in mcd/m². Tile
  offsets are not 4-byte aligned (read bytes per tile).
- **License:** the 2016 README says further distribution is "generally prohibited" and that policy
  changes are announced on the data access page; that page (checked 2026-09-30) lists
  **CC BY-NC 4.0**, which permits non-commercial redistribution with attribution. SkyTrust is a
  non-commercial student project; the committed file is a derived regional subset
  (`artifacts/light_pollution.npz`), credited on every page.
- **Conversions (from the paper):** natural background 22.0 mag/arcsec² = 174 µcd/m², so
  SQM = 22.0 − 2.5·log₁₀((artificial + 174)/174); colour levels are artificial/natural ratios
  doubling from 0.01; model vs SQM measurements σ = 0.15 mag/arcsec². Bortle classes use the SQM
  ranges tabulated in the Bortle scale article (Wikipedia); the scale itself is visual.
- **Checks on the cropped grid:** Las Vegas Strip 16.24 mag/arcsec² (200× natural), downtown Los
  Angeles 17.33, San Francisco 18.15, Seattle 17.71; Death Valley 21.98, Great Basin NP 22.00
  (certified dark-sky parks), Pacific Ocean off Big Sur 21.98. Sites: SAC 18.85 (Bortle 6), FAT
  18.78 (6), AUN 20.52 (4.5), TRK 21.48 (3), BIH 21.64 (2).
- **Caveats:** VIIRS data from 2014–2015 (skies have brightened since); zenith only; ~1 km grid;
  "darkest nearby" is straight-line distance and ignores roads, access and terrain.

---

## 12. Night lights 2015 / 2025 and city glow, added 2026-09-30

- **Night lights:** NASA Black Marble annual composites (VNP46A4 ≤ 2019 from Suomi NPP, VJ146A4
  > 2019 from NOAA-20; CC0), as raw GeoTIFFs from lightpollutionmap.info
  (`viirs_{year}_raw.zip`, ~0.9 GB each, no account). Checked 2026-09-30: 86400 × 33600 float32,
  15″ grid from (−180°, 75°), **LZW-compressed, one row per strip**, nodata −999.9. The brightest
  pixel in the region decodes at 36.115°N, 115.173°W (the Las Vegas Strip). The LZW reader was
  verified byte-for-byte against libtiff.
- **Kernel fit (2015 lights → 2016 atlas):** 9 distance rings to 300 km on a 1 km Lambert
  azimuthal equal-area grid; spatial 5-fold CV by longitude band: RMS 0.044 mag, 98% of places
  within 0.15 mag (atlas σ = 0.15). The kernel falls monotonically with distance.
- **Change 2015 → 2025:** median ×1.12 artificial light in lit areas; 77% of places brighter.
  The sensor changed between the two years (Suomi NPP → NOAA-20); NASA harmonizes the products,
  but some of the change could be instrumental.
- **Independent check:** David Lorenz's *World Atlas of the Artificial Night Sky Brightness 2025*
  (djlorenz.github.io; NorthAmerica2025.png, 7–75°N, 180–51°W at 1/120°, 15 colour zones whose
  index bounds are read from its colour bar). No license is stated on that site, so it is used
  only to validate, never shipped. Within one zone in lit areas: updated grid 80%, 2016 atlas
  59%. In lit areas Lorenz reads 0.43 mag brighter than SkyTrust (0.51 for the 2016 atlas).
- **LED caveat:** VIIRS is nearly blind to the blue light of white LEDs; Kyba et al. (2023),
  *Science* 379:265, found skies brightening 9.6%/yr (2011–2022) from 51,000+ citizen
  observations, faster than satellites show. Satellite-based updates are a lower bound on change.
- **Places:** GeoNames `cities1000` (CC BY 4.0), places of 1,000+ people within 3° of the region
  (3,058), for naming light domes.
- **Walker's law** (Walker 1977, *PASP* 89:405): glow toward a city ∝ distance^−2.5, used for the
  *relative* strength of light domes. The absolute constant could not be confirmed from a
  primary source, so strengths are relative to Sacramento's dome seen from 50 km.

## 13. The sky guide and sky events, added 2026-09-30

**Places** (`config/places.yaml`): nine well-known California places, coordinates and elevation
(SRTM3 `dem`) from GeoNames cities1000 (CC BY 4.0); each row keeps its `geonames_id`. "Lake
Tahoe" is the GeoNames record for South Lake Tahoe; "Death Valley" is the populated place of that
name (GeoNames 12523068). How dark each place is comes from the 2025 light-pollution grid at run
time (SQM: Los Angeles 17.29, San Francisco 18.11, Sacramento 18.08, Santa Barbara 19.38, Big Bear
Lake 20.93, Joshua Tree 21.09, South Lake Tahoe 21.37, Yosemite Valley 21.96, Death Valley 21.98).
They were never part of the evaluation, so forecasts there use the site-agnostic blend.

**Stars**: Hipparcos main catalogue (ESA 1997, CDS I/239, `hip_main.dat`), every star with
V ≤ 6.5 (8,874): position (ICRS, epoch J1991.25; proper motion ignored, < 0.1° for any naked-eye
star in 35 years), V, B−V. A few entries have no astrometric solution; their sexagesimal columns
are used. Proper names: Skyfield's `named_star_dict` (100 of its names are among these stars).

**Constellations and the Milky Way**: d3-celestial by Olaf Frohn (BSD-3-Clause):
`constellations.lines.json` (stick figures), `constellations.json` (label positions, a 1–3
importance rank), `mw.json` (five nested brightness levels of the Milky Way outline). Constellation
membership of every star and object comes from Skyfield's bundled IAU boundaries (Roman 1987), an
independent check on the third-party coordinates.

**Deep-sky showpieces**: d3-celestial `messier.json` and `dsos.bright.json`. **Finding:** the
bright-DSO list files M4 at RA 83.82°, M42's right ascension (M4 is at 245.9°, in Scorpius). The
build checks every object against the constellation it must fall in and refuses bad ones, so
Messier objects come from the Messier list. Clusters wider than 100′ (Pleiades, Hyades, α Persei,
Coma) are judged by their third-brightest catalogue star within the cluster radius, not their
combined magnitude.

**Meteor showers** (`config/meteor_showers.yaml`): the IMO's *2026 Meteor Shower Calendar*
(J. Rendtel, ed., IMO INFO(3-25), DOI 10.13140/RG.2.2.36179.08480), Table 5. Peaks are stored as
solar longitudes and converted to times with Skyfield; for 2026 they reproduce the calendar's
times within 25 minutes (Geminids 13:49 vs 14h UT, Leonids 23:52 vs 23:45, Draconids 01:21 vs 01h,
Quadrantids 21:24 vs ~21h). Notes on parent bodies are from the calendar text or NASA's shower
pages. **Findings:** (1) the calendar's Draconid radiant is 262°, +54° in Table 5 but 263°, +56° in
the text; Table 5 is used. (2) Table 4's lunar phase dates follow Central European time: the
June 2026 full Moon (23:57 UTC on June 29, confirmed by other sources) is listed as June 30.

**Validation of the computed events** (tests/test_events.py): every 2026 new and full Moon matches
IMO Table 4; Saturn's 2026 opposition is computed at 12:29 UTC on October 4 (published ~12:00 UTC,
definitions differ); the 2026 lunar eclipses are found on March 3 (total) and August 28 (partial,
greatest at 04:12:54 UTC, umbral magnitude 0.929; NASA/Espenak: 04:13 UTC, 0.9299). News reports'
"96%" for that eclipse is by area, not diameter.

**Formulas** (sources in `sky.py`): limiting magnitude NELM = 7.93 − 5 log10(10^(4.316 − B/5) + 1)
(Schaefer 1990 via Unihedron); moonlight per Krisciunas & Schaefer (1991) as reproduced in Yao et
al. (2013, RAA 13:1255); B[nL] = 34.08 exp(20.7233 − 0.92104 V) (Garstang 1989, confirmed in
arXiv:1710.06755); observed meteor rate = ZHR · sin(h) / r^(6.5 − LM) (the IMO formula, zenith
exponent 1; LM capped at 6.5). Assumed: extinction 0.20 mag/airmass (K&S: 0.172 on Mauna Kea; Yao
et al.: 0.23 at Xinglong). Wikipedia's Bortle table lists naked-eye limits 0.3–1.2 mag fainter than
the Schaefer formula at the same SQM: Bortle describes experienced observers, the formula a
typical one; the app uses the formula and says so.

## 14. Every place in California (the location menu), added 2026-10-01

Built by `python -m skytrust build-places` (`src/skytrust/gazetteer.py`) into
`artifacts/places_ca.json` (3,561 entries, 276 KB). Raw files in `data/raw/places/` (gitignored).

| Source | What it gives | Licence |
|---|---|---|
| Census Bureau 2025 Gazetteer, places (`2025_Gaz_place_national.zip`) | the 1,619 California places: 462 cities, 21 towns (483 incorporated: the state's 482 plus Mountain House, incorporated in 2024), 1,136 census-designated places; boundary internal points, land area | public domain |
| USGS GNIS Domestic Names, California (`DomesticNames_CA_Text.zip`) | county of each place (joined on the GNIS id the Census lists as ANSICODE: 1,619 of 1,619 matched) and the settlement's own point (feature class "Populated Place") | public domain |
| GeoNames US dump (`US.zip`, `admin2Codes.txt`) | 83 named neighbourhoods (feature code PPLX with a population) and 57 communities that aren't census places; population of unincorporated places; SRTM heights (`dem`) | CC BY 4.0 |
| Census Bureau Vintage 2024 estimates (`sub-est2024.csv`) | population of incorporated places (orders search results only) | public domain |
| Census Bureau 2025 Gazetteer, ZCTAs | 1,802 ZIP code areas 900xx–961xx with internal points | public domain |
| USGS 3DEP via the Elevation Point Query Service | terrain height at every point (1–10 m lidar/DEM) | public domain |

**Town centres.** The Census internal point is the middle of a place's boundary, which can be far
from town: San Francisco's is at 37.727, −123.032, in the Pacific (the city includes the Farallon
Islands); GNIS's record for the City of Los Angeles sits in Woodland Hills. The point used is the
GNIS "Populated Place" of the same name in the same county, if it lies within 2.5 "boundary
radii" (radius of a circle with the place's land area; at least 3 km) of the internal point,
otherwise the place's own GNIS point (1,444 places use a populated-place centre). The radius rule
rejects same-name features nearby that are different places (San Simeon, East Whittier, Santa
Susana, Calipatria, all 4–6 km away from tiny places) while keeping Anaheim's downtown (2.3 radii).

**Cross-checks** (`data/raw/places/crosscheck_report.json`):
- Centre vs boundary (independent of the point's source): median 0.47 boundary radii from the
  internal point; 88.6 % within 1 radius, 98.9 % within 2. The 18 beyond 2 are San Francisco (the
  islands) and small places within the 3 km floor.
- GNIS vs GeoNames centres: median 0.0004 km, 98.1 % within 1 km, 1,491 matched. **Not an
  independent check:** GeoNames copies its US populated places from GNIS, so this confirms the
  join, not the coordinates. Five places differ by more than 5 km; each is a same-name feature that
  the radius rule correctly did not use.
- Heights: USGS 3DEP vs GeoNames SRTM: median |difference| 2.4 m, 95th percentile 13.3 m, 98.2 %
  within 25 m (n = 1,631); USGS vs Open-Meteo (Copernicus 90 m DEM): median 1.5 m, 95th percentile
  13.8 m (n = 601). GeoNames gives Furnace Creek 0 m; USGS −58.7 m (it is below sea level).

**ZIP codes** are named after the place they most plausibly lie in: distance to each town centre
divided by the town's boundary radius, preferring a city or town within 2 radii (so ZIP 90004 is
"near Los Angeles", not "near West Hollywood", whose centre is closer; 95616 is "near Davis", not
the UC Davis campus community).

**Lesson:** Open-Meteo's elevation API counts every coordinate in a request as one call: 600
points in six requests triggered HTTP 429 while the network backfill was running. Heights moved to
the USGS service (one point per request, ~2 s each, four in parallel, ~30 minutes for all).

## 15. The live sky chart (browser) vs the Python chart, added 2026-10-01

The Sky Guide's chart is drawn in the browser from data computed in Python (`app/views/skylive.py`)
and is checked against the Python chart (`app/views/skychart.py`, still used on Tonight) at the
same moments: Death Valley and Los Angeles, three times of night, both modes, on 2026-10-01.
- Every named star and planet drawn by both: largest position difference 0.14 units on the
  1,000-unit chart (about 0.03° of sky).
- Stars drawn: the browser's rounded count matches the Python count (e.g. Los Angeles 45/45, 41/42,
  54/54; Death Valley "about 2,800" vs 2,771–2,855). The only star-level difference seen was one
  borderline star (Alcor) in one frame, at the visibility limit.
- Visibility tables: across light pollution 17.2–21.9 mag/arcsec², Moon heights −3° to 80° and
  phases 0–140°, 99 % of the sky is within 0.05 mag of the exact model, worst case 0.15 mag
  beside a bright Moon (test in `tests/test_sky.py`).
- **K&S finding:** the published moonlight equations switch the Mie term from 10^(6.15−ρ/40) to
  6.2×10⁷ρ⁻² below ρ = 10° from the Moon; the two differ by 28 % at 10° (7.9×10⁵ vs 6.2×10⁵). The
  model keeps the published form; the browser table has nodes on both sides of 10°.
- Smoothness: redrawing ~3,000 stars, the Milky Way and the figures takes about one screen refresh
  in Chrome on the development Mac (frames measured over 40 slider positions).

## 16. Twilight, added 2026-10-01

The sky model now includes leftover sunlight. Source: Patat, Ugolnikov & Postylyakov (2006),
"UBVRI twilight sky brightness at ESO-Paranal", *A&A* 455:385, Table 1: zenith V brightness
m = 11.84 + 1.518 (ζ − 95) − 0.057 (ζ − 95)² mag/arcsec² for Sun zenith distance 95° ≤ ζ ≤ 105°,
from >2,000 FORS1 frames. (The PDF's text layer drops the minus sign of the quadratic term; only
the negative sign reaches the night level, 21.32 at ζ = 105°, where the paper says the night sky
takes over at ζ ≈ 105°–106°.) SkyTrust takes the twilight glow as the brightness above the fit's
105° value, zero below a 15° solar depression, and brightens it towards the horizon like the dark
sky. **Limits:** the extra glow on the side of the set Sun isn't modelled (low western sky after
sunset is brighter than shown), and the paper finds deep twilight ~30 % brighter at a 600 m site
than at Paranal (2,600 m), so near sea level twilight lasts slightly longer than shown. Result at
a dark site: faintest star −0.4 at civil dusk (Sun −6°), 4.0 at −10°, 5.3 at −12°, full darkness
by −15°; in a city the light pollution takes over by about −12°. The Sky Guide's slider now runs
from civil dusk to civil dawn with the fully dark part (Sun below −18°) marked.
