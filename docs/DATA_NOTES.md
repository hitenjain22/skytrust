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

Implications, raised at the Phase 0 checkpoint:
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
