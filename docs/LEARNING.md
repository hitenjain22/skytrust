# Learning Notes

One entry per module: what it does, why it's built that way, the key concept, and interview
questions to be ready for. Terms follow the glossary in SPEC.md Section 18.

---

## Project setup (`pyproject.toml`, `Makefile`, CI)

**What:** `pyproject.toml` pins every dependency to an exact version; `uv sync` builds an identical
environment anywhere. `ruff` lints/formats, `pytest` runs offline tests, and GitHub Actions runs
both on every push.

**Why:** Reproducibility. If a number in RESULTS.md changes, it should be because the data or code
changed, not because scikit-learn silently upgraded. CI proves the tests pass on a clean machine,
not just on this laptop.

**Key concept:** *Tests must not hit the network by default.* Live APIs change, go down, and
rate-limit. We record real responses once (`tests/fixtures/`) and test the parsing against those.
`pytest -m network` is the opt-in for live contract checks.

**Interview questions:**
1. Why pin exact versions instead of `>=`?
2. Why are your tests offline? How do you know your code still works against the real API?
3. What does CI run, and what would a red build tell you?

---

## `skytrust/data/http.py`: the shared HTTP client

**What:** Every request in the project goes through `HttpClient.get()`. It adds a 30 s timeout and a
User-Agent, waits politely between calls, retries 429/5xx and connection errors up to 5 times with
exponential backoff + jitter, and raises exactly two error types: `SourceUnavailableError`
(couldn't get an answer) or `BadResponseError` (got an answer, but it's garbage).

**Why:** Free public APIs (IEM especially) rate-limit. During Phase 0 IEM returned HTTP 429 several
times in a row and the retry logic recovered each time. Two typed errors let callers decide:
"source down → use cached/last-good data" vs "bad payload → don't cache it, fail loudly".

**Key concept:** *Exponential backoff with jitter.* After failure *n*, wait a random time in
[0, base·2ⁿ]. Doubling gives the server room to recover; randomness stops many clients from
retrying at the exact same instant (a "thundering herd").

**Interview questions:**
1. Why retry a 503 but not a 400?
2. What is jitter and what problem does it solve?
3. How do you test retry logic without actually waiting? (Answer: inject `sleep` and a fake session.)

---

## `skytrust/data/cache.py`: disk cache

**What:** Saves each raw response to `data/raw/{source}/{site}/{model}/{start}_{end}.{ext}`. A
second run reads from disk, making zero network calls.

**Why:** The backfill is thousands of requests; re-running it on every code change would be slow
and rude to free services. Writes are atomic (write temp file, then rename) so a crash never leaves
a half-file that later *looks* cached. Payloads are validated *before* caching.

**Interview questions:**
1. What happens if the program crashes mid-download? Why doesn't that corrupt the cache?
2. How do you prove a complete cache makes zero network calls? (A test with a mock client asserts
   `get` is never called.)

---

## `skytrust/data/iem.py`: ASOS observations → hourly cloud cover

**What:** Downloads METAR reports, converts sky codes (CLR/FEW/SCT/BKN/OVC/VV) to cloud fractions
via okta midpoints, takes the max over layers per report, then the max over all reports in each
hour's window (H−30 min, H+30 min].

**Why it's built that way:**
- **Max over layers:** METAR amounts are cumulative, so the top layer's amount *is* total coverage.
- **Nearest report to the hour (not max):** the spec originally took the max over all reports in
  the window. But AUN reports 3×/hour and SAC 1×, so "max" made AUN look cloudier purely because it
  reports more. Using the report nearest H gives every station one reading per hour, which keeps the
  labels comparable. Ties go to the cloudier report. `max` is kept as a sensitivity option.
- **`report_type=3,4`:** IEM by default includes 5-minute readings; a max over 12 readings per
  hour would be much stricter than intended, and stricter at some stations than others.
- **The window trick:** "t is in (H−30, H+30]" rearranges to "H = ceil_to_hour(t − 30 min)". That
  gives each report exactly one hour with a single vectorised line instead of a loop.

**Key concept:** *Know your ground truth's blind spots.* ASOS ceilometers only see clouds below
12,000 ft, so "CLR" can still mean a sky full of cirrus. That's why the primary label also uses ERA5.
We also found FAT is human-augmented and *does* report high clouds, and AUN (an AWOS) reports 3×
per hour, so the ASOS label isn't equally strict everywhere.

**Interview questions:**
1. Why use the report nearest the top of the hour instead of the max over the hour? What's the trade-off?
2. What can't ASOS see, and how does your design compensate?
3. An obs at exactly H−30 min: which hour does it belong to, and why?
4. Why did you pass `report_type=3,4` to IEM?

---

## `skytrust/sites.py` + `validate-sites`

**What:** Looks up each candidate station in IEM's CA_ASOS metadata (coordinates and elevation
come from IEM, never hand-typed), downloads its history one year at a time, and measures the share
of night hours with a valid sky report. Stations ≥ 85 % go into `config/sites.yaml`.

**Why:** A backtest is only as good as its labels. A station with big gaps would silently drop
nights, and if gaps cluster in bad weather (outages during storms) they'd bias the base rate.

**Interview questions:**
1. Why not type the coordinates yourself?
2. Why might missing data be *non-random*, and why does that matter?

---

## Approximation to remember: what "lead d" means (SPEC 4.7)

`cloud_cover_previous_day1` at 03:00 UTC is the value predicted ~24 h before *03:00*, and at 06:00 it's
~24 h before *06:00*. So one night's lead-1 hours come from slightly different model runs. That's
close to "the forecast you'd check the day before", but not identical to looking at one forecast at
5 pm. Also: 3-hourly models (ECMWF 0.25°) are interpolated to hourly by Open-Meteo, so their
"consecutive clear hours" are partly an interpolation artifact.

---

## `skytrust/astro.py`: dusk, dawn, dark hours, the Moon

**What:** For each site and night, finds astronomical dusk and dawn (Sun centre 18° below the
horizon), lists the whole UTC hours in between ("dark hours"), and computes Moon illumination,
Moon altitude, and moonrise/moonset.

**Why it's built that way:**
- **Search by event, not by formula:** Skyfield's `find_discrete` samples "is the Sun below −18°?"
  hourly over the whole date range and then narrows each change down to the exact moment. One call
  gives every dusk and dawn for a year (~5 s per site-year).
- **Night key:** a dusk is assigned to the *local* date it happens on, so "night of Jan 10" =
  dusk on the evening of Jan 10 local, even though that's already Jan 11 in UTC.
- **Dark hours built in UTC:** UTC has no daylight saving, so every hour exists exactly once.
  On spring-forward night the local clock jumps from 01:00 to 03:00; in UTC the hours are still
  consecutive, so there are no duplicates or gaps (tested for 4 DST nights).
- **Offline ephemeris:** a committed 880 KB excerpt of JPL DE421 covering 2023–2030.
- **Verified independently:** dusk/dawn match the `astral` library (different algorithm) within
  2 minutes on both solstices.

**Key concept:** *Astronomical twilight.* Until the Sun is 18° below the horizon, scattered sunlight
still brightens the sky enough to wash out faint galaxies and nebulae. That's why the "dark window"
is shorter than sunset→sunrise: at SAC it's 5 hours in June and 11 in December.

**Interview questions:**
1. Why is a night keyed by the local evening date, and how do you handle it crossing midnight UTC?
2. How do you guarantee no duplicated/missing hours on daylight-saving nights?
3. How did you verify your dusk times are right? Why is a *different* library a better check?
4. What happens at a latitude with no astronomical darkness?

---

## `skytrust/data/openmeteo.py`: forecasts and ERA5

**What:** Downloads Previous Runs forecasts (per site × model, only the leads that model has) and
ERA5 reanalysis, month by month, through the shared HTTP client and disk cache. `parse_hourly`
turns any Open-Meteo response into a UTC-indexed DataFrame of cloud *fractions* (0–1).

**Why it's built that way:**
- **Convert percent → fraction in exactly one place,** so no other module has to remember which
  scale a number is on.
- **Handle both key styles:** multi-model responses name columns `cloud_cover_gfs_global`;
  single-model ones just `cloud_cover`. The parser accepts both.
- **Nulls are "missing", never 0 %.** A forecast that doesn't exist must not look like a clear sky.
  (A test caught a bug here: pandas 3 keeps NaN when stacking, which broke the range check on
  HRRR's all-null long leads.)
- **Don't cache what isn't finished:** if a chunk's last day has no data (ERA5's ~6-day lag), it's
  used but not cached, so a later run fetches the completed version.

**Interview questions:**
1. Why do you treat a null forecast differently from 0 % cloud?
2. How do you avoid caching a month that the provider hasn't finished publishing?
3. Why pass the station elevation to Open-Meteo?

---

## `skytrust/data/backfill.py` + `python -m skytrust fetch`

**What:** Works out which dates each source needs for a range of nights and runs the fetchers for
every site. It prints how many real network requests were made.

**Why:** *Idempotent* commands: running `fetch` twice gives the same result, and the second run
makes **0** requests (verified). If one site fails (outage/rate limit), the others continue and a
re-run resumes from the cache instead of starting over.

**Interview questions:**
1. What does "idempotent" mean and why does it matter for data pipelines?
2. How do you *prove* the second run didn't touch the network?
