// SkyTrust · live sky chart for the Sky Guide.
//
// Python (app/views/skylive.py, src/skytrust/sky.py) computes every position and the visibility
// tables; this file only interpolates between the night's 20-minute steps, turns RA/Dec of date
// into altitude/azimuth with the local sidereal time (the formula of
// sky.Observer.altaz_from_radec), projects onto the planisphere (skychart.project) and draws.
// The chart is redrawn on every movement of the slider, so it follows the finger or mouse.

const SIZE = 1000;
const R = 440;
const CX = 500;
const CY = 500;
const D2R = Math.PI / 180;
const COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW",
  "W", "WNW", "NW", "NNW"];
const KIND_ORDER = { moon: 0, planet: 1, star: 2, pattern: 3, cluster: 4, galaxy: 4, nebula: 4 };
const INK = "#C9D2EA";
const GOLD = "#F0C987";
const COLS = "minmax(0,1.4fr) minmax(0,1fr) 96px";

// ---------- decoding ----------

function decode(b64, Type) {
  if (!b64) return new Type(0);
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Type(bytes.buffer);
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;",
    '"': "&quot;", "'": "&#39;" }[c]));
}

function prepare(d) {
  const st = d.stars;
  const stars = {
    ra: Float64Array.from(decode(st.ra, Uint16Array), (v) => v / 100),
    dec: Float64Array.from(decode(st.dec, Int16Array), (v) => v / 100),
    mag: Float64Array.from(decode(st.mag, Int16Array), (v) => v / 100),
    tint: decode(st.tint, Uint8Array),
    names: st.names,
  };
  Object.assign(stars, trig(stars.ra, stars.dec));
  const rings = (block) => {
    const pts = Int16Array.from(decode(block.pts, Int16Array));
    const out = [];
    let k = 0;
    for (const len of block.len) {
      const ra = new Float64Array(len);
      const dec = new Float64Array(len);
      for (let j = 0; j < len; j++, k++) {
        ra[j] = pts[2 * k] / 10;
        dec[j] = pts[2 * k + 1] / 10;
      }
      out.push({ ra, dec, ...trig(ra, dec) });
    }
    return out;
  };
  return {
    d,
    stars,
    milkyWay: d.milkyWay.map(rings),
    lines: rings(d.figures.lines),
    tables: d.tables.map((t) => (t ? Float64Array.from(decode(t, Int16Array), (v) => v / 100)
      : null)),
    maxHere: Math.max(...d.curveHere) + 0.05,
    maxDark: Math.max(...d.curveDark) + 0.05,
    latRad: d.lat * D2R,
    sinLat: Math.sin(d.lat * D2R),
    cosLat: Math.cos(d.lat * D2R),
  };
}

// ---------- geometry ----------

// sin/cos of RA and Dec, computed once: then a frame needs only one asin and one sqrt per point.
function trig(ra, dec) {
  const n = ra.length;
  const out = { sr: new Float64Array(n), cr: new Float64Array(n), sd: new Float64Array(n),
    cd: new Float64Array(n) };
  for (let i = 0; i < n; i++) {
    out.sr[i] = Math.sin(ra[i] * D2R);
    out.cr[i] = Math.cos(ra[i] * D2R);
    out.sd[i] = Math.sin(dec[i] * D2R);
    out.cd[i] = Math.cos(dec[i] * D2R);
  }
  return out;
}

// Point i of a trig() set at the frame's sidereal time -> [alt (deg), x, y on the chart, and the
// horizontal unit vector (north, east, up)]. Same formula as sky.Observer.altaz_from_radec, with
// cos/sin of the hour angle from the angle-difference identities and azimuth taken from the
// north/east components directly (no atan2).
const OUT = new Float64Array(6);
function place(P, F, set, i) {
  const cosH = F.cosL * set.cr[i] + F.sinL * set.sr[i];
  const sinH = F.sinL * set.cr[i] - F.cosL * set.sr[i];
  const sinAlt = P.sinLat * set.sd[i] + P.cosLat * set.cd[i] * cosH;
  const north = set.sd[i] * P.cosLat - set.cd[i] * P.sinLat * cosH;
  const east = -set.cd[i] * sinH;
  const alt = Math.asin(Math.max(-1, Math.min(1, sinAlt))) / D2R;
  const cosAlt = Math.sqrt(north * north + east * east) || 1e-9;
  const r = ((90 - alt) / 90) * R;
  OUT[0] = alt;
  OUT[1] = CX - (r * east) / cosAlt;
  OUT[2] = CY - (r * north) / cosAlt;
  OUT[3] = north;
  OUT[4] = east;
  OUT[5] = sinAlt;
  return OUT;
}

// Angle from the Moon (deg) for a point with horizontal unit vector (north, east, up).
function moonSep(F, north, east, up) {
  if (!F.moonUp) return 180;
  const c = north * F.mN + east * F.mE + up * F.mU;
  return Math.acos(Math.max(-1, Math.min(1, c))) / D2R;
}

function lstAt(P, t) {
  return (((P.d.lst0 + P.d.lstRate * (t - P.d.t0)) % 360) + 360) % 360;
}

function altaz(P, raDeg, sinDec, cosDec, lst) {
  const ha = (lst - raDeg) * D2R;
  const cosHa = Math.cos(ha);
  const sinAlt = P.sinLat * sinDec + P.cosLat * cosDec * cosHa;
  const alt = Math.asin(Math.max(-1, Math.min(1, sinAlt))) / D2R;
  const y = -cosDec * Math.sin(ha);
  const x = sinDec * P.cosLat - cosDec * P.sinLat * cosHa;
  const az = ((Math.atan2(y, x) / D2R) % 360 + 360) % 360;
  return [alt, az];
}

function altazDeg(P, ra, dec, lst) {
  return altaz(P, ra, Math.sin(dec * D2R), Math.cos(dec * D2R), lst);
}

function project(alt, az) {
  const r = ((90 - alt) / 90) * R;
  return [CX - r * Math.sin(az * D2R), CY - r * Math.cos(az * D2R)];
}

function separation(alt1, az1, alt2, az2) {
  const c = Math.sin(alt1 * D2R) * Math.sin(alt2 * D2R)
    + Math.cos(alt1 * D2R) * Math.cos(alt2 * D2R) * Math.cos((az1 - az2) * D2R);
  return Math.acos(Math.max(-1, Math.min(1, c))) / D2R;
}

// ---------- time interpolation ----------

function stepAt(P, t) {
  const s = P.d.steps;
  if (t <= s[0]) return [0, 0];
  if (t >= s[s.length - 1]) return [s.length - 2, 1];
  let i = 0;
  while (i < s.length - 2 && s[i + 1] <= t) i++;
  return [i, (t - s[i]) / (s[i + 1] - s[i])];
}

function lerp(a, b, f) {
  return a + (b - a) * f;
}

function lerpAngle(a, b, f) {
  const d = ((b - a + 540) % 360) - 180;
  return ((a + d * f) % 360 + 360) % 360;
}

// ---------- visibility (tables from sky.limit_table) ----------

function bracket(grid, v) {
  if (v <= grid[0]) return [0, 0];
  const last = grid.length - 1;
  if (v >= grid[last]) return [last - 1, 1];
  let lo = 0;
  let hi = last;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (grid[mid] <= v) lo = mid;
    else hi = mid;
  }
  return [lo, (v - grid[lo]) / (grid[lo + 1] - grid[lo])];
}

function curveAt(P, curve, alt) {
  const [i, f] = bracket(P.d.alts, alt);
  return lerp(curve[i], curve[i + 1], f);
}

function tableAt(P, table, alt, sep) {
  const nSep = P.d.seps.length;
  const [i, fa] = bracket(P.d.alts, alt);
  const [j, fs] = bracket(P.d.seps, sep);
  const top = lerp(table[i * nSep + j], table[i * nSep + j + 1], fs);
  const bottom = lerp(table[(i + 1) * nSep + j], table[(i + 1) * nSep + j + 1], fs);
  return lerp(top, bottom, fa);
}

// Faintest visible magnitude at (alt, separation from the Moon) at the current moment.
function limitAt(P, F, alt, sep) {
  if (F.mode === "dark") {  // no light pollution or Moon; twilight still counts
    const dk = (k) => curveAt(P, P.d.darkTwilight[k] || P.d.curveDark, alt);
    return lerp(dk(F.s), dk(F.s + 1), F.f);
  }
  const one = (k) => (P.tables[k] ? tableAt(P, P.tables[k], alt, sep)
    : curveAt(P, P.d.curveHere, alt));
  return lerp(one(F.s), one(F.s + 1), F.f);
}

// ---------- words ----------

function compass(az) {
  return COMPASS[Math.floor(((az % 360) + 11.25) / 22.5) % 16];
}

function heightWords(alt) {
  if (alt >= 70) return "nearly overhead";
  if (alt >= 45) return "high";
  if (alt >= 20) return "halfway up";
  if (alt >= 5) return "low";
  return "on the horizon";
}

function whereWords(alt, az) {
  const h = heightWords(alt);
  return h === "nearly overhead" ? h : `${h} in the ${compass(az)}`;
}

function cap(s) {
  return s.slice(0, 1).toUpperCase() + s.slice(1);
}

function roundHalfEven(x) {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

function fmtCount(n) {
  // as lookup.fmt_count (Python rounds halves to even)
  if (n >= 1000) return (roundHalfEven(n / 100) * 100).toLocaleString("en-US");
  if (n >= 100) return (roundHalfEven(n / 10) * 10).toLocaleString("en-US");
  return String(n);
}

const FORMATS = new Map();
function clock(t, tz) {
  // as lookup.clock: rounded to 5 minutes, "9 PM" on the hour
  const r = Math.round(t / 300000) * 300000;
  if (!FORMATS.has(tz)) {
    FORMATS.set(tz, new Intl.DateTimeFormat("en-US", { timeZone: tz, hour: "numeric",
      minute: "2-digit", hour12: true }));
  }
  const parts = FORMATS.get(tz).formatToParts(new Date(r));
  const get = (type) => (parts.find((p) => p.type === type) || {}).value || "";
  const mm = get("minute");
  return mm === "00" ? `${get("hour")} ${get("dayPeriod")}` : `${get("hour")}:${mm} ${get("dayPeriod")}`;
}

// ---------- drawing ----------

function skyFillStops(zen) {
  const t = Math.max(0, Math.min(1, (zen - 17) / (22 - 17)));
  const mix = (a, b) => "#" + a.map((x, i) => Math.round(x + (b[i] - x) * t)
    .toString(16).padStart(2, "0")).join("");
  const centre = mix([0x3a, 0x45, 0x5e], [0x10, 0x18, 0x2e]);
  const edge = mix([0x5a, 0x58, 0x5e], [0x16, 0x20, 0x3a]);
  return [centre, edge];
}

function milkyWaySvg(P, F, strength) {
  if (strength <= 0.02) return "";
  let out = "";
  P.milkyWay.forEach((rings, level) => {
    const d = [];
    for (const ring of rings) {
      let maxAlt = -90;
      let path = "";
      for (let j = 0; j < ring.ra.length; j++) {
        const o = place(P, F, ring, j);
        if (o[0] > maxAlt) maxAlt = o[0];
        path += (j ? "L" : "M") + Math.round(o[1]) + " " + Math.round(o[2]);
      }
      if (maxAlt >= -10) d.push(path + "Z");
    }
    if (d.length) {
      const op = (0.05 + 0.02 * level) * strength;
      out += `<path d="${d.join(" ")}" fill="#C9D4F2" fill-opacity="${op.toFixed(3)}" fill-rule="evenodd"/>`;
    }
  });
  return `<g filter="url(#mwblur-${P.uid})">${out}</g>`;
}

function linesSvg(P, F) {
  let out = "";
  for (const seg of P.lines) {
    let minAlt = 90;
    const pts = [];
    for (let j = 0; j < seg.ra.length; j++) {
      const o = place(P, F, seg, j);
      if (o[0] < minAlt) minAlt = o[0];
      pts.push(Math.round(o[1]) + "," + Math.round(o[2]));
    }
    if (minAlt > -15) out += `<polyline points="${pts.join(" ")}"/>`;
  }
  return `<g fill="none" stroke="${INK}" stroke-opacity="0.22" stroke-width="1.6" stroke-linejoin="round">${out}</g>`;
}

function labelsSvg(P, F) {
  let out = "";
  for (const [name, ra, dec] of P.d.figures.labels) {
    const [alt, az] = altazDeg(P, ra, dec, F.lst);
    if (alt > 12) {
      const [x, y] = project(alt, az);
      out += `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}">${esc(name)}</text>`;
    }
  }
  return `<g fill="${INK}" fill-opacity=".42" font-family="Geist Mono, monospace" font-size="15" letter-spacing="2.4" text-anchor="middle">${out}</g>`;
}

function starsSvg(P, F) {
  const s = P.stars;
  const tints = P.d.tints;
  const batches = new Map();
  const named = [];
  let count = 0;
  for (let i = 0; i < s.ra.length; i++) {
    const mg = s.mag[i];
    if (mg > F.maxLimit) continue; // fainter than anything visible anywhere right now
    const o = place(P, F, s, i);
    const alt = o[0];
    if (alt <= 0) continue;
    const lm = limitAt(P, F, alt, F.mode === "dark" ? 180 : moonSep(F, o[3], o[4], o[5]));
    if (mg > lm) continue;
    count++;
    const x = o[1];
    const y = o[2];
    const r = Math.min(5.2, Math.max(0.7, 3.4 - 0.52 * mg));
    const op = Math.min(1, Math.max(0.42, 0.45 + (0.55 * (lm - mg)) / 2));
    const tint = tints[s.tint[i]];
    const name = s.names[i];
    if (name !== undefined || mg < 2.0) {
      const cls = mg < 2.2 ? ' class="sk-tw"' : "";
      const tip = name !== undefined ? `<title>${esc(name)} (mag ${mg.toFixed(1)})</title>` : "";
      named.push(`<circle${cls} cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${r.toFixed(2)}" fill="${tint}" style="--o:${op.toFixed(2)}" fill-opacity="${op.toFixed(2)}">${tip}</circle>`);
      continue;
    }
    const key = `${tint}|${Math.round(r * 4) / 4}|${Math.round(op * 10) / 10}`;
    const pts = batches.get(key);
    const seg = `M${Math.round(x)} ${Math.round(y)}h0`;
    if (pts) pts.push(seg);
    else batches.set(key, [seg]);
  }
  let out = "";
  for (const [key, pts] of batches) {
    const [tint, r, o] = key.split("|");
    out += `<path d="${pts.join("")}" stroke="${tint}" stroke-width="${(2 * Number(r)).toFixed(2)}" stroke-opacity="${o}" stroke-linecap="round" fill="none"/>`;
  }
  return [out + named.join(""), count];
}

function planetsSvg(P, F) {
  let out = "";
  for (const p of P.d.planets) {
    const ra = lerpAngle(p.ra[F.s], p.ra[F.s + 1], F.f);
    const dec = lerp(p.dec[F.s], p.dec[F.s + 1], F.f);
    const mg = lerp(p.mag[F.s], p.mag[F.s + 1], F.f);
    const [alt, az] = altazDeg(P, ra, dec, F.lst);
    if (alt <= 0) continue;
    const sep = F.moonUp ? separation(alt, az, F.moonAlt, F.moonAz) : 180;
    if (mg > limitAt(P, F, alt, sep)) continue;
    const [x, y] = project(alt, az);
    const r = Math.min(7.5, Math.max(3.2, 4.6 - 0.55 * mg));
    out += `<g><title>${esc(p.name)} (mag ${mg.toFixed(1)})</title>`
      + `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${r.toFixed(1)}" fill="${GOLD}"/>`
      + `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${(r + 5).toFixed(1)}" fill="none" stroke="${GOLD}" stroke-opacity=".35"/>`
      + `<text x="${(x + r + 9).toFixed(1)}" y="${(y + 5).toFixed(1)}" fill="${GOLD}" font-size="22" font-family="Geist, sans-serif">${esc(p.name)}</text></g>`;
  }
  return out;
}

function moonSvg(P, F) {
  if (!F.moonUp) return "";
  const [x, y] = project(F.moonAlt, F.moonAz);
  const r = 16;
  const lit = (1 - Math.cos(F.moonPhase * D2R)) / 2;
  const waxing = F.moonPhase < 180;
  const sx = 1 - 2 * lit;
  const rx = Math.abs(sx) * r;
  const half = `M ${x.toFixed(1)} ${(y - r).toFixed(1)} A ${r} ${r} 0 0 ${waxing ? 1 : 0} ${x.toFixed(1)} ${(y + r).toFixed(1)} Z`;
  const ell = sx > 0 ? "#2A3248" : "#F3EEDF";
  return `<g><circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${(r * 2.6).toFixed(1)}" fill="url(#moonglow-${P.uid})"/>`
    + `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${r}" fill="#2A3248"/>`
    + `<path d="${half}" fill="#F3EEDF"/>`
    + `<ellipse cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" rx="${rx.toFixed(1)}" ry="${r}" fill="${ell}"/></g>`
    + `<text x="${(x + 22).toFixed(1)}" y="${(y + 6).toFixed(1)}" fill="#E8E2D0" font-size="22" font-family="Geist, sans-serif">Moon</text>`;
}

function frame(P, t, mode) {
  const [s, f] = stepAt(P, t);
  const lst = lstAt(P, t);
  const m = P.d.moon;
  const mra = lerpAngle(m.ra[s], m.ra[s + 1], f);
  const mdec = lerp(m.dec[s], m.dec[s + 1], f);
  const [moonAlt, moonAz] = altazDeg(P, mra, mdec, lst);
  const F = { t, s, f, lst, mode, moonAlt, moonAz, moonUp: moonAlt > 0,
    moonPhase: lerp(m.phase[s], m.phase[s + 1], f), moonIllum: lerp(m.illum[s], m.illum[s + 1], f) };
  F.cosL = Math.cos(lst * D2R);
  F.sinL = Math.sin(lst * D2R);
  F.mN = Math.cos(moonAlt * D2R) * Math.cos(moonAz * D2R);
  F.mE = Math.cos(moonAlt * D2R) * Math.sin(moonAz * D2R);
  F.mU = Math.sin(moonAlt * D2R);
  F.zen = mode === "dark" ? lerp(P.d.zenDark[s], P.d.zenDark[s + 1], f)
    : lerp(P.d.zen[s], P.d.zen[s + 1], f);
  F.twilight = t < P.d.dark0 - 60000 || t > P.d.dark1 + 60000;
  // nothing anywhere can beat the moonless sky's best (moonlight only brightens the sky)
  F.maxLimit = mode === "dark" ? P.maxDark : P.maxHere;
  return F;
}

function chartSvg(P) {
  const ticks = [];
  for (let a = 0; a < 360; a += 15) {
    const s = Math.sin(a * D2R);
    const c = Math.cos(a * D2R);
    ticks.push(`<line x1="${(CX - R * s).toFixed(1)}" y1="${(CY - R * c).toFixed(1)}" x2="${(CX - (R + 9) * s).toFixed(1)}" y2="${(CY - (R + 9) * c).toFixed(1)}" stroke="currentColor" stroke-opacity=".35"/>`);
  }
  const cardinals = [["N", 0], ["E", 90], ["S", 180], ["W", 270]].map(([txt, a]) => {
    const x = CX - (R + 30) * Math.sin(a * D2R);
    const y = CY - (R + 30) * Math.cos(a * D2R) + 9;
    return `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" text-anchor="middle" fill="currentColor" fill-opacity=".7" font-size="26" font-family="Geist, sans-serif" font-weight="500">${txt}</text>`;
  }).join("");
  const rings = [30, 60].map((a) => `<circle cx="${CX}" cy="${CY}" r="${(R * (90 - a) / 90).toFixed(1)}" fill="none" stroke="${INK}" stroke-opacity=".08" stroke-dasharray="3 7"/>`).join("");
  const u = P.uid;
  return `<svg viewBox="0 0 ${SIZE} ${SIZE}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="${esc("The sky over " + P.d.place)}">`
    + `<defs><clipPath id="horizon-${u}"><circle cx="${CX}" cy="${CY}" r="${R}"/></clipPath>`
    + `<radialGradient id="skyfill-${u}" cx="50%" cy="50%" r="50%"><stop offset="0" class="sk-live-c"/><stop offset=".78" class="sk-live-c"/><stop offset="1" class="sk-live-e"/></radialGradient>`
    + `<radialGradient id="moonglow-${u}"><stop offset="0" stop-color="#F3EEDF" stop-opacity=".35"/><stop offset="1" stop-color="#F3EEDF" stop-opacity="0"/></radialGradient>`
    + `<filter id="mwblur-${u}" x="-5%" y="-5%" width="110%" height="110%"><feGaussianBlur stdDeviation="6"/></filter></defs>`
    + `<circle cx="${CX}" cy="${CY}" r="${R}" fill="url(#skyfill-${u})"/>`
    + `<g clip-path="url(#horizon-${u})"><g class="sk-live-sky"></g>${rings}<g class="sk-live-dyn"></g></g>`
    + `<circle cx="${CX}" cy="${CY}" r="${R}" fill="none" stroke="${INK}" stroke-opacity=".28" stroke-width="1.5"/>`
    + ticks.join("") + cardinals + "</svg>";
}

// ---------- the "Up at ..." list ----------

function itemAt(P, it, F) {
  const alt = lerp(it.alt[F.s], it.alt[F.s + 1], F.f);
  const az = lerpAngle(it.az[F.s], it.az[F.s + 1], F.f);
  return [alt, az];
}

function listHtml(P, F) {
  const rows = [];
  for (const it of P.d.items) {
    const [alt, az] = itemAt(P, it, F);
    if (alt <= 3) continue;
    let seen = true;
    if (it.k !== "moon" && F.mode !== "dark") {
      const sep = F.moonUp ? separation(alt, az, F.moonAlt, F.moonAz) : 180;
      seen = it.v <= limitAt(P, F, alt, sep) - (it.x ? P.d.extendedMargin : 0);
    }
    rows.push({ it, alt, az, seen });
  }
  rows.sort((a, b) => (KIND_ORDER[a.it.k] ?? 5) - (KIND_ORDER[b.it.k] ?? 5)
    || (a.it.m ?? 9) - (b.it.m ?? 9));
  const groups = [
    ["Moon and planets", rows.filter((r) => r.it.k === "moon" || r.it.k === "planet")],
    ["Brightest stars", rows.filter((r) => r.it.k === "star").slice(0, 4)],
    ["Star patterns", rows.filter((r) => r.it.k === "pattern").slice(0, 4)],
    ["Clusters, galaxies, nebulae", rows.filter((r) => ["cluster", "galaxy", "nebula"].includes(r.it.k)).slice(0, 4)],
  ];
  let html = `<div class="sk-eyebrow">Up at ${esc(clock(F.t, P.d.tz))}</div>`;
  for (const [title, group] of groups) {
    if (!group.length) continue;
    html += `<div class="sk-eyebrow" style="margin-top:14px">${title}</div><div class="sk-list">`;
    for (const r of group) {
      const tag = r.seen
        ? '<span class="sk-badge" style="--c:var(--sk-go)">Visible</span>'
        : '<span class="sk-badge" title="Too faint to see from here now">Too faint</span>';
      html += `<div class="sk-row" style="--cols:${COLS}"><div class="sk-main"><div class="sk-row-title">${esc(r.it.n)}</div>`
        + `<div class="sk-row-sub">${esc(r.it.s)}</div></div>`
        + `<div class="sk-row-sub" style="font-size:.86rem">${esc(cap(whereWords(r.alt, r.az)))}</div>`
        + `<div style="text-align:right">${tag}</div></div>`;
    }
    html += "</div>";
  }
  if (!rows.length) html += '<p class="sk-row-sub">Nothing bright is above the horizon right now.</p>';
  return html;
}

// ---------- the widget ----------

function build(root, P) {
  const d = P.d;
  const step = d.stepMinutes * 60000;
  const n = Math.max(1, Math.round((d.t1 - d.t0) / step));
  const startIdx = Math.max(0, Math.min(n, Math.round((d.start - d.t0) / step)));
  const now = Date.now();
  const nowIdx = now > d.t0 && now < d.t1 ? Math.round((now - d.t0) / step) : null;
  root.innerHTML = `
    <div class="sk-live">
      <div class="sk-live-controls">
        <div class="sk-live-time">
          <div class="sk-live-time-top">
            <span class="sk-eyebrow">Time tonight</span>
            <span class="sk-live-clock" aria-live="off"></span>
            ${nowIdx !== null ? '<button type="button" class="sk-live-now">Now</button>' : ""}
          </div>
          <input class="sk-live-range" type="range" min="0" max="${n}" step="1" value="${startIdx}" aria-label="Time tonight">
          <div class="sk-live-ticks" aria-hidden="true"></div>
          <div class="sk-live-dark">Fully dark ${esc(clock(d.dark0, d.tz))} – ${esc(clock(d.dark1, d.tz))}.
            Before and after, leftover sunlight (twilight) hides the fainter stars.</div>
        </div>
        <div class="sk-live-mode" role="radiogroup" aria-label="Show">
          <button type="button" role="radio" aria-checked="true" data-mode="here">Your sky</button>
          <button type="button" role="radio" aria-checked="false" data-mode="dark">A perfectly dark sky</button>
        </div>
      </div>
      <div class="sk-live-body">
        <div class="sk-live-chart"><div class="sk-chart">${chartSvg(P)}</div><div class="sk-chart-help"></div></div>
        <div class="sk-live-list"></div>
      </div>
    </div>`;
  const range = root.querySelector(".sk-live-range");
  const clockEl = root.querySelector(".sk-live-clock");
  const help = root.querySelector(".sk-chart-help");
  const listEl = root.querySelector(".sk-live-list");
  const dyn = root.querySelector(".sk-live-dyn");
  const skyLayer = root.querySelector(".sk-live-sky");
  const stopsC = root.querySelectorAll(".sk-live-c");
  const stopE = root.querySelector(".sk-live-e");
  const ticks = root.querySelector(".sk-live-ticks");

  // hour labels under the slider (every hour, or every two on a narrow screen)
  const hourly = [];
  const firstHour = Math.ceil(d.t0 / 3600000) * 3600000;
  for (let t = firstHour; t < d.t1; t += 3600000) hourly.push(t);
  const drawTicks = () => {
    const every = root.clientWidth < 560 ? 2 : 1;
    ticks.innerHTML = hourly.filter((_, i) => i % every === 0).map((t) => {
      const left = ((t - d.t0) / (d.t1 - d.t0)) * 100;
      return `<span style="left:${left.toFixed(2)}%">${esc(clock(t, d.tz))}</span>`;
    }).join("");
  };
  drawTicks();
  // the twilight parts of the night, shaded on the slider's track
  const pct = (x) => `${(((x - d.t0) / (d.t1 - d.t0)) * 100).toFixed(2)}%`;
  range.style.setProperty("--d0", pct(d.dark0));
  range.style.setProperty("--d1", pct(d.dark1));

  let mode = "here";
  let pending = false;
  let lastList = "";
  const draw = () => {
    pending = false;
    const idx = Number(range.value);
    const t = Math.min(d.t1, d.t0 + idx * step);
    const F = frame(P, t, mode);
    const strength = Math.max(0, Math.min(1, (F.zen - d.minMilkyWaySqm) / (21.6 - d.minMilkyWaySqm)));
    const [centre, edge] = skyFillStops(F.zen);
    stopsC.forEach((el) => el.setAttribute("stop-color", centre));
    stopE.setAttribute("stop-color", edge);
    skyLayer.innerHTML = milkyWaySvg(P, F, strength);
    const [stars, count] = starsSvg(P, F);
    dyn.innerHTML = linesSvg(P, F) + labelsSvg(P, F) + stars + planetsSvg(P, F) + moonSvg(P, F);
    const label = F.twilight
      ? `${clock(t, d.tz)} · ${t < d.dark0 ? "evening" : "morning"} twilight`
      : clock(t, d.tz);
    clockEl.textContent = label;
    root.querySelector(".sk-live").classList.toggle("is-twilight", F.twilight);
    range.setAttribute("aria-valuetext", label);
    range.style.setProperty("--p", `${((idx / n) * 100).toFixed(2)}%`);
    const what = mode === "here" ? "you can see" : "a perfectly dark sky would show";
    const stars = count === 0 ? "No stars show yet"
      : count === 1 ? "1 star" : `About ${fmtCount(count)} stars`;
    help.textContent = (count === 0
      ? `${stars} at ${clock(t, d.tz)}: the sky is still too bright.`
      : `${stars} ${what} at ${clock(t, d.tz)}.`)
      + " Hold it overhead with north at the top, or turn it so the direction you face is at "
      + "the bottom.";
    // the list changes only every few minutes: rebuild it only when its content would change
    const key = `${mode}|${clock(t, d.tz)}`;
    if (key !== lastList) {
      listEl.innerHTML = listHtml(P, F);
      lastList = key;
    }
  };
  const schedule = () => {
    if (!pending) {
      pending = true;
      requestAnimationFrame(draw);
    }
  };
  range.addEventListener("input", schedule);
  range.addEventListener("change", schedule);
  root.querySelectorAll(".sk-live-mode button").forEach((b) => {
    b.addEventListener("click", () => {
      mode = b.dataset.mode;
      root.querySelectorAll(".sk-live-mode button").forEach((x) =>
        x.setAttribute("aria-checked", String(x === b)));
      schedule();
    });
  });
  const nowBtn = root.querySelector(".sk-live-now");
  if (nowBtn) {
    nowBtn.addEventListener("click", () => {
      range.value = String(Math.round((Date.now() - d.t0) / step));
      schedule();
    });
  }
  if (typeof ResizeObserver !== "undefined") new ResizeObserver(drawTicks).observe(root);
  draw();
}

export default function (component) {
  const { data, parentElement } = component;
  if (!data || !parentElement) return;
  let root = parentElement.querySelector(".sk-live-root");
  if (!root) {
    root = document.createElement("div");
    root.className = "sk-live-root";
    parentElement.appendChild(root);
  }
  // Streamlit calls this again on every rerun of the page: keep the chart (and where the
  // slider is) unless the place or the night changed.
  if (root.dataset.id === data.id) return;
  root.dataset.id = data.id;
  try {
    const P = prepare(data);
    P.uid = "lv" + Math.random().toString(36).slice(2, 9);
    build(root, P);
  } catch (err) {
    root.innerHTML = '<p class="sk-row-sub">The live chart could not be drawn in this browser.</p>';
    console.error("SkyTrust live chart:", err);
  }
}
