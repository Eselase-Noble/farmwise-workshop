// Crop recommendation page.
"use strict";

async function postJSON(url, body) {
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(data.detail || `Request failed (${resp.status})`);
  return data;
}

const fmtPct = (x) => (x * 100).toFixed(1) + "%";
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const form = document.getElementById("crop-form");
const result = document.getElementById("crop-result");

const NAMES = { N: "Nitrogen", P: "Phosphorus", K: "Potassium", temperature: "Temperature",
  humidity: "Humidity", ph: "Soil pH", rainfall: "Rainfall" };
let CONDITIONS = null;

function fmtVal(feature, v) {
  const unit = (CONDITIONS && CONDITIONS.units[feature]) || "";
  const step = feature === "N" || feature === "P" || feature === "K" || feature === "rainfall" ? 0 : 1;
  const num = step === 0 ? Math.round(v) : v.toFixed(1);
  return unit ? `${num} ${unit}` : `${num}`;
}

// Live value readouts for the sliders, and refine slider bounds from the data.
function initSliders() {
  form.querySelectorAll('input[type="range"]').forEach((input) => {
    const feature = input.name;
    const valEl = form.querySelector(`[data-val="${feature}"]`);
    const sync = () => { if (valEl) valEl.textContent = fmtVal(feature, parseFloat(input.value)); };
    input.addEventListener("input", sync);
    if (CONDITIONS && CONDITIONS.ranges[feature]) {
      const r = CONDITIONS.ranges[feature];
      const pad = Math.max((r.max - r.min) * 0.12, 1);
      input.min = Math.max(0, Math.floor(r.min - pad));
      input.max = Math.ceil(r.max + pad);
    }
    sync();
  });
}

fetch("/api/crop-conditions")
  .then((r) => r.json())
  .then((d) => { CONDITIONS = d && d.features && d.features.length ? d : null; initSliders(); initRequirements(); })
  .catch(() => initSliders());

// --- tabs: Recommend a crop  /  What a crop needs ---
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => {
      const on = t === tab;
      t.classList.toggle("is-active", on);
      t.setAttribute("aria-selected", on ? "true" : "false");
    });
    document.getElementById("tab-recommend").hidden = tab.dataset.tab !== "recommend";
    document.getElementById("tab-requirements").hidden = tab.dataset.tab !== "requirements";
  });
});

// --- crop -> requirements (the reverse of the recommender) ---
function fmtRange(feature, lo, hi) {
  const unit = (CONDITIONS && CONDITIONS.units[feature]) || "";
  const whole = feature === "N" || feature === "P" || feature === "K" || feature === "rainfall";
  const n = (x) => (whole ? Math.round(x) : x.toFixed(1));
  return `${n(lo)}–${n(hi)}${unit ? " " + unit : ""}`;
}

function initRequirements() {
  const sel = document.getElementById("req-crop");
  if (!sel || !CONDITIONS || !CONDITIONS.crops) return;
  const crops = Object.keys(CONDITIONS.crops).sort();
  sel.innerHTML = crops
    .map((c) => `<option value="${c}">${c.charAt(0).toUpperCase() + c.slice(1)}</option>`)
    .join("");
  sel.addEventListener("change", () => renderRequirements(sel.value));
  renderRequirements(sel.value);
}

function renderRequirements(crop) {
  const out = document.getElementById("req-output");
  const cond = CONDITIONS && CONDITIONS.crops[crop];
  if (!cond) { out.innerHTML = '<p class="result__empty">No data for this crop.</p>'; return; }

  let rows = "";
  CONDITIONS.features.forEach((f) => {
    const c = cond[f];
    const r = CONDITIONS.ranges[f];
    const sMin = Math.min(r.min, c.min), sMax = Math.max(r.max, c.max);
    const span = (sMax - sMin) || 1;
    const pct = (x) => ((x - sMin) / span) * 100;
    const bandL = pct(c.min), bandW = Math.max(1.5, pct(c.max) - pct(c.min));
    const meanL = Math.max(0, Math.min(100, pct(c.mean)));
    rows +=
      `<div class="cond-row"><span class="cond-label">${NAMES[f]}</span>` +
      `<span class="cond-track"><span class="cond-band" style="left:${bandL}%;width:${bandW}%"></span>` +
      `<span class="cond-mark mean" style="left:${meanL}%"></span></span>` +
      `<span class="cond-val">${fmtRange(f, c.min, c.max)}</span></div>`;
  });
  out.innerHTML =
    `<div class="conditions__title">What <strong style="text-transform:capitalize">${crop}</strong> usually needs</div>` +
    rows +
    `<div class="conditions__key"><span><i class="k-band"></i> typical range</span>` +
    `<span><i class="k-mark"></i> average</span></div>`;

  const ask = document.getElementById("req-ask");
  if (ask) ask.href = "/chat?ask=" + encodeURIComponent(`How do I grow ${crop}, and what soil and weather does it need?`);
}

// Re-play a CSS entrance animation on an element that is already in the DOM.
function replay(el, cls) {
  el.classList.remove(cls);
  void el.offsetWidth; // force reflow so the animation restarts
  el.classList.add(cls);
}

// Animate a percentage from 0 to the target value.
function countUp(el, target) {
  if (reduceMotion) { el.textContent = fmtPct(target); return; }
  const start = performance.now();
  const dur = 600;
  function step(now) {
    const t = Math.min(1, (now - start) / dur);
    const eased = 1 - Math.pow(1 - t, 3);
    el.textContent = fmtPct(target * eased);
    if (t < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const btn = form.querySelector("button");
  const body = {};
  new FormData(form).forEach((v, k) => (body[k] = parseFloat(v)));

  btn.disabled = true;
  btn.classList.add("is-loading");
  try {
    const pred = await postJSON("/api/predict", body);
    render(pred, body);
    saveHistoryEntry(body, pred);
  } catch (err) {
    showError(err.message);
  } finally {
    btn.disabled = false;
    btn.classList.remove("is-loading");
  }
});

const empty = document.getElementById("crop-empty");

function render(pred, inputs) {
  if (empty) empty.hidden = true;
  result.hidden = false;
  replay(result, "reveal");
  document.getElementById("crop-name").textContent = pred.crop;
  countUp(document.getElementById("crop-conf"), pred.confidence);
  const bar = document.getElementById("crop-bar");
  bar.style.width = "0";
  requestAnimationFrame(() => { bar.style.width = fmtPct(pred.confidence); });

  const chip = document.getElementById("crop-chip");
  chip.hidden = false;
  chip.textContent = pred.out_of_distribution ? "out of range" : "in range";
  chip.className = "chip " + (pred.out_of_distribution ? "chip--warn" : "chip--good");

  const top3 = document.getElementById("crop-top3");
  top3.innerHTML = "";
  pred.top3.forEach((s) => {
    const pct = fmtPct(s.probability);
    const li = document.createElement("li");
    li.innerHTML =
      `<span class="rank__crop">${s.crop}</span>` +
      `<span class="rank__track"><span class="rank__bar" style="width:${pct}"></span></span>` +
      `<span class="rank__pct">${pct}</span>`;
    top3.appendChild(li);
  });

  const ood = document.getElementById("crop-ood");
  if (pred.out_of_distribution) {
    ood.hidden = false;
    ood.innerHTML = `<strong>Out of range.</strong> Some inputs fall outside what the ` +
      `model was trained on, so trust this less:<br>` +
      pred.ood_warnings.map((w) => "• " + w).join("<br>");
  } else {
    ood.hidden = true;
  }

  renderConditions(pred.crop, inputs);
}

// Chart: for the recommended crop, show each feature's typical range and where
// the user's field sits relative to it (green inside the range, red outside).
function renderConditions(crop, inputs) {
  const box = document.getElementById("crop-conditions");
  const cond = CONDITIONS && CONDITIONS.crops[crop];
  if (!cond || !inputs) { box.hidden = true; return; }

  document.getElementById("cond-crop").textContent = crop;
  const rows = document.getElementById("cond-rows");
  rows.innerHTML = "";

  CONDITIONS.features.forEach((f) => {
    const c = cond[f];
    const v = inputs[f];
    const lo = Math.min(c.min, v), hi = Math.max(c.max, v);
    const span = (hi - lo) || 1;
    const pad = span * 0.12;
    const sMin = lo - pad, sMax = hi + pad;
    const pct = (x) => ((x - sMin) / (sMax - sMin)) * 100;
    const bandL = pct(c.min), bandW = Math.max(1.5, pct(c.max) - pct(c.min));
    const mark = Math.max(0, Math.min(100, pct(v)));
    const inBand = v >= c.min && v <= c.max;

    const row = document.createElement("div");
    row.className = "cond-row";
    row.innerHTML =
      `<span class="cond-label">${NAMES[f]}</span>` +
      `<span class="cond-track">` +
        `<span class="cond-band" style="left:${bandL}%;width:${bandW}%"></span>` +
        `<span class="cond-mark ${inBand ? "in" : "out"}" style="left:${mark}%"></span>` +
      `</span>` +
      `<span class="cond-val">${fmtVal(f, v)}</span>`;
    rows.appendChild(row);
  });
  box.hidden = false;
}

function showError(msg) {
  if (empty) empty.hidden = true;
  result.hidden = false;
  document.getElementById("crop-top3").innerHTML = "";
  document.getElementById("crop-conditions").hidden = true;
  const ood = document.getElementById("crop-ood");
  ood.hidden = false;
  ood.innerHTML = `<span class="error">${msg}</span>`;
}

// --- weather auto-fill ------------------------------------------------------
// Pull temperature, humidity and rainfall for a place (or the device location)
// and drop them into the matching sliders.
const weatherBtn = document.getElementById("weather-btn");
const weatherGeo = document.getElementById("weather-geo");
const weatherLoc = document.getElementById("weather-loc");
const weatherNote = document.getElementById("weather-note");

function setWeatherNote(msg, kind) {
  if (!weatherNote) return;
  weatherNote.hidden = !msg;
  weatherNote.textContent = msg || "";
  weatherNote.className = "weather-note" + (kind ? " is-" + kind : "");
}

// Move a slider to a value (clamped to its range) and update its readout.
function setSlider(feature, value) {
  if (value === null || value === undefined) return false;
  const input = form.querySelector(`input[name="${feature}"]`);
  if (!input) return false;
  const v = Math.max(parseFloat(input.min), Math.min(parseFloat(input.max), value));
  input.value = v;
  input.dispatchEvent(new Event("input"));
  return true;
}

async function applyWeather(query) {
  setWeatherNote("Fetching weather…", "busy");
  [weatherBtn, weatherGeo].forEach((b) => b && (b.disabled = true));
  try {
    const resp = await fetch("/api/weather?" + query);
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(data.detail || `Could not fetch weather (${resp.status})`);
    setSlider("temperature", data.temperature);
    setSlider("humidity", data.humidity);
    const gotRain = setSlider("rainfall", data.rainfall);
    const place = data.location || "your location";
    setWeatherNote(
      `Filled from ${place}: ${data.temperature}°C, ${data.humidity}% humidity` +
        (gotRain ? `, ${data.rainfall} mm rain/month (12-month average).` : ". Rainfall unavailable — set it by hand.") +
        " Soil values (N, P, K, pH) still need measuring.",
      "ok"
    );
  } catch (err) {
    setWeatherNote(err.message, "error");
  } finally {
    [weatherBtn, weatherGeo].forEach((b) => b && (b.disabled = false));
  }
}

if (weatherBtn) {
  weatherBtn.addEventListener("click", () => {
    const loc = (weatherLoc.value || "").trim();
    if (!loc) { setWeatherNote("Type a town or region first, or use 📍 for your location.", "error"); return; }
    applyWeather("location=" + encodeURIComponent(loc));
  });
}
if (weatherLoc) {
  weatherLoc.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); weatherBtn.click(); }
  });
}
if (weatherGeo) {
  weatherGeo.addEventListener("click", () => {
    if (!navigator.geolocation) { setWeatherNote("This browser cannot share a location.", "error"); return; }
    setWeatherNote("Getting your location…", "busy");
    navigator.geolocation.getCurrentPosition(
      (pos) => applyWeather(`latitude=${pos.coords.latitude}&longitude=${pos.coords.longitude}`),
      () => setWeatherNote("Could not get your location. Type a town instead.", "error"),
      { timeout: 10000 }
    );
  });
}

// --- recommendation history (saved in the browser, exportable as CSV) -------
const HISTORY_KEY = "farmwise_crop_history";
const HISTORY_MAX = 50;
const histSection = document.getElementById("history-section");
const histList = document.getElementById("history-list");

function loadHistory() {
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY)) || []; }
  catch (_) { return []; }
}

function saveHistoryEntry(inputs, pred) {
  const entry = {
    at: new Date().toISOString(),
    crop: pred.crop,
    confidence: pred.confidence,
    out_of_distribution: !!pred.out_of_distribution,
    inputs: inputs,
  };
  const items = loadHistory();
  items.unshift(entry);
  if (items.length > HISTORY_MAX) items.length = HISTORY_MAX;
  try { localStorage.setItem(HISTORY_KEY, JSON.stringify(items)); } catch (_) {}
  renderHistory();
}

function renderHistory() {
  if (!histSection || !histList) return;
  const items = loadHistory();
  histSection.hidden = items.length === 0;
  histList.innerHTML = items
    .map((e) => {
      const when = new Date(e.at).toLocaleString();
      const pct = (e.confidence * 100).toFixed(0) + "%";
      const flag = e.out_of_distribution ? '<span class="chip chip--warn">out of range</span>' : "";
      const fields = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        .map((f) => `${f} ${e.inputs[f]}`).join(" · ");
      return (
        `<div class="hist-row"><div class="hist-row__main">` +
        `<span class="hist-row__crop">${e.crop}</span> ` +
        `<span class="hist-row__conf">${pct}</span> ${flag}` +
        `<div class="hist-row__fields">${fields}</div></div>` +
        `<time class="hist-row__time">${when}</time></div>`
      );
    })
    .join("");
}

function historyToCSV(items) {
  const cols = ["at", "crop", "confidence", "out_of_distribution",
    "N", "P", "K", "temperature", "humidity", "ph", "rainfall"];
  const esc = (v) => {
    const s = String(v);
    return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
  };
  const rows = items.map((e) =>
    cols.map((c) => esc(c in e.inputs ? e.inputs[c] : e[c])).join(",")
  );
  return [cols.join(","), ...rows].join("\n");
}

const histCsvBtn = document.getElementById("history-csv");
const histClearBtn = document.getElementById("history-clear");
if (histCsvBtn) {
  histCsvBtn.addEventListener("click", () => {
    const items = loadHistory();
    if (!items.length) return;
    const blob = new Blob([historyToCSV(items)], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "farmwise-recommendations.csv";
    a.click();
    URL.revokeObjectURL(url);
  });
}
if (histClearBtn) {
  histClearBtn.addEventListener("click", () => {
    localStorage.removeItem(HISTORY_KEY);
    renderHistory();
  });
}
renderHistory();
