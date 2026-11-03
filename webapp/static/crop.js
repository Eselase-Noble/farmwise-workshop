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
    render(pred);
  } catch (err) {
    showError(err.message);
  } finally {
    btn.disabled = false;
    btn.classList.remove("is-loading");
  }
});

const empty = document.getElementById("crop-empty");

function render(pred) {
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
}

function showError(msg) {
  if (empty) empty.hidden = true;
  result.hidden = false;
  document.getElementById("crop-top3").innerHTML = "";
  const ood = document.getElementById("crop-ood");
  ood.hidden = false;
  ood.innerHTML = `<span class="error">${msg}</span>`;
}
