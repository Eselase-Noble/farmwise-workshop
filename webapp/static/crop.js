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

const form = document.getElementById("crop-form");
const result = document.getElementById("crop-result");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const btn = form.querySelector("button");
  const body = {};
  new FormData(form).forEach((v, k) => (body[k] = parseFloat(v)));

  btn.disabled = true;
  const label = btn.textContent;
  btn.textContent = "Working…";
  try {
    const pred = await postJSON("/api/predict", body);
    render(pred);
  } catch (err) {
    showError(err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = label;
  }
});

function render(pred) {
  result.hidden = false;
  document.getElementById("crop-name").textContent = pred.crop;
  document.getElementById("crop-conf").textContent = fmtPct(pred.confidence);
  document.getElementById("crop-bar").style.width = fmtPct(pred.confidence);

  const chip = document.getElementById("crop-chip");
  chip.hidden = false;
  chip.textContent = pred.out_of_distribution ? "out of range" : "in range";
  chip.style.background = pred.out_of_distribution ? "#fff5f5" : "#eefaf0";
  chip.style.color = pred.out_of_distribution ? "#8a1220" : "#10692c";

  const top3 = document.getElementById("crop-top3");
  top3.innerHTML = "";
  pred.top3.forEach((s) => {
    const li = document.createElement("li");
    li.innerHTML = `<span class="rank__crop">${s.crop}</span><span class="rank__pct">${fmtPct(s.probability)}</span>`;
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
  result.hidden = false;
  const ood = document.getElementById("crop-ood");
  ood.hidden = false;
  ood.innerHTML = `<span class="error">${msg}</span>`;
}
