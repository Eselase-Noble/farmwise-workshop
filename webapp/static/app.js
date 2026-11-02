// FarmWise web app — talks to the FastAPI backend.
"use strict";

async function postJSON(url, body) {
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    throw new Error(data.detail || `Request failed (${resp.status})`);
  }
  return data;
}

function fmtPct(x) {
  return (x * 100).toFixed(1) + "%";
}

// ---------- Crop recommendation ----------
const cropForm = document.getElementById("crop-form");
const cropResult = document.getElementById("crop-result");

cropForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const btn = cropForm.querySelector("button");
  const fd = new FormData(cropForm);
  const body = {};
  for (const [k, v] of fd.entries()) body[k] = parseFloat(v);

  btn.disabled = true;
  const label = btn.textContent;
  btn.textContent = "Working…";
  try {
    const pred = await postJSON("/api/predict", body);
    renderCrop(pred);
  } catch (err) {
    showCropError(err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = label;
  }
});

function renderCrop(pred) {
  cropResult.hidden = false;
  document.getElementById("crop-name").textContent = pred.crop;
  document.getElementById("crop-conf").textContent = fmtPct(pred.confidence);
  document.getElementById("crop-bar").style.width = fmtPct(pred.confidence);

  const top3 = document.getElementById("crop-top3");
  top3.innerHTML = "";
  pred.top3.forEach((s) => {
    const li = document.createElement("li");
    li.innerHTML = `<span class="rank__crop">${s.crop}</span>` +
                   `<span class="rank__pct">${fmtPct(s.probability)}</span>`;
    top3.appendChild(li);
  });

  const ood = document.getElementById("crop-ood");
  if (pred.out_of_distribution) {
    ood.hidden = false;
    ood.innerHTML = `<strong>Out of range.</strong> ` +
      `Some inputs fall outside what the model was trained on, so trust this less:` +
      `<br>` + pred.ood_warnings.map((w) => "• " + w).join("<br>");
  } else {
    ood.hidden = true;
    ood.innerHTML = "";
  }
}

function showCropError(msg) {
  cropResult.hidden = false;
  document.getElementById("crop-name").textContent = "—";
  document.getElementById("crop-top3").innerHTML = "";
  document.getElementById("crop-bar").style.width = "0";
  const ood = document.getElementById("crop-ood");
  ood.hidden = false;
  ood.innerHTML = `<span class="error">${msg}</span>`;
}

// ---------- Farming Q&A ----------
const qaForm = document.getElementById("qa-form");
const qaResult = document.getElementById("qa-result");

qaForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const btn = qaForm.querySelector("button");
  const question = qaForm.question.value.trim();
  const useRag = qaForm.querySelector('input[name="mode"]:checked').value === "rag";
  if (!question) return;

  btn.disabled = true;
  const label = btn.textContent;
  btn.textContent = "Thinking…";
  qaResult.hidden = false;
  document.getElementById("qa-answer").textContent = "";
  document.getElementById("qa-answer").className = "answer loading";
  document.getElementById("qa-answer").textContent = "Generating answer…";
  document.getElementById("qa-sources").hidden = true;
  document.getElementById("qa-mode").textContent = useRag ? "Grounded · RAG" : "Plain model";

  try {
    const ans = await postJSON("/api/ask", { question, use_rag: useRag });
    renderAnswer(ans);
  } catch (err) {
    const a = document.getElementById("qa-answer");
    a.className = "answer error";
    a.textContent = err.message;
  } finally {
    btn.disabled = false;
    btn.textContent = label;
  }
});

function renderAnswer(ans) {
  const a = document.getElementById("qa-answer");
  const idk = /don't know based on the farming guides/i.test(ans.answer);
  a.className = "answer" + (idk ? " answer--idk" : "");
  a.textContent = ans.answer;

  const box = document.getElementById("qa-sources");
  const list = document.getElementById("qa-source-list");
  list.innerHTML = "";
  if (ans.used_context && ans.sources && ans.sources.length) {
    const seen = new Set();
    ans.sources.forEach((s) => {
      const key = s.source;
      if (seen.has(key)) return;
      seen.add(key);
      const li = document.createElement("li");
      li.innerHTML = `${s.source} <span class="src__score">${(s.score).toFixed(2)}</span>`;
      list.appendChild(li);
    });
    box.hidden = false;
  } else {
    box.hidden = true;
  }
}
