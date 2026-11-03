// Knowledge base page.
"use strict";

const kbStatus = document.getElementById("kb-status");
const kbProgress = document.getElementById("kb-progress");
const kbToken = document.getElementById("kb-token");

// Kept for this browser tab only, and sent only on requests that change things.
kbToken.value = sessionStorage.getItem("farmwise-admin-token") || "";
kbToken.addEventListener("input", () =>
  sessionStorage.setItem("farmwise-admin-token", kbToken.value.trim())
);

function adminHeaders() {
  const token = kbToken.value.trim();
  return token ? { "X-Admin-Token": token } : {};
}

async function showTokenFieldIfNeeded() {
  try {
    const status = await (await fetch("/api/admin/status")).json();
    document.getElementById("kb-token-wrap").hidden = !status.token_required;
  } catch (_) {
    /* field stays hidden; a failed change will explain itself */
  }
}

async function postJSON(url, body) {
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...adminHeaders() },
    body: JSON.stringify(body),
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(data.detail || `Request failed (${resp.status})`);
  return data;
}

function setStatus(msg, kind) {
  kbStatus.hidden = false;
  kbStatus.className = "kb-status" + (kind ? " is-" + kind : "");
  kbStatus.textContent = msg;
}
function setBusy(on) { kbProgress.hidden = !on; }

// --- source rendering ---
const ICONS = {
  web: '<svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="1.8"/><path d="M3 12h18M12 3c3 3 3 15 0 18M12 3c-3 3-3 15 0 18" stroke="currentColor" stroke-width="1.8"/></svg>',
  pdf: '<svg viewBox="0 0 24 24" fill="none"><path d="M14 3H7a2 2 0 00-2 2v14a2 2 0 002 2h10a2 2 0 002-2V8l-5-5Z" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/><path d="M14 3v5h5" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/></svg>',
  file: '<svg viewBox="0 0 24 24" fill="none"><path d="M14 3H7a2 2 0 00-2 2v14a2 2 0 002 2h10a2 2 0 002-2V8l-5-5Z" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/><path d="M9 13h6M9 16h4" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>',
};

function sourceType(src) {
  const s = src.toLowerCase();
  if (/^https?:\/\//.test(src)) return s.split("?")[0].endsWith(".pdf") ? "pdf" : "web";
  return s.endsWith(".pdf") ? "pdf" : "file";
}
function prettyName(src) {
  if (/^https?:\/\//.test(src)) {
    try { return new URL(src).hostname.replace(/^www\./, ""); } catch (_) { return src; }
  }
  return src;
}

function renderSources(data) {
  const list = document.getElementById("kb-list");
  document.getElementById("kb-total").textContent = `${(data && data.total_chunks) || 0} chunks`;
  list.innerHTML = "";
  const items = (data && data.sources) || [];
  if (!items.length) {
    list.innerHTML = '<li class="kb-empty">Nothing indexed yet. Add a link or upload a document above.</li>';
    return;
  }
  const max = Math.max(1, ...items.map((s) => s.chunks));
  items.forEach((s) => {
    const li = document.createElement("li");
    li.className = "kb-card";

    const icon = document.createElement("span");
    icon.className = "kb-card__icon";
    icon.innerHTML = ICONS[sourceType(s.source)];

    const body = document.createElement("div");
    body.className = "kb-card__body";
    const name = document.createElement("div");
    name.className = "kb-card__name";
    name.textContent = prettyName(s.source);
    name.title = s.source;
    const track = document.createElement("div");
    track.className = "kb-card__track";
    track.innerHTML = `<span class="kb-card__bar" style="width:${Math.round((s.chunks / max) * 100)}%"></span>`;
    const meta = document.createElement("div");
    meta.className = "kb-card__meta";
    meta.innerHTML = `<span>${s.chunks} chunk${s.chunks === 1 ? "" : "s"}</span>`;
    body.append(name, track, meta);

    const del = document.createElement("button");
    del.className = "src__remove";
    del.textContent = "Remove";
    del.title = "Remove this source";
    del.addEventListener("click", () => removeSource(s.source));

    li.append(icon, body, del);
    list.appendChild(li);
  });
}

async function removeSource(source) {
  if (!confirm(`Remove this source from the knowledge base?\n\n${source}`)) return;
  setStatus(`Removing ${prettyName(source)} …`, "busy");
  try {
    const data = await postJSON("/api/sources/remove", { source });
    setStatus(`Removed ${data.removed} chunk(s) from ${prettyName(source)}.`, "ok");
    renderSources(data);
  } catch (err) {
    setStatus(err.message, "error");
  }
}

async function loadSources() {
  try {
    renderSources(await (await fetch("/api/sources")).json());
  } catch (_) {
    document.getElementById("kb-list").innerHTML = '<li class="kb-empty">Could not load sources.</li>';
  }
}

// --- add from URL ---
document.getElementById("url-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const form = e.target;
  const url = form.url.value.trim();
  if (!url) return;
  const crawl = form.crawl.checked;
  const btn = form.querySelector("button");
  btn.classList.add("is-loading"); btn.disabled = true; setBusy(true);
  setStatus(crawl ? `Searching ${url} and its linked pages …` : `Fetching ${url} …`, "busy");
  try {
    const data = await postJSON("/api/ingest/url", { url, crawl });
    const pages = data.pages_added || 1;
    setStatus(`Added ${data.chunks_added} chunk(s) from ${pages} page(s).`, "ok");
    renderSources(data);
    form.reset();
  } catch (err) {
    setStatus(err.message, "error");
  } finally {
    btn.classList.remove("is-loading"); btn.disabled = false; setBusy(false);
  }
});

// --- upload a file (button or drag-and-drop) ---
const fileForm = document.getElementById("file-form");
const fileInput = fileForm.file;

async function uploadFile(file) {
  if (!file) return;
  const btn = fileForm.querySelector("button");
  btn.classList.add("is-loading"); btn.disabled = true; setBusy(true);
  setStatus(`Uploading ${file.name} …`, "busy");
  try {
    const fd = new FormData();
    fd.append("file", file);
    const resp = await fetch("/api/ingest/file", { method: "POST", headers: adminHeaders(), body: fd });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(data.detail || `Upload failed (${resp.status})`);
    setStatus(`Added ${data.chunks_added} chunk(s) from ${data.source}.`, "ok");
    renderSources(data);
    fileForm.reset();
  } catch (err) {
    setStatus(err.message, "error");
  } finally {
    btn.classList.remove("is-loading"); btn.disabled = false; setBusy(false);
  }
}

fileForm.addEventListener("submit", (e) => { e.preventDefault(); uploadFile(fileInput.files[0]); });

["dragenter", "dragover"].forEach((ev) =>
  fileForm.addEventListener(ev, (e) => { e.preventDefault(); fileForm.classList.add("is-drop"); })
);
["dragleave", "dragend", "drop"].forEach((ev) =>
  fileForm.addEventListener(ev, (e) => { e.preventDefault(); if (ev !== "dragleave" || e.target === fileForm) fileForm.classList.remove("is-drop"); })
);
fileForm.addEventListener("drop", (e) => {
  const file = e.dataTransfer && e.dataTransfer.files[0];
  if (file) { fileInput.files = e.dataTransfer.files; uploadFile(file); }
});

showTokenFieldIfNeeded();
loadSources();
