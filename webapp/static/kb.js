// Knowledge base page.
"use strict";

const kbStatus = document.getElementById("kb-status");
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
    // The field stays hidden; a failed change will explain itself.
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

function renderSources(data) {
  const list = document.getElementById("kb-list");
  document.getElementById("kb-total").textContent = `${(data && data.total_chunks) || 0} chunks`;
  list.innerHTML = "";
  const items = (data && data.sources) || [];
  if (!items.length) {
    list.innerHTML = '<li class="kb-empty">Nothing indexed yet.</li>';
    return;
  }
  items.forEach((s) => {
    const li = document.createElement("li");
    const meta = document.createElement("span");
    meta.className = "src__meta";
    meta.innerHTML = `<span class="src__chunks">${s.chunks} chunk${s.chunks === 1 ? "" : "s"}</span>`;
    const del = document.createElement("button");
    del.className = "src__remove";
    del.title = "Remove this source";
    del.textContent = "Remove";
    del.addEventListener("click", () => removeSource(s.source));
    meta.appendChild(del);

    const name = document.createElement("span");
    name.className = "src__name";
    name.textContent = s.source;
    li.appendChild(name);
    li.appendChild(meta);
    list.appendChild(li);
  });
}

async function removeSource(source) {
  if (!confirm(`Remove this source from the knowledge base?\n\n${source}`)) return;
  setStatus(`Removing ${source} …`, "busy");
  try {
    const data = await postJSON("/api/sources/remove", { source });
    setStatus(`Removed ${data.removed} chunk(s) from ${source}.`, "ok");
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

document.getElementById("url-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const form = e.target;
  const url = form.url.value.trim();
  if (!url) return;
  const crawl = form.crawl.checked;
  const btn = form.querySelector("button");
  btn.disabled = true;
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
    btn.disabled = false;
  }
});

document.getElementById("file-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const form = e.target;
  const file = form.file.files[0];
  if (!file) return;
  const btn = form.querySelector("button");
  btn.disabled = true;
  setStatus(`Uploading ${file.name} …`, "busy");
  try {
    const fd = new FormData();
    fd.append("file", file);
    const resp = await fetch("/api/ingest/file", { method: "POST", headers: adminHeaders(), body: fd });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(data.detail || `Upload failed (${resp.status})`);
    setStatus(`Added ${data.chunks_added} chunk(s) from ${data.source}.`, "ok");
    renderSources(data);
    form.reset();
  } catch (err) {
    setStatus(err.message, "error");
  } finally {
    btn.disabled = false;
  }
});

showTokenFieldIfNeeded();
loadSources();
