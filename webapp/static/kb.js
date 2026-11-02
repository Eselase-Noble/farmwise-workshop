// Knowledge base page.
"use strict";

const kbStatus = document.getElementById("kb-status");

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
    li.innerHTML = `<span class="src__name">${s.source}</span>` +
                   `<span class="src__chunks">${s.chunks} chunk${s.chunks === 1 ? "" : "s"}</span>`;
    list.appendChild(li);
  });
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
  const btn = form.querySelector("button");
  btn.disabled = true;
  setStatus(`Fetching ${url} …`, "busy");
  try {
    const data = await postJSON("/api/ingest/url", { url });
    setStatus(`Added ${data.chunks_added} chunk(s) from ${data.source}.`, "ok");
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
    const resp = await fetch("/api/ingest/file", { method: "POST", body: fd });
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

loadSources();
