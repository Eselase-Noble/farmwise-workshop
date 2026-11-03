// Chat page. Keeps the conversation client-side and posts it to /api/chat.
"use strict";

const log = document.getElementById("chat-log");
const form = document.getElementById("chat-form");
const input = document.getElementById("chat-input");
const sendBtn = document.getElementById("chat-send");

const history = []; // [{role, content}]
const MAX_HISTORY = 20; // the assistant only needs the recent turns

function currentMode() {
  const el = document.querySelector('input[name="mode"]:checked');
  return el && el.value === "bare" ? "bare" : "rag";
}

// Update the hint line when the mode changes.
document.querySelectorAll('input[name="mode"]').forEach((r) =>
  r.addEventListener("change", () => {
    document.getElementById("mode-hint").textContent =
      currentMode() === "rag"
        ? "Grounded in the farming guides, and it cites its sources."
        : "Bare chatbot: the model answers on its own, with no documents and no sources. It may guess.";
  })
);

// Grow the textarea with its content.
input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 140) + "px";
});

// Suggested questions fill the box and send.
document.getElementById("suggest").addEventListener("click", (e) => {
  const chip = e.target.closest(".suggest__chip");
  if (!chip) return;
  input.value = chip.textContent;
  form.requestSubmit();
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;

  const useRag = currentMode() === "rag";
  addMessage("user", text);
  history.push({ role: "user", content: text });
  input.value = "";
  input.style.height = "auto";

  const thinking = addThinking();
  sendBtn.disabled = true;
  try {
    const data = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: history.slice(-MAX_HISTORY), use_rag: useRag }),
    }).then(async (r) => {
      const d = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(d.detail || `Request failed (${r.status})`);
      return d;
    });
    thinking.remove();
    const answer = cleanMarkdown(data.answer);
    addMessage("bot", answer, data.sources, data.used_context, false, useRag ? "rag" : "bare");
    history.push({ role: "assistant", content: answer });
  } catch (err) {
    thinking.remove();
    addMessage("bot", err.message, [], false, true);
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
});

const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// The model is told to avoid Markdown, but strip any stray markers just in case
// so the plain-text bubble never shows raw ** or # characters.
function cleanMarkdown(t) {
  return t
    .replace(/\*\*(.*?)\*\*/g, "$1")
    .replace(/__(.*?)__/g, "$1")
    .replace(/`([^`]*)`/g, "$1")
    .replace(/(^|\n)\s{0,3}#{1,6}\s*/g, "$1")
    .replace(/(^|\n)\s*[*-]\s+/g, "$1• ")
    .trim();
}

const SRC_ICONS = {
  web: '<svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="2"/><path d="M3 12h18M12 3c2.5 2.5 2.5 15.5 0 18M12 3c-2.5 2.5-2.5 15.5 0 18" stroke="currentColor" stroke-width="2"/></svg>',
  pdf: '<svg viewBox="0 0 24 24" fill="none"><path d="M14 3H7a2 2 0 00-2 2v14a2 2 0 002 2h10a2 2 0 002-2V8l-5-5Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/></svg>',
  file: '<svg viewBox="0 0 24 24" fill="none"><path d="M14 3H7a2 2 0 00-2 2v14a2 2 0 002 2h10a2 2 0 002-2V8l-5-5Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/></svg>',
};
function sourceType(src) {
  const s = src.toLowerCase();
  if (/^https?:\/\//.test(src)) return s.split("?")[0].endsWith(".pdf") ? "pdf" : "web";
  return s.endsWith(".pdf") ? "pdf" : "file";
}
function prettySource(src) {
  if (/^https?:\/\//i.test(src)) {
    try { return new URL(src).hostname.replace(/^www\./, ""); } catch (_) { return src; }
  }
  return src;
}

// Reveal bot text character by character (skipped for reduced motion / long text).
function typeOut(el, text, done) {
  if (reduceMotion || text.length > 500) { el.textContent = text; done && done(); return; }
  let i = 0;
  (function tick() {
    el.textContent = text.slice(0, i++);
    log.scrollTop = log.scrollHeight;
    if (i <= text.length) setTimeout(tick, 11);
    else done && done();
  })();
}

function addMessage(role, text, sources, usedContext, isError, mode) {
  const wrap = document.createElement("div");
  wrap.className = "msg " + (role === "user" ? "msg--user" : "msg--bot");

  if (role === "bot" && mode) {
    const badge = document.createElement("span");
    badge.className = "msg__mode " + (mode === "rag" ? "is-rag" : "is-bare");
    badge.textContent = mode === "rag" ? "RAG · grounded" : "Bare chatbot";
    wrap.appendChild(badge);
  }

  const bubble = document.createElement("div");
  bubble.className = "msg__bubble";
  if (isError || (role === "bot" && /don't know based on the farming guides/i.test(text))) {
    bubble.classList.add("is-idk");
  }
  wrap.appendChild(bubble);
  log.appendChild(wrap);
  log.scrollTop = log.scrollHeight;

  if (role !== "bot") { bubble.textContent = text; return wrap; }

  // Bot: type the answer, then add the copy button and source chips.
  typeOut(bubble, text, () => {
    if (isError) return;
    const foot = document.createElement("div");
    foot.className = "msg__foot";

    const copy = document.createElement("button");
    copy.className = "msg__copy";
    copy.textContent = "Copy";
    copy.addEventListener("click", async () => {
      try { await navigator.clipboard.writeText(text); copy.textContent = "Copied"; }
      catch (_) { copy.textContent = "Copy failed"; }
      setTimeout(() => (copy.textContent = "Copy"), 1500);
    });
    foot.appendChild(copy);
    wrap.appendChild(foot);

    if (usedContext && sources && sources.length) {
      const box = document.createElement("div");
      box.className = "msg__sources";
      const seen = new Set();
      sources.forEach((s) => {
        if (seen.has(s.source)) return;
        seen.add(s.source);
        const chip = document.createElement("span");
        chip.className = "src-chip";
        chip.title = s.source;
        chip.innerHTML = SRC_ICONS[sourceType(s.source)];
        chip.appendChild(document.createTextNode(prettySource(s.source)));
        box.appendChild(chip);
      });
      wrap.appendChild(box);
    }
    log.scrollTop = log.scrollHeight;
  });
  return wrap;
}

function addThinking() {
  const wrap = document.createElement("div");
  wrap.className = "msg msg--bot";
  wrap.innerHTML = '<div class="msg__bubble"><span class="typing"><span></span><span></span><span></span></span></div>';
  log.appendChild(wrap);
  log.scrollTop = log.scrollHeight;
  return wrap;
}

// A question handed over from another page (e.g. the crop page) auto-asks it.
const askParam = new URLSearchParams(location.search).get("ask");
if (askParam) {
  input.value = askParam;
  input.dispatchEvent(new Event("input"));
  form.requestSubmit();
}
