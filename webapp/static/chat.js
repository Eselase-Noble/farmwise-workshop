// Chat page. Keeps the conversation client-side and posts it to /api/chat.
"use strict";

const log = document.getElementById("chat-log");
const form = document.getElementById("chat-form");
const input = document.getElementById("chat-input");
const sendBtn = document.getElementById("chat-send");

const history = []; // [{role, content}]
const MAX_HISTORY = 20; // the assistant only needs the recent turns

const langSel = document.getElementById("chat-lang");

function currentMode() {
  const el = document.querySelector('input[name="mode"]:checked');
  return el && el.value === "bare" ? "bare" : "rag";
}

function currentLanguage() {
  return langSel ? langSel.value : "en";
}

// The languages are listed in the HTML, so the selector always renders even
// offline. Here we just restore the last choice and remember new ones.
(function initLanguages() {
  if (!langSel) return;
  const saved = localStorage.getItem("farmwise_lang");
  if (saved && [...langSel.options].some((o) => o.value === saved)) langSel.value = saved;
  langSel.addEventListener("change", () =>
    localStorage.setItem("farmwise_lang", langSel.value)
  );
})();

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
  const lang = currentLanguage();
  addMessage("user", text);
  history.push({ role: "user", content: text });
  input.value = "";
  input.style.height = "auto";

  sendBtn.disabled = true;
  try {
    await streamAnswer(useRag, lang);
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
});

// Stream the answer token by token from /api/chat/stream (server-sent events),
// showing it as it arrives instead of waiting for the whole reply.
async function streamAnswer(useRag, lang) {
  const wrap = document.createElement("div");
  wrap.className = "msg msg--bot";
  const badge = document.createElement("span");
  badge.className = "msg__mode " + (useRag ? "is-rag" : "is-bare");
  badge.textContent = useRag ? "RAG · grounded" : "Bare chatbot";
  wrap.appendChild(badge);
  const bubble = document.createElement("div");
  bubble.className = "msg__bubble";
  bubble.innerHTML = '<span class="typing"><span></span><span></span><span></span></span>';
  wrap.appendChild(bubble);
  log.appendChild(wrap);
  log.scrollTop = log.scrollHeight;

  let raw = "";
  let started = false;
  let final = null;

  try {
    const resp = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: history.slice(-MAX_HISTORY), use_rag: useRag, language: lang }),
    });
    if (!resp.ok || !resp.body) {
      const d = await resp.json().catch(() => ({}));
      throw new Error(d.detail || `Request failed (${resp.status})`);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const chunks = buffer.split("\n\n");
      buffer = chunks.pop();
      for (const chunk of chunks) {
        const line = chunk.split("\n").find((l) => l.startsWith("data:"));
        if (!line) continue;
        let evt;
        try { evt = JSON.parse(line.slice(5).trim()); } catch (_) { continue; }
        if (evt.type === "token") {
          if (!started) { started = true; bubble.textContent = ""; }
          raw += evt.text;
          bubble.textContent = raw;
          log.scrollTop = log.scrollHeight;
        } else if (evt.type === "status") {
          // Others are being answered first; say so instead of showing dots.
          if (!started) bubble.textContent = evt.message;
        } else if (evt.type === "done") {
          final = evt;
        } else if (evt.type === "error") {
          throw new Error(evt.detail || "The assistant failed.");
        }
      }
    }
  } catch (err) {
    bubble.textContent = err.message;
    bubble.classList.add("is-idk");
    return;
  }

  const answer = cleanMarkdown(final ? final.answer : raw);
  bubble.textContent = answer;
  if (/don't know based on the farming guides/i.test(answer)) bubble.classList.add("is-idk");
  history.push({ role: "assistant", content: answer });
  finishBotMessage(wrap, answer, final && final.sources, final && final.used_context);
  log.scrollTop = log.scrollHeight;
}

// Add the Copy button and source chips under a finished bot message.
function finishBotMessage(wrap, text, sources, usedContext) {
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
}

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

// Append a simple message bubble. Bot answers are rendered by streamAnswer();
// this handles the user's own messages (and any plain bot notice).
function addMessage(role, text) {
  const wrap = document.createElement("div");
  wrap.className = "msg " + (role === "user" ? "msg--user" : "msg--bot");
  const bubble = document.createElement("div");
  bubble.className = "msg__bubble";
  bubble.textContent = text;
  wrap.appendChild(bubble);
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
