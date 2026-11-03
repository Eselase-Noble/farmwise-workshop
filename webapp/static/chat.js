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
    addMessage("bot", data.answer, data.sources, data.used_context, false, useRag ? "rag" : "bare");
    history.push({ role: "assistant", content: data.answer });
  } catch (err) {
    thinking.remove();
    addMessage("bot", err.message, [], false, true);
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
});

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
  if (isError) bubble.classList.add("is-idk");
  if (role === "bot" && /don't know based on the farming guides/i.test(text)) {
    bubble.classList.add("is-idk");
  }
  bubble.textContent = text;
  wrap.appendChild(bubble);

  if (role === "bot" && usedContext && sources && sources.length) {
    const box = document.createElement("div");
    box.className = "msg__sources";
    const seen = new Set();
    sources.forEach((s) => {
      if (seen.has(s.source)) return;
      seen.add(s.source);
      const chip = document.createElement("span");
      chip.className = "src-chip";
      chip.title = s.source;
      chip.textContent = prettySource(s.source);
      box.appendChild(chip);
    });
    wrap.appendChild(box);
  }

  log.appendChild(wrap);
  log.scrollTop = log.scrollHeight;
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

function prettySource(src) {
  if (/^https?:\/\//i.test(src)) {
    try { return new URL(src).hostname.replace(/^www\./, ""); } catch (_) { return src; }
  }
  return src;
}
