// Cross-cutting behaviour loaded on every page:
//  1. a live service-status banner driven by /api/health
//  2. service-worker registration for offline/installable use
"use strict";

(function () {
  // --- 1. service status banner -------------------------------------------
  // Shown only when something is wrong (Ollama down, model not trained, empty
  // index, or the server unreachable), so a healthy app stays uncluttered.
  const banner = document.createElement("div");
  banner.className = "status-banner";
  banner.setAttribute("role", "status");
  banner.hidden = true;
  const header = document.querySelector(".nav");
  if (header && header.parentNode) {
    header.parentNode.insertBefore(banner, header.nextSibling);
  } else {
    document.body.insertBefore(banner, document.body.firstChild);
  }

  function show(level, text) {
    banner.className = "status-banner is-" + level;
    banner.textContent = text;
    banner.hidden = false;
  }
  function hide() {
    banner.hidden = true;
  }

  async function checkHealth() {
    let h;
    try {
      const resp = await fetch("/api/health", { cache: "no-store" });
      h = await resp.json();
    } catch (_) {
      show("error", "Cannot reach FarmWise. You may be offline — showing a cached page.");
      return;
    }
    if (!h.ollama_running) {
      show("error", "The local AI service (Ollama) is not running, so chat and Q&A are unavailable. Start it with: ollama serve");
    } else if (!h.model_trained) {
      show("warn", "The crop model has not been trained yet. Run: python scripts/train_model.py");
    } else if (!h.index_chunks) {
      show("warn", "The knowledge base is empty, so the assistant has no guides to cite yet. Add sources on the Knowledge page.");
    } else {
      hide();
    }
  }

  checkHealth();
  setInterval(checkHealth, 30000);
  window.addEventListener("online", checkHealth);
  window.addEventListener("offline", () =>
    show("error", "You are offline. Cached pages still work; live answers need a connection.")
  );

  // --- 2. service worker ---------------------------------------------------
  if ("serviceWorker" in navigator) {
    // When an updated worker replaces an existing one, reload once so the page
    // picks up fresh assets instead of whatever the old worker had cached.
    const hadController = !!navigator.serviceWorker.controller;
    let reloaded = false;
    navigator.serviceWorker.addEventListener("controllerchange", () => {
      if (!hadController || reloaded) return;
      reloaded = true;
      window.location.reload();
    });
    window.addEventListener("load", () => {
      navigator.serviceWorker.register("/sw.js", { scope: "/" }).catch(() => {
        /* offline support is a progressive enhancement; ignore failures */
      });
    });
  }
})();
