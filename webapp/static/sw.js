// FarmWise service worker: makes the app installable and the pages usable
// offline. Live AI calls still need a connection; the UI surfaces that.
"use strict";

const CACHE = "farmwise-v4";

// The shell: pages and static assets that make the app load without a network.
const PRECACHE = [
  "/",
  "/crop",
  "/chat",
  "/knowledge",
  "/static/styles.css",
  "/static/nav.js",
  "/static/app.js",
  "/static/crop.js",
  "/static/chat.js",
  "/static/kb.js",
  "/static/logo.svg",
  "/static/favicon.svg",
  "/static/manifest.webmanifest",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE).then((cache) => cache.addAll(PRECACHE)).then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return;

  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;

  // Never cache dynamic API responses (predictions, chat, weather, health).
  if (url.pathname.startsWith("/api/")) return;

  // Network-first for everything else (pages and static assets): serve fresh
  // content when online — so code and style updates land on the next load —
  // and fall back to the cache only when the network is unavailable. This is
  // what keeps the app usable offline without ever pinning stale JS/CSS.
  event.respondWith(
    fetch(req)
      .then((resp) => {
        if (resp && resp.ok) {
          const copy = resp.clone();
          caches.open(CACHE).then((c) => c.put(req, copy));
        }
        return resp;
      })
      .catch(() =>
        caches.match(req).then((hit) => hit || (req.mode === "navigate" ? caches.match("/") : hit))
      )
  );
});
