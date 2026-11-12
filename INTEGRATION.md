# FarmWise — API integration guide

How to call the FarmWise backend from another application. Covers the two parts
you asked for: **crop recommendation** and **chat**. Everything here is served by
the FastAPI app in `webapp/main.py`.

## Base URL and setup

```
http://localhost:8000      # local
https://<your-tunnel>      # e.g. ngrok, for remote callers
```

The backend needs the Session 1 setup running: Ollama up (`ollama serve`) with
`llama3.2:3b` and `nomic-embed-text` pulled, a trained crop model, and (for RAG)
a built index. Start it with:

```bash
uvicorn webapp.main:app --reload --host 0.0.0.0 --port 8000
```

### Calling from a browser app on a different origin (CORS)

By default the API answers **same-origin only**. To let a separate front-end
(e.g. a React app on `http://localhost:3000`) call it, set the allowed origins:

```bash
export FARMWISE_CORS_ORIGINS="http://localhost:3000,https://yourapp.com"
# or, to allow any origin:
export FARMWISE_CORS_ORIGINS="*"
```

**Server-to-server** callers (another backend, mobile app, curl) don't need this —
CORS is a browser rule only.

### Rate limiting

The AI endpoints (`/api/predict`, `/api/ask`, `/api/chat*`, `/api/ingest*`) are
rate limited per client IP: 40 requests / 60 s by default. Over the limit returns
`429` with a `Retry-After` header. Tune with `FARMWISE_RATE_LIMIT` and
`FARMWISE_RATE_LIMIT_WINDOW` (set `FARMWISE_RATE_LIMIT=0` to disable).

### Errors

All errors return JSON `{ "detail": "<message>" }` with an HTTP status:

| Status | Meaning |
|--------|---------|
| 422 | Invalid input (missing/NaN field, empty question) |
| 429 | Rate limited — back off and retry after `Retry-After` seconds |
| 503 | Model not trained, or Ollama not running |
| 413 | Body / upload too large |

---

## 1. Crop recommendation

### `POST /api/predict`

Recommend a crop from 7 soil/weather measurements.

**Request body** (all fields required, all numbers):

```json
{
  "N": 90,
  "P": 42,
  "K": 43,
  "temperature": 20.9,
  "humidity": 82.0,
  "ph": 6.5,
  "rainfall": 202.0
}
```

| Field | Meaning | Unit |
|-------|---------|------|
| `N` | Nitrogen in soil | kg/ha |
| `P` | Phosphorus in soil | kg/ha |
| `K` | Potassium in soil | kg/ha |
| `temperature` | Avg temperature | °C |
| `humidity` | Relative humidity | % |
| `ph` | Soil pH | 0–14 |
| `rainfall` | Rainfall | mm |

**Response** `200`:

```json
{
  "crop": "rice",
  "confidence": 0.98,
  "top3": [
    { "crop": "rice",   "probability": 0.98 },
    { "crop": "maize",  "probability": 0.01 },
    { "crop": "jute",   "probability": 0.005 }
  ],
  "out_of_distribution": false,
  "ood_warnings": []
}
```

- `crop` — the top recommendation.
- `confidence` — probability of the top crop (0–1).
- `top3` — the three most likely crops, ranked.
- `out_of_distribution` — `true` when an input is outside the model's training
  range. **Show a warning to the user and treat the result as untrustworthy** —
  the dataset was not collected in Ghana. `ood_warnings` lists which fields.

**Example:**

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"N":90,"P":42,"K":43,"temperature":20.9,"humidity":82,"ph":6.5,"rainfall":202}'
```

### Helper endpoints for the crop UI

- `GET /api/crop-conditions` → per-crop and overall feature ranges (slider
  bounds, "your input vs this crop" chart data).
- `GET /api/weather?location=Accra` (or `?latitude=..&longitude=..`) → current
  temperature, humidity and ~annual rainfall, to auto-fill the form.

---

## 2. Chat / farming assistant

Two shapes: **one-shot Q&A** (`/api/ask`) and **multi-turn chat** (`/api/chat`,
with an optional streaming variant). Both can run in two modes:

- `use_rag: true` (default) — **grounded**: answers only from the indexed farming
  guides, and returns the source passages. Says "I don't know based on the farming
  guides provided." when the guides don't cover it.
- `use_rag: false` — **bare LLM**: answers from the model's own knowledge, no
  sources. May hallucinate; useful for comparison.

### `POST /api/ask` — single question

**Request:**

```json
{
  "question": "When should I plant maize in southern Ghana?",
  "use_rag": true,
  "language": "en"
}
```

`language` accepts a code from `GET /api/languages` (e.g. `en`, `tw`, `ee`, `ha`);
defaults to `en`.

**Response** `200`:

```json
{
  "question": "When should I plant maize in southern Ghana?",
  "answer": "Plant maize at the onset of the major rains, typically...",
  "used_context": true,
  "sources": [
    {
      "source": "maize_guide.pdf",
      "chunk_index": 4,
      "text": "The major season in the forest zone begins...",
      "score": 0.82
    }
  ],
  "model": "llama3.2:3b"
}
```

- `used_context` — `true` for RAG, `false` for the bare LLM.
- `sources` — the guide chunks the answer was grounded in; empty for a refusal or
  a bare-LLM answer. Use these to show citations.

### `POST /api/chat` — multi-turn conversation

Send the running message history; the last message must be from the user. Keep
the previous turns so follow-ups work.

**Request:**

```json
{
  "messages": [
    { "role": "user",      "content": "What crop suits sandy soil?" },
    { "role": "assistant", "content": "Cassava and groundnut do well..." },
    { "role": "user",      "content": "How much rainfall does cassava need?" }
  ],
  "use_rag": true,
  "language": "en"
}
```

Constraints: 1–50 messages; each `content` 1–4000 chars; `role` is `"user"` or
`"assistant"`.

**Response** — same shape as `/api/ask` (`question` is the latest user message):

```json
{
  "question": "How much rainfall does cassava need?",
  "answer": "Cassava tolerates 1000–1500 mm per year...",
  "used_context": true,
  "sources": [ { "source": "...", "chunk_index": 2, "text": "...", "score": 0.79 } ],
  "model": "llama3.2:3b"
}
```

### `POST /api/chat/stream` — streaming chat (Server-Sent Events)

Same request body as `/api/chat`. Response is `text/event-stream`; each line is
`data: {json}`:

```
data: {"type": "token", "text": "Cassava"}
data: {"type": "token", "text": " tolerates"}
...
data: {"type": "done", "answer": "Cassava tolerates...", "used_context": true, "sources": [...], "model": "llama3.2:3b"}
```

An error mid-stream arrives as `data: {"type": "error", "detail": "..."}` (the
HTTP status is already `200`, so always check the event `type`).

**Browser example:**

```js
const res = await fetch("http://localhost:8000/api/chat/stream", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ messages, use_rag: true, language: "en" }),
});
const reader = res.body.getReader();
const decoder = new TextDecoder();
let buffer = "";
while (true) {
  const { value, done } = await reader.read();
  if (done) break;
  buffer += decoder.decode(value, { stream: true });
  for (const line of buffer.split("\n")) {
    if (!line.startsWith("data: ")) continue;
    const evt = JSON.parse(line.slice(6));
    if (evt.type === "token") appendToken(evt.text);
    else if (evt.type === "done") finish(evt.answer, evt.sources);
    else if (evt.type === "error") showError(evt.detail);
  }
  buffer = buffer.slice(buffer.lastIndexOf("\n") + 1);
}
```

### Helper endpoint

- `GET /api/languages` → `{ "languages": [ { "code": "en", "name": "English" }, ... ] }`
  for a language selector.

---

## 3. Knowledge base (optional — admin only)

These change the RAG index, so they need admin rights (`X-Admin-Token` header
when `FARMWISE_ADMIN_TOKEN` is set, otherwise localhost only):

- `GET  /api/sources` — list indexed sources + chunk counts (open, no auth).
- `POST /api/ingest/url` — `{ "url": "...", "crawl": false }` add a web page/PDF.
- `POST /api/ingest/file` — multipart upload of a `.pdf`/`.md`/`.txt` guide.
- `POST /api/sources/remove` — `{ "source": "name" }` remove a source.

---

## 4. Health check

`GET /api/health` (no auth) — poll this to show status before enabling the UI:

```json
{
  "ollama_running": true,
  "llm_model": "llama3.2:3b",
  "embedding_model": "nomic-embed-text",
  "index_chunks": 128,
  "model_trained": true
}
```

- `model_trained: false` → `/api/predict` returns `503`.
- `ollama_running: false` → chat/ask endpoints return `503`.
- `index_chunks: 0` → RAG has nothing to ground on; answers will refuse.

---

## Quick reference

| Endpoint | Method | Auth | Purpose |
|----------|--------|------|---------|
| `/api/predict` | POST | none | Crop recommendation |
| `/api/crop-conditions` | GET | none | Feature ranges for the crop UI |
| `/api/weather` | GET | none | Auto-fill soil/weather from a location |
| `/api/ask` | POST | none | One-shot question (RAG or bare LLM) |
| `/api/chat` | POST | none | Multi-turn chat |
| `/api/chat/stream` | POST | none | Multi-turn chat, streamed (SSE) |
| `/api/languages` | GET | none | Supported answer languages |
| `/api/health` | GET | none | Backend / model status |
| `/api/sources` | GET | none | List indexed sources |
| `/api/ingest/url` | POST | admin | Add a URL to the index |
| `/api/ingest/file` | POST | admin | Upload a guide to the index |
| `/api/sources/remove` | POST | admin | Remove a source |

Interactive docs (auto-generated) are at `GET /docs` while the app runs.
