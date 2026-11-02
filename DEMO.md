# DEMO.md — FarmWise Session 1 Presenter Runbook

A 2-hour virtual workshop. You demo live; participants watch. This is your
minute-by-minute script, the commands to run **before** going live, talking
points per notebook section, the comparison questions, and a **fallback for every
step**.

---

## 0. Before you go live (do this ~30 min early)

Run each command and confirm it passes. The last one is the single source of truth.

```bash
cd farmwise-workshop
source .venv/bin/activate            # or: uv venv && uv pip install -r requirements.txt

# 1. Start Ollama (leave it running in its own terminal)
ollama serve

# 2. Confirm models are pulled (one-time, ~2.3 GB)
ollama pull llama3.2:3b
ollama pull nomic-embed-text

# 3. Make sure the dataset is in place
ls data/raw/Crop_recommendation.csv

# 4. Train the model + build the RAG index
python scripts/train_model.py
python scripts/build_index.py

# 5. THE pre-flight check — every line must say [ OK ]
python scripts/check_setup.py
```

Then open Jupyter and **restart-and-run-all each notebook once** to warm caches
and confirm everything is green:

```bash
jupyter lab      # or: jupyter notebook
```

> The notebooks are committed **with outputs already saved**. If anything fails
> live, you can simply scroll the pre-executed cells — see fallbacks below.

Keep a second terminal open for `ollama serve` logs, and a browser tab on the
GitHub repo in case you want to show the code.

---

## 1. Timed run order (matches the session flow)

| Time | Segment | What you do | Notebook |
|------|---------|-------------|----------|
| 0:00–0:25 | **Theory** | Slides (not covered here) | — |
| 0:25–0:55 | **Demo 1 — Crop ML** | load → explore → train → evaluate → predict → save | `01_crop_recommendation.ipynb` |
| 0:55–1:05 | Break / Q&A | — | — |
| 1:05–1:40 | **Demo 2 — RAG** | build RAG step by step; plain LLM vs RAG | `02_rag_farming_assistant.ipynb` |
| 1:40–1:50 | **Responsible AI** | data origin, OOD, confidence, hallucination | `03_responsible_ai.ipynb` |
| 1:50–2:00 | **Close** | what's missing → Session 2 | (talk track below) |

---

## 2. Demo 1 — Crop Recommendation (0:25–0:55)

**Notebook:** `01_crop_recommendation.ipynb`

Talking points by section:
- **Load & explore:** "Seven simple numbers any extension officer can measure."
  Show `df.shape`, the class-balance bar chart (22 crops, evenly sized), the
  feature histograms.
- **Split:** why we hold out a test set — "grading the model on questions it
  hasn't seen."
- **Train Random Forest:** "a crowd of decision trees that vote." Show accuracy.
- **Baseline comparison:** "always check a simple model first." RF ≈ 0.99 vs
  LogReg ≈ 0.97 — the forest earns its keep.
- **Evaluate:** classification report (precision/recall per crop), confusion
  matrix (where crops get confused), feature importance (**rainfall & humidity
  dominate** — remember this for Demo 3).
- **Predict + save:** run `predict_crop` on a sample farm; show top-3 and
  confidence; show the saved `metadata.json` — point at `dataset_source`.

**Comparison / reveal:** the inline code first, then the one-line
`farmwise` package call that does the same thing — "this is the exact function
Session 2's API will call."

**Fallback:**
- Plots slow / kernel wedged → scroll the committed outputs (already rendered).
- `DatasetNotFoundError` → `ls data/raw/`; the file must be
  `Crop_recommendation.csv`. Re-run `python scripts/train_model.py`.
- Don't retrain live if short on time — the saved model in `models/` is enough
  for the prediction cell.

---

## 3. Demo 2 — RAG Farming Assistant (1:05–1:40)

**Notebook:** `02_rag_farming_assistant.ipynb`

Talking points by section:
- **Part A, plain LLM:** ask a specific question with no documents. "Sounds
  confident — but what is it based on? Nothing we can check."
- **Step 1 Load:** show the guide files and a snippet.
- **Step 2 Chunk:** show a real chunk and its char range. "Models read short
  passages; overlap keeps split sentences whole." (size 800 / overlap 150)
- **Step 3 Embed:** show the vector length and the cosine-similarity example —
  "maize vs corn" scores high, "maize vs goat" low. *This is the whole idea.*
- **Step 4 Store:** `store.count()` chunks in ChromaDB. Mention the
  backend-agnostic interface → pgvector in Session 2.
- **Step 5 Retrieve:** show top-k chunks **with scores**.
- **Step 6 Prompt:** **print the actual prompt.** "No magic — we literally paste
  the passages in and tell it to answer only from them."
- **Step 7 Answer:** the grounded answer **with citations**.
- **Part C table:** plain LLM vs RAG vs sources, side by side.

**Comparison questions to ask (the Part C list):**
1. "What is the critical weed-free period for maize?" (in the maize guide)
2. "How can I tell if my tomato plant has early blight?" (in the tomato guide)
3. **"How do I raise tilapia fish in a pond?"** — *not* in any guide. RAG must
   answer **"I don't know based on the farming guides provided."** This is the
   money moment: a grounded assistant refuses to guess.

**Fallback:**
- Ollama slow / first token lagging → the notebook outputs are pre-executed;
  narrate the committed answers.
- `OllamaNotRunningError` → start `ollama serve` in the other terminal; re-run
  the setup cell.
- Model missing → `ollama pull llama3.2:3b` / `ollama pull nomic-embed-text`.
- Empty retrieval → `python scripts/build_index.py`, then re-run.
- If generation is too slow on the day: lower the limit —
  `export FARMWISE_MAX_TOKENS=128` and `export FARMWISE_TOP_K=2` before launching
  Jupyter, or switch to a smaller model: `export FARMWISE_LLM_MODEL=llama3.2:1b`
  (pull it first).

---

## 4. Responsible AI (1:40–1:50)

**Notebook:** `03_responsible_ai.ipynb`

Talking points:
- **Provenance:** read `dataset_source` out loud. "This data was **not collected
  in Ghana.**"
- **Ranges vs Ghana:** the table flags features where local conditions exceed the
  training range (e.g. northern temperatures). "The model never saw those."
- **OOD flag:** run the risky input → `out_of_distribution = True` with a warning.
  "The model still answers — our job is to flag when not to trust it."
- **Confidence:** the near-tie example — "confidence is not correctness."
- **Per-class:** the weakest crops by f1 — "averages hide the crops it struggles
  with."
- **Hallucination/RAG:** re-show the unanswerable question — plain LLM invents,
  RAG cites or declines.

**Fallback:** all cells are pre-executed; if the kernel is busy, scroll outputs.
The OOD table and the warning are the two things you must show — they're static
once run.

---

## 5. Close — what's missing → Session 2 (1:50–2:00)

Say the limitations out loud (bottom of notebook 03), then frame Session 2:

- **No memory** — each question is isolated; no conversation history.
- **No users** — nothing is saved per person; no accounts.
- **Not shareable** — it runs on my laptop in a notebook, not on the web.

> "In Session 2 we take this exact `farmwise` package — the same `predict_crop`
> and `ask` functions — and wrap it in a **FastAPI** backend with **PostgreSQL +
> pgvector** for storage and a **Streamlit** app so anyone can use it in a
> browser. Nothing we built today gets thrown away."

Show the `se` branch on GitHub as the starting point.

---

## 6. Quick reference — knobs (all env vars)

| Variable | Default | Use in demo |
|----------|---------|-------------|
| `FARMWISE_LLM_MODEL` | `llama3.2:3b` | drop to `llama3.2:1b` if slow |
| `FARMWISE_EMBEDDING_MODEL` | `nomic-embed-text` | — |
| `FARMWISE_TOP_K` | `3` | lower to `2` for speed |
| `FARMWISE_MAX_TOKENS` | `256` | lower to `128` for speed |
| `FARMWISE_CHUNK_SIZE` / `_OVERLAP` | `800` / `150` | rebuild index if changed |

**Golden rule:** if a live cell misbehaves, don't debug on stage — scroll to the
committed output and keep the story moving.
