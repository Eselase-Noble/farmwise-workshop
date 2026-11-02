# 🌱 FarmWise — a smart farming assistant

Demo project for **Session 1** of a 2-hour workshop at the **Africa Digital
Skills Conference**. Everything runs **locally and offline** (once set up).

Two AI features:

1. **Crop recommendation** — a classic ML model (Random Forest) that recommends a
   crop from soil and weather inputs (N, P, K, temperature, humidity, pH, rainfall).
2. **Farming Q&A** — a **RAG** assistant that answers farmers' questions from
   farming-guide documents using a local LLM (via Ollama), with citations.

The `farmwise/` package is deliberately written for **reuse**: Session 2 (branch
`se`) wraps the same functions in a **FastAPI** backend with **PostgreSQL +
pgvector** and a **Streamlit** app — without rewriting the core.

> **Presenter?** See **[DEMO.md](DEMO.md)** for the timed runbook, talking points,
> comparison questions, and a fallback for every step.

---

## Prerequisites

- Python **3.11+**
- [Ollama](https://ollama.com) (local LLM runtime)
- ~3 GB disk for the two models

## Setup

```bash
# 1. Create the environment and install dependencies
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
#   (plain pip works too: python -m venv .venv && pip install -r requirements.txt)

# 2. Start Ollama and pull the models (one-time, ~2.3 GB)
ollama serve                       # leave running in its own terminal
ollama pull llama3.2:3b
ollama pull nomic-embed-text

# 3. Add the crop dataset  ->  data/raw/Crop_recommendation.csv
#    (see data/raw/README.md for the download link)

# 4. Train the model and build the RAG index
python scripts/train_model.py
python scripts/build_index.py

# 5. Verify everything is ready (every line should say [ OK ])
python scripts/check_setup.py

# 6. Open the notebooks
jupyter lab
```

## The three notebooks

| Notebook | What it shows |
|----------|---------------|
| `notebooks/01_crop_recommendation.ipynb` | Explore data → train RF (+ baseline) → evaluate → predict → save |
| `notebooks/02_rag_farming_assistant.ipynb` | Build RAG step by step; plain LLM **vs** RAG, side by side |
| `notebooks/03_responsible_ai.ipynb` | Data provenance, out-of-distribution flags, confidence, hallucination |

Notebooks are committed **with their outputs saved** so they double as a backup
if anything misbehaves live.

## Using the package directly

```python
# Crop recommendation — plain typed function, FastAPI-ready
from farmwise.ml.predict import predict_crop, CropFeatures

pred = predict_crop(CropFeatures(N=90, P=42, K=43, temperature=21,
                                 humidity=82, ph=6.5, rainfall=200))
print(pred.crop, pred.confidence, pred.top3, pred.out_of_distribution)

# Farming Q&A — two entry points, one typed answer shape
from farmwise.rag.assistant import ask, ask_without_rag

ask_without_rag("What is maize?")          # bare LLM, no sources
answer = ask("How much nitrogen for maize?")  # RAG, grounded + cited
print(answer.answer, [s.source for s in answer.sources])
```

## Configuration

All knobs are environment variables with sensible defaults (see
`farmwise/config.py`):

| Variable | Default |
|----------|---------|
| `FARMWISE_LLM_MODEL` | `llama3.2:3b` |
| `FARMWISE_EMBEDDING_MODEL` | `nomic-embed-text` |
| `FARMWISE_TOP_K` | `3` |
| `FARMWISE_CHUNK_SIZE` / `FARMWISE_CHUNK_OVERLAP` | `800` / `150` |
| `FARMWISE_MAX_TOKENS` | `256` |
| `OLLAMA_HOST` | `http://localhost:11434` |

## Project layout

```
farmwise/            reusable package (Session 2 imports this)
  config.py          paths, model names, chunk size, top_k — via env vars
  ml/
    data.py          load + validate the crop dataset
    train.py         train, evaluate, save model + metadata
    predict.py       predict_crop(inputs) -> crop, confidence, top-3, OOD flag
  rag/
    loader.py        load .pdf/.txt/.md guides
    chunker.py       simple, explainable chunking with overlap
    embeddings.py    embeddings via Ollama (+ Chroma embedding function)
    vector_store.py  backend-agnostic VectorStore ABC + ChromaVectorStore
    indexer.py       load -> chunk -> embed -> store pipeline
    assistant.py     ask() (RAG) and ask_without_rag() (bare LLM)
  llm/
    client.py        thin Ollama wrapper (clear error if not running)
notebooks/           the three live demos (committed with outputs)
scripts/             train_model.py · build_index.py · check_setup.py
data/
  raw/               crop dataset CSV (you download it; git-ignored)
  guides/            farming guides (sample .md files for testing)
models/              saved model + metadata JSON
tests/               pytest suite (ML + RAG, runs offline)
DEMO.md              presenter runbook
```

## Design for Session 2 reuse

- **`VectorStore` is backend-agnostic** (`add / search / count / reset`), so a
  `PgVectorStore` (pgvector) drops in without touching `assistant.py`.
- **`predict_crop` and `ask` are plain typed functions** (dataclasses in/out), so
  FastAPI routes can wrap them directly.
- **Model metadata is saved alongside the model** (feature names, classes,
  accuracy, training date, dataset provenance, feature ranges).

## Tests

```bash
python -m pytest        # 25 tests, fully offline (Ollama mocked)
```

## ⚠️ Responsible-AI note

The crop dataset's collection location is undocumented and is widely understood
to reflect **Indian** conditions — it is **not validated for Ghana**. The model
flags out-of-distribution inputs, and notebook 03 explores what this means. The
guide documents shipped here are **SAMPLES for testing only, not agronomic
advice** — replace them with real MoFA/FAO materials before presenting.

## Branches

- **`ai`** — Session 1 (this project)
- **`se`** — Session 2 (FastAPI + PostgreSQL/pgvector + Streamlit), built later
- **`java`** — a full Java port of FarmWise

## License

MIT — © Eselase-Noble
