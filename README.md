# FarmWise, a smart farming assistant

This is the Session 1 project for a two hour workshop at the Africa Digital Skills
Conference. Once it is set up, everything runs locally and offline.

It does two things:

1. Crop recommendation. A classic machine learning model (a Random Forest) that
   suggests a crop from a field's soil and weather numbers (N, P, K, temperature,
   humidity, pH, rainfall).
2. Farming Q&A. A retrieval based assistant (RAG) that answers questions from
   farming guides using a local language model through Ollama, and cites the
   guide it used.

The `farmwise/` package is written so it can be reused. In Session 2 (branch
`se`) the same functions sit behind a FastAPI backend and a web page, with
PostgreSQL and pgvector for storage, and none of the core has to be rewritten.

If you are presenting, read [DEMO.md](DEMO.md) first. It has the timed run order,
the commands to run before going live, talking points for each section, the
comparison questions, and a fallback for every step.

## What you need

* Python 3.11 or newer
* [Ollama](https://ollama.com) for the local language model
* About 3 GB of disk for the two models

## Setup

```bash
# 1. Create the environment and install the dependencies
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
#   plain pip works too: python -m venv .venv && pip install -r requirements.txt

# 2. Start Ollama and pull the two models (one time, about 2.3 GB)
ollama serve                       # leave this running in its own terminal
ollama pull llama3.2:3b
ollama pull nomic-embed-text

# 3. Put the crop dataset at data/raw/Crop_recommendation.csv
#    (data/raw/README.md has the download link)

# 4. Train the model and build the RAG index
python scripts/train_model.py
python scripts/build_index.py

# 5. Check that everything is ready (every line should say OK)
python scripts/check_setup.py

# 6. Open the notebooks
jupyter lab
```

## The three notebooks

| Notebook | What it shows |
|----------|---------------|
| `notebooks/01_crop_recommendation.ipynb` | Explore the data, train the forest and a baseline, evaluate, predict, save |
| `notebooks/02_rag_farming_assistant.ipynb` | Build RAG step by step, then compare the plain model with RAG side by side |
| `notebooks/03_responsible_ai.ipynb` | Where the data came from, out of range warnings, confidence, hallucination |

The notebooks are committed with their outputs saved, so they also work as a
backup if something misbehaves during the live demo.

## Using the package directly

```python
# Crop recommendation. A plain typed function, ready to wrap in an API.
from farmwise.ml.predict import predict_crop, CropFeatures

pred = predict_crop(CropFeatures(N=90, P=42, K=43, temperature=21,
                                 humidity=82, ph=6.5, rainfall=200))
print(pred.crop, pred.confidence, pred.top3, pred.out_of_distribution)

# Farming Q&A. Two entry points that return the same answer type.
from farmwise.rag.assistant import ask, ask_without_rag

ask_without_rag("What is maize?")             # the plain model, no sources
answer = ask("How much nitrogen for maize?")  # grounded in the guides, with citations
print(answer.answer, [s.source for s in answer.sources])
```

## Configuration

Everything you might want to change is an environment variable with a sensible
default. See `farmwise/config.py`.

| Variable | Default |
|----------|---------|
| `FARMWISE_LLM_MODEL` | `llama3.2:3b` |
| `FARMWISE_EMBEDDING_MODEL` | `nomic-embed-text` |
| `FARMWISE_TOP_K` | `3` |
| `FARMWISE_CHUNK_SIZE` and `FARMWISE_CHUNK_OVERLAP` | `800` and `150` |
| `FARMWISE_MAX_TOKENS` | `256` |
| `OLLAMA_HOST` | `http://localhost:11434` |

## How it is organised

```
farmwise/            the reusable package (Session 2 imports this)
  config.py          paths, model names, chunk size and top_k from env vars
  ml/
    data.py          load and validate the crop dataset
    train.py         train, evaluate, save the model and its metadata
    predict.py       predict_crop(inputs) with confidence, top 3 and an out of range flag
  rag/
    loader.py        read .pdf/.txt/.md guides
    chunker.py       simple overlapping chunks you can explain on a slide
    embeddings.py    embeddings through Ollama, plus a Chroma embedding function
    vector_store.py  a backend agnostic VectorStore interface and a Chroma version
    indexer.py       load, chunk, embed, store
    assistant.py     ask() for RAG and ask_without_rag() for the plain model
  llm/
    client.py        a small Ollama wrapper with a clear error when it is not running
notebooks/           the three live demos, committed with outputs
scripts/             train_model.py, build_index.py, check_setup.py
data/
  raw/               the crop dataset CSV (you download it, kept out of git)
  guides/            farming guides (sample files for testing)
models/              the saved model and its metadata
tests/               the pytest suite (ML and RAG, runs offline)
DEMO.md              the presenter runbook
```

## Why it is built for reuse

* The `VectorStore` interface is backend agnostic (add, search, count, reset), so
  a pgvector version can replace Chroma without touching `assistant.py`.
* `predict_crop` and `ask` are plain typed functions with dataclasses in and out,
  so an API can wrap them directly.
* The model metadata is saved next to the model: feature names, classes,
  accuracy, training date, where the data came from, and the feature ranges.

## Tests

```bash
python -m pytest        # 25 tests, all offline (Ollama is mocked)
```

## A note on responsible AI

The crop dataset has no documented origin and is generally taken to reflect
Indian conditions, so it is not validated for Ghana. The model flags inputs that
fall outside the range it was trained on, and notebook 03 digs into what that
means. The guides shipped here are samples for testing only and are not real
agronomic advice. Replace them with real MoFA or FAO material before presenting.

## Branches

* `ai` is Session 1, this project.
* `se` is Session 2, the FastAPI backend and web page, built later.
* `java` is a full Java port of FarmWise.

© Eselase-Noble. MIT License.
