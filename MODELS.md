# Models and setup for another machine

FarmWise runs its AI locally through [Ollama](https://ollama.com). To reproduce
this build on a different computer, install the same two models at the versions
below, then follow the run steps. Everything is CPU-friendly; no GPU required.

## Exact models used

| Role | Ollama model | Params | Quant | Size | Image ID (digest) |
|------|--------------|--------|-------|------|-------------------|
| Chat / generation (`FARMWISE_LLM_MODEL`) | `llama3.2:3b` | 3.2B | Q4_K_M | ~2.0 GB | `a80c4f17acd5` |
| Embeddings (`FARMWISE_EMBEDDING_MODEL`) | `nomic-embed-text` (`:latest`) | 137M | F16 | ~274 MB | `0a109f422b47` |

- `llama3.2:3b` — Llama 3.2, 3B instruct, context length 131072, embedding length
  3072. Licensed under the Llama 3.2 Community License (release 2024‑09‑25).
- `nomic-embed-text` — Nomic BERT text-embedding model, used to embed both the
  guides and each question.

The model names are the defaults in `farmwise/config.py` (`LLM_MODEL`,
`EMBEDDING_MODEL`) and can be overridden with the env vars shown above.

> The Image IDs pin the exact builds this project was tested against. `ollama pull`
> fetches the current tag; if a tag has moved and you need a byte-identical model,
> compare `ollama list` against the IDs here.

## Install the models

```bash
# install Ollama first (https://ollama.com/download), then:
ollama serve                      # leave running in its own terminal
ollama pull llama3.2:3b           # ~2.0 GB
ollama pull nomic-embed-text      # ~274 MB
ollama list                       # confirm the IDs match the table above
```

## Run the app on the new machine

```bash
# 1. Python 3.11+ and dependencies
pip install -r requirements.txt

# 2. Train the crop model and build the knowledge-base index
python scripts/train_model.py     # writes models/crop_model.joblib
python scripts/build_index.py     # embeds data/guides/* into chroma/

# 3. Start the web app
uvicorn webapp.main:app --reload
# open http://localhost:8000
```

`scripts/check_setup.py` verifies all of the above in one go (Ollama reachable,
both models present, model trained, index built).

## Verifying it matches this build

```bash
curl -s http://localhost:8000/api/health
# expect: ollama_running=true, llm_model=llama3.2:3b,
#         embedding_model=nomic-embed-text, model_trained=true, index_chunks>0
```

The index is rebuilt from `data/guides/` on each machine, so the chunk count
depends on which guides are present. With the committed guides (including
`regional_producers.md`) and no extra sources added, expect a few thousand chunks.

Notes:
- `nomic-embed-text` must be the **same** on both machines — if you ever swap the
  embedding model, rebuild the index, because embeddings from different models are
  not comparable.
- Default retrieval is `TOP_K=4` (see `farmwise/config.py`). On a CPU-only laptop
  the prompt is the slow part (about 40 tokens/s), and 4 passages answered all five
  `DEMO_QA.md` questions correctly at about 2.5x the speed of 8. Raise it with
  `FARMWISE_TOP_K` on a faster machine.

## Latency on a laptop

- The web app loads both models when it starts and touches them every 4 minutes
  (`FARMWISE_KEEP_WARM_SECONDS`), and asks Ollama to keep them loaded for 2 hours
  (`FARMWISE_OLLAMA_KEEP_ALIVE`). A cold or swapped-out model took over a minute to
  read its first prompt.
- Free memory before presenting: close IDEs and other heavy apps. The two models
  need about 3 GB of RAM, and if the system swaps them out every answer slows down.
- Plug in the charger. The `performance` power profile
  (`powerprofilesctl set performance`) gave about 15% more speed in testing.
- Start the app a few minutes before going live and ask one question, so the first
  answer the audience sees is a warm one.
- The Intel Arc GPU through Ollama's Vulkan backend was tried and is not usable
  with the current driver (it crashed and produced garbage), so stay on CPU.
