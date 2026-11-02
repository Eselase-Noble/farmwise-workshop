# FarmWise — a smart farming assistant

Demo project for **Session 1** of a 2-hour workshop at the Africa Digital Skills
Conference. Two local, offline-capable AI features:

1. **Crop recommendation** — a classic ML model that recommends a crop from soil
   and weather inputs (N, P, K, temperature, humidity, pH, rainfall).
2. **Farming Q&A** — a RAG assistant that answers farmers' questions from farming
   guide documents, using a local LLM (Ollama).

The `farmwise/` package is written for reuse: Session 2 wraps the same functions
in a FastAPI backend with PostgreSQL + pgvector and a Streamlit app.

> Full setup, run instructions, and the presenter runbook are being finalized.
> See **[DEMO.md](DEMO.md)** for the timed run order (added in the final stage).

## Quick start (outline)

```bash
# 1. Install deps
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# 2. Put the crop dataset at data/raw/Crop_recommendation.csv  (see data/raw/README.md)
# 3. Start Ollama and pull models
ollama serve &
ollama pull llama3.2:3b
ollama pull nomic-embed-text

# 4. Verify everything is ready
python scripts/check_setup.py

# 5. Train the model and build the RAG index
python scripts/train_model.py
python scripts/build_index.py

# 6. Open the notebooks
jupyter lab
```
