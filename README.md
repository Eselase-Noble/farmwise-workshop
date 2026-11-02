# FarmWise, Session 2 web app

This is the `se` branch. It takes the Session 1 FarmWise package and puts a web
interface in front of it, so the crop model and the farming assistant can be used
from a browser instead of a notebook. The point of Session 2 is to make the same
work shareable: nothing from Session 1 is rewritten, it is reused.

Built with Africode Lab, Accra Digital Skills, and Mobile Web Ghana.

## What is here

* A FastAPI backend in `webapp/main.py` that wraps the existing functions:
  * `POST /api/predict` calls `farmwise.ml.predict.predict_crop`
  * `POST /api/ask` calls `farmwise.rag.assistant.ask` (grounded, with sources)
    or `ask_without_rag` (the plain model), depending on the request
* A single black and white web page in `webapp/static/` with two panels, one for
  crop recommendation and one for farming questions.
* Everything from Session 1 (the `farmwise` package, notebooks, scripts, tests).

## Running it

You need the Session 1 setup first: Ollama running with the two models pulled,
the dataset in `data/raw/`, a trained model, and the guides indexed.

```bash
# from Session 1, if you have not already done it
pip install -r requirements.txt
ollama serve
ollama pull llama3.2:3b
ollama pull nomic-embed-text
python scripts/train_model.py
python scripts/build_index.py

# start the web app
uvicorn webapp.main:app --reload
```

Then open http://localhost:8000 in a browser. The crop form is prefilled with a
sample field so you can try it in one click, and the question box starts with a
maize question. Switch the Q&A mode between Grounded and Plain to see the
difference between an answer backed by the guides and an answer from the model
alone.

## How it talks to the core

```
browser  ->  FastAPI (webapp/main.py)  ->  farmwise package
             /api/predict              ->  predict_crop(CropFeatures)
             /api/ask (grounded)       ->  ask(question)
             /api/ask (plain)          ->  ask_without_rag(question)
```

The web layer holds no AI logic of its own. It validates the request, calls the
function, and returns the typed result as JSON. That is exactly why Session 1 was
written with plain typed functions and a backend agnostic vector store.

## What comes next on this branch

The vector store is still the local ChromaDB from Session 1. Because the store
sits behind a small interface (add, search, count, reset), the planned next step
is a PostgreSQL with pgvector implementation that drops in without changing the
assistant or this web app. A Streamlit version of the same two panels is also a
natural addition for people who prefer it.

## A note on trust

The crop dataset was not collected in Ghana, so the model flags inputs that fall
outside its training range, and the Q&A assistant only answers from the supplied
guides and says it does not know when they do not cover a question. The sample
guides are for testing only and are not real agronomic advice.

© Eselase-Noble. MIT License.
