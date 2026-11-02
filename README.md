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
  * `POST /api/chat` calls `farmwise.rag.assistant.chat`, a conversational version
    that keeps the recent turns in mind so follow-up questions work
  * `POST /api/ingest/url`, `POST /api/ingest/file`, `GET /api/sources` for the
    knowledge base
* Three pages in `webapp/static/`, styled after the Africa Digital Skills
  Conference, AfricodeLab and Mobile Web Ghana (Montserrat and Inter, a navy and
  cyan palette):
  * `/` crop recommendation
  * `/chat` the farming assistant chat
  * `/knowledge` the knowledge base, on its own page
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

## Growing the knowledge base

The assistant only knows what it has read. The web page has a Knowledge base
panel where you can grow it without touching code:

* Paste a link to a guide, article or PDF and it is fetched, reduced to readable
  text, chunked, embedded and added to the index straight away.
* Upload a document and the same thing happens. Supported types are PDF, Word
  (`.docx`), Excel (`.xlsx`/`.xls`), CSV, HTML and plain text or markdown.
  Uploaded files are also saved into `data/guides/` so a full rebuild keeps them.

Use credible sources for anything you demonstrate. The rice example below is
built from a United States Department of Agriculture Foreign Agricultural Service
report and a MoFA and IFPRI rice market brief. Government, FAO, MoFA and research
institute material is what belongs here, not blog posts.

The panel lists every source currently indexed and its chunk count, so you can
see the knowledge base grow. Behind the panel are three endpoints:

```
POST /api/ingest/url    { "url": "..." }        add a web page or PDF
POST /api/ingest/file   (multipart file upload) add a document
GET  /api/sources                               list what is indexed
```

They call `farmwise.rag.loader.load_url` / `load_file` and
`farmwise.rag.indexer.index_document`, which chunks the new source and upserts it
into the same persistent store the assistant reads from. Re-adding a source
replaces its chunks rather than duplicating them.

A worked example: the question "When is the best time to grow rice in the western
part of Ghana?" returns "I don't know based on the farming guides provided" with
only the sample guides loaded. Add a public rice guide through the panel and the
same question is answered, with the new source cited.

Note: Session 2 raises the default `top_k` from 3 to 5, because real guides are
larger than the Session 1 samples and the useful passage is often not in the top
three. Tune it with `FARMWISE_TOP_K`.

## How it talks to the core

```
browser  ->  FastAPI (webapp/main.py)  ->  farmwise package
             /api/predict              ->  predict_crop(CropFeatures)
             /api/ask (grounded)       ->  ask(question)
             /api/ask (plain)          ->  ask_without_rag(question)
             /api/chat                 ->  chat(messages)        conversational RAG
             /api/ingest/url|file      ->  loader + indexer      grow the knowledge base
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
