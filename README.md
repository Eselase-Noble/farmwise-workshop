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
sample field so you can try it in one click. On the chat page, switch between
"RAG · grounded" and "Bare chatbot" to see the difference between an answer
backed by the guides and an answer from the model alone.

## Added features

Beyond the three core pages, this build adds:

* **Installable, works offline.** The app ships a web manifest and a service
  worker, so it can be installed on a phone and its pages open offline (live AI
  answers still need a connection; the UI says so). The brand mark is a single
  SVG (`static/logo.svg`) reused in the header, footer and favicon.
* **Live status banner.** Every page polls `GET /api/health` and shows a banner
  when Ollama is down, the crop model is untrained, or the knowledge base is
  empty — so a failed demo tells you why instead of erroring mid-request.
* **Weather auto-fill.** On the crop page, enter a town (or use your device
  location) and `GET /api/weather` fills temperature, humidity and rainfall from
  Open-Meteo (free, no API key). Soil values still need measuring.
* **Recommendation history.** Each crop recommendation is saved in the browser
  (not on the server) and can be downloaded as CSV from the crop page.
* **Local-language answers.** The chat page has a language selector; `ask` and
  `chat` take a `language` of `en`, `twi`, `fante`, `ewe` or `ga`. The model is
  told to answer in that language, primed with a few aligned example sentences
  (`data/language_primers.json`, built by `scripts/build_language_primers.py`
  from a parallel corpus). Note: `llama3.2:3b` has only partial coverage of
  these low-resource languages, so non-English answers are approximate.

### Robustness

* Ollama calls now have a connect/read timeout and retry transient failures
  (`FARMWISE_OLLAMA_TIMEOUT`, `FARMWISE_OLLAMA_MAX_RETRIES`, …).
* A per-IP rate limit guards the AI and ingest endpoints
  (`FARMWISE_RATE_LIMIT`, default 40 requests / `FARMWISE_RATE_LIMIT_WINDOW`
  seconds; set the limit to 0 to disable). Rejected requests return 429 with a
  `Retry-After` header.
* Each request is logged (client, method, path, status, duration) to the console
  and a rotating file under `FARMWISE_LOG_DIR` (default `logs/webapp.log`).

## Growing the knowledge base

The assistant only knows what it has read. The `/knowledge` page lets you grow
it without touching code:

* Paste a link to a guide, article or PDF and it is fetched, reduced to readable
  text, chunked, embedded and added to the index straight away.
* Upload a document and the same thing happens. Supported types are PDF, Word
  (`.docx`), Excel (`.xlsx`/`.xls`), CSV, HTML and plain text or markdown.
  Uploaded files are also saved into `data/guides/` so a full rebuild keeps them.

Two helpers make curation easier:

* Tick "Search linked pages too" when adding a link and the crawler follows the
  page's relevant links (reports and PDFs) and indexes those as well, so a thin
  landing page still contributes the documents it points to. Note that some
  government sites render their report lists with JavaScript, so their PDFs are
  not reachable this way; add those PDFs by their direct link instead.
* Each indexed source has a Remove button, so you can prune anything that is not
  credible. Remove only takes it out of the index: an uploaded file stays in
  `data/guides/` and comes back on the next rebuild, so delete it there as well.

Use credible sources for anything you demonstrate. The index is built from
United States Department of Agriculture, World Bank, MoFA (Agriculture in Ghana
Facts and Figures), FAO and MoFA-IFPRI material. Government, FAO, MoFA and
research institute documents are what belong here, not blog posts or paywalled
aggregators.

The panel lists every source currently indexed and its chunk count, so you can
see the knowledge base grow. Behind the panel are three endpoints:

```
POST /api/ingest/url      { "url": "..." }        add a web page or PDF
POST /api/ingest/file     (multipart file upload) add a document
POST /api/sources/remove  { "source": "..." }     remove a source
GET  /api/sources                                 list what is indexed
```

They call `farmwise.rag.loader.load_url` / `load_file` and
`farmwise.rag.indexer.index_document`, which chunks the new source and upserts it
into the same persistent store the assistant reads from. Re-adding a source
replaces its chunks rather than duplicating them. An upload only joins
`data/guides/` after it has been read and indexed, so a broken file is rejected
instead of breaking the next rebuild.

A worked example: the question "When is the best time to grow rice in the western
part of Ghana?" returns "I don't know based on the farming guides provided" with
only the sample guides loaded. Add a public rice guide through the panel and the
same question is answered, with the new source cited.

Note: Session 2 uses a default `top_k` of 4 (Session 1 used 3), because real guides are
larger than the Session 1 samples and the useful passage is often not in the top
three. Tune it with `FARMWISE_TOP_K`.

## Answers, relevance and follow-ups

* Retrieved passages scoring under `FARMWISE_MIN_SCORE` (default 0.62, tuned for
  `nomic-embed-text`) are dropped. If nothing relevant is left, the assistant says
  it does not know without calling the model at all.
* When the answer is "I don't know", no sources are shown, because none were used.
* In the chat, a follow-up such as "how much nitrogen does it need?" is also
  searched together with the previous question, so it finds the right guide.

## Who can change the knowledge base

Anyone who can reach the app can predict, ask, chat and see the source list.
Adding or removing sources is restricted:

* With no `FARMWISE_ADMIN_TOKEN` set, only a browser on the same machine can
  change it (the default `uvicorn` setup, which listens on localhost only).
* To share the app on a network, set a token and run on all interfaces:

  ```bash
  export FARMWISE_ADMIN_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(24))')"
  uvicorn webapp.main:app --host 0.0.0.0
  ```

  The knowledge page then shows a field for the token.
* Requests from another website are refused. Links are only fetched from public
  addresses, never the local network. Uploads are capped at
  `FARMWISE_MAX_UPLOAD_MB` (25) and downloads at `FARMWISE_MAX_DOWNLOAD_MB` (50).

There are no user accounts and no rate limits, so put it behind a proper login
before exposing it to the internet.

## Tests

```bash
python -m pytest        # all offline: Ollama and DNS are faked
```

## Reproducing the knowledge base on another machine

The vector index (`chroma/`) is machine-local and git-ignored, so a fresh clone
starts with an empty knowledge base. The curated sources themselves are committed
in `data/knowledge_sources.txt`, so you can rebuild the whole index in one
command (Ollama must be running):

```bash
python scripts/rebuild_kb.py --reset
```

That loads the sample guides and then ingests every credible source in the
manifest (USDA, GSS, World Bank, MoFA, FAO, MoFA-IFPRI). Add a line to the
manifest to make a new source part of the reproducible set. The script exits
with an error if any source fails, so a partial rebuild is easy to spot.
`scripts/build_index.py` only indexes `data/guides/`, which removes the sources
added by link.

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
sits behind a small interface (add, search, count, reset, plus listing and
removing sources), the planned next step
is a PostgreSQL with pgvector implementation that drops in without changing the
assistant or this web app. A Streamlit version of the same two panels is also a
natural addition for people who prefer it.

## A note on trust

The crop dataset was not collected in Ghana, so the model flags inputs that fall
outside its training range, and the Q&A assistant only answers from the supplied
guides and says it does not know when they do not cover a question. The sample
guides are for testing only and are not real agronomic advice.

© Eselase-Noble. MIT License.
