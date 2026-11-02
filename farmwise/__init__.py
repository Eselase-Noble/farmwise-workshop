"""FarmWise — a smart farming assistant.

Two AI features, built to be reused by a FastAPI backend in Session 2:

* Crop recommendation — a classic ML model (``farmwise.ml``).
* Farming Q&A — a RAG assistant over farming guides (``farmwise.rag``).

The public, FastAPI-friendly entry points are plain typed functions:

* :func:`farmwise.ml.predict.predict_crop`
* :func:`farmwise.rag.assistant.ask` / :func:`farmwise.rag.assistant.ask_without_rag`
"""

__version__ = "0.1.0"
