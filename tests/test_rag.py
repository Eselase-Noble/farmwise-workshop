"""Tests for the RAG pipeline (loader, chunker, vector store, assistant, client).

Everything here runs offline: the ChromaVectorStore is driven with precomputed
embeddings, and all Ollama network calls are monkeypatched.
"""

from __future__ import annotations

import pytest

from farmwise.llm.client import model_available
from farmwise.rag.assistant import (
    RagAnswer,
    ask,
    ask_without_rag,
    build_prompt,
)
from farmwise.rag.chunker import chunk_text
from farmwise.rag.loader import Document, load_file, load_guides
from farmwise.rag.vector_store import SearchResult, VectorStore


# --- chunker ---------------------------------------------------------------

def test_chunk_overlap_and_indexing():
    text = "word " * 400  # 2000 chars
    chunks = chunk_text(text, source="x.md", chunk_size=800, overlap=150)
    assert len(chunks) >= 2
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(c.source == "x.md" for c in chunks)
    assert all(len(c.text) <= 800 for c in chunks)
    # consecutive chunks should overlap in the source (start steps back).
    assert chunks[1].start < chunks[0].end


def test_chunk_short_text_single_chunk():
    chunks = chunk_text("just a short guide", chunk_size=800, overlap=150)
    assert len(chunks) == 1
    assert chunks[0].text == "just a short guide"


def test_chunk_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=100, overlap=100)


# --- loader ----------------------------------------------------------------

def test_load_txt_and_md(tmp_path):
    (tmp_path / "a.md").write_text("# Title\nhello")
    doc = load_file(tmp_path / "a.md")
    assert isinstance(doc, Document)
    assert doc.source == "a.md"
    assert "hello" in doc.text


def test_load_unsupported_type(tmp_path):
    (tmp_path / "a.rtf").write_text("nope")  # .rtf is not a supported type
    with pytest.raises(ValueError):
        load_file(tmp_path / "a.rtf")


def test_load_guides_skips_readme(tmp_path):
    (tmp_path / "README.md").write_text("readme")
    (tmp_path / "guide1.md").write_text("content one")
    (tmp_path / "guide2.txt").write_text("content two")
    docs = load_guides(tmp_path)
    sources = {d.source for d in docs}
    assert sources == {"guide1.md", "guide2.txt"}


# --- vector store (real Chroma, precomputed vectors, no Ollama) ------------

def test_chroma_store_add_search_count_reset(tmp_path):
    from farmwise.rag.vector_store import ChromaVectorStore

    store = ChromaVectorStore(persist_dir=tmp_path, collection_name="test_guides")
    store.reset()
    store.add(
        ids=["a::0", "b::0", "c::0"],
        documents=["maize needs nitrogen", "tomato early blight", "soil ph lime"],
        embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        metadatas=[
            {"source": "maize.md", "chunk_index": 0},
            {"source": "tomato.md", "chunk_index": 0},
            {"source": "soil.md", "chunk_index": 0},
        ],
    )
    assert store.count() == 3

    results = store.search([0.9, 0.1, 0.0], top_k=2)
    assert len(results) == 2
    assert isinstance(results[0], SearchResult)
    assert results[0].metadata["source"] == "maize.md"  # nearest to [1,0,0]
    assert 0.0 <= results[0].score <= 1.0
    assert results[0].score >= results[1].score  # sorted by similarity

    store.reset()
    assert store.count() == 0


# --- assistant (fake store + mocked LLM) -----------------------------------

class FakeStore(VectorStore):
    """In-memory stand-in that returns fixed search results."""

    def __init__(self, results):
        self._results = results

    def add(self, ids, documents, embeddings, metadatas):  # pragma: no cover
        pass

    def search(self, query_embedding, top_k):
        return self._results[:top_k]

    def count(self):  # pragma: no cover
        return len(self._results)

    def reset(self):  # pragma: no cover
        pass


@pytest.fixture
def fake_results():
    return [
        SearchResult(
            id="maize.md::0",
            document="Maize responds strongly to nitrogen.",
            metadata={"source": "maize.md", "chunk_index": 0},
            score=0.82,
            distance=0.18,
        ),
        SearchResult(
            id="soil.md::1",
            document="Lime raises soil pH over time.",
            metadata={"source": "soil.md", "chunk_index": 1},
            score=0.40,
            distance=0.60,
        ),
    ]


def test_build_prompt_contains_context_and_question(fake_results):
    prompt = build_prompt("How much nitrogen for maize?", fake_results)
    assert "[1]" in prompt and "[2]" in prompt
    assert "maize.md" in prompt
    assert "How much nitrogen for maize?" in prompt


def test_ask_rag_returns_sources(monkeypatch, fake_results):
    monkeypatch.setattr("farmwise.rag.assistant.embed_query", lambda q, **k: [0.1, 0.2])
    monkeypatch.setattr(
        "farmwise.rag.assistant.llm_client.generate",
        lambda prompt, **k: "Apply nitrogen; see maize guide.",
    )
    answer = ask("How much nitrogen for maize?", store=FakeStore(fake_results), top_k=2)
    assert isinstance(answer, RagAnswer)
    assert answer.used_context is True
    assert len(answer.sources) == 2
    assert answer.sources[0].source == "maize.md"
    assert answer.sources[0].score == pytest.approx(0.82)


def test_ask_without_rag_has_no_sources(monkeypatch):
    monkeypatch.setattr(
        "farmwise.rag.assistant.llm_client.generate",
        lambda prompt, **k: "Generic answer from the model.",
    )
    answer = ask_without_rag("What is maize?")
    assert answer.used_context is False
    assert answer.sources == []
    assert answer.answer == "Generic answer from the model."


# --- llm client errors (mocked network) ------------------------------------

def test_generate_raises_clear_error_when_ollama_down(monkeypatch):
    import requests

    from farmwise.llm import client

    def boom(*a, **k):
        raise requests.exceptions.ConnectionError()

    monkeypatch.setattr(client.requests, "post", boom)
    with pytest.raises(client.OllamaNotRunningError):
        client.generate("hello")


def test_is_ollama_running_false_on_connection_error(monkeypatch):
    import requests

    from farmwise.llm import client

    def boom(*a, **k):
        raise requests.exceptions.ConnectionError()

    monkeypatch.setattr(client.requests, "get", boom)
    assert client.is_ollama_running() is False


@pytest.mark.parametrize(
    "model, available, expected",
    [
        ("llama3.2:3b", ["llama3.2:3b"], True),
        ("llama3.2:3b", ["llama3.2:1b"], False),
        ("nomic-embed-text", ["nomic-embed-text:latest"], True),
        ("nomic-embed-text:latest", ["nomic-embed-text"], True),
    ],
)
def test_model_available_compares_tags(model, available, expected):
    assert model_available(model, available) is expected
