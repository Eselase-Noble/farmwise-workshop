"""Tests for config resolution and the ingestion pipeline (offline)."""

from __future__ import annotations

from farmwise import config
from farmwise.rag.indexer import index_guides
from farmwise.rag.vector_store import SearchResult, VectorStore


def test_config_as_dict_has_core_keys():
    d = config.as_dict()
    for key in ("LLM_MODEL", "EMBEDDING_MODEL", "TOP_K", "CHUNK_SIZE", "CHUNK_OVERLAP"):
        assert key in d
    assert d["CHUNK_OVERLAP"] < d["CHUNK_SIZE"]


def test_config_env_override(monkeypatch):
    # config reads env at import; re-import a fresh module to see the override.
    import importlib

    monkeypatch.setenv("FARMWISE_TOP_K", "7")
    fresh = importlib.reload(config)
    try:
        assert fresh.TOP_K == 7
    finally:
        monkeypatch.delenv("FARMWISE_TOP_K", raising=False)
        importlib.reload(config)  # restore defaults for other tests


class RecordingStore(VectorStore):
    """Captures what index_guides adds, so we can assert on it without a backend."""

    def __init__(self):
        self.added = None
        self.was_reset = False

    def add(self, ids, documents, embeddings, metadatas):
        self.added = {
            "ids": ids,
            "documents": documents,
            "embeddings": embeddings,
            "metadatas": metadatas,
        }

    def search(self, query_embedding, top_k):  # pragma: no cover
        return []

    def count(self):
        return len(self.added["ids"]) if self.added else 0

    def distinct_sources(self):  # pragma: no cover
        return {}

    def delete_source(self, source):
        return 0

    def reset(self):
        self.was_reset = True


def test_index_guides_pipeline(tmp_path, monkeypatch):
    # two tiny guides on disk
    (tmp_path / "a.md").write_text("Maize needs nitrogen. " * 20)
    (tmp_path / "b.txt").write_text("Tomatoes need steady water. " * 20)

    # don't call Ollama: fake the embedder with a fixed-length vector per chunk
    monkeypatch.setattr(
        "farmwise.rag.indexer.embed_texts",
        lambda texts, **k: [[0.1, 0.2, 0.3] for _ in texts],
    )

    store = RecordingStore()
    report = index_guides(store, directory=tmp_path, chunk_size=100, overlap=20)

    assert store.was_reset is True
    assert report.documents == 2
    assert report.chunks == len(store.added["ids"]) > 0
    # ids are stable "source::index" and metadata carries the source
    assert all("::" in i for i in store.added["ids"])
    assert {m["source"] for m in store.added["metadatas"]} == {"a.md", "b.txt"}
    # one embedding per chunk
    assert len(store.added["embeddings"]) == report.chunks
