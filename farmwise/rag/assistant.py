"""The farming Q&A assistant: two entry points, one typed answer shape.

* :func:`ask_without_rag` — the **bare LLM**: answer from the model's own
  knowledge, no documents. Shows how a plain chatbot can hallucinate.
* :func:`ask` — **RAG**: retrieve relevant guide chunks, put them in the prompt,
  and answer *only* from them, with citations. Says it doesn't know when the
  guides don't cover the question.

Both return a :class:`RagAnswer`, so Session 2 can expose them as two FastAPI
endpoints (a "raw chatbot" API and a "RAG" API) with the same response model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

from farmwise import config
from farmwise.llm import client as llm_client
from farmwise.rag.embeddings import embed_query
from farmwise.rag.vector_store import SearchResult, VectorStore


@dataclass
class SourceChunk:
    """A retrieved guide chunk cited in a RAG answer."""

    source: str
    chunk_index: int
    text: str
    score: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RagAnswer:
    """A typed answer from either entry point."""

    question: str
    answer: str
    used_context: bool              # True for RAG, False for the bare LLM
    sources: list[SourceChunk] = field(default_factory=list)
    model: str = ""

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "used_context": self.used_context,
            "sources": [s.to_dict() for s in self.sources],
            "model": self.model,
        }


# --- prompts ---------------------------------------------------------------

PLAIN_SYSTEM_PROMPT = (
    "You are FarmWise, a helpful assistant for farmers. "
    "Answer concisely in plain language."
)

RAG_SYSTEM_PROMPT = (
    "You are FarmWise, a helpful assistant for farmers. "
    "Answer the question using only the numbered context passages provided. "
    "You may apply a fact about a larger area to a place inside it. In Ghana the "
    "southern (forest and coastal) zone includes the Western, Central, Greater "
    "Accra, Eastern, Ashanti, Bono and Volta regions, and the northern (savanna) "
    "zone includes the Northern, Savannah, North East, Upper East and Upper West "
    "regions; so a planting time given for 'southern Ghana' applies to the Western "
    "Region. "
    "If the passages contain nothing relevant to the question, reply exactly: "
    "\"I don't know based on the farming guides provided.\" "
    "Do not add agronomic facts that are not supported by the passages. Keep the "
    "answer short, and mention which source(s) the answer came from."
)


def build_prompt(question: str, results: list[SearchResult]) -> str:
    """Assemble the RAG prompt from retrieved chunks (printed in the demo)."""
    context_blocks = []
    for i, r in enumerate(results, start=1):
        source = r.metadata.get("source", "?")
        context_blocks.append(f"[{i}] (from {source})\n{r.document}")
    context = "\n\n".join(context_blocks) if context_blocks else "(no context found)"
    return (
        f"Context passages:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer (use only the context above):"
    )


# --- default store ---------------------------------------------------------

def get_default_store() -> VectorStore:
    """Return the configured persistent Chroma store (lazy import)."""
    from farmwise.rag.vector_store import ChromaVectorStore

    return ChromaVectorStore()


# --- entry points ----------------------------------------------------------

def ask_without_rag(
    question: str,
    *,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> RagAnswer:
    """Answer from the LLM's own knowledge — no documents, no retrieval."""
    model_name = model or config.LLM_MODEL
    answer = llm_client.generate(
        question, system=PLAIN_SYSTEM_PROMPT, model=model_name, max_tokens=max_tokens
    )
    return RagAnswer(
        question=question,
        answer=answer,
        used_context=False,
        sources=[],
        model=model_name,
    )


def ask(
    question: str,
    *,
    store: Optional[VectorStore] = None,
    top_k: Optional[int] = None,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> RagAnswer:
    """Answer the question with RAG: retrieve, ground, cite.

    Returns a :class:`RagAnswer` whose ``sources`` are the chunks the answer was
    grounded in. If the guides don't cover the question, the model is instructed
    to say so rather than guess.
    """
    store = store or get_default_store()
    k = top_k if top_k is not None else config.TOP_K
    model_name = model or config.LLM_MODEL

    query_vec = embed_query(question)
    results = store.search(query_vec, top_k=k)

    prompt = build_prompt(question, results)
    answer = llm_client.generate(
        prompt, system=RAG_SYSTEM_PROMPT, model=model_name, max_tokens=max_tokens
    )

    sources = [
        SourceChunk(
            source=r.metadata.get("source", "?"),
            chunk_index=int(r.metadata.get("chunk_index", -1)),
            text=r.document,
            score=r.score,
        )
        for r in results
    ]
    return RagAnswer(
        question=question,
        answer=answer,
        used_context=True,
        sources=sources,
        model=model_name,
    )
