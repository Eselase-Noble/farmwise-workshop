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

NO_ANSWER = "I don't know based on the farming guides provided."

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
    f"\"{NO_ANSWER}\" "
    "Do not add agronomic facts that are not supported by the passages. "
    "Answer in short, plain prose. Do NOT put citation markers such as [1] or [2] "
    "in your answer, and do NOT name the source files or write phrases like "
    "'according to the guide'; the sources are listed separately for the reader."
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


# --- retrieval and grounding -----------------------------------------------

def _retrieve(queries: list[str], store: VectorStore, top_k: int) -> list[SearchResult]:
    """Search for each query and return the best relevant chunks across them."""
    best: dict[str, SearchResult] = {}
    for query in queries:
        for result in store.search(embed_query(query), top_k=top_k):
            if result.score < config.MIN_RELEVANCE_SCORE:
                continue
            current = best.get(result.id)
            if current is None or result.score > current.score:
                best[result.id] = result
    return sorted(best.values(), key=lambda r: r.score, reverse=True)[:top_k]


def _is_refusal(answer: str) -> bool:
    normalised = answer.replace("\u2019", "'").lower()
    return NO_ANSWER.rstrip(".").lower() in normalised


def _grounded_answer(
    question: str,
    results: list[SearchResult],
    *,
    model_name: str,
    max_tokens: Optional[int],
    history: str = "",
) -> RagAnswer:
    """Answer from ``results``. A refusal never lists sources, as none were used."""
    if not results:
        return RagAnswer(
            question=question, answer=NO_ANSWER, used_context=True, sources=[], model=model_name
        )

    prompt = build_prompt(question, results)
    if history:
        prompt = f"Conversation so far:\n{history}\n\n{prompt}"
    answer = llm_client.generate(
        prompt, system=RAG_SYSTEM_PROMPT, model=model_name, max_tokens=max_tokens
    )

    sources = [] if _is_refusal(answer) else [
        SourceChunk(
            source=r.metadata.get("source", "?"),
            chunk_index=int(r.metadata.get("chunk_index", -1)),
            text=r.document,
            score=r.score,
        )
        for r in results
    ]
    return RagAnswer(
        question=question, answer=answer, used_context=True, sources=sources, model=model_name
    )


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
    grounded in. If nothing relevant is found the model is not called, and if the
    model finds the passages do not cover the question, ``sources`` is empty.
    """
    store = store or get_default_store()
    k = top_k if top_k is not None else config.TOP_K
    model_name = model or config.LLM_MODEL

    results = _retrieve([question], store, k)
    return _grounded_answer(question, results, model_name=model_name, max_tokens=max_tokens)


def chat(
    messages: list[dict],
    *,
    use_rag: bool = True,
    store: Optional[VectorStore] = None,
    top_k: Optional[int] = None,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> RagAnswer:
    """Conversational assistant with the recent history in mind.

    ``messages`` is a list of ``{"role": "user"|"assistant", "content": str}``.

    * ``use_rag=True`` (default): retrieve on the latest user message, answer only
      from the guides, and cite the sources. Says it doesn't know when uncovered.
    * ``use_rag=False``: the bare chatbot. Answer from the model's own knowledge
      with no documents and no sources, so the demo can show how it may guess.
    """
    model_name = model or config.LLM_MODEL

    user_turns = [m for m in messages if m.get("role") == "user"]
    latest = user_turns[-1]["content"] if user_turns else ""

    recent = messages[-6:]
    history = "\n".join(
        f"{'Farmer' if m.get('role') == 'user' else 'FarmWise'}: {m.get('content', '')}"
        for m in recent
    )

    if not use_rag:
        prompt = f"Conversation so far:\n{history}\n\nReply as FarmWise to the farmer's last message."
        answer = llm_client.generate(
            prompt, system=PLAIN_SYSTEM_PROMPT, model=model_name, max_tokens=max_tokens
        )
        return RagAnswer(
            question=latest, answer=answer, used_context=False, sources=[], model=model_name
        )

    store = store or get_default_store()
    k = top_k if top_k is not None else config.TOP_K

    # A follow-up such as "what about in the north?" means little on its own, so
    # also search with the previous question attached and keep the best matches.
    queries = [latest]
    if len(user_turns) > 1:
        queries.append(f"{user_turns[-2]['content']} {latest}")
    results = _retrieve(queries, store, k)

    earlier = "\n".join(
        f"{'Farmer' if m.get('role') == 'user' else 'FarmWise'}: {m.get('content', '')}"
        for m in messages[:-1][-6:]
    )
    return _grounded_answer(
        latest, results, model_name=model_name, max_tokens=max_tokens, history=earlier
    )
