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

import re
import threading
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Optional

from farmwise import config
from farmwise.i18n import DEFAULT_LANGUAGE, language_directive, normalise_language
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
    "Write as a professional agricultural advisor: clear, formal, and practical. "
    "State the guidance directly, with no preamble such as 'Based on the provided "
    "context' or 'According to the passages'. Do NOT put citation markers such as "
    "[1] or [2], do NOT name source files, and do NOT refer to datasets, figures, "
    "tables, reports, passages or 'the guide'; the sources are listed separately "
    "for the reader. When giving a crop's requirements, describe them as "
    "recommended growing conditions, not as raw statistics. If the question covers "
    "more than one crop, address each crop in its own short paragraph beginning "
    "with the crop name, and never mix one crop's figures into another's. "
    "Write plain sentences with no Markdown formatting: no asterisks (*), hashes "
    "(#), backticks or bullet characters."
)


def _with_language(system_prompt: str, language: str | None) -> str:
    """Append a language directive to a system prompt, or return it unchanged."""
    directive = language_directive(language)
    return f"{system_prompt}\n\n{directive}" if directive else system_prompt


# llama3.2:3b tends to loop on low-resource languages; a repetition penalty on
# non-English generation curbs the runaway repeats. English is left untouched.
_NON_ENGLISH_REPEAT_PENALTY = 1.3


def _repeat_penalty(language: str | None) -> Optional[float]:
    return None if normalise_language(language) == DEFAULT_LANGUAGE else _NON_ENGLISH_REPEAT_PENALTY


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


# --- answer tidying --------------------------------------------------------

# Small models sometimes ignore the "no citation markers" rule. Sources are shown
# separately, so drop markers like [2] and openers like "According to Figure 4 of
# the report," that point at passages the reader cannot see.
_MARKER = re.compile(r"\s*\[\d+(?:\s*[,-]\s*\d+)*\]")
_POINTER_WORDS = (
    r"(?:passages?|context|figures?|tables?|guides?|reports?|data|documents?|sources?|"
    r"statistics|survey|facts|text|information)"
)
_OPENER = re.compile(
    r"(^|(?<=[.!?]\s)|(?<=\n))"
    r"(?:according to|based on|as (?:stated|shown|mentioned|noted) in)\b"
    r"[^.,\n]{0,160}?\b" + _POINTER_WORDS + r"\b[^.,\n]{0,160}?,\s*",
    re.IGNORECASE,
)


def tidy_answer(answer: str) -> str:
    """Remove citation markers and "according to the passage" style openers."""
    text = _MARKER.sub("", answer)
    text = _OPENER.sub(lambda m: m.group(1), text)
    # Re-capitalise a sentence whose opener was removed.
    text = re.sub(r"(^|[.!?]\s+|\n)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


# --- repeated questions ----------------------------------------------------

# A first grounded question (no earlier turns) gets the same answer each time while
# the knowledge base is unchanged, so recent ones are kept and repeats answered at
# once. The key includes the index size, and the web app clears the cache whenever
# it changes the knowledge base.
_CACHE_SIZE = 128
_answer_cache: "OrderedDict[tuple, dict]" = OrderedDict()
_cache_lock = threading.Lock()


def clear_answer_cache() -> None:
    with _cache_lock:
        _answer_cache.clear()


def _cache_key(question: str, store: VectorStore, k: int, model_name: str, language) -> tuple:
    normalised = " ".join(question.lower().split())
    return (
        normalised, language or DEFAULT_LANGUAGE, model_name, k,
        config.MIN_RELEVANCE_SCORE, store.count(),
    )


def _cache_get(key: tuple) -> Optional[RagAnswer]:
    with _cache_lock:
        if key not in _answer_cache:
            return None
        _answer_cache.move_to_end(key)
        return _answer_cache[key]


def _cache_put(key: tuple, answer: RagAnswer) -> None:
    with _cache_lock:
        _answer_cache[key] = answer
        _answer_cache.move_to_end(key)
        while len(_answer_cache) > _CACHE_SIZE:
            _answer_cache.popitem(last=False)


# --- waiting for the model -------------------------------------------------

# The laptop answers one question at a time. Count the answers in progress so a
# waiting user is told how many are ahead instead of watching dots.
_in_progress = 0
_in_progress_lock = threading.Lock()


def _tokens_with_queue_status(stream_factory):
    """Yield a status event if others are being answered, then the model's tokens."""
    global _in_progress
    with _in_progress_lock:
        ahead = _in_progress
        _in_progress += 1
    try:
        if ahead:
            noun = "question" if ahead == 1 else "questions"
            yield {
                "type": "status",
                "ahead": ahead,
                "message": f"FarmWise is busy: {ahead} {noun} ahead of yours. "
                           "Each takes about 20 to 30 seconds.",
            }
        for delta in stream_factory():
            yield {"type": "token", "text": delta}
    finally:
        with _in_progress_lock:
            _in_progress -= 1


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
    language: str | None = None,
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
        prompt, system=_with_language(RAG_SYSTEM_PROMPT, language),
        model=model_name, max_tokens=max_tokens, repeat_penalty=_repeat_penalty(language),
    )
    return _finish(question, answer, results, model_name)


def _finish(question: str, answer: str, results: list[SearchResult], model_name: str) -> RagAnswer:
    """Tidy a grounded answer and attach its sources (none for a refusal)."""
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
        question=question, answer=tidy_answer(answer), used_context=True,
        sources=sources, model=model_name,
    )


# --- entry points ----------------------------------------------------------

def warm_up() -> None:
    """Load both models into memory and prime the grounded system prompt.

    Ollama reuses the work done on a prompt's opening, so running the RAG system
    prompt once makes the first real question as fast as the later ones.
    """
    embed_query("warm up")
    llm_client.generate("Reply with OK.", system=RAG_SYSTEM_PROMPT, max_tokens=1)


def ask_without_rag(
    question: str,
    *,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    language: str | None = None,
) -> RagAnswer:
    """Answer from the LLM's own knowledge — no documents, no retrieval."""
    model_name = model or config.LLM_MODEL
    answer = llm_client.generate(
        question, system=_with_language(PLAIN_SYSTEM_PROMPT, language),
        model=model_name, max_tokens=max_tokens, repeat_penalty=_repeat_penalty(language),
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
    language: str | None = None,
) -> RagAnswer:
    """Answer the question with RAG: retrieve, ground, cite.

    Returns a :class:`RagAnswer` whose ``sources`` are the chunks the answer was
    grounded in. If nothing relevant is found the model is not called, and if the
    model finds the passages do not cover the question, ``sources`` is empty.
    """
    store = store or get_default_store()
    k = top_k if top_k is not None else config.TOP_K
    model_name = model or config.LLM_MODEL

    key = _cache_key(question, store, k, model_name, language) if max_tokens is None else None
    if key and (cached := _cache_get(key)):
        return cached

    results = _retrieve([question], store, k)
    answer = _grounded_answer(
        question, results, model_name=model_name, max_tokens=max_tokens, language=language
    )
    if key:
        _cache_put(key, answer)
    return answer


def chat(
    messages: list[dict],
    *,
    use_rag: bool = True,
    store: Optional[VectorStore] = None,
    top_k: Optional[int] = None,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    language: str | None = None,
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
            prompt, system=_with_language(PLAIN_SYSTEM_PROMPT, language),
            model=model_name, max_tokens=max_tokens, repeat_penalty=_repeat_penalty(language),
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
        latest, results, model_name=model_name, max_tokens=max_tokens,
        history=earlier, language=language,
    )


def chat_stream(
    messages: list[dict],
    *,
    use_rag: bool = True,
    store: Optional[VectorStore] = None,
    top_k: Optional[int] = None,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    language: str | None = None,
):
    """Streaming twin of :func:`chat`. Yields event dicts as the answer forms:

    * ``{"type": "token", "text": ...}`` — a piece of the answer
    * ``{"type": "done", "answer", "used_context", "sources", "model"}`` — the end
    * ``{"type": "error", "detail": ...}`` — Ollama was unreachable or failed

    The grounding and citation rules match :func:`chat` exactly; only the
    delivery differs. Sources are known only once the full answer is in (a
    refusal lists none), so they ride on the final ``done`` event.
    """
    model_name = model or config.LLM_MODEL
    user_turns = [m for m in messages if m.get("role") == "user"]
    latest = user_turns[-1]["content"] if user_turns else ""
    recent = messages[-6:]
    history = "\n".join(
        f"{'Farmer' if m.get('role') == 'user' else 'FarmWise'}: {m.get('content', '')}"
        for m in recent
    )

    def _done(answer: str, used_context: bool, sources: list[SourceChunk]) -> dict:
        return {
            "type": "done",
            "answer": answer,
            "used_context": used_context,
            "sources": [s.to_dict() for s in sources],
            "model": model_name,
        }

    try:
        if not use_rag:
            prompt = f"Conversation so far:\n{history}\n\nReply as FarmWise to the farmer's last message."
            parts: list[str] = []
            for event in _tokens_with_queue_status(lambda: llm_client.generate_stream(
                prompt, system=_with_language(PLAIN_SYSTEM_PROMPT, language),
                model=model_name, max_tokens=max_tokens, repeat_penalty=_repeat_penalty(language),
            )):
                if event["type"] == "token":
                    parts.append(event["text"])
                yield event
            yield _done("".join(parts).strip(), False, [])
            return

        store = store or get_default_store()
        k = top_k if top_k is not None else config.TOP_K

        # Only a first question can be answered from the cache: with earlier
        # turns the conversation shapes the answer.
        key = None
        if len(user_turns) == 1 and max_tokens is None:
            key = _cache_key(latest, store, k, model_name, language)
            cached = _cache_get(key)
            if cached:
                yield {"type": "token", "text": cached.answer}
                yield _done(cached.answer, True, cached.sources)
                return
        queries = [latest]
        if len(user_turns) > 1:
            queries.append(f"{user_turns[-2]['content']} {latest}")
        results = _retrieve(queries, store, k)

        if not results:
            yield {"type": "token", "text": NO_ANSWER}
            yield _done(NO_ANSWER, True, [])
            return

        prompt = build_prompt(latest, results)
        earlier = "\n".join(
            f"{'Farmer' if m.get('role') == 'user' else 'FarmWise'}: {m.get('content', '')}"
            for m in messages[:-1][-6:]
        )
        if earlier:
            prompt = f"Conversation so far:\n{earlier}\n\n{prompt}"

        parts = []
        for event in _tokens_with_queue_status(lambda: llm_client.generate_stream(
            prompt, system=_with_language(RAG_SYSTEM_PROMPT, language),
            model=model_name, max_tokens=max_tokens, repeat_penalty=_repeat_penalty(language),
        )):
            if event["type"] == "token":
                parts.append(event["text"])
            yield event
        # The page swaps in this tidied answer when the stream ends.
        final = _finish(latest, "".join(parts).strip(), results, model_name)
        if key:
            _cache_put(key, final)
        yield _done(final.answer, True, final.sources)
    except llm_client.OllamaError as exc:
        yield {"type": "error", "detail": str(exc)}
