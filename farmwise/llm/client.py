"""A thin wrapper around a local Ollama server.

Only what the demo needs: :func:`generate` (single prompt) and :func:`chat`
(message list), plus helpers to check the server is up and list models.

If Ollama isn't running (or a model isn't pulled) you get a clear, actionable
error instead of a raw connection traceback — important for a live demo.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Iterator, Optional

import requests

from farmwise import config

logger = logging.getLogger("farmwise.llm")


class OllamaError(RuntimeError):
    """Base class for Ollama problems."""


class OllamaNotRunningError(OllamaError):
    """The Ollama server could not be reached."""


class OllamaModelNotFoundError(OllamaError):
    """A requested model is not available locally."""


_NOT_RUNNING_HINT = (
    "Could not reach the Ollama server at {host}.\n"
    "Start it in a terminal with:  ollama serve\n"
    "(Verify with:  python scripts/check_setup.py)"
)

_MODEL_HINT = (
    "Ollama model '{model}' is not available.\n"
    "Pull it with:  ollama pull {model}"
)


def _post(path: str, payload: dict, host: Optional[str] = None, timeout: Optional[int] = None) -> dict:
    """POST JSON to the Ollama server and return the decoded response.

    A transient failure — the server still warming up, a dropped connection, or a
    5xx — is retried up to ``config.OLLAMA_MAX_RETRIES`` times with a short linear
    backoff. Generation has no side effects, so a retry only ever repeats work.
    A model-not-found (404) or a timeout are not transient, so they fail fast.
    """
    base = host or config.OLLAMA_HOST
    url = f"{base.rstrip('/')}{path}"
    read_timeout = timeout if timeout is not None else config.OLLAMA_TIMEOUT
    timeouts = (config.OLLAMA_CONNECT_TIMEOUT, read_timeout)

    attempts = config.OLLAMA_MAX_RETRIES + 1
    for attempt in range(1, attempts + 1):
        try:
            resp = requests.post(url, json=payload, timeout=timeouts)
        except requests.exceptions.ConnectionError as exc:
            if attempt < attempts:
                logger.warning("Ollama connection failed (attempt %d/%d), retrying", attempt, attempts)
                time.sleep(config.OLLAMA_RETRY_BACKOFF * attempt)
                continue
            raise OllamaNotRunningError(_NOT_RUNNING_HINT.format(host=base)) from exc
        except requests.exceptions.Timeout as exc:
            raise OllamaError(
                f"Ollama request timed out after {read_timeout}s. The model may be "
                "loading for the first time; try again in a moment."
            ) from exc

        if resp.status_code == 404:
            model = payload.get("model", "?")
            raise OllamaModelNotFoundError(_MODEL_HINT.format(model=model))
        if resp.status_code >= 500 and attempt < attempts:
            logger.warning("Ollama returned %d (attempt %d/%d), retrying", resp.status_code, attempt, attempts)
            time.sleep(config.OLLAMA_RETRY_BACKOFF * attempt)
            continue
        if not resp.ok:
            raise OllamaError(f"Ollama returned {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        # Ollama reports model-not-found via a JSON 'error' field too.
        if isinstance(data, dict) and data.get("error"):
            err = data["error"]
            if "not found" in err.lower():
                raise OllamaModelNotFoundError(_MODEL_HINT.format(model=payload.get("model", "?")))
            raise OllamaError(f"Ollama error: {err}")
        return data

    # Unreachable: the loop either returns or raises on the final attempt.
    raise OllamaError("Ollama request failed.")


def is_ollama_running(host: Optional[str] = None) -> bool:
    """Return True if the Ollama server responds to a tags request."""
    base = host or config.OLLAMA_HOST
    try:
        resp = requests.get(f"{base.rstrip('/')}/api/tags", timeout=5)
        return resp.ok
    except requests.exceptions.RequestException:
        return False


def list_models(host: Optional[str] = None) -> list[str]:
    """Return the names of models available locally."""
    base = host or config.OLLAMA_HOST
    try:
        resp = requests.get(f"{base.rstrip('/')}/api/tags", timeout=10)
    except requests.exceptions.ConnectionError as exc:
        raise OllamaNotRunningError(_NOT_RUNNING_HINT.format(host=base)) from exc
    resp.raise_for_status()
    return [m["name"] for m in resp.json().get("models", [])]


def model_available(model: str, available: list[str]) -> bool:
    """Return True if ``model`` is pulled. A name with no tag means ``:latest``.

    Tags are compared exactly, so ``llama3.2:1b`` does not satisfy ``llama3.2:3b``.
    """

    def normalise(name: str) -> str:
        return name if ":" in name else f"{name}:latest"

    return normalise(model) in {normalise(name) for name in available}


def generate(
    prompt: str,
    *,
    system: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    repeat_penalty: Optional[float] = None,
    host: Optional[str] = None,
) -> str:
    """Generate a completion for a single prompt. Returns the text response."""
    options = {
        "num_predict": max_tokens if max_tokens is not None else config.MAX_TOKENS,
        "temperature": temperature if temperature is not None else config.LLM_TEMPERATURE,
    }
    if repeat_penalty is not None:
        options["repeat_penalty"] = repeat_penalty
    payload = {
        "model": model or config.LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": config.OLLAMA_KEEP_ALIVE,
        "options": options,
    }
    if system:
        payload["system"] = system
    data = _post("/api/generate", payload, host=host)
    return (data.get("response") or "").strip()


def generate_stream(
    prompt: str,
    *,
    system: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    repeat_penalty: Optional[float] = None,
    host: Optional[str] = None,
) -> Iterator[str]:
    """Yield the completion for a prompt token-by-token as Ollama produces it.

    Unlike :func:`generate` this does not retry: a stream cannot be safely
    resumed mid-way. Connection and model errors raise before any token is
    yielded, so the caller can still turn an early failure into a clean message.
    """
    options = {
        "num_predict": max_tokens if max_tokens is not None else config.MAX_TOKENS,
        "temperature": temperature if temperature is not None else config.LLM_TEMPERATURE,
    }
    if repeat_penalty is not None:
        options["repeat_penalty"] = repeat_penalty
    payload = {
        "model": model or config.LLM_MODEL,
        "prompt": prompt,
        "stream": True,
        "keep_alive": config.OLLAMA_KEEP_ALIVE,
        "options": options,
    }
    if system:
        payload["system"] = system

    base = host or config.OLLAMA_HOST
    url = f"{base.rstrip('/')}/api/generate"
    try:
        resp = requests.post(url, json=payload, stream=True, timeout=(config.OLLAMA_CONNECT_TIMEOUT, config.OLLAMA_TIMEOUT))
    except requests.exceptions.ConnectionError as exc:
        raise OllamaNotRunningError(_NOT_RUNNING_HINT.format(host=base)) from exc
    except requests.exceptions.Timeout as exc:
        raise OllamaError(f"Ollama request timed out after {config.OLLAMA_TIMEOUT}s.") from exc

    if resp.status_code == 404:
        raise OllamaModelNotFoundError(_MODEL_HINT.format(model=payload["model"]))
    if not resp.ok:
        raise OllamaError(f"Ollama returned {resp.status_code}: {resp.text[:300]}")

    for line in resp.iter_lines():
        if not line:
            continue
        try:
            data = json.loads(line)
        except ValueError:
            continue
        if data.get("error"):
            err = data["error"]
            if "not found" in err.lower():
                raise OllamaModelNotFoundError(_MODEL_HINT.format(model=payload["model"]))
            raise OllamaError(f"Ollama error: {err}")
        chunk = data.get("response")
        if chunk:
            yield chunk
        if data.get("done"):
            break


def chat(
    messages: list[dict],
    *,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    host: Optional[str] = None,
) -> str:
    """Chat completion over a list of ``{"role", "content"}`` messages."""
    payload = {
        "model": model or config.LLM_MODEL,
        "messages": messages,
        "stream": False,
        "keep_alive": config.OLLAMA_KEEP_ALIVE,
        "options": {
            "num_predict": max_tokens if max_tokens is not None else config.MAX_TOKENS,
            "temperature": temperature if temperature is not None else config.LLM_TEMPERATURE,
        },
    }
    data = _post("/api/chat", payload, host=host)
    return (data.get("message", {}).get("content") or "").strip()
