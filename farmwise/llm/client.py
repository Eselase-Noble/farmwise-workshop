"""A thin wrapper around a local Ollama server.

Only what the demo needs: :func:`generate` (single prompt) and :func:`chat`
(message list), plus helpers to check the server is up and list models.

If Ollama isn't running (or a model isn't pulled) you get a clear, actionable
error instead of a raw connection traceback — important for a live demo.
"""

from __future__ import annotations

from typing import Optional

import requests

from farmwise import config


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


def _post(path: str, payload: dict, host: Optional[str] = None, timeout: int = 120) -> dict:
    """POST JSON to the Ollama server and return the decoded response."""
    base = host or config.OLLAMA_HOST
    url = f"{base.rstrip('/')}{path}"
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
    except requests.exceptions.ConnectionError as exc:
        raise OllamaNotRunningError(_NOT_RUNNING_HINT.format(host=base)) from exc
    except requests.exceptions.Timeout as exc:
        raise OllamaError(f"Ollama request timed out after {timeout}s.") from exc

    if resp.status_code == 404:
        model = payload.get("model", "?")
        raise OllamaModelNotFoundError(_MODEL_HINT.format(model=model))
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
    host: Optional[str] = None,
) -> str:
    """Generate a completion for a single prompt. Returns the text response."""
    payload = {
        "model": model or config.LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens if max_tokens is not None else config.MAX_TOKENS,
            "temperature": temperature if temperature is not None else config.LLM_TEMPERATURE,
        },
    }
    if system:
        payload["system"] = system
    data = _post("/api/generate", payload, host=host)
    return (data.get("response") or "").strip()


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
        "options": {
            "num_predict": max_tokens if max_tokens is not None else config.MAX_TOKENS,
            "temperature": temperature if temperature is not None else config.LLM_TEMPERATURE,
        },
    }
    data = _post("/api/chat", payload, host=host)
    return (data.get("message", {}).get("content") or "").strip()
