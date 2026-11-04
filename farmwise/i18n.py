"""Local-language support for the assistant.

FarmWise answers in English by default, and can be asked to answer in a Ghanaian
language instead (Twi, Fante, Ewe or Ga). ``llama3.2:3b`` has only partial
coverage of these low-resource languages, so :func:`language_directive` appends a
short instruction plus a few aligned English -> language sentence pairs as
in-context style examples. The pairs come from ``data/language_primers.json``
(see ``scripts/build_language_primers.py``) and anchor spelling and orthography
only; they carry no agronomic content.
"""

from __future__ import annotations

import json
from functools import lru_cache

from farmwise import config

# key -> human-readable name shown in the UI and used in the prompt.
LANGUAGES: dict[str, str] = {
    "en": "English",
    "twi": "Twi",
    "fante": "Fante",
    "ewe": "Ewe",
    "ga": "Ga",
}

DEFAULT_LANGUAGE = "en"


def normalise_language(language: str | None) -> str:
    """Return a supported language key, falling back to English."""
    if not language:
        return DEFAULT_LANGUAGE
    key = language.strip().lower()
    return key if key in LANGUAGES else DEFAULT_LANGUAGE


@lru_cache(maxsize=1)
def _primers() -> dict[str, list[dict]]:
    path = config.DATA_DIR / "language_primers.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError):
        return {}


def language_directive(language: str | None) -> str:
    """An instruction (and style examples) telling the model which language to use.

    Empty for English, so the default behaviour is unchanged.
    """
    key = normalise_language(language)
    if key == DEFAULT_LANGUAGE:
        return ""

    name = LANGUAGES[key]
    # We deliberately do NOT few-shot the model with example sentences here:
    # llama3.2:3b tends to copy them verbatim (the parallel corpus is Bible
    # text, so it parrots scripture). A plain instruction keeps answers on the
    # farming topic. The primer data is still built and kept for future use
    # (e.g. fine-tuning); see scripts/build_language_primers.py and i18n_examples().
    return (
        f"Write your entire answer in {name}, using correct {name} spelling and "
        f"special characters. Answer only the farmer's question using the given "
        f"farming facts; do not change the subject."
    )


def i18n_examples(language: str | None) -> list[dict]:
    """Aligned example sentences for a language (unused in prompts; see above)."""
    return _primers().get(normalise_language(language), [])
