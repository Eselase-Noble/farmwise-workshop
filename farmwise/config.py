"""Central configuration for FarmWise.

Everything that might change between a laptop demo (Session 1) and a server
deployment (Session 2) lives here and is driven by environment variables with
sensible local defaults. Import these constants rather than hard-coding paths
or model names elsewhere.

    from farmwise import config
    print(config.LLM_MODEL, config.TOP_K)

Override any value without touching code, e.g.::

    export FARMWISE_LLM_MODEL=llama3.2:1b
    export FARMWISE_TOP_K=5
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# The package lives at <project_root>/farmwise/, so the project root is one up.
PACKAGE_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = PACKAGE_DIR.parent


def _env_path(name: str, default: Path) -> Path:
    """Return a Path from an env var, or ``default`` if the var is unset."""
    value = os.getenv(name)
    return Path(value).expanduser() if value else default


DATA_DIR: Path = _env_path("FARMWISE_DATA_DIR", PROJECT_ROOT / "data")
RAW_DATA_DIR: Path = _env_path("FARMWISE_RAW_DATA_DIR", DATA_DIR / "raw")
GUIDES_DIR: Path = _env_path("FARMWISE_GUIDES_DIR", DATA_DIR / "guides")
MODELS_DIR: Path = _env_path("FARMWISE_MODELS_DIR", PROJECT_ROOT / "models")
CHROMA_DIR: Path = _env_path("FARMWISE_CHROMA_DIR", PROJECT_ROOT / "chroma")

# Specific artifacts
CROP_DATASET_PATH: Path = _env_path(
    "FARMWISE_CROP_DATASET", RAW_DATA_DIR / "Crop_recommendation.csv"
)
MODEL_PATH: Path = _env_path("FARMWISE_MODEL_PATH", MODELS_DIR / "crop_model.joblib")
MODEL_METADATA_PATH: Path = _env_path(
    "FARMWISE_MODEL_METADATA", MODELS_DIR / "crop_model_metadata.json"
)

# --------------------------------------------------------------------------
# Ollama / models  (names are configurable so Session 2 can swap them)
# --------------------------------------------------------------------------
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL: str = os.getenv("FARMWISE_LLM_MODEL", "llama3.2:3b")
EMBEDDING_MODEL: str = os.getenv("FARMWISE_EMBEDDING_MODEL", "nomic-embed-text")

# --------------------------------------------------------------------------
# RAG parameters  (kept small for fast responses on a laptop)
# --------------------------------------------------------------------------
CHUNK_SIZE: int = int(os.getenv("FARMWISE_CHUNK_SIZE", "800"))          # characters
CHUNK_OVERLAP: int = int(os.getenv("FARMWISE_CHUNK_OVERLAP", "150"))    # characters
TOP_K: int = int(os.getenv("FARMWISE_TOP_K", "5"))                      # chunks retrieved (Session 2 uses real, larger docs)
MAX_TOKENS: int = int(os.getenv("FARMWISE_MAX_TOKENS", "256"))         # generation cap
LLM_TEMPERATURE: float = float(os.getenv("FARMWISE_LLM_TEMPERATURE", "0.2"))
# Chunks scoring below this cosine similarity are treated as not relevant. Tuned
# for nomic-embed-text: covered questions scored 0.70+, off-topic ones up to 0.60.
MIN_RELEVANCE_SCORE: float = float(os.getenv("FARMWISE_MIN_SCORE", "0.62"))

# --------------------------------------------------------------------------
# Web app limits and access
# --------------------------------------------------------------------------
MAX_UPLOAD_MB: int = int(os.getenv("FARMWISE_MAX_UPLOAD_MB", "25"))
MAX_DOWNLOAD_MB: int = int(os.getenv("FARMWISE_MAX_DOWNLOAD_MB", "50"))
# With a token set, changing the knowledge base requires it. With no token, only
# requests made directly on this machine may change it.
ADMIN_TOKEN: str = os.getenv("FARMWISE_ADMIN_TOKEN", "")

# Vector store collection name
COLLECTION_NAME: str = os.getenv("FARMWISE_COLLECTION", "farming_guides")

# --------------------------------------------------------------------------
# Dataset provenance  (surfaced in model metadata + the Responsible-AI notebook)
# --------------------------------------------------------------------------
CROP_DATASET_SOURCE: str = os.getenv(
    "FARMWISE_DATASET_SOURCE",
    "Kaggle 'Crop Recommendation' dataset (Atharva Ingle). Collection location "
    "and methodology are undocumented; widely understood to reflect Indian "
    "agronomic conditions — NOT validated for Ghana.",
)


def as_dict() -> dict:
    """Return the resolved configuration as a plain dict (handy for debugging)."""
    return {
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "DATA_DIR": str(DATA_DIR),
        "RAW_DATA_DIR": str(RAW_DATA_DIR),
        "GUIDES_DIR": str(GUIDES_DIR),
        "MODELS_DIR": str(MODELS_DIR),
        "CHROMA_DIR": str(CHROMA_DIR),
        "CROP_DATASET_PATH": str(CROP_DATASET_PATH),
        "MODEL_PATH": str(MODEL_PATH),
        "MODEL_METADATA_PATH": str(MODEL_METADATA_PATH),
        "OLLAMA_HOST": OLLAMA_HOST,
        "LLM_MODEL": LLM_MODEL,
        "EMBEDDING_MODEL": EMBEDDING_MODEL,
        "CHUNK_SIZE": CHUNK_SIZE,
        "CHUNK_OVERLAP": CHUNK_OVERLAP,
        "TOP_K": TOP_K,
        "MAX_TOKENS": MAX_TOKENS,
        "LLM_TEMPERATURE": LLM_TEMPERATURE,
        "MIN_RELEVANCE_SCORE": MIN_RELEVANCE_SCORE,
        "MAX_UPLOAD_MB": MAX_UPLOAD_MB,
        "MAX_DOWNLOAD_MB": MAX_DOWNLOAD_MB,
        "COLLECTION_NAME": COLLECTION_NAME,
    }
