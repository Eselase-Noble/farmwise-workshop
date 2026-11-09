#!/usr/bin/env python
"""Pre-flight check before the live demo.

Usage:
    python scripts/check_setup.py

Verifies, with a clear PASS/FAIL per item:
  1. Python dependencies import
  2. Ollama server is running
  3. The LLM and embedding models are pulled
  4. The crop dataset is present
  5. The trained model exists
  6. The installed scikit-learn matches the version that saved the model
  7. The RAG index has been built

It also warns when the sample guides, which are not real agronomic advice, are
still in the guides folder.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from farmwise import config  # noqa: E402

OK = "  [ OK ]"
FAIL = "  [FAIL]"
WARN = "  [WARN]"


def _check_imports() -> bool:
    try:
        import chromadb  # noqa: F401
        import joblib  # noqa: F401
        import pandas  # noqa: F401
        import pypdf  # noqa: F401
        import sklearn  # noqa: F401

        print(f"{OK} Python dependencies import")
        return True
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"{FAIL} importing dependencies: {exc}")
        print("        Fix: uv pip install -r requirements.txt")
        return False


def _check_ollama() -> bool:
    from farmwise.llm.client import is_ollama_running

    if is_ollama_running():
        print(f"{OK} Ollama server reachable at {config.OLLAMA_HOST}")
        return True
    print(f"{FAIL} Ollama not reachable at {config.OLLAMA_HOST}")
    print("        Fix: run  ollama serve")
    return False


def _check_models(ollama_up: bool) -> bool:
    if not ollama_up:
        print(f"{WARN} skipping model check (Ollama not running)")
        return False
    from farmwise.llm.client import list_models, model_available

    available = list_models()
    ok = True
    for model in (config.LLM_MODEL, config.EMBEDDING_MODEL):
        if model_available(model, available):
            print(f"{OK} model present: {model}")
        else:
            print(f"{FAIL} model missing: {model}")
            print(f"        Fix: ollama pull {model}")
            ok = False
    return ok


def _check_path(label: str, path: Path, fix: str) -> bool:
    if path.exists():
        print(f"{OK} {label}: {path}")
        return True
    print(f"{FAIL} {label} missing: {path}")
    print(f"        Fix: {fix}")
    return False


def _check_index() -> bool:
    try:
        from farmwise.rag.vector_store import ChromaVectorStore

        count = ChromaVectorStore().count()
    except Exception as exc:
        print(f"{FAIL} could not open vector store: {exc}")
        return False
    if count > 0:
        print(f"{OK} RAG index built ({count} chunks)")
        return True
    print(f"{FAIL} RAG index empty")
    print("        Fix: python scripts/build_index.py")
    return False


def _check_model_version() -> bool:
    """A model saved by another scikit-learn version can warn or fail on load."""
    import json

    import sklearn

    if not config.MODEL_METADATA_PATH.exists():
        print(f"{WARN} no model metadata, skipping the scikit-learn version check")
        return True
    saved = json.loads(config.MODEL_METADATA_PATH.read_text()).get("sklearn_version")
    if saved == sklearn.__version__:
        print(f"{OK} scikit-learn {saved} matches the saved model")
        return True
    print(f"{FAIL} model saved with scikit-learn {saved}, installed {sklearn.__version__}")
    print("        Fix: pip install -r requirements.txt, or python scripts/train_model.py")
    return False


def _warn_about_sample_guides() -> None:
    samples = sorted(config.GUIDES_DIR.glob("sample_*"))
    if samples and config.INCLUDE_SAMPLE_GUIDES:
        names = ", ".join(path.name for path in samples)
        print(f"{WARN} sample guides are not real agronomic advice: {names}")


def main() -> int:
    print("FarmWise setup check\n")
    deps = _check_imports()
    ollama_up = _check_ollama()
    models_ok = _check_models(ollama_up)
    dataset_ok = _check_path(
        "crop dataset",
        config.CROP_DATASET_PATH,
        "download it into data/raw/ (see data/raw/README.md)",
    )
    model_ok = _check_path(
        "trained model", config.MODEL_PATH, "python scripts/train_model.py"
    )
    version_ok = _check_model_version() if deps else False
    index_ok = _check_index()
    _warn_about_sample_guides()

    print()
    all_ok = all([deps, ollama_up, models_ok, dataset_ok, model_ok, version_ok, index_ok])
    if all_ok:
        print("All checks passed — ready to demo.")
        return 0
    print("Some checks failed — see the fixes above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
