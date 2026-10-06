"""Central configuration for the Banking FAQ Chatbot.

Every tunable value lives here and can be overridden with an environment
variable, or with a line in a local `.env` file. This keeps paths and model
names out of the rest of the code instead of hardcoding them in several files.

Environment variables:
    BANKING_CHROMA_DIR   Where the persistent ChromaDB lives.
    BANKING_DATASET      Path to the banking FAQ CSV.
    BANKING_COLLECTION   ChromaDB collection name.
    BANKING_EMBED_MODEL  Sentence Transformers model name.
    BANKING_LLM_MODEL    Default Ollama model.
    BANKING_TOP_K        How many FAQ chunks to retrieve per question.
    BANKING_OLLAMA_HOST  Ollama server URL.
    BANKING_OLLAMA_KEEP_ALIVE  How long Ollama keeps the model in memory.
    BANKING_LOG_LEVEL    DEBUG / INFO / WARNING / ERROR.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

# src/config.py -> src -> project root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
ENV_FILE: Path = PROJECT_ROOT / ".env"


def _load_dotenv_if_present() -> None:
    """Load a local `.env` file when python-dotenv is installed.

    python-dotenv is optional. Without it the defaults below are used and
    real environment variables still work.
    """

    if not ENV_FILE.is_file():
        return

    try:
        from dotenv import load_dotenv
    except ImportError:
        # Not an error: the project runs fine without a .env file.
        return

    # override=False keeps real environment variables winning over the file.
    load_dotenv(ENV_FILE, override=False)


_load_dotenv_if_present()


def _env_path(name: str, default: Path) -> Path:
    """Read a filesystem path from the environment, falling back to a default."""

    raw = os.getenv(name)

    if not raw:
        return default

    path = Path(raw).expanduser()

    # Relative paths resolve from the project root, so the app behaves the
    # same no matter which folder it is launched from.
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def _env_int(name: str, default: int) -> int:
    """Read an integer from the environment, falling back to a default."""

    try:
        return int(os.getenv(name, default))
    except ValueError:
        logging.getLogger(__name__).warning(
            "Ignoring invalid integer for %s; using %s.", name, default
        )
        return default


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CHROMA_DIR: Path = _env_path("BANKING_CHROMA_DIR", PROJECT_ROOT / "chroma_db")
DATASET_PATH: Path = _env_path(
    "BANKING_DATASET",
    PROJECT_ROOT / "data" / "raw" / "banking_knowledge_base_1000.csv",
)

# ---------------------------------------------------------------------------
# Model settings
# ---------------------------------------------------------------------------

COLLECTION_NAME: str = os.getenv("BANKING_COLLECTION", "banking_faq")
EMBEDDING_MODEL_NAME: str = os.getenv("BANKING_EMBED_MODEL", "all-MiniLM-L6-v2")
LLM_MODEL_NAME: str = os.getenv("BANKING_LLM_MODEL", "llama3.2:3b")
OLLAMA_HOST: str = os.getenv("BANKING_OLLAMA_HOST", "http://localhost:11434")
OLLAMA_KEEP_ALIVE: str = os.getenv("BANKING_OLLAMA_KEEP_ALIVE", "30m")
TOP_K: int = _env_int("BANKING_TOP_K", 3)

# Options shown in the Streamlit sidebar.
AVAILABLE_LLM_MODELS: tuple[str, ...] = ("llama3.2:3b", "gemma2:2b")

# Sentence Transformers is chatty and contacts the Hugging Face Hub on every
# model load. These must be set before that library is imported.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

LOG_LEVEL_NAME: str = os.getenv("BANKING_LOG_LEVEL", "INFO").upper()
LOG_LEVEL: int = getattr(logging, LOG_LEVEL_NAME, logging.INFO)

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
# These libraries log a line for every request, which drowns out our own logs.
logging.getLogger("httpx").setLevel(logging.WARNING)

logger: logging.Logger = logging.getLogger("banking_chatbot")