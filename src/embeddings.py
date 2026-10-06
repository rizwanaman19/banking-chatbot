"""Local embedding model wrapper.

The Sentence Transformer is expensive to load (roughly 10 seconds on this
laptop) so it is loaded exactly once per process and then reused. Both the
offline ingestion script and the live chatbot share this one cached instance.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from config import EMBEDDING_MODEL_NAME, logger
from perf import timed


class EmbeddingModel:
    """Thin wrapper around a local Sentence Transformers model.

    Args:
        model_name: Hugging Face model id, e.g. "all-MiniLM-L6-v2".
        device: Torch device such as "cpu" or "cuda".
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        device: str = "cpu",
    ) -> None:
        # Imported here rather than at module level so that importing this
        # file stays cheap; torch and transformers together take several
        # seconds to import.
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s (device=%s)", model_name, device)

        # local_files_only=True skips the network round-trip to the Hugging
        # Face Hub. That check alone cost ~9s per load on a warm disk. If the
        # model has never been downloaded we fall back to downloading it.
        try:
            self.model = SentenceTransformer(
                model_name, device=device, local_files_only=True
            )
        except Exception:
            logger.info(
                "Embedding model not found locally; downloading %s.", model_name
            )
            self.model = SentenceTransformer(model_name, device=device)

        logger.info("Embedding model ready.")

    def generate_embeddings(self, texts: list[str]) -> np.ndarray:
        """Convert many texts into vectors.

        Used by the offline ingestion script. Not used at chatbot runtime.
        """

        if not texts:
            return np.empty((0, self.model.get_sentence_embedding_dimension()))

        with timed(f"embed {len(texts)} texts"):
            return self.model.encode(texts, show_progress_bar=True)

    def generate_embedding(self, text: str) -> np.ndarray:
        """Convert a single text (the user's question) into a vector."""

        return self.model.encode(text)


@lru_cache(maxsize=2)
def get_embedding_model(model_name: str | None = None) -> EmbeddingModel:
    """Return a shared EmbeddingModel so the model loads only once.

    `lru_cache` keys on the arguments, so the default is `None` rather than
    the model name. Using the name as a default value would create two cache
    entries for the same model and load it twice.

    This is the single most important optimisation in the project: without it
    every new `BankingRAG` object reloads the model from disk.

    Args:
        model_name: Model to load, or None for the configured default.
    """

    return EmbeddingModel(model_name=model_name or EMBEDDING_MODEL_NAME)