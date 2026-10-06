"""ChromaDB vector store.

The database is persistent, so it is built once by `src/ingest.py` and then
only read at chatbot runtime. Opening it is cheap (~1s, mostly the `chromadb`
import); it never regenerates embeddings and never re-inserts documents.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from config import CHROMA_DIR, COLLECTION_NAME, logger
from perf import timed


class VectorStoreError(RuntimeError):
    """Raised when the vector database is missing or empty."""


class ChromaVectorStore:
    """Persistent ChromaDB collection used for retrieval.

    Args:
        persist_directory: Folder holding the ChromaDB files.
        collection_name: Collection inside that folder.
        create_if_missing: When False (the default) an absent or empty
            collection is an error instead of being silently created. This
            stops the app from pretending to be ready when ingestion has not
            run yet. Ingestion passes True because it is the process that is
            allowed to work with an empty collection.
    """

    def __init__(
        self,
        persist_directory: str | Path = CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
        create_if_missing: bool = False,
    ) -> None:
        import chromadb

        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name

        if not self.persist_directory.exists():
            if not create_if_missing:
                raise VectorStoreError(
                    f"Vector database not found at {self.persist_directory}. "
                    "Run `python src/ingest.py` to build it."
                )

            self.persist_directory.mkdir(parents=True, exist_ok=True)

        with timed("chroma client"):
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory)
            )

        with timed("chroma collection"):
            # embedding_function=None: we always pass our own vectors, so
            # ChromaDB must never try to load its own default ONNX model.
            if create_if_missing:
                self.collection = self.client.get_or_create_collection(
                    name=collection_name, embedding_function=None
                )
            else:
                self.collection = self.client.get_collection(
                    name=collection_name, embedding_function=None
                )

        self._count: int | None = None

        with timed("chroma count"):
            self._count = self.collection.count()

        if self._count == 0 and not create_if_missing:
            raise VectorStoreError(
                f"Collection '{collection_name}' is empty. "
                "Run `python src/ingest.py` to build it."
            )

        logger.info(
            "ChromaDB ready: %s (%d chunks) at %s",
            collection_name,
            self._count,
            self.persist_directory,
        )

    def add_documents(
        self,
        documents: list[dict],
        embeddings: np.ndarray,
        ids: list[str],
        batch_size: int = 256,
    ) -> None:
        """Insert documents and their vectors.

        Only used by the offline ingestion script. IDs are deterministic, so
        running ingestion twice updates the same rows instead of duplicating
        them.

        Args:
            documents: Chunks with "text" and "metadata" keys.
            embeddings: Matching embedding matrix.
            ids: One stable ID per chunk.
            batch_size: Rows per write; keeps memory use flat.
        """

        texts = [document["text"] for document in documents]
        metadatas = [document["metadata"] for document in documents]

        vectors = np.asarray(embeddings).tolist()

        with timed("chroma upsert"):
            self.collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=vectors,
                metadatas=metadatas,
            )

        self._count = None
        logger.info("Upserted %d documents into ChromaDB.", len(ids))

    def count(self) -> int:
        """Return the number of stored chunks."""

        if self._count is None:
            self._count = self.collection.count()

        return self._count

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> dict:
        """Return the `top_k` chunks closest to `query_embedding`."""

        total = self.count()

        if total == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        vector = np.asarray(query_embedding, dtype=float).tolist()

        with timed("retrieval"):
            return self.collection.query(
                query_embeddings=[vector],
                n_results=min(top_k, total),
                include=["documents", "metadatas", "distances"],
            )