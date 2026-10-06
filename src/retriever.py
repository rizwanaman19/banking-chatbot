"""Retrieval layer: turns a question into matching FAQ chunks."""

from __future__ import annotations

from config import TOP_K, logger
from embeddings import EmbeddingModel, get_embedding_model
from vector_store import ChromaVectorStore


class BankingRetriever:
    """Finds the most relevant banking FAQ chunks for a question.

    The embedding model and vector store are shared process-wide, so creating
    several retrievers does not reload anything.

    Args:
        top_k: Number of chunks to return.
        embedding_model: Optional pre-built model (otherwise the shared one).
        vector_store: Optional pre-built store (otherwise a fresh handle).
    """

    def __init__(
        self,
        top_k: int = TOP_K,
        embedding_model: EmbeddingModel | None = None,
        vector_store: ChromaVectorStore | None = None,
    ) -> None:
        self.top_k = top_k
        self.embedding_model = embedding_model or get_embedding_model()
        self.vector_store = vector_store or ChromaVectorStore()
        logger.info("Retriever ready (top_k=%d).", self.top_k)

    def retrieve(self, question: str) -> list[dict]:
        """Return the closest FAQ chunks for `question`.

        Only the question itself is embedded here. Document embeddings were
        already generated and stored offline.
        """

        question = question.strip()

        if not question:
            return []

        query_embedding = self.embedding_model.generate_embedding(question)

        results = self.vector_store.search(query_embedding, top_k=self.top_k)

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        ids = results["ids"][0]

        return [
            {
                "id": doc_id,
                "text": document,
                # `.get` keeps a missing metadata field from crashing a query.
                "section": (metadata or {}).get("section", "Unknown"),
                "distance": float(distance),
            }
            for doc_id, document, metadata, distance in zip(
                ids, documents, metadatas, distances
            )
        ]