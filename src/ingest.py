"""Offline ingestion: CSV -> documents -> chunks -> embeddings -> ChromaDB.

Run this script whenever the dataset changes. The chatbot itself never does
this work; at runtime it only embeds the user's question.

Usage (from the project root):
    python src/ingest.py

Re-running is safe. Chunk IDs are deterministic, so ingestion updates the same
rows instead of creating duplicates.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python src/ingest.py` as well as `python -m src.ingest`.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chunker import create_chunks  # noqa: E402
from config import DATASET_PATH, logger  # noqa: E402
from document_loader import create_documents, load_banking_data  # noqa: E402
from embeddings import get_embedding_model  # noqa: E402
from perf import timed  # noqa: E402
from vector_store import ChromaVectorStore  # noqa: E402


def chunk_id(index: int) -> str:
    """Stable ID for a chunk, matching the existing ChromaDB rows."""

    return f"banking_{index}"


def reset_collection(vector_store: ChromaVectorStore) -> None:
    """Delete the collection so it can be rebuilt from scratch.

    Only needed when the dataset shrank, because upsert cannot remove rows
    that are no longer produced by the CSV.
    """

    logger.warning("Deleting collection '%s'.", vector_store.collection_name)

    vector_store.client.delete_collection(name=vector_store.collection_name)


def main(reset: bool = False) -> int:
    """Build or refresh the vector database.

    Args:
        reset: Delete the existing collection first. Use this when the
            dataset shrank or the stored text needs to be rebuilt.

    Returns:
        A process exit code.
    """

    logger.info("Starting ingestion from %s", DATASET_PATH)

    with timed("load csv"):
        dataframe = load_banking_data(DATASET_PATH)

    logger.info("Loaded %d rows.", len(dataframe))

    documents = create_documents(dataframe)
    chunks = create_chunks(documents)

    if not chunks:
        logger.error("The dataset produced no chunks; nothing to ingest.")
        return 1

    logger.info("Prepared %d chunks.", len(chunks))

    embedding_model = get_embedding_model()

    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.generate_embeddings(texts)

    logger.info(
        "Generated embeddings with shape %s (dimension=%d).",
        embeddings.shape,
        embeddings.shape[1],
    )

    # create_if_missing=True so a first run can create the database.
    vector_store = ChromaVectorStore(create_if_missing=True)

    if reset:
        reset_collection(vector_store)
        vector_store = ChromaVectorStore(create_if_missing=True)

    ids = [chunk_id(index) for index in range(len(chunks))]

    with timed("store in chroma"):
        vector_store.add_documents(chunks, embeddings, ids)

    total = vector_store.count()
    logger.info("Ingestion complete. ChromaDB now holds %d chunks.", total)

    if total != len(chunks):
        # Not fatal, but worth a warning: it means the database holds rows
        # from a different or larger dataset.
        logger.warning(
            "ChromaDB holds %d chunks but the dataset produced %d. "
            "Run with --reset to rebuild it from scratch.",
            total,
            len(chunks),
        )

    print(f"\nIngestion complete. ChromaDB now holds {total} chunks.")

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing collection and rebuild it from scratch.",
    )
    arguments = parser.parse_args()

    raise SystemExit(main(reset=arguments.reset))