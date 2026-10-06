"""Turn documents into retrieval chunks.

Each FAQ row is already a self-contained Question + Answer pair, so one
document equals one chunk. Splitting further would fragment the answer and
make retrieval less accurate.
"""

from __future__ import annotations

import sys
from pathlib import Path

from config import DATASET_PATH, logger
from document_loader import create_documents, load_banking_data


def create_chunks(documents: list[dict]) -> list[dict]:
    """Return one chunk per document.

    Args:
        documents: Documents from `create_documents`.

    Returns:
        A list of {"text": str, "metadata": dict} chunks.
    """

    chunks = [
        {"text": document["text"], "metadata": document["metadata"]}
        for document in documents
        if document.get("text", "").strip()
    ]

    logger.debug("Created %d chunks.", len(chunks))

    return chunks


def main() -> int:
    """Print a short summary of the chunking result. Returns an exit code."""

    documents = create_documents(load_banking_data(DATASET_PATH))
    chunks = create_chunks(documents)

    print(f"Total documents: {len(documents)}")
    print(f"Total chunks: {len(chunks)}")
    print("\nFirst chunk:")
    print(chunks[0]["text"])
    print("\nChunk metadata:")
    print(chunks[0]["metadata"])

    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())