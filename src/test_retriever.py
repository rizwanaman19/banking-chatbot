"""Manual retrieval check.

Verifies that a question is embedded, matched against ChromaDB, and returns
sensible FAQ chunks. Run it after building the database:

    python src/test_retriever.py

Exits non-zero if retrieval fails, so it can also be used in CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import TOP_K, logger  # noqa: E402
from embeddings import get_embedding_model  # noqa: E402
from retriever import BankingRetriever  # noqa: E402

TEST_QUESTION = "How can I use banking services on my phone?"


def main() -> int:
    """Run one retrieval and print the results. Returns an exit code."""

    retriever = BankingRetriever(top_k=TOP_K, embedding_model=get_embedding_model())

    print(f"\nQuestion:\n{TEST_QUESTION}")

    results = retriever.retrieve(TEST_QUESTION)

    if not results:
        logger.error("No results returned.")
        print("\nNo results returned.")
        return 1

    print(f"\nTop {len(results)} retrieved results:\n")

    for index, result in enumerate(results, start=1):
        print("-" * 60)
        print(f"Result {index}")
        print(f"ID: {result['id']}")
        print(f"Section: {result['section']}")
        print(f"Distance: {result['distance']:.4f}")
        print(f"\n{result['text']}\n")

    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())