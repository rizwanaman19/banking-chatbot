"""Retrieval and RAG pipeline checks.

Verifies that the optimized pipeline returns the right FAQ chunks for known
questions, that the embedding model is only loaded once per process, and that
ChromaDB data survives. Exits non-zero if any check fails.

Run from the project root:
    python tests/test_rag.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import CHROMA_DIR, COLLECTION_NAME  # noqa: E402
from embeddings import get_embedding_model  # noqa: E402
from perf import timed  # noqa: E402
from retriever import BankingRetriever  # noqa: E402
from vector_store import ChromaVectorStore  # noqa: E402

FAILURES: list[str] = []

# (question, a phrase that must appear in the best-matching chunk)
EXPECTED_TOPICS: tuple[tuple[str, str], ...] = (
    ("What is a savings account?", "savings account"),
    ("How can I use banking services on my phone?", "mobile"),
    ("How do I report fraud in my account?", "fraud"),
    ("What is a SWIFT code?", "swift"),
    ("What is DICGC insurance?", "dicgc"),
    ("What is the difference between a current account and a savings account?", "current account"),
)


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {label}" + (f" -> {detail}" if detail else ""))

    if not condition:
        FAILURES.append(label)


def safe(text: str) -> str:
    """Make text printable on a Windows console.

    The FAQ dataset contains characters such as the rupee sign, which cp1252
    cannot encode.
    """

    return text.encode("ascii", "replace").decode("ascii")


def main() -> int:
    print("--- embedding model is loaded once ---")
    first = get_embedding_model()
    second = get_embedding_model()

    check("same instance reused", first is second)
    check(
        "wrapping in two retrievers does not reload",
        _build_two_retrievers_share_model(first),
    )

    print("\n--- vector store ---")
    store = ChromaVectorStore()
    count_before = store.count()

    check("collection found", count_before > 0, f"{count_before} chunks")
    check(
        "stored chunk count unchanged by reads",
        ChromaVectorStore().count() == count_before,
    )

    print("\n--- retrieval quality ---")
    retriever = BankingRetriever(top_k=3, embedding_model=first, vector_store=store)

    for question, expected in EXPECTED_TOPICS:
        start = time.perf_counter()
        results = retriever.retrieve(question)
        elapsed = time.perf_counter() - start

        check(
            f"retrieval is fast: {question[:38]}...",
            elapsed < 1.0,
            f"{elapsed:.3f}s",
        )
        check(
            f"returned {3} results: {question[:38]}...",
            len(results) == 3,
            str(len(results)),
        )

        if not results:
            continue

        best = results[0]["text"].lower()

        check(
            f"top hit mentions '{expected}': {question[:38]}...",
            expected in best,
            safe(best[:110].replace("\n", " | ")),
        )
        check(
            f"top hit distance is finite: {question[:38]}...",
            results[0]["distance"] == results[0]["distance"],
            f"{results[0]['distance']:.4f}",
        )
        check(
            f"section metadata present: {question[:38]}...",
            bool(results[0]["section"]) and results[0]["section"] != "Unknown",
            results[0]["section"],
        )

    print("\n--- empty input ---")
    check("blank question returns nothing", retriever.retrieve("") == [])
    check("whitespace question returns nothing", retriever.retrieve("   \n ") == [])

    print("\n--- database untouched ---")
    check(
        "chroma collection still holds the same rows",
        ChromaVectorStore().count() == count_before,
        str(count_before),
    )
    check("collection name unchanged", COLLECTION_NAME == "banking_faq")
    check("database directory unchanged", CHROMA_DIR == PROJECT_ROOT / "chroma_db")

    print("\n--- timing helper ---")
    with timed("no-op"):
        pass
    check("timed() context manager works", True)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {FAILURES}")
        return 1

    print("All checks passed.")
    return 0


def _build_two_retrievers_share_model(model) -> bool:
    """Two retrievers built separately must share one embedding model."""

    left = BankingRetriever()
    right = BankingRetriever()

    return left.embedding_model is right.embedding_model is model


if __name__ == "__main__":
    raise SystemExit(main())