"""Error-handling checks for the optimized RAG pipeline.

Each check runs the real code paths with a deliberately broken input and
verifies that the failure is a clear, friendly message rather than a stack
trace. Exits non-zero if any check fails.

Run from the project root:
    python tests/test_error_handling.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from document_loader import create_documents, load_banking_data  # noqa: E402
from embeddings import get_embedding_model  # noqa: E402
from llm import BankingLLM, LLMUnavailableError  # noqa: E402
from rag import BankingRAG  # noqa: E402
from retriever import BankingRetriever  # noqa: E402
from vector_store import ChromaVectorStore, VectorStoreError  # noqa: E402

FAILURES: list[str] = []


@contextmanager
def temp_dir() -> Iterator[str]:
    """Yield a temporary directory that is removed even if ChromaDB holds it.

    ChromaDB keeps its SQLite file open for the life of the client, so on
    Windows the usual auto-cleanup raises PermissionError. Cleanup errors are
    therefore ignored.
    """

    path = tempfile.mkdtemp()

    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {label}" + (f" -> {detail}" if detail else ""))

    if not condition:
        FAILURES.append(label)


def main() -> int:
    print("--- missing vector database ---")
    with temp_dir() as directory:
        missing = Path(directory) / "no_such_chroma_db"
        try:
            ChromaVectorStore(persist_directory=missing)
            check("missing database raises", False, "no error raised")
        except VectorStoreError as error:
            message = str(error)
            check("missing database raises VectorStoreError", True)
            check("message mentions ingest.py", "ingest.py" in message, message)

        check(
            "no directory was created as a side effect",
            not missing.exists(),
        )

    print("\n--- empty collection ---")
    with temp_dir() as directory:
        store = ChromaVectorStore(persist_directory=directory, create_if_missing=True)
        check("fresh collection reports zero chunks", store.count() == 0)

        try:
            store.search([0.0] * 384, top_k=3)
            check("search on empty collection returns nothing", True)
        except Exception as error:  # noqa: BLE001
            check("search on empty collection returns nothing", False, str(error))

    print("\n--- unreadable / invalid dataset ---")
    try:
        load_banking_data(Path(tempfile.gettempdir()) / "definitely_missing.csv")
        check("missing CSV raises FileNotFoundError", False)
    except FileNotFoundError as error:
        check("missing CSV raises FileNotFoundError", True, str(error)[:60])

    with temp_dir() as directory:
        bad_csv = Path(directory) / "bad.csv"
        bad_csv.write_text("Foo,Bar\n1,2\n", encoding="utf-8")

        try:
            load_banking_data(bad_csv)
            check("wrong columns raise ValueError", False)
        except ValueError as error:
            check("wrong columns raise ValueError", True, str(error))

        empty_csv = Path(directory) / "empty.csv"
        empty_csv.write_text("Section,Question,Answer\n", encoding="utf-8")

        try:
            load_banking_data(empty_csv)
            check("empty dataset raises ValueError", False)
        except ValueError as error:
            check("empty dataset raises ValueError", True, str(error))

    print("\n--- dataset with blank answers is filtered ---")
    frame = pd.DataFrame(
        {
            "Section": ["A", "B", "C"],
            "Question": ["q1", "q2", "q3"],
            "Answer": ["a1", "   ", None],
        }
    )
    filtered = load_banking_data_from_frame(frame)
    documents = create_documents(filtered)
    check("blank and missing answers dropped", len(documents) == 1, str(len(documents)))

    print("\n--- empty question ---")
    retriever = BankingRetriever(embedding_model=get_embedding_model())
    check("empty question returns no documents", retriever.retrieve("   ") == [])

    rag = BankingRAG(retriever=retriever, llm=BankingLLM())
    answer = rag.answer_question("   ")
    check("empty question gets a prompt to type one", "enter a question" in answer.lower(), answer)

    print("\n--- Ollama unavailable ---")
    unreachable = BankingLLM(model_name="llama3.2:3b", client=_FakeClient())
    try:
        unreachable.warm_up()
        check("unreachable Ollama raises LLMUnavailableError", False)
    except LLMUnavailableError as error:
        check("unreachable Ollama raises LLMUnavailableError", True)
        check(
            "message names the model and suggests checking Ollama",
            "llama3.2:3b" in str(error) and "Ollama" in str(error),
            str(error),
        )

    try:
        unreachable.generate_answer("q", "ctx")
        check("failed generation raises LLMUnavailableError", False)
    except LLMUnavailableError as error:
        check("failed generation raises LLMUnavailableError", True)

    print("\n--- missing LLM model ---")
    empty_server = BankingLLM(model_name="not-installed:99b", client=_FakeClient(models=[]))
    try:
        empty_server.warm_up()
        check("missing model raises LLMUnavailableError", False)
    except LLMUnavailableError as error:
        check("missing model raises LLMUnavailableError", True)
        check("message suggests ollama pull", "ollama pull" in str(error), str(error))

    print("\n--- private data refusal needs no retrieval ---")
    refusal = rag.answer_question("What is my current balance?")
    check("private data refused", "can't access" in refusal.lower(), refusal[:80])

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {FAILURES}")
        return 1

    print("All checks passed.")
    return 0


def load_banking_data_from_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Run the loader's filtering rules on an in-memory frame."""

    import document_loader

    with temp_dir() as directory:
        path = Path(directory) / "frame.csv"
        frame.to_csv(path, index=False)
        return document_loader.load_banking_data(path)


class _FakeClient:
    """Stands in for `ollama.Client` to test the failure paths offline."""

    def __init__(self, models: list[str] | None = None) -> None:
        self._models = models if models is not None else ["llama3.2:3b"]
        self._fail = models is None

    def list(self) -> dict:
        if self._fail:
            raise ConnectionError("connection refused")
        return {"models": [{"model": name} for name in self._models]}

    def generate(self, **kwargs):
        raise ConnectionError("connection refused")

    def chat(self, **kwargs):
        raise ConnectionError("connection refused")


if __name__ == "__main__":
    raise SystemExit(main())