"""The RAG pipeline: question -> retrieval -> context -> local LLM -> answer."""

from __future__ import annotations

from config import LLM_MODEL_NAME, TOP_K, logger
from llm import BankingLLM
from perf import timed
from retriever import BankingRetriever

# Questions about the user's own account cannot be answered from an FAQ
# dataset, so they are handled before any retrieval happens.
PRIVATE_ACCOUNT_PHRASES: tuple[str, ...] = (
    "my current balance",
    "my account balance",
    "balance in my account",
    "my bank balance",
    "my bank statement",
    "my transactions",
    "my transaction history",
    "my account details",
)

PRIVATE_ACCOUNT_ANSWER = (
    "I can't access your personal bank account or see your balance, "
    "statements, or transactions. Please check your bank's official mobile "
    "app, website, or contact its official customer support. Never share "
    "your PIN, password, or OTP."
)

NO_CONTEXT_ANSWER = (
    "I couldn't find enough relevant information to answer this question "
    "accurately."
)


class BankingRAG:
    """Coordinates retrieval and answer generation.

    Args:
        model_name: Ollama model tag.
        top_k: Number of FAQ chunks to retrieve.
        retriever: Optional shared retriever (avoids reloading the embedding
            model and re-opening ChromaDB).
        llm: Optional pre-built LLM wrapper.
    """

    def __init__(
        self,
        model_name: str = LLM_MODEL_NAME,
        top_k: int = TOP_K,
        retriever: BankingRetriever | None = None,
        llm: BankingLLM | None = None,
    ) -> None:
        with timed("pipeline init"):
            self.retriever = retriever or BankingRetriever(top_k=top_k)
            self.llm = llm or BankingLLM(model_name=model_name)

    def warm_up(self) -> None:
        """Load the embedding model, open ChromaDB, and load the LLM.

        Called once at startup so the first real question is not the slow one.
        """

        with timed("pipeline warm-up"):
            self.llm.warm_up()

    def answer_question(self, question: str) -> str:
        """Answer `question` using the banking FAQ dataset."""

        question = question.strip()

        if not question:
            return "Please enter a question."

        if self._asks_for_private_data(question):
            return PRIVATE_ACCOUNT_ANSWER

        retrieved_documents = self.retriever.retrieve(question)

        if not retrieved_documents:
            logger.info("No documents retrieved for question.")
            return NO_CONTEXT_ANSWER

        # Each chunk already starts with "Section: ...", so the text is used
        # as-is; repeating the section here would waste prompt tokens.
        context = "\n\n---\n\n".join(
            document["text"] for document in retrieved_documents
        )

        logger.debug(
            "Prompting with %d chunks (%d characters).",
            len(retrieved_documents),
            len(context),
        )

        with timed("rag total"):
            return self.llm.generate_answer(question=question, context=context)

    @staticmethod
    def _asks_for_private_data(question: str) -> bool:
        """Return True when the question is about personal account data."""

        lowered = question.lower()

        return any(phrase in lowered for phrase in PRIVATE_ACCOUNT_PHRASES)


def main() -> None:
    """Simple terminal chat loop, useful for testing without Streamlit."""

    with timed("startup"):
        rag = BankingRAG()
        rag.warm_up()

    print("\nBanking FAQ Chatbot")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if question.lower() in {"exit", "quit"}:
            print("Chatbot: Goodbye!")
            break

        print(f"\nChatbot: {rag.answer_question(question)}\n")


if __name__ == "__main__":
    main()