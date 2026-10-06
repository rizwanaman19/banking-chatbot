"""Streamlit chat UI for the Banking FAQ Assistant.

Startup cost is paid once and reused:

- The Sentence Transformer, ChromaDB handle, and Ollama client are all built
  inside `@st.cache_resource`, so Streamlit reruns reuse them instead of
  reloading the models.
- The pipeline is warmed up right away so the first question is not the slow
  one.

Conversation state lives in `st.session_state`, which survives reruns.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import (  # noqa: E402
    AVAILABLE_LLM_MODELS,
    LLM_MODEL_NAME,
    logger,
)
from embeddings import get_embedding_model  # noqa: E402
from llm import BankingLLM, LLMUnavailableError  # noqa: E402
from rag import BankingRAG  # noqa: E402
from retriever import BankingRetriever  # noqa: E402
from vector_store import ChromaVectorStore, VectorStoreError  # noqa: E402

# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Banking FAQ Assistant",
    page_icon="🏦",
    layout="centered",
)

st.markdown(
    """
    <style>
        .block-container {
            max-width: 900px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        [data-testid="stChatMessage"] {
            border-radius: 12px;
        }

        .app-subtitle {
            color: #777;
            margin-top: -12px;
            margin-bottom: 24px;
        }

        .privacy-note {
            font-size: 0.85rem;
            color: #777;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# Cached resources
#
# Everything expensive lives in these two functions. Streamlit keeps the
# returned objects in memory and hands back the same object on every rerun.
# ---------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_retriever() -> BankingRetriever:
    """Load the embedding model and open ChromaDB once per server process."""

    logger.info("Building retriever (first call only).")

    # The embedding model is cached process-wide by get_embedding_model, so
    # switching the LLM model does not reload it.
    retriever = BankingRetriever(
        embedding_model=get_embedding_model(),
        vector_store=ChromaVectorStore(),
    )

    logger.info("Retriever ready.")
    return retriever


@st.cache_resource(show_spinner="Loading the local language model...")
def get_llm(model_name: str) -> BankingLLM:
    """Create the Ollama wrapper and load the model into memory once."""

    logger.info("Preparing Ollama model '%s'.", model_name)

    llm = BankingLLM(model_name=model_name)
    llm.warm_up()

    return llm


@st.cache_resource(show_spinner=False)
def get_rag(model_name: str) -> BankingRAG:
    """Assemble the RAG pipeline for one LLM model, reusing cached pieces."""

    logger.info("Building RAG pipeline for '%s'.", model_name)

    rag = BankingRAG(
        model_name=model_name,
        retriever=get_retriever(),
        llm=get_llm(model_name),
    )

    logger.info("RAG pipeline ready for '%s'.", model_name)
    return rag


# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
with st.sidebar:
    st.title("🏦 Banking FAQ")
    st.markdown("### About this assistant")

    st.write(
        "Ask general questions about banking, accounts, "
        "digital banking, security, and financial terminology "
        "covered by the FAQ dataset."
    )

    st.divider()

    default_index = (
        AVAILABLE_LLM_MODELS.index(LLM_MODEL_NAME)
        if LLM_MODEL_NAME in AVAILABLE_LLM_MODELS
        else 0
    )

    model_name = st.selectbox(
        "Local language model",
        options=list(AVAILABLE_LLM_MODELS),
        index=default_index,
        help="Choose a model that is already installed in Ollama.",
    )

    st.caption("The selected model runs through your local Ollama setup.")

    st.divider()

    st.markdown("### Privacy and safety")
    st.write(
        "This chatbot does not connect to your bank account "
        "or retrieve personal balances, statements, or transactions."
    )
    st.warning(
        "Never enter your PIN, password, OTP, CVV, "
        "or other confidential banking credentials."
    )

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------
# Warm-up
#
# Runs once. Doing it here means the loading cost is paid before the first
# question instead of during it.
# ---------------------------------------------------------------------
warmup_error: str | None = None

if not st.session_state.get("warmed_up"):
    with st.spinner("Preparing the chatbot..."):
        try:
            get_rag(model_name)
            st.session_state["warmed_up"] = True
        except VectorStoreError as error:
            warmup_error = str(error)
            logger.error("Vector database unavailable: %s", error)
        except LLMUnavailableError as error:
            warmup_error = str(error)
            logger.error("Ollama unavailable: %s", error)
        except Exception as error:  # noqa: BLE001 - shown to the user, logged in full
            logger.exception("Unexpected startup failure.")
            warmup_error = str(error)

# ---------------------------------------------------------------------
# Main interface
# ---------------------------------------------------------------------
st.title("Banking FAQ Assistant")
st.markdown(
    '<p class="app-subtitle">'
    "Your local AI assistant for general banking questions."
    "</p>",
    unsafe_allow_html=True,
)

if warmup_error:
    st.error(warmup_error)
    st.info(
        "If the vector database is missing, build it once with:\n\n"
        "```\npython src/ingest.py\n```"
    )
else:
    st.info(
        "Answers are generated using your banking FAQ dataset. "
        "If the dataset does not contain the answer, the assistant "
        "may be unable to help."
    )

# Show suggested questions when the chat is empty.
if not st.session_state.messages and not warmup_error:
    st.markdown("#### Try asking")

    suggestions = [
        "What is a savings account?",
        "What is the difference between current and savings accounts?",
        "How can I use banking services on my phone?",
        "How do I report fraud in my account?",
    ]

    columns = st.columns(2)

    for index, suggestion in enumerate(suggestions):
        if columns[index % 2].button(
            suggestion,
            key=f"suggestion_{index}",
            use_container_width=True,
        ):
            st.session_state.pending_question = suggestion
            st.rerun()

# Show the conversation history.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept a typed question or a clicked suggestion.
typed_question = st.chat_input("Ask a general banking question...")

pending_question = st.session_state.pop("pending_question", None)
question = typed_question or pending_question

if question and not warmup_error:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching your FAQs and generating an answer..."):
                rag = get_rag(model_name)
                answer = rag.answer_question(question)

            st.markdown(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )

        except LLMUnavailableError:
            # The LLM layer already produces a user-friendly message.
            logger.exception("Ollama call failed.")
            st.error(
                "The local language model is not available right now. "
                "Check that Ollama is running and the selected model "
                "is installed."
            )

        except VectorStoreError:
            logger.exception("Vector database unavailable.")
            st.error(
                "The FAQ vector database is unavailable. "
                "Run `python src/ingest.py` to rebuild it."
            )

        except Exception:  # noqa: BLE001 - logged in full, summarised for the user
            # The traceback goes to the terminal; the chat window only shows
            # a short, friendly message.
            logger.exception("Unexpected error while answering the question.")
            st.error(
                "The chatbot could not process your question. "
                "Check the terminal for the technical error details."
            )

st.divider()

st.markdown(
    '<p class="privacy-note">'
    "For general information only. Confirm important banking "
    "decisions with your bank through its official channels."
    "</p>",
    unsafe_allow_html=True,
)