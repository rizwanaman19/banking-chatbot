"""Smoke test for the optimized Streamlit app.

Uses Streamlit's AppTest to exercise the real app end to end: startup, chat
state, a FAQ answer, the private-data refusal, and error handling.

Run from the project root:
    python tests/test_app_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "app" / "streamlit_app.py"

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from streamlit.testing.v1 import AppTest  # noqa: E402

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    """Record a pass or failure without stopping the run."""

    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" -> {detail}" if detail else ""))

    if not condition:
        FAILURES.append(label)


def last_assistant_text(at: AppTest) -> str:
    """Return the text of the most recent assistant message."""

    if not at.chat_message:
        return ""

    for message in reversed(at.chat_message):
        if message.name == "assistant":
            return " ".join(
                block.value for block in message.markdown if block.value
            )

    return ""


def main() -> int:
    at = AppTest.from_file(str(APP_PATH), default_timeout=300)

    print("\n--- startup ---")
    at.run()
    check("app starts without exceptions", not at.exception, str(at.exception))
    check("chat input is present", len(at.chat_input) == 1)
    check("no startup error banner", not at.error, str(at.error))

    print("\n--- FAQ answer ---")
    at.chat_input[0].set_value("What is a savings account?").run()
    check("no exception on question", not at.exception, str(at.exception))
    answer = last_assistant_text(at)
    check("produced an answer", len(answer) > 20, answer[:120])
    check(
        "answer mentions saving/savings",
        "saving" in answer.lower() or "interest" in answer.lower(),
        answer[:120],
    )
    check("history has 2 messages", len(at.session_state.messages) == 2)

    print("\n--- history persists across reruns ---")
    at.chat_input[0].set_value("What is a SWIFT code?").run()
    check("no exception on second question", not at.exception, str(at.exception))
    check("history has 4 messages", len(at.session_state.messages) == 4)
    check("4 chat bubbles rendered", len(at.chat_message) == 4)

    print("\n--- private data refusal ---")
    at.chat_input[0].set_value("What is my current balance?").run()
    refusal = last_assistant_text(at).lower()
    check(
        "refuses personal account data",
        "can't access" in refusal or "cannot access" in refusal,
        refusal[:120],
    )

    print("\n--- out of scope question ---")
    at.chat_input[0].set_value("What is the weather in Islamabad?").run()
    check("no exception on out-of-scope question", not at.exception, str(at.exception))
    check("answers or safely declines", len(last_assistant_text(at)) > 10)

    print("\n--- clear conversation ---")
    # at.button also contains sidebar buttons, so match on label.
    clear_button = next(b for b in at.button if "Clear conversation" in b.label)
    clear_button.click().run()
    check("history cleared", len(at.session_state.messages) == 0)
    check("no exception after clearing", not at.exception, str(at.exception))

    print("\n--- suggested question button ---")
    # Suggestions are only rendered while the conversation is empty.
    check("suggestions shown on empty chat", len(at.button) > 1)
    suggestion = next(b for b in at.button if b.key == "suggestion_0")
    suggestion.click().run()
    check("no exception after suggestion click", not at.exception, str(at.exception))
    check(
        "suggestion added a question and answer",
        len(at.session_state.messages) == 2,
        str(len(at.session_state.messages)),
    )
    check(
        "suggested question is echoed back",
        at.session_state.messages[0]["content"]
        == "What is a savings account?",
        str(at.session_state.messages[0]["content"]),
    )

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {FAILURES}")
        return 1

    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())