"""Very small timing helper used to find slow steps.

Timing is off by default. Set `BANKING_TIMING=1` (or `BANKING_LOG_LEVEL=DEBUG`)
to print how long each step took. This keeps normal runs quiet while making it
easy to see exactly where the time goes:

    set BANKING_TIMING=1
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Iterator

from config import logger

TIMING_ENABLED: bool = os.getenv("BANKING_TIMING", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


@contextmanager
def timed(stage: str) -> Iterator[None]:
    """Measure a block of code and log how long it took.

    Args:
        stage: Short label describing the step, e.g. "retrieval".
    """

    if not TIMING_ENABLED:
        yield
        return

    start = time.perf_counter()

    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("TIMING %-22s %7.3fs", stage, elapsed)


def log_timing(stage: str, seconds: float) -> None:
    """Log a single measured duration, if timing is enabled."""

    if TIMING_ENABLED:
        logger.info("TIMING %-22s %7.3fs", stage, seconds)