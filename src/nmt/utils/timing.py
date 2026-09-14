"""Timing utilities for profiling training and inference."""

import logging
import time
from contextlib import contextmanager
from typing import Dict, Generator, Optional

logger = logging.getLogger(__name__)


class Timer:
    """Accumulates elapsed time across multiple start/stop calls.

    Usage::

        timer = Timer()
        timer.start()
        # ... do work ...
        elapsed = timer.stop()  # seconds

        # Or as a context manager:
        with timer:
            # ... do work ...
    """

    def __init__(self) -> None:
        self._start: Optional[float] = None
        self.elapsed: float = 0.0
        self.laps: Dict[str, float] = {}

    def start(self) -> None:
        """Start or resume the timer."""
        self._start = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer and return elapsed seconds since last start."""
        if self._start is None:
            raise RuntimeError("Timer was not started.")
        delta = time.perf_counter() - self._start
        self.elapsed += delta
        self._start = None
        return delta

    def lap(self, name: str) -> float:
        """Record a named lap time (does not stop the timer).

        Args:
            name: Label for this lap.

        Returns:
            Elapsed seconds since last start.
        """
        if self._start is None:
            raise RuntimeError("Timer is not running.")
        delta = time.perf_counter() - self._start
        self.laps[name] = delta
        logger.debug("Lap '%s': %.4fs", name, delta)
        return delta

    def reset(self) -> None:
        """Reset the timer to zero."""
        self._start = None
        self.elapsed = 0.0
        self.laps = {}

    def __enter__(self) -> "Timer":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()


@contextmanager
def timed(label: str = "block") -> Generator[None, None, None]:
    """Context manager that logs the execution time of a code block.

    Args:
        label: Label to display in the log message.

    Usage::

        with timed("data loading"):
            dataset = load_data(...)
    """
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    logger.info("[%s] completed in %.3fs", label, elapsed)
