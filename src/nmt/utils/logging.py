"""Logging configuration for the NMT package."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_str: Optional[str] = None,
) -> None:
    """Configure the root logger.

    Args:
        level: Log level string ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        log_file: Optional path to write log output to a file.
        format_str: Optional custom format string.
    """
    fmt = format_str or "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
        force=True,
    )

    # Suppress overly verbose third-party loggers
    for noisy in ("transformers", "datasets", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
