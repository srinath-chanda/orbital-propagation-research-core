"""Logging helpers for experiment runs."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def create_run_logger(
    log_path: str | Path, *, console: bool = True
) -> logging.Logger:
    """Create a logger that writes both to the terminal and a run-log file."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger_name = f"research_core.run.{path.parent.name}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        fmt="%(asctime)sZ | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def close_run_logger(logger: logging.Logger) -> None:
    """Flush and close every handler attached to a run logger."""
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)
