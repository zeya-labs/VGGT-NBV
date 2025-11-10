"""Centralized logging helpers for the NBV framework."""

from __future__ import annotations

import builtins
import logging
import os
import sys
from typing import Optional

_LOGGER_NAME = "nbv"
_LOGGING_INITIALIZED = False
_ORIGINAL_PRINT = builtins.print


def configure_logging(
    *,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    rank: int = 0,
    enable_console: bool = True,
    replace_print: bool = True,
) -> logging.Logger:
    """Configure the shared NBV logger once per process."""
    global _LOGGING_INITIALIZED

    logger = logging.getLogger(_LOGGER_NAME)
    if not _LOGGING_INITIALIZED:
        logger.setLevel(level)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        if log_file:
            log_path = os.path.abspath(log_file)
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        if enable_console:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

        logger.propagate = False
        _LOGGING_INITIALIZED = True

    if replace_print:
        _override_print(logger, rank)

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a child logger under the shared NBV namespace."""
    if not name:
        return logging.getLogger(_LOGGER_NAME)
    return logging.getLogger(f"{_LOGGER_NAME}.{name}")


def _override_print(logger: logging.Logger, rank: int) -> None:
    """Redirect built-in print to the shared logger for stdout/stderr."""

    def _logged_print(*args, sep: str = " ", end: str = "\n", file=None, flush: bool = False):
        if file not in (None, sys.stdout, sys.stderr):
            _ORIGINAL_PRINT(*args, sep=sep, end=end, file=file, flush=flush)
            return

        message = sep.join(str(arg) for arg in args)
        if end and end != "\n":
            message = f"{message}{end}"
        logger.info("[rank%d] %s", rank, message)

    builtins.print = _logged_print
