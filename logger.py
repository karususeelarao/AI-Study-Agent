"""
logger.py
==============================================================================
Centralized logging configuration for the AI Personal Study & Interview
Assistant.

Design:
    - One function, `get_logger(name)`, used everywhere via
      `logger = get_logger(__name__)`.
    - Logs go to BOTH a rotating file (data/app.log) and the console.
    - Rotating file handler prevents unbounded log growth in production.
    - Log level is controlled centrally via config.settings.log_level.
==============================================================================
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from config import settings

# Guard against re-configuring handlers multiple times when Streamlit
# re-executes this module on every UI interaction (Streamlit re-runs the
# whole script top-to-bottom on each widget event).
_CONFIGURED = False


def _configure_root_logger() -> None:
    """Attach a console handler and a rotating file handler to the root logger."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    root_logger = logging.getLogger()
    root_logger.setLevel(settings.log_level.upper())

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # --- Console handler (stdout) ---
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(settings.log_level.upper())

    # --- Rotating file handler: 5 MB per file, keep last 5 files ---
    file_handler = RotatingFileHandler(
        filename=str(settings.log_file),
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(settings.log_level.upper())

    # Avoid duplicate log lines if this ever runs twice.
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Quiet down noisy third-party libraries.
    for noisy_logger in ("httpx", "httpcore", "urllib3", "sentence_transformers"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a module-scoped logger, ensuring root logging is configured
    exactly once.

    Usage:
        from logger import get_logger
        logger = get_logger(__name__)
        logger.info("Something happened")
    """
    _configure_root_logger()
    return logging.getLogger(name)
