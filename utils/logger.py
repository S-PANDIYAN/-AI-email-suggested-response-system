"""
Centralised logging.

Why not ``print``?  print() has no levels, no timestamps, no module
attribution, and can't be silenced or redirected per-environment. A pipeline
that may call an LLM API needs an audit trail (which provider ran? did a
fallback trigger?). ``logging`` gives that for free.

Usage
-----
    from utils.logger import get_logger
    log = get_logger(__name__)
    log.info("retrieved %d neighbours", k)
"""
from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    """Install a single root handler. Idempotent (safe to call repeatedly)."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()
    root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger (configures root on first use)."""
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
