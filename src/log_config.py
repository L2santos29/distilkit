"""Logging configuration for DistilKit.

Usage:
    from src.log_config import logger
    logger.info("Training started")
    logger.error("Something went wrong")

Contextual information (request ID, task ID) can be attached via
``logger.bind()`` which returns a ``LoggerAdapter`` with extra fields:

    ctx_log = logger.bind(request_id="abc123")
    ctx_log.info("Processing request")   # → "request_id=abc123 Processing request"
"""

import logging
import sys
from contextvars import ContextVar
from typing import Any

# Thread-safe context variable for the current request ID
_request_id: ContextVar[str] = ContextVar("_request_id", default="")


def get_request_id() -> str:
    """Return the request ID for the current thread (or empty string)."""
    return _request_id.get()


def set_request_id(rid: str) -> None:
    """Set the request ID for the current thread."""
    _request_id.set(rid)


class ContextAdapter(logging.LoggerAdapter):
    """A logger adapter that appends contextual fields to every message.

    Extra kwargs passed to ``logger.info()`` etc. are forwarded as-is;
    bound context from ``logger.bind()`` is prepended as ``key=value``
    tokens.
    """

    def __init__(self, logger: logging.Logger, context: dict[str, Any] | None = None) -> None:
        super().__init__(logger, context or {})

    def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        # Build prefix tokens from bound context
        tokens = []
        for k, v in self.extra.items():
            if v:
                tokens.append(f"{k}={v}")
        # Also include any request ID set via the context var
        rid = _request_id.get()
        if rid:
            tokens.append(f"request_id={rid}")
        prefix = " ".join(tokens) + "  " if tokens else ""
        return prefix + msg, kwargs

    def bind(self, **kwargs: Any) -> "ContextAdapter":
        """Return a new adapter with merged context (does not mutate self)."""
        merged = {**self.extra, **kwargs}
        return ContextAdapter(self.logger, merged)


def setup_logger(name: str = "distilkit", level: int = logging.INFO) -> ContextAdapter:
    """Create a logger with structured output and contextual fields.

    Args:
        name: Logger name.
        level: Logging level (default: INFO).

    Returns:
        ``ContextAdapter`` wrapping a ``logging.Logger``.
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers when the module is reloaded
    if logger.handlers:
        return ContextAdapter(logger)

    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(fmt)

    logger.addHandler(handler)
    return ContextAdapter(logger)


logger = setup_logger()
