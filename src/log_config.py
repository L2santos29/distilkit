"""Logging configuration for DistilKit.

Usage:
    from src.log_config import logger
    logger.info("Training started")
    logger.error("Something went wrong")
"""

import logging
import sys


def setup_logger(name: str = "distilkit", level: int = logging.INFO) -> logging.Logger:
    """Create a logger that writes to stderr with a clean format.

    Args:
        name: Logger name.
        level: Logging level (default: INFO).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers when the module is reloaded
    if logger.handlers:
        return logger

    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)

    fmt = logging.Formatter(
        "%(message)s",  # Just the message — clean output
    )
    handler.setFormatter(fmt)

    logger.addHandler(handler)
    return logger


logger = setup_logger()
