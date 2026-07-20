"""Centralized application settings, sourced from environment variables.

All configuration lives here instead of being scattered across modules.
Create a ``.env`` file in the project root to override defaults (see
``.env.example`` for the full list).

Usage::

    from src.settings import settings

    device = settings.device
    if settings.api_only:
        ...
"""

import os
from dataclasses import dataclass


def _bool_env(key: str, default: bool) -> bool:
    """Parse an environment variable as a boolean.

    ``"1"``, ``"true"``, ``"yes"`` (case-insensitive) → ``True``.
    Everything else → ``False``.
    """
    return os.environ.get(key, str(default)).lower() in ("1", "true", "yes")


@dataclass
class Settings:
    """Settings bag populated from environment variables.

    Every field has a sensible default so the project works out of the box
    without any configuration.

    Fields are modifiable at runtime (e.g. ``launch()`` may toggle
    ``api_only`` after import).
    """

    # ── Device ────────────────────────────────────────────────────
    # Target device for training/inference: "cpu", "cuda", "npu"
    device: str = "cpu"

    # ── Server ────────────────────────────────────────────────────
    # Host address to bind the web server
    host: str = "127.0.0.1"
    # Port to bind the web server
    port: int = 7860
    # If true, only expose the REST API (no frontend)
    api_only: bool = False

    # ── Paths ─────────────────────────────────────────────────────
    # Directory for persisted training-run history (JSON files)
    runs_dir: str = "runs"
    # Root directory for downloaded datasets
    data_dir: str = "./data"
    # Directory for exported models and checkpoints
    checkpoints_dir: str = "checkpoints"

    # ── Logging ───────────────────────────────────────────────────
    # Maximum size of the in-memory training log buffer (characters)
    max_log_size: int = 100_000
    # Log level: DEBUG, INFO, WARNING, ERROR
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Settings":
        """Build a ``Settings`` instance from the current environment."""
        return cls(
            device=os.environ.get("DEVICE", "cpu"),
            host=os.environ.get("HOST", "127.0.0.1"),
            port=int(os.environ.get("PORT", "7860")),
            api_only=_bool_env("API_ONLY", False),
            runs_dir=os.environ.get("RUNS_DIR", "runs"),
            data_dir=os.environ.get("DATA_DIR", "./data"),
            checkpoints_dir=os.environ.get("CHECKPOINTS_DIR", "checkpoints"),
            max_log_size=int(os.environ.get("MAX_LOG_SIZE", "100000")),
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
        )


# Singleton — import this from anywhere in the application.
settings = Settings.from_env()
