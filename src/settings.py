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

    # ── Security ──────────────────────────────────────────────────
    # API key for protecting endpoints. Empty = no auth (local mode).
    # Set via the API_KEY environment variable.
    api_key: str = ""
    # CORS allowed origins. Comma-separated list.
    # Set via the CORS_ORIGINS environment variable.
    cors_origins: str = "http://127.0.0.1:7860"
    # Rate limit: max requests per minute per IP. 0 = disabled.
    # Set via the RATE_LIMIT_PER_MINUTE environment variable.
    rate_limit_per_minute: int = 30
    # HSTS max-age in seconds (set >0 to enable). Only meaningful behind HTTPS.
    # Set via the HSTS_MAX_AGE environment variable.
    hsts_max_age: int = 0
    # Path to SSL certificate file for HTTPS. Empty = no TLS.
    # Set via the SSL_CERTFILE environment variable.
    ssl_certfile: str = ""
    # Path to SSL private key file for HTTPS.
    # Set via the SSL_KEYFILE environment variable.
    ssl_keyfile: str = ""

    # ── Monitoring ────────────────────────────────────────────────
    # Webhook URL for alert notifications (Slack, Discord, etc.).
    # Empty = alerts are logged only.
    # Set via the ALERT_WEBHOOK_URL environment variable.
    alert_webhook_url: str = ""

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
            api_key=os.environ.get("API_KEY", ""),
            cors_origins=os.environ.get("CORS_ORIGINS", "http://127.0.0.1:7860"),
            rate_limit_per_minute=int(os.environ.get("RATE_LIMIT_PER_MINUTE", "30")),
            hsts_max_age=int(os.environ.get("HSTS_MAX_AGE", "0")),
            ssl_certfile=os.environ.get("SSL_CERTFILE", ""),
            ssl_keyfile=os.environ.get("SSL_KEYFILE", ""),
            alert_webhook_url=os.environ.get("ALERT_WEBHOOK_URL", ""),
        )


# Singleton — import this from anywhere in the application.
settings = Settings.from_env()
