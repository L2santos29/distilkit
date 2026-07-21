"""In-process alert manager for DistilKit.

Provides lightweight, self-contained alerting that works without external
infrastructure.  Alerts are always logged; if ``ALERT_WEBHOOK_URL`` is
configured, they are also posted as JSON to that endpoint (Slack, Discord,
or any custom webhook).

Alert rules are evaluated on a background interval (default 60 s).
"""

import asyncio
import json
import time
import urllib.request
from typing import Any

from src.log_config import logger
from src.settings import settings

# ---------------------------------------------------------------------------
# Sliding-window counters
# ---------------------------------------------------------------------------

_error_window: list[float] = []  # timestamps of 5xx responses
_request_window: list[float] = []  # timestamps of all responses
_task_failures: dict[str, float] = {}  # task_id → timestamp of failure
_last_alert: dict[str, float] = {}  # alert_name → last fired timestamp
_consecutive_failures: int = 0  # consecutive alert cycles with issues


def record_response(status_code: int, duration: float) -> None:
    """Record an HTTP response for alert evaluation."""
    now = time.time()
    _request_window.append(now)
    if status_code >= 500:
        _error_window.append(now)


def record_task_failure(task_id: str) -> None:
    """Record a failed training task."""
    _task_failures[task_id] = time.time()


# ---------------------------------------------------------------------------
# Alert evaluation
# ---------------------------------------------------------------------------

_ALERT_SUPPRESSION_SEC = 300  # minimum seconds between re-firing the same alert


def _check_error_rate() -> str | None:
    """Return an alert message if 5xx rate exceeds threshold."""
    now = time.time()
    cutoff = now - 300  # 5-minute window

    # Purge old entries
    while _error_window and _error_window[0] < cutoff:
        _error_window.pop(0)
    while _request_window and _request_window[0] < cutoff:
        _request_window.pop(0)

    if not _request_window:
        return None

    error_rate = len(_error_window) / max(len(_request_window), 1)
    if error_rate > 0.05:  # > 5% error rate
        return (
            f"High 5xx error rate: {error_rate:.1%} "
            f"({len(_error_window)} errors in last 5 min, "
            f"{len(_request_window)} total requests)"
        )
    if len(_error_window) >= 5:  # at least 5 errors even if rate is low
        return f"Multiple 5xx errors: {len(_error_window)} in last 5 min (rate: {error_rate:.1%})"
    return None


def _check_task_failures() -> str | None:
    """Return an alert message if training tasks are failing."""
    now = time.time()
    cutoff = now - 3600  # 1-hour window

    recent = [ts for ts in _task_failures.values() if ts >= cutoff]
    if len(recent) >= 3:
        return f"{len(recent)} training tasks failed in the last hour"
    return None


def _should_suppress(alert_name: str) -> bool:
    """Prevent the same alert from firing too often."""
    now = time.time()
    last = _last_alert.get(alert_name, 0.0)
    if now - last < _ALERT_SUPPRESSION_SEC:
        return True
    _last_alert[alert_name] = now
    return False


def _post_webhook(payload: dict[str, Any]) -> None:
    """POST an alert payload to the configured webhook URL."""
    url = settings.alert_webhook_url
    if not url:
        return
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception as exc:
        logger.warning("Alert webhook failed: %s", exc)


# ---------------------------------------------------------------------------
# Background evaluation loop
# ---------------------------------------------------------------------------

_INTERVAL_SEC = 60


async def evaluate_alerts() -> None:
    """Periodic alert evaluation — runs forever as a background task."""
    while True:
        await asyncio.sleep(_INTERVAL_SEC)
        try:
            _evaluate_once()
        except Exception:
            logger.exception("Alert evaluation crashed")


def _evaluate_once() -> None:
    """Run a single alert evaluation cycle."""
    global _consecutive_failures
    alerts: list[dict[str, str]] = []

    # 1. Error rate
    msg = _check_error_rate()
    if msg and not _should_suppress("error_rate"):
        alerts.append({"name": "high_error_rate", "message": msg, "severity": "warning"})

    # 2. Task failures
    msg = _check_task_failures()
    if msg and not _should_suppress("task_failures"):
        alerts.append({"name": "task_failures", "message": msg, "severity": "warning"})

    # 3. Consecutive failure escalation
    if alerts:
        _consecutive_failures += 1
    else:
        _consecutive_failures = 0

    if _consecutive_failures >= 5 and not _should_suppress("escalation"):
        alerts.append(
            {
                "name": "sustained_issues",
                "message": f"Consecutive alert cycles: {_consecutive_failures}. "
                f"System may require manual intervention.",
                "severity": "critical",
            }
        )

    # Fire all pending alerts
    for alert in alerts:
        sev = alert["severity"].upper()
        log_msg = f"[ALERT][{sev}] {alert['message']}"
        if alert["severity"] == "critical":
            logger.critical(log_msg)
        else:
            logger.warning(log_msg)

        if settings.alert_webhook_url:
            _post_webhook(
                {
                    "text": f"*DistilKit Alert* `{alert['name']}`\n"
                    f"Severity: {sev}\n{alert['message']}\n"
                    f"Instance: {settings.host}:{settings.port}",
                }
            )
