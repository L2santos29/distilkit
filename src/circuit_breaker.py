"""Circuit breaker — prevents cascading failures from repeated I/O errors.

``CircuitBreaker`` tracks consecutive failures of a wrapped callable.
After *failure_threshold* failures the circuit *opens* and subsequent
calls are short-circuited (raising ``CircuitOpenError``) instead of
executing the failing operation, giving the downstream system time to
recover.  After *recovery_timeout* seconds the circuit transitions to
*half-open* and allows a single probe call to decide whether to close
or open again.
"""

import time
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


class CircuitOpenError(Exception):
    """Raised when a call is blocked because the circuit is open."""


class CircuitBreaker:
    """Consecutive-failure circuit breaker with automatic recovery.

    Thread-safe for typical GIL-guarded use (all state mutations are
    plain attribute assignments).

    Parameters
    ----------
    failure_threshold:
        Number of consecutive failures before the circuit opens.
    recovery_timeout:
        Seconds to wait before transitioning to half-open.
    name:
        Optional label (included in log messages and exception text)
        to identify which circuit fired.
    """

    __slots__ = (
        "_failure_threshold",
        "_recovery_timeout",
        "_name",
        "_failure_count",
        "_state",
        "_last_failure_time",
    )

    OPEN = "open"
    HALF_OPEN = "half_open"
    CLOSED = "closed"

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        name: str = "",
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._name = name or "unnamed"
        self._failure_count = 0
        self._state = self.CLOSED
        self._last_failure_time = 0.0

    # ── Public API ────────────────────────────────────────────────────

    @property
    def state(self) -> str:
        """Current circuit state: ``closed``, ``open``, or ``half_open``."""
        self._maybe_half_open()
        return self._state

    @property
    def failure_count(self) -> int:
        return self._failure_count

    def call(self, fn: F, *args: Any, **kwargs: Any) -> Any:
        """Execute *fn* if the circuit is closed/half-open; raise otherwise.

        Returns the result of *fn* on success.

        Raises
        ------
        CircuitOpenError
            If the circuit is open and the recovery timeout has not elapsed.
        """
        self._maybe_half_open()

        if self._state == self.OPEN:
            remaining = max(0, self._recovery_timeout - (time.time() - self._last_failure_time))
            raise CircuitOpenError(
                f"Circuit [{self._name}] is open — "
                f"retry in {remaining:.0f}s "
                f"({self._failure_count} consecutive failures)"
            )

        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            self._on_failure()
            raise exc

        self._on_success()
        return result

    def reset(self) -> None:
        """Manually close the circuit and clear the failure count."""
        self._failure_count = 0
        self._state = self.CLOSED
        self._last_failure_time = 0.0

    # ── Internal helpers ──────────────────────────────────────────────

    def _on_success(self) -> None:
        if self._state == self.HALF_OPEN:
            # A single success in half-open closes the circuit.
            self.reset()
        elif self._state == self.CLOSED:
            self._failure_count = 0

    def _on_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.time()
        if self._failure_count >= self._failure_threshold:
            self._state = self.OPEN

    def _maybe_half_open(self) -> None:
        if self._state == self.OPEN:
            elapsed = time.time() - self._last_failure_time
            if elapsed >= self._recovery_timeout:
                self._state = self.HALF_OPEN
