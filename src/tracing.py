"""Distributed tracing for DistilKit — W3C Trace Context + optional OpenTelemetry.

Provides end-to-end request tracing across service boundaries.  A lightweight
built-in tracer works without any extra dependencies.  When the ``opentelemetry``
packages are installed, the tracer automatically delegates to OpenTelemetry and
exports spans via OTLP.

Usage::

    from src.tracing import tracer

    with tracer.start_span("pipeline.distill") as span:
        span.set_attribute("epochs", 10)
        result = run_training()
        span.set_attribute("final_loss", result["loss"])
"""

from __future__ import annotations

import os
import time
import uuid
from contextvars import ContextVar
from typing import Any

from src.log_config import logger

# ---------------------------------------------------------------------------
# Context variable — carries the current span across async boundaries
# ---------------------------------------------------------------------------

_current_span: ContextVar["Span | None"] = ContextVar("_current_span", default=None)


def current_span() -> "Span | None":
    """Return the active span (if any) for the current async context."""
    return _current_span.get()


# ---------------------------------------------------------------------------
# Span — lightweight implementation (used when OpenTelemetry is absent)
# ---------------------------------------------------------------------------


class Span:
    """A single tracing span with start/end time, attributes, and parent linkage.

    When OpenTelemetry is available this wraps an ``opentelemetry Span``;
    otherwise it records timing in-process for log correlation.
    """

    def __init__(
        self,
        name: str,
        trace_id: str,
        span_id: str,
        parent_span_id: str | None = None,
    ) -> None:
        self.name = name
        self.trace_id = trace_id
        self.span_id = span_id
        self.parent_span_id = parent_span_id
        self._start = time.perf_counter()
        self._end: float | None = None
        self.attributes: dict[str, Any] = {}
        self._otel_span: Any = None  # set if OpenTelemetry is active

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a key-value attribute on this span."""
        self.attributes[key] = value
        if self._otel_span is not None:
            self._otel_span.set_attribute(key, value)

    def end(self) -> None:
        """Finish the span and record its duration."""
        if self._end is not None:
            return
        self._end = time.perf_counter()
        if self._otel_span is not None:
            self._otel_span.end()

    @property
    def duration_ms(self) -> float:
        """Return the span duration in milliseconds (0 if not ended)."""
        if self._end is None:
            return 0.0
        return (self._end - self._start) * 1000

    def to_traceparent(self) -> str:
        """W3C ``traceparent`` header value: ``00-{trace_id}-{span_id}-01``."""
        return f"00-{self.trace_id.replace('-', '')}-{self.span_id.replace('-', '')}-01"

    def __enter__(self) -> Span:
        return self

    def __exit__(self, *args: Any) -> None:
        self.end()

    def __repr__(self) -> str:
        return f"Span({self.name!r}, trace={self.trace_id[:8]}..., span={self.span_id[:8]}...)"


# ---------------------------------------------------------------------------
# Tracer — context manager & span factory
# ---------------------------------------------------------------------------


def _new_id() -> str:
    """Generate a 16-byte hex trace/span ID (32 hex chars, W3C-compatible)."""
    return uuid.uuid4().hex


class Tracer:
    """Tracer that creates spans and manages the active span context.

    If OpenTelemetry is installed, spans are also exported via OTLP to the
    endpoint configured in ``OTEL_EXPORTER_OTLP_ENDPOINT`` (defaults to
    ``http://localhost:4318``).
    """

    def __init__(self, service_name: str = "distilkit") -> None:
        self.service_name = service_name
        self._otel_tracer: Any = None
        self._try_init_otel()

    # ------------------------------------------------------------------
    # OpenTelemetry initialisation (best-effort)
    # ------------------------------------------------------------------

    def _try_init_otel(self) -> None:
        try:
            from opentelemetry import trace as otel_trace
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            resource = Resource.create({"service.name": self.service_name})
            provider = TracerProvider(resource=resource)

            endpoint = os.environ.get(
                "OTEL_EXPORTER_OTLP_ENDPOINT",
                "http://localhost:4318",
            )
            exporter = OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces")
            provider.add_span_processor(BatchSpanProcessor(exporter))
            otel_trace.set_tracer_provider(provider)

            self._otel_tracer = otel_trace.get_tracer(self.service_name)
            logger.info(
                "OpenTelemetry tracing enabled — exporting to %s/v1/traces",
                endpoint,
            )
        except ImportError:
            self._otel_tracer = None
        except Exception as exc:
            logger.warning("OpenTelemetry initialisation failed: %s", exc)
            self._otel_tracer = None

    # ------------------------------------------------------------------
    # Span creation
    # ------------------------------------------------------------------

    def _make_otel_span(self, name: str, parent: Span | None) -> Any | None:
        """Create an OpenTelemetry span if the SDK is loaded."""
        if self._otel_tracer is None:
            return None
        try:
            from opentelemetry import trace as otel_trace

            context = None
            if parent is not None and parent._otel_span is not None:
                # Use the parent's OpenTelemetry context
                context = otel_trace.set_span_in_context(parent._otel_span)
            return self._otel_tracer.start_span(name, context=context)
        except Exception:
            return None

    def start_span(
        self,
        name: str,
        parent: Span | None = None,
        trace_id: str | None = None,
    ) -> Span:
        """Create and activate a new span.

        The span becomes the current context span (via ``_current_span``)
        so that child operations inherit it.

        Args:
            name: Span name (e.g. ``"http.request"``, ``"pipeline.distill"``).
            parent: Explicit parent span (defaults to ``current_span()``).
            trace_id: Explicit trace ID (auto-generated if omitted).

        Returns:
            A started ``Span`` — call ``.end()`` or use as context manager.
        """
        parent = parent or current_span()
        parent_id = parent.span_id if parent else None
        trace_id = trace_id or (parent.trace_id if parent else _new_id())
        span_id = _new_id()

        span = Span(name, trace_id, span_id, parent_span_id=parent_id)

        # Hook into OpenTelemetry if available
        if self._otel_tracer is not None:
            otel_span = self._make_otel_span(name, parent)
            span._otel_span = otel_span
            if otel_span is not None:
                otel_span.set_attribute("service.name", self.service_name)

        # Make this the active span
        _current_span.set(span)
        return span

    def span_from_traceparent(self, name: str, traceparent: str) -> Span:
        """Parse a W3C ``traceparent`` header and create a child span.

        Format: ``00-{trace_id}-{parent_span_id}-{flags}``

        Args:
            name: Span name.
            traceparent: Raw ``traceparent`` header value.

        Returns:
            A new span linked to the incoming trace.
        """
        parts = traceparent.strip().split("-")
        trace_id = parts[1] if len(parts) >= 2 else _new_id()
        parent_span_id = parts[2] if len(parts) >= 3 else None
        span_id = _new_id()

        span = Span(name, trace_id, span_id, parent_span_id=parent_span_id)
        _current_span.set(span)
        return span


# ---------------------------------------------------------------------------
# Module-level singleton tracer
# ---------------------------------------------------------------------------

tracer = Tracer()
