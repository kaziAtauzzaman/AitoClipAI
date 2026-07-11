"""Observer engine lifecycle and execution contracts."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from core import ObserverResult


@dataclass(frozen=True, slots=True)
class ObserverContext:
    """Input context shared with observers during one engine run.

    Attributes:
        source_path: Optional path to source media or an artifact.
        source: Optional in-memory source object for future workflows.
        metadata: Workflow metadata available to observers.
    """

    source_path: Path | None = None
    source: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ObserverExecutionFailure:
    """Failure captured for one observer without stopping the engine."""

    observer: str
    error_type: str
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ObserverEngineResult:
    """Result of executing all registered observers."""

    results: list[ObserverResult] = field(default_factory=list)
    failures: list[ObserverExecutionFailure] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def observer_results(self) -> list[ObserverResult]:
        """Alias for compatibility with timeline aggregation terminology."""

        return self.results


class ObserverTelemetryHook(Protocol):
    """Extension point for logging, metrics, tracing, and future telemetry."""

    def before_observer(self, observer_name: str, context: ObserverContext) -> None:
        """Called immediately before an observer runs."""

    def after_observer(
        self,
        observer_name: str,
        context: ObserverContext,
        result: ObserverResult,
    ) -> None:
        """Called after an observer returns a valid result."""

    def on_observer_error(
        self,
        observer_name: str,
        context: ObserverContext,
        failure: ObserverExecutionFailure,
    ) -> None:
        """Called when observer setup, execution, validation, or teardown fails."""


class NullObserverTelemetryHook:
    """No-op telemetry hook used when no extension hook is injected."""

    def before_observer(self, observer_name: str, context: ObserverContext) -> None:
        """Ignore observer start events."""

    def after_observer(
        self,
        observer_name: str,
        context: ObserverContext,
        result: ObserverResult,
    ) -> None:
        """Ignore observer success events."""

    def on_observer_error(
        self,
        observer_name: str,
        context: ObserverContext,
        failure: ObserverExecutionFailure,
    ) -> None:
        """Ignore observer failure events."""
