"""Deterministic observer execution engine."""

import logging
from typing import Iterable

from core import ObserverResult
from observers.base import Observer
from observers.contracts import (
    NullObserverTelemetryHook,
    ObserverContext,
    ObserverEngineResult,
    ObserverExecutionFailure,
    ObserverTelemetryHook,
)
from observers.registry import ObserverRegistry
from observers.validation import ObserverResultValidator


class ObserverEngine:
    """Execute registered observers in deterministic order."""

    def __init__(
        self,
        registry: ObserverRegistry | None = None,
        validator: ObserverResultValidator | None = None,
        telemetry_hook: ObserverTelemetryHook | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._registry = registry or ObserverRegistry()
        self._validator = validator or ObserverResultValidator()
        self._telemetry_hook = telemetry_hook or NullObserverTelemetryHook()
        self._logger = logger or logging.getLogger(__name__)

    def run(self, context: ObserverContext) -> ObserverEngineResult:
        """Run all discovered observers and isolate per-observer failures."""

        results: list[ObserverResult] = []
        failures: list[ObserverExecutionFailure] = []

        for observer in self._ordered_observers(self._registry.discover()):
            result, failure = self._run_observer(observer, context)
            if result is not None:
                results.append(result)
            if failure is not None:
                failures.append(failure)

        return ObserverEngineResult(
            results=results,
            failures=failures,
            metadata={"observer_count": len(results) + len(failures)},
        )

    def _ordered_observers(self, observers: Iterable[Observer]) -> list[Observer]:
        return sorted(observers, key=lambda observer: (observer.order, observer.name))

    def _run_observer(
        self,
        observer: Observer,
        context: ObserverContext,
    ) -> tuple[ObserverResult | None, ObserverExecutionFailure | None]:
        observer_name = observer.name
        self._logger.debug("Starting observer %s", observer_name)
        self._telemetry_hook.before_observer(observer_name, context)

        try:
            observer.setup(context)
            result = self._validator.validate(observer_name, observer.observe(context))
            observer.teardown(context)
        except Exception as exc:
            failure = ObserverExecutionFailure(
                observer=observer_name,
                error_type=type(exc).__name__,
                message=str(exc),
            )
            self._logger.exception("Observer %s failed", observer_name)
            self._telemetry_hook.on_observer_error(observer_name, context, failure)
            return None, failure

        self._logger.debug("Finished observer %s", observer_name)
        self._telemetry_hook.after_observer(observer_name, context, result)
        return result, None
