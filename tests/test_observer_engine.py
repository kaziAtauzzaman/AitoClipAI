from typing import Iterable

from core import Observation, ObserverResult
from observers import (
    Observer,
    ObserverContext,
    ObserverEngine,
    ObserverExecutionFailure,
    ObserverProvider,
    ObserverRegistry,
)


class FakeObserver(Observer):
    def __init__(
        self,
        name: str,
        *,
        order: int = 1000,
        output: object | None = None,
        should_fail: bool = False,
        calls: list[str] | None = None,
    ) -> None:
        self._name = name
        self._order = order
        self._output = output
        self._should_fail = should_fail
        self._calls = calls

    @property
    def name(self) -> str:
        return self._name

    @property
    def order(self) -> int:
        return self._order

    def setup(self, context: ObserverContext) -> None:
        if self._calls is not None:
            self._calls.append(f"{self.name}:setup")

    def observe(self, context: ObserverContext) -> ObserverResult:
        if self._calls is not None:
            self._calls.append(f"{self.name}:observe")

        if self._should_fail:
            raise RuntimeError(f"{self.name} failed")

        if self._output is not None:
            return self._output  # type: ignore[return-value]

        return ObserverResult(
            observer=self.name,
            observations=[
                Observation(
                    timestamp_seconds=float(self.order),
                    observer=self.name,
                    type="fake",
                    value=True,
                )
            ],
        )

    def teardown(self, context: ObserverContext) -> None:
        if self._calls is not None:
            self._calls.append(f"{self.name}:teardown")


class FakeProvider(ObserverProvider):
    def __init__(self, observers: Iterable[Observer]) -> None:
        self._observers = list(observers)

    def get_observers(self) -> Iterable[Observer]:
        return self._observers


class FakeTelemetryHook:
    def __init__(self) -> None:
        self.events: list[str] = []

    def before_observer(self, observer_name: str, context: ObserverContext) -> None:
        self.events.append(f"before:{observer_name}")

    def after_observer(
        self,
        observer_name: str,
        context: ObserverContext,
        result: ObserverResult,
    ) -> None:
        self.events.append(f"after:{observer_name}")

    def on_observer_error(
        self,
        observer_name: str,
        context: ObserverContext,
        failure: ObserverExecutionFailure,
    ) -> None:
        self.events.append(f"error:{observer_name}:{failure.error_type}")


def test_observer_engine_executes_successful_observers() -> None:
    registry = ObserverRegistry(
        observers=[
            FakeObserver("audio", order=20),
            FakeObserver("speech", order=10),
        ]
    )

    result = ObserverEngine(registry).run(ObserverContext())

    assert [observer_result.observer for observer_result in result.results] == [
        "speech",
        "audio",
    ]
    assert result.failures == []


def test_observer_failure_does_not_stop_remaining_observers() -> None:
    registry = ObserverRegistry(
        observers=[
            FakeObserver("good-before", order=10),
            FakeObserver("bad", order=20, should_fail=True),
            FakeObserver("good-after", order=30),
        ]
    )

    result = ObserverEngine(registry).run(ObserverContext())

    assert [observer_result.observer for observer_result in result.results] == [
        "good-before",
        "good-after",
    ]
    assert len(result.failures) == 1
    assert result.failures[0].observer == "bad"
    assert result.failures[0].error_type == "RuntimeError"


def test_observer_engine_uses_deterministic_execution_order() -> None:
    registry = ObserverRegistry(
        observers=[
            FakeObserver("zeta", order=10),
            FakeObserver("alpha", order=10),
            FakeObserver("middle", order=5),
        ]
    )

    first = ObserverEngine(registry).run(ObserverContext())
    second = ObserverEngine(registry).run(ObserverContext())

    expected_order = ["middle", "alpha", "zeta"]
    assert [result.observer for result in first.results] == expected_order
    assert [result.observer for result in second.results] == expected_order


def test_observer_engine_handles_empty_observer_list() -> None:
    result = ObserverEngine(ObserverRegistry()).run(ObserverContext())

    assert result.results == []
    assert result.failures == []
    assert result.metadata == {"observer_count": 0}


def test_observer_engine_isolates_invalid_observer_outputs() -> None:
    registry = ObserverRegistry(
        observers=[
            FakeObserver("invalid", output={"not": "an observer result"}),
            FakeObserver("valid"),
        ]
    )

    result = ObserverEngine(registry).run(ObserverContext())

    assert [observer_result.observer for observer_result in result.results] == ["valid"]
    assert len(result.failures) == 1
    assert result.failures[0].observer == "invalid"
    assert result.failures[0].error_type == "InvalidObserverOutputError"


def test_observer_engine_isolates_invalid_observation_items() -> None:
    invalid_result = ObserverResult(
        observer="invalid",
        observations=["not an observation"],  # type: ignore[list-item]
    )
    registry = ObserverRegistry(observers=[FakeObserver("invalid", output=invalid_result)])

    result = ObserverEngine(registry).run(ObserverContext())

    assert result.results == []
    assert len(result.failures) == 1
    assert result.failures[0].error_type == "InvalidObserverOutputError"


def test_observer_registry_discovers_provider_observers() -> None:
    registry = ObserverRegistry(
        providers=[FakeProvider([FakeObserver("provided", order=1)])]
    )

    result = ObserverEngine(registry).run(ObserverContext())

    assert [observer_result.observer for observer_result in result.results] == [
        "provided"
    ]


def test_observer_lifecycle_and_telemetry_hooks_are_called() -> None:
    calls: list[str] = []
    telemetry = FakeTelemetryHook()
    registry = ObserverRegistry(observers=[FakeObserver("audio", calls=calls)])

    result = ObserverEngine(registry, telemetry_hook=telemetry).run(ObserverContext())

    assert result.failures == []
    assert calls == ["audio:setup", "audio:observe", "audio:teardown"]
    assert telemetry.events == ["before:audio", "after:audio"]


def test_observer_telemetry_receives_failures() -> None:
    telemetry = FakeTelemetryHook()
    registry = ObserverRegistry(observers=[FakeObserver("bad", should_fail=True)])

    result = ObserverEngine(registry, telemetry_hook=telemetry).run(ObserverContext())

    assert len(result.failures) == 1
    assert telemetry.events == ["before:bad", "error:bad:RuntimeError"]
