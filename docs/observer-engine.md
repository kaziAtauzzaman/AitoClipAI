# Observer Engine

The `src/observers/` package provides production infrastructure for running
AitoClipAI observers. It is generic by design: audio, video, OCR, caption, face,
chat, emotion, scene-change, speech, and future observers all plug into the same
engine contract.

## Architecture

The engine is split into small responsibilities:

- `Observer` defines the base interface every observer implements.
- `ObserverContext` carries source media, in-memory source data, and workflow
  metadata into an engine run.
- `ObserverRegistry` stores direct observers and discovery providers.
- `ObserverProvider` lets new observers be discovered without modifying the
  engine.
- `ObserverResultValidator` verifies that observer outputs are valid
  `ObserverResult` objects containing only `Observation` objects.
- `ObserverEngine` orders observers, runs lifecycle methods, validates output,
  isolates failures, and returns an `ObserverEngineResult`.
- `ObserverTelemetryHook` is an extension point for metrics, tracing, logs, and
  future observability.

The engine does not score, rank, filter, aggregate, or interpret observations.
Observers produce `ObserverResult` objects. The aggregation framework can then
consume `ObserverEngineResult.results` or `ObserverEngineResult.observer_results`
without compatibility changes.

## Extension Process

To add an observer:

1. Implement `Observer`.
2. Provide a stable unique `name`.
3. Optionally override `order`; lower values run first.
4. Return an `ObserverResult` from `observe()`.
5. Register the observer directly with `ObserverRegistry.register()` or expose
   it through an `ObserverProvider`.

Direct registration works for explicit wiring. Providers are preferred when a
module owns multiple observers or wants to expose observers to the engine
without changing engine code.

## Execution Flow

1. `ObserverEngine.run()` asks the registry to discover observers.
2. Observers are sorted deterministically by `(order, name)`.
3. For each observer, the engine calls `setup()`, `observe()`, validates the
   returned `ObserverResult`, then calls `teardown()`.
4. Valid results are appended to `ObserverEngineResult.results`.
5. Any setup, execution, validation, or teardown exception is captured as an
   `ObserverExecutionFailure`; the engine logs it and continues to the next
   observer.
6. Telemetry hooks receive before, after, and error callbacks.

## Future Observer Guide

Observer implementations should keep domain logic inside the observer and emit
neutral observations:

```python
from core import Observation, ObserverResult
from observers import Observer, ObserverContext


class SpeechKeywordObserver(Observer):
    @property
    def name(self) -> str:
        return "speech-keywords"

    @property
    def order(self) -> int:
        return 200

    def observe(self, context: ObserverContext) -> ObserverResult:
        return ObserverResult(
            observer=self.name,
            observations=[
                Observation(
                    timestamp_seconds=12.5,
                    observer=self.name,
                    type="keyword",
                    value="example",
                )
            ],
        )
```

Use `metadata` on `Observation`, `ObserverResult`, and `ObserverContext` for
observer-specific details. Do not place scoring, ranking, or clip-selection
decisions in observers; those belong in downstream stages.
