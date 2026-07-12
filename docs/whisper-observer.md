# Whisper Observer

The `src/whisper_observer/` package integrates timestamped speech
transcription with the existing observer engine. `WhisperObserver` reads the
extracted WAV path from `ObserverContext.source_path` and emits neutral
`Observation` objects with type `speech`.

`TranscriptionBackend` is the dependency-injection boundary. Backends return a
normalized `TranscriptionResult` containing timestamped `TranscriptionSegment`
objects. Segment speaker labels, confidence, and backend metadata are preserved
on the resulting observations.

`OpenAIWhisperBackend` is an optional adapter that lazy-loads the `whisper`
Python package. `WhisperObserverConfig` selects the model, language, task,
device, execution order, deterministic behavior, and additional backend
options. Models are cached per model name and device within a backend instance.

Deterministic mode is enabled by default. The OpenAI Whisper backend uses a
scalar `temperature=0.0` to disable stochastic temperature fallback, preserves
`condition_on_previous_text=True`, and uses `fp16=False` when the loaded model
runs on CPU. Other configured backend options remain intact. Set
`deterministic=False` to retain Whisper's legacy fallback behavior when
reproducibility is not required. Deterministic mode does not change global
PyTorch random state or thread configuration.

Expected failures use `TranscriptionError`, `WhisperUnavailableError`, and
`InvalidTranscriptionError`. The observer engine isolates these failures in the
same way as other observer failures.

Whisper is not registered in the default pipeline automatically because its
runtime and model are optional. Inject an `ObserverEngine` containing
`WhisperObserver` into `PipelineOrchestrator` to add transcription without
changing pipeline orchestration or aggregation.
