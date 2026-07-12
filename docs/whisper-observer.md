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
device, execution order, and additional backend options. Models are cached per
model name and device within a backend instance.

Expected failures use `TranscriptionError`, `WhisperUnavailableError`, and
`InvalidTranscriptionError`. The observer engine isolates these failures in the
same way as other observer failures.

Whisper is not registered in the default pipeline automatically because its
runtime and model are optional. Inject an `ObserverEngine` containing
`WhisperObserver` into `PipelineOrchestrator` to add transcription without
changing pipeline orchestration or aggregation.
