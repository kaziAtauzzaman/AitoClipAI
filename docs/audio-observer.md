# Audio Observer

The `src/audio_observer/` package implements audio analysis as a normal
Observer Engine observer. It does not modify the engine and returns the existing
`ObserverResult` contract consumed by aggregation.

## Architecture

The package separates responsibilities into focused services:

- `AudioObserver` implements `observers.Observer` and coordinates the workflow.
- `AudioExtractor` resolves an audio artifact from `ObserverContext`.
- `ContextAudioExtractor` uses `ObserverContext.source_path` as the audio file.
- `FFmpegAudioExtractor` converts context media into deterministic PCM WAV
  audio through an injectable command runner.
- `FFmpegAudioExtractorConfig` controls sample rate, channels, output location,
  overwrite behavior, and the FFmpeg executable name.
- `AudioLoader` loads normalized samples from an `AudioSource`.
- `WavAudioLoader` loads PCM WAV files using only the Python standard library.
- `TimestampGenerator` creates deterministic analysis windows.
- `LoudnessAnalyzer` measures RMS dBFS loudness.
- `SilenceDetector` emits low-loudness intervals.
- `PeakDetector` emits high-amplitude peaks.
- `SpeakingIntensityAnalyzer` emits active speaking-intensity windows.
- `AudioAnalyzer` composes the focused analysis services.
- `AudioObserverConfig` owns thresholds and window settings.

This keeps future FFmpeg extraction, model-based diarization, speech activity
detection, or ML loudness models pluggable behind the same extractor, loader,
and analyzer interfaces.

## Execution Flow

1. `AudioObserver.observe()` receives an `ObserverContext`.
2. The extractor resolves an `AudioSource`.
3. The loader returns normalized mono `AudioData`.
4. `AudioAnalyzer` runs loudness, silence, peak, and speaking-intensity
   analysis.
5. The observer maps analysis outputs into `Observation` objects.
6. The observer returns an `ObserverResult` with metadata describing the audio
   input and thresholds used.

The observer emits these observation types:

- `loudness`: whole-file loudness and peak amplitude.
- `silence`: detected low-loudness intervals.
- `peak`: high-amplitude events.
- `speaking_intensity`: active windows with normalized intensity values.

## Extension Points

Use dependency injection to replace infrastructure without changing
`AudioObserver`:

- Replace `AudioExtractor` to extract audio from video, remote media, or cached
  artifacts.
- Use `FFmpegAudioExtractor` for downloaded video or media paths. It validates
  FFmpeg availability and emits `FFmpegNotFoundError` or
  `AudioExtractionError` for expected infrastructure failures.
- Replace `AudioLoader` to support non-WAV formats or external decoders.
- Replace `AudioAnalyzer` or its focused services to use more advanced DSP or
  ML models.
- Adjust `AudioObserverConfig` for different content types or quality targets.

## Error Handling

Audio-specific failures raise `AudioObserverError`. When the observer is run by
`ObserverEngine`, the engine captures that failure as an
`ObserverExecutionFailure` and continues running remaining observers.

## Future Implementation Guide

Future audio models should return neutral `Observation` objects rather than
clip decisions. Keep model-specific details in `value` and `metadata`, and let
downstream aggregation, scoring, and clipping stages decide how to use them.
