"""Observer implementation for audio features."""

import logging

from audio_observer.analysis import AudioAnalysisResult, AudioAnalyzer
from audio_observer.config import AudioObserverConfig
from audio_observer.errors import AudioObserverError
from audio_observer.extraction import AudioExtractor, ContextAudioExtractor
from audio_observer.loading import AudioLoader, WavAudioLoader
from core import Observation, ObserverResult
from observers import Observer, ObserverContext


class AudioObserver(Observer):
    """Run audio analysis and emit aggregation-compatible observations."""

    def __init__(
        self,
        config: AudioObserverConfig | None = None,
        extractor: AudioExtractor | None = None,
        loader: AudioLoader | None = None,
        analyzer: AudioAnalyzer | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._config = config or AudioObserverConfig()
        self._extractor = extractor or ContextAudioExtractor()
        self._loader = loader or WavAudioLoader()
        self._analyzer = analyzer or AudioAnalyzer()
        self._logger = logger or logging.getLogger(__name__)

    @property
    def name(self) -> str:
        """Stable observer name."""

        return self._config.observer_name

    @property
    def order(self) -> int:
        """Deterministic observer execution order."""

        return self._config.order

    def observe(self, context: ObserverContext) -> ObserverResult:
        """Analyze audio and return structured observer output."""

        try:
            source = self._extractor.extract(context)
            audio = self._loader.load(source)
            analysis = self._analyzer.analyze(audio, self._config)
        except AudioObserverError:
            raise
        except Exception as exc:
            raise AudioObserverError(f"Audio observer failed: {exc}") from exc

        self._logger.debug(
            "Audio observer analyzed %.3f seconds from %s",
            audio.duration_seconds,
            source.path,
        )

        return ObserverResult(
            observer=self.name,
            observations=self._build_observations(analysis, audio.duration_seconds),
            metadata={
                "sample_rate_hz": audio.sample_rate_hz,
                "channels": audio.channels,
                "duration_seconds": audio.duration_seconds,
                "source_path": str(source.path),
                "config": {
                    "window_seconds": self._config.window_seconds,
                    "hop_seconds": self._config.hop_seconds,
                    "silence_threshold_dbfs": self._config.silence_threshold_dbfs,
                    "peak_threshold": self._config.peak_threshold,
                    "speaking_intensity_threshold_dbfs": (
                        self._config.speaking_intensity_threshold_dbfs
                    ),
                },
            },
        )

    def _build_observations(
        self,
        analysis: AudioAnalysisResult,
        duration_seconds: float,
    ) -> list[Observation]:
        observations = [
            Observation(
                timestamp_seconds=0.0,
                duration_seconds=duration_seconds,
                observer=self.name,
                type="loudness",
                value={
                    "loudness_dbfs": analysis.loudness_dbfs,
                    "peak_amplitude": analysis.peak_amplitude,
                },
            )
        ]

        observations.extend(
            Observation(
                timestamp_seconds=segment.start_seconds,
                duration_seconds=segment.duration_seconds,
                observer=self.name,
                type="silence",
                value={"loudness_dbfs": segment.loudness_dbfs},
            )
            for segment in analysis.silence_segments
        )
        observations.extend(
            Observation(
                timestamp_seconds=event.timestamp_seconds,
                observer=self.name,
                type="peak",
                value={"amplitude": event.amplitude},
            )
            for event in analysis.peak_events
        )
        observations.extend(
            Observation(
                timestamp_seconds=window.start_seconds,
                duration_seconds=window.duration_seconds,
                observer=self.name,
                type="speaking_intensity",
                value={
                    "intensity": window.intensity,
                    "loudness_dbfs": window.loudness_dbfs,
                },
            )
            for window in analysis.speaking_intensity
        )

        return observations
