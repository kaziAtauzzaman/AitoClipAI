"""Audio analysis services."""

import math
from dataclasses import dataclass, field

from audio_observer.config import AudioObserverConfig
from audio_observer.contracts import AudioData
from audio_observer.timestamping import AudioWindow, TimestampGenerator


MIN_DBFS = -120.0


@dataclass(frozen=True, slots=True)
class SilenceSegment:
    """Detected silence interval."""

    start_seconds: float
    duration_seconds: float
    loudness_dbfs: float


@dataclass(frozen=True, slots=True)
class PeakEvent:
    """Detected audio peak."""

    timestamp_seconds: float
    amplitude: float


@dataclass(frozen=True, slots=True)
class SpeakingIntensityWindow:
    """Speaking intensity measured over one analysis window."""

    start_seconds: float
    duration_seconds: float
    intensity: float
    loudness_dbfs: float


@dataclass(frozen=True, slots=True)
class AudioAnalysisResult:
    """Combined output from audio analysis services."""

    loudness_dbfs: float
    peak_amplitude: float
    silence_segments: list[SilenceSegment] = field(default_factory=list)
    peak_events: list[PeakEvent] = field(default_factory=list)
    speaking_intensity: list[SpeakingIntensityWindow] = field(default_factory=list)


class LoudnessAnalyzer:
    """Measure RMS loudness in dBFS."""

    def analyze(self, samples: tuple[float, ...]) -> float:
        """Return RMS loudness for samples."""

        return rms_dbfs(samples)


class SilenceDetector:
    """Detect contiguous low-loudness windows."""

    def __init__(
        self,
        timestamp_generator: TimestampGenerator | None = None,
    ) -> None:
        self._timestamp_generator = timestamp_generator or TimestampGenerator()

    def detect(
        self,
        audio: AudioData,
        config: AudioObserverConfig,
    ) -> list[SilenceSegment]:
        """Return silence intervals using configured window thresholds."""

        windows = self._timestamp_generator.windows(
            len(audio.samples),
            audio.sample_rate_hz,
            config.window_seconds,
            config.hop_seconds,
        )
        segments: list[SilenceSegment] = []
        current_start: float | None = None
        current_end = 0.0
        loudness_values: list[float] = []

        for window in windows:
            loudness = rms_dbfs(audio.samples[window.start_index : window.end_index])
            if loudness <= config.silence_threshold_dbfs:
                if current_start is None:
                    current_start = window.start_seconds
                    loudness_values = []
                current_end = window.start_seconds + window.duration_seconds
                loudness_values.append(loudness)
            elif current_start is not None:
                self._append_segment(
                    segments,
                    current_start,
                    current_end,
                    loudness_values,
                    config.min_silence_seconds,
                )
                current_start = None
                loudness_values = []

        if current_start is not None:
            self._append_segment(
                segments,
                current_start,
                current_end,
                loudness_values,
                config.min_silence_seconds,
            )

        return segments

    def _append_segment(
        self,
        segments: list[SilenceSegment],
        start_seconds: float,
        end_seconds: float,
        loudness_values: list[float],
        min_duration_seconds: float,
    ) -> None:
        duration_seconds = max(0.0, end_seconds - start_seconds)
        if duration_seconds < min_duration_seconds:
            return

        average_loudness = (
            sum(loudness_values) / len(loudness_values)
            if loudness_values
            else MIN_DBFS
        )
        segments.append(
            SilenceSegment(
                start_seconds=start_seconds,
                duration_seconds=duration_seconds,
                loudness_dbfs=average_loudness,
            )
        )


class PeakDetector:
    """Detect high-amplitude audio peaks."""

    def detect(
        self,
        audio: AudioData,
        config: AudioObserverConfig,
    ) -> list[PeakEvent]:
        """Return peaks above threshold with minimum distance applied."""

        min_distance_samples = max(
            1,
            int(round(config.min_peak_distance_seconds * audio.sample_rate_hz)),
        )
        events: list[PeakEvent] = []
        last_peak_index = -min_distance_samples

        for sample_index, sample in enumerate(audio.samples):
            amplitude = abs(sample)
            if amplitude < config.peak_threshold:
                continue
            if sample_index - last_peak_index < min_distance_samples:
                if events and amplitude > events[-1].amplitude:
                    events[-1] = PeakEvent(
                        timestamp_seconds=sample_index / audio.sample_rate_hz,
                        amplitude=amplitude,
                    )
                    last_peak_index = sample_index
                continue

            events.append(
                PeakEvent(
                    timestamp_seconds=sample_index / audio.sample_rate_hz,
                    amplitude=amplitude,
                )
            )
            last_peak_index = sample_index

        return events


class SpeakingIntensityAnalyzer:
    """Measure normalized speaking intensity over analysis windows."""

    def __init__(
        self,
        timestamp_generator: TimestampGenerator | None = None,
    ) -> None:
        self._timestamp_generator = timestamp_generator or TimestampGenerator()

    def analyze(
        self,
        audio: AudioData,
        config: AudioObserverConfig,
    ) -> list[SpeakingIntensityWindow]:
        """Return intensity windows above the configured loudness threshold."""

        windows = self._timestamp_generator.windows(
            len(audio.samples),
            audio.sample_rate_hz,
            config.window_seconds,
            config.hop_seconds,
        )
        intensities: list[SpeakingIntensityWindow] = []

        for window in windows:
            loudness = rms_dbfs(audio.samples[window.start_index : window.end_index])
            if loudness < config.speaking_intensity_threshold_dbfs:
                continue

            intensities.append(
                SpeakingIntensityWindow(
                    start_seconds=window.start_seconds,
                    duration_seconds=window.duration_seconds,
                    intensity=dbfs_to_intensity(loudness),
                    loudness_dbfs=loudness,
                )
            )

        return intensities


class AudioAnalyzer:
    """Coordinate focused audio analysis services."""

    def __init__(
        self,
        loudness_analyzer: LoudnessAnalyzer | None = None,
        silence_detector: SilenceDetector | None = None,
        peak_detector: PeakDetector | None = None,
        speaking_intensity_analyzer: SpeakingIntensityAnalyzer | None = None,
    ) -> None:
        self._loudness_analyzer = loudness_analyzer or LoudnessAnalyzer()
        self._silence_detector = silence_detector or SilenceDetector()
        self._peak_detector = peak_detector or PeakDetector()
        self._speaking_intensity_analyzer = (
            speaking_intensity_analyzer or SpeakingIntensityAnalyzer()
        )

    def analyze(
        self,
        audio: AudioData,
        config: AudioObserverConfig,
    ) -> AudioAnalysisResult:
        """Run all configured audio analysis steps."""

        return AudioAnalysisResult(
            loudness_dbfs=self._loudness_analyzer.analyze(audio.samples),
            peak_amplitude=max((abs(sample) for sample in audio.samples), default=0.0),
            silence_segments=self._silence_detector.detect(audio, config),
            peak_events=self._peak_detector.detect(audio, config),
            speaking_intensity=self._speaking_intensity_analyzer.analyze(audio, config),
        )


def rms_dbfs(samples: tuple[float, ...]) -> float:
    """Return RMS loudness in dBFS for normalized audio samples."""

    if not samples:
        return MIN_DBFS

    rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
    if rms <= 0:
        return MIN_DBFS

    return max(MIN_DBFS, 20.0 * math.log10(rms))


def dbfs_to_intensity(loudness_dbfs: float) -> float:
    """Map dBFS loudness to a zero-to-one intensity value."""

    return max(0.0, min(1.0, (loudness_dbfs - MIN_DBFS) / abs(MIN_DBFS)))
