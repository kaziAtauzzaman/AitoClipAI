import math
import struct
import wave
from pathlib import Path

from aggregation import FeatureAggregator
from audio_observer import (
    AudioData,
    AudioObserver,
    AudioObserverConfig,
    AudioObserverError,
    AudioSource,
    ContextAudioExtractor,
    TimestampGenerator,
    WavAudioLoader,
)
from audio_observer.analysis import (
    AudioAnalyzer,
    LoudnessAnalyzer,
    PeakDetector,
    SilenceDetector,
    SpeakingIntensityAnalyzer,
)
from core import ObserverResult
from observers import ObserverContext, ObserverEngine, ObserverRegistry


class FakeExtractor:
    def __init__(self, source: AudioSource) -> None:
        self.source = source

    def extract(self, context: ObserverContext) -> AudioSource:
        return self.source


class FakeLoader:
    def __init__(self, audio: AudioData) -> None:
        self.audio = audio

    def load(self, source: AudioSource) -> AudioData:
        return self.audio


class FailingLoader:
    def load(self, source: AudioSource) -> AudioData:
        raise AudioObserverError("audio unavailable")


def make_audio(samples: list[float], sample_rate_hz: int = 10) -> AudioData:
    return AudioData(samples=tuple(samples), sample_rate_hz=sample_rate_hz)


def test_audio_observer_generates_aggregation_compatible_result(tmp_path: Path) -> None:
    audio = make_audio(
        [
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.95,
            0.1,
            0.1,
            0.1,
            0.1,
        ]
    )
    observer = AudioObserver(
        config=AudioObserverConfig(
            window_seconds=0.5,
            hop_seconds=0.5,
            min_silence_seconds=0.5,
            silence_threshold_dbfs=-60.0,
            peak_threshold=0.9,
            speaking_intensity_threshold_dbfs=-20.0,
        ),
        extractor=FakeExtractor(AudioSource(tmp_path / "source.wav")),
        loader=FakeLoader(audio),
    )

    result = observer.observe(ObserverContext())
    timeline = FeatureAggregator().aggregate([result])

    assert isinstance(result, ObserverResult)
    assert result.observer == "audio"
    assert result.metadata["sample_rate_hz"] == 10
    assert result.metadata["duration_seconds"] == 1.5
    assert {observation.type for observation in result.observations} == {
        "loudness",
        "peak",
        "silence",
        "speaking_intensity",
    }
    assert timeline.observer_results == [result]


def test_audio_thresholds_are_configurable(tmp_path: Path) -> None:
    audio = make_audio([0.5, 0.95, 0.5, 0.0, 0.0])
    observer = AudioObserver(
        config=AudioObserverConfig(
            window_seconds=0.5,
            hop_seconds=0.5,
            silence_threshold_dbfs=-120.0,
            peak_threshold=1.0,
            speaking_intensity_threshold_dbfs=0.0,
        ),
        extractor=FakeExtractor(AudioSource(tmp_path / "source.wav")),
        loader=FakeLoader(audio),
    )

    result = observer.observe(ObserverContext())

    assert [observation.type for observation in result.observations] == ["loudness"]


def test_audio_observer_failure_is_isolated_by_engine(tmp_path: Path) -> None:
    observer = AudioObserver(
        extractor=FakeExtractor(AudioSource(tmp_path / "missing.wav")),
        loader=FailingLoader(),
    )
    engine = ObserverEngine(ObserverRegistry(observers=[observer]))

    result = engine.run(ObserverContext())

    assert result.results == []
    assert len(result.failures) == 1
    assert result.failures[0].observer == "audio"
    assert result.failures[0].error_type == "AudioObserverError"


def test_context_audio_extractor_requires_existing_source_path(tmp_path: Path) -> None:
    extractor = ContextAudioExtractor()

    try:
        extractor.extract(ObserverContext())
    except AudioObserverError as exc:
        assert "source_path" in str(exc)
    else:
        raise AssertionError("Expected missing source_path to fail")

    missing_path = tmp_path / "missing.wav"
    try:
        extractor.extract(ObserverContext(source_path=missing_path))
    except AudioObserverError as exc:
        assert "does not exist" in str(exc)
    else:
        raise AssertionError("Expected missing file to fail")


def test_wav_audio_loader_loads_and_downmixes_pcm(tmp_path: Path) -> None:
    wav_path = tmp_path / "stereo.wav"
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(4)
        frames = [
            struct.pack("<hh", 16384, 16384),
            struct.pack("<hh", -16384, -16384),
            struct.pack("<hh", 0, 0),
            struct.pack("<hh", 32767, 32767),
        ]
        wav_file.writeframes(b"".join(frames))

    audio = WavAudioLoader().load(AudioSource(wav_path))

    assert audio.sample_rate_hz == 4
    assert audio.channels == 2
    assert audio.duration_seconds == 1.0
    assert audio.samples[0] == 0.5
    assert audio.samples[1] == -0.5
    assert audio.samples[2] == 0.0
    assert math.isclose(audio.samples[3], 0.999969482421875)


def test_timestamp_generator_produces_deterministic_windows() -> None:
    windows = TimestampGenerator().windows(
        sample_count=12,
        sample_rate_hz=10,
        window_seconds=0.5,
        hop_seconds=0.25,
    )

    assert [(window.start_index, window.end_index) for window in windows] == [
        (0, 5),
        (2, 7),
        (4, 9),
        (6, 11),
        (8, 12),
    ]


def test_loudness_analyzer_returns_dbfs() -> None:
    loudness = LoudnessAnalyzer().analyze((0.5, -0.5, 0.5, -0.5))

    assert math.isclose(loudness, -6.020599913279624)


def test_silence_detector_finds_contiguous_silence() -> None:
    audio = make_audio([0.3] * 5 + [0.0] * 10 + [0.3] * 5)

    segments = SilenceDetector().detect(
        audio,
        AudioObserverConfig(
            window_seconds=0.5,
            hop_seconds=0.5,
            silence_threshold_dbfs=-80.0,
            min_silence_seconds=0.5,
        ),
    )

    assert len(segments) == 1
    assert segments[0].start_seconds == 0.5
    assert segments[0].duration_seconds == 1.0


def test_peak_detector_applies_minimum_distance() -> None:
    audio = make_audio([0.0, 0.95, 0.9, 0.0, 0.0, 0.98])

    peaks = PeakDetector().detect(
        audio,
        AudioObserverConfig(peak_threshold=0.9, min_peak_distance_seconds=0.3),
    )

    assert [(peak.timestamp_seconds, peak.amplitude) for peak in peaks] == [
        (0.1, 0.95),
        (0.5, 0.98),
    ]


def test_speaking_intensity_analyzer_reports_active_windows() -> None:
    audio = make_audio([0.5] * 5 + [0.0] * 5)

    intensities = SpeakingIntensityAnalyzer().analyze(
        audio,
        AudioObserverConfig(
            window_seconds=0.5,
            hop_seconds=0.5,
            speaking_intensity_threshold_dbfs=-20.0,
        ),
    )

    assert len(intensities) == 1
    assert intensities[0].start_seconds == 0.0
    assert 0.0 < intensities[0].intensity <= 1.0


def test_audio_analyzer_combines_analysis_outputs() -> None:
    audio = make_audio([0.5] * 5 + [0.0] * 5 + [0.95] + [0.1] * 4)

    analysis = AudioAnalyzer().analyze(
        audio,
        AudioObserverConfig(
            window_seconds=0.5,
            hop_seconds=0.5,
            silence_threshold_dbfs=-60.0,
            peak_threshold=0.9,
            speaking_intensity_threshold_dbfs=-20.0,
        ),
    )

    assert analysis.loudness_dbfs > -20.0
    assert analysis.peak_amplitude == 0.95
    assert len(analysis.silence_segments) == 1
    assert len(analysis.peak_events) == 1
    assert len(analysis.speaking_intensity) >= 1
