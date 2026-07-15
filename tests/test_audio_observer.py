import math
import struct
import wave
from pathlib import Path

import pytest

from aggregation import FeatureAggregator
from audio_observer import (
    AudioData,
    AudioObserver,
    AudioObserverConfig,
    AudioObserverError,
    AudioSource,
    ContextAudioExtractor,
    IncrementalAudioObserverConfig,
    IncrementalWavAudioObserver,
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


def write_wav(path: Path, samples: list[float], sample_rate_hz: int = 10) -> Path:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate_hz)
        wav_file.writeframes(
            b"".join(
                struct.pack("<h", max(-32768, min(32767, round(sample * 32768))))
                for sample in samples
            )
        )
    return path


def incremental_batches(
    path: Path,
    analysis: AudioObserverConfig,
    *,
    chunk_frames: int,
):
    return list(
        IncrementalWavAudioObserver(
            IncrementalAudioObserverConfig(
                chunk_frames=chunk_frames,
                analysis=analysis,
            )
        ).batches(path)
    )


def observation_signature(observation):
    return (
        observation.type,
        observation.timestamp_seconds,
        observation.duration_seconds,
        observation.value,
    )


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


def test_incremental_audio_matches_whole_file_local_observations(tmp_path: Path) -> None:
    samples = [0.5] * 5 + [0.0] * 10 + [0.95, 0.92] + [0.4] * 8
    path = write_wav(tmp_path / "audio.wav", samples)
    config = AudioObserverConfig(
        window_seconds=0.5,
        hop_seconds=0.3,
        silence_threshold_dbfs=-60.0,
        min_silence_seconds=0.5,
        peak_threshold=0.9,
        min_peak_distance_seconds=0.3,
        speaking_intensity_threshold_dbfs=-20.0,
    )
    whole = AudioObserver(
        config=config,
        extractor=FakeExtractor(AudioSource(path)),
    ).observe(ObserverContext())
    batches = incremental_batches(path, config, chunk_frames=4)
    incremental = [item for batch in batches for item in batch.observations]

    expected = sorted(
        (
            observation_signature(item)
            for item in whole.observations
            if item.type != "loudness"
        ),
        key=lambda item: (item[1], item[0]),
    )
    actual = sorted(
        (observation_signature(item) for item in incremental),
        key=lambda item: (item[1], item[0]),
    )
    assert actual == expected
    assert all(item.type != "loudness" for item in incremental)
    assert batches[-1].metadata["whole_file_loudness_is_candidate_signal"] is False
    assert batches[-1].metadata["overall_loudness_dbfs"] == pytest.approx(
        next(item for item in whole.observations if item.type == "loudness").value[
            "loudness_dbfs"
        ]
    )


def test_incremental_silence_spanning_chunks_closes_once(tmp_path: Path) -> None:
    path = write_wav(tmp_path / "silence.wav", [0.4] * 5 + [0.0] * 12)
    config = AudioObserverConfig(
        window_seconds=0.5,
        hop_seconds=0.5,
        silence_threshold_dbfs=-60.0,
        min_silence_seconds=0.5,
    )

    observations = [
        item
        for batch in incremental_batches(path, config, chunk_frames=3)
        for item in batch.observations
        if item.type == "silence"
    ]

    assert len(observations) == 1
    assert observations[0].timestamp_seconds == 0.5
    assert observations[0].duration_seconds == 1.2


def test_incremental_peak_suppression_crosses_chunk_boundary(tmp_path: Path) -> None:
    path = write_wav(
        tmp_path / "peaks.wav",
        [0.0, 0.0, 0.9, 0.0, 0.98, 0.0, 0.0, 0.0, 0.95],
    )
    config = AudioObserverConfig(
        peak_threshold=0.85,
        min_peak_distance_seconds=0.3,
    )

    peaks = [
        item
        for batch in incremental_batches(path, config, chunk_frames=4)
        for item in batch.observations
        if item.type == "peak"
    ]

    assert [(item.timestamp_seconds, item.value["amplitude"]) for item in peaks] == [
        (0.4, pytest.approx(0.98, abs=0.0001)),
        (0.8, pytest.approx(0.95, abs=0.0001)),
    ]

    batches = incremental_batches(path, config, chunk_frames=4)
    for batch in batches:
        emitted = tuple(
            item.timestamp_seconds for item in batch.observations if item.type == "peak"
        )
        if emitted:
            assert batch.metadata["finalized_peak_timestamps_seconds"] == emitted


def test_incremental_analysis_windows_are_not_lost_or_duplicated(tmp_path: Path) -> None:
    path = write_wav(tmp_path / "windows.wav", [0.5] * 12)
    config = AudioObserverConfig(
        window_seconds=0.5,
        hop_seconds=0.2,
        speaking_intensity_threshold_dbfs=-30.0,
    )
    whole_audio = WavAudioLoader().load(AudioSource(path))
    expected = SpeakingIntensityAnalyzer().analyze(whole_audio, config)

    actual = [
        item
        for batch in incremental_batches(path, config, chunk_frames=3)
        for item in batch.observations
        if item.type == "speaking_intensity"
    ]

    assert [(item.timestamp_seconds, item.duration_seconds) for item in actual] == [
        (item.start_seconds, item.duration_seconds) for item in expected
    ]


def test_incremental_sparse_windows_remain_aligned_across_chunks(
    tmp_path: Path,
) -> None:
    path = write_wav(tmp_path / "sparse-windows.wav", [0.5] * 24)
    config = AudioObserverConfig(
        window_seconds=0.3,
        hop_seconds=0.7,
        speaking_intensity_threshold_dbfs=-30.0,
    )
    whole_audio = WavAudioLoader().load(AudioSource(path))
    expected = SpeakingIntensityAnalyzer().analyze(whole_audio, config)

    actual = [
        item
        for batch in incremental_batches(path, config, chunk_frames=4)
        for item in batch.observations
        if item.type == "speaking_intensity"
    ]

    assert [(item.timestamp_seconds, item.duration_seconds) for item in actual] == [
        (item.start_seconds, item.duration_seconds) for item in expected
    ]


def test_incremental_watermarks_hold_open_state_and_advance_monotonically(
    tmp_path: Path,
) -> None:
    path = write_wav(tmp_path / "watermarks.wav", [0.0] * 8 + [0.95] + [0.4] * 11)
    config = AudioObserverConfig(
        window_seconds=0.4,
        hop_seconds=0.2,
        silence_threshold_dbfs=-60.0,
        min_silence_seconds=0.2,
        peak_threshold=0.9,
        min_peak_distance_seconds=0.4,
    )

    batches = incremental_batches(path, config, chunk_frames=3)
    watermarks = [batch.watermark_seconds for batch in batches]

    assert watermarks == sorted(watermarks)
    assert watermarks[1] == 0.0  # The silence beginning at zero is still open.
    peak_batches = [batch for batch in batches if batch.frames_processed in {9, 12}]
    assert all(batch.watermark_seconds <= 0.8 for batch in peak_batches)
    assert batches[-1].eof is True
    assert batches[-1].watermark_seconds == 2.0


def test_incremental_eof_flushes_partial_state_exactly_once(tmp_path: Path) -> None:
    path = write_wav(tmp_path / "partial.wav", [0.5] * 7)
    observer = IncrementalWavAudioObserver(
        IncrementalAudioObserverConfig(
            chunk_frames=20,
            analysis=AudioObserverConfig(
                window_seconds=0.5,
                hop_seconds=0.5,
                speaking_intensity_threshold_dbfs=-30.0,
            ),
        )
    )

    with observer.session(path) as session:
        data_batch = session.read_batch()
        eof_batch = session.read_batch()
        assert session.read_batch() is None
        assert session.flush() is None

    assert data_batch is not None and data_batch.eof is False
    assert eof_batch is not None and eof_batch.eof is True
    intensity = [item for item in eof_batch.observations if item.type == "speaking_intensity"]
    assert [(item.timestamp_seconds, item.duration_seconds) for item in intensity] == [
        (0.5, 0.2),
    ]


def test_incremental_rejects_flush_before_wav_is_exhausted(tmp_path: Path) -> None:
    path = write_wav(tmp_path / "premature-eof.wav", [0.5] * 12)
    observer = IncrementalWavAudioObserver(
        IncrementalAudioObserverConfig(chunk_frames=4)
    )

    with observer.session(path) as session:
        first_batch = session.read_batch()
        with pytest.raises(AudioObserverError, match="before the source is exhausted"):
            session.flush()
        remaining = []
        while (batch := session.read_batch()) is not None:
            remaining.append(batch)

    assert first_batch is not None and first_batch.eof is False
    assert sum(batch.eof for batch in remaining) == 1
    assert remaining[-1].watermark_seconds == 1.2
    assert remaining[-1].metadata["duration_seconds"] == 1.2


def test_existing_whole_file_audio_still_emits_loudness(tmp_path: Path) -> None:
    path = write_wav(tmp_path / "whole.wav", [0.5] * 10)

    result = AudioObserver(
        extractor=FakeExtractor(AudioSource(path)),
    ).observe(ObserverContext())

    assert sum(item.type == "loudness" for item in result.observations) == 1
