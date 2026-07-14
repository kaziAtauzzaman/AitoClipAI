import math
import shutil
import struct
import wave
from pathlib import Path

import pytest

from aggregation import FeatureAggregator
from audio_observer import AudioObserver, FFmpegAudioExtractor, FFmpegAudioExtractorConfig
from candidate_generation import (
    CandidateEvent,
    CandidateGenerationConfig,
    CandidateGenerationError,
    CandidateGenerator,
    EventBoundaryRole,
)
from core import FeatureTimeline, Observation, ObserverResult
from observers import ObserverContext, ObserverEngine, ObserverRegistry
from pipeline import PipelineOrchestrator
from whisper_observer import (
    TranscriptionResult,
    TranscriptionSegment,
    WhisperObserver,
    WhisperObserverConfig,
)


def observation(
    timestamp: float,
    observer: str,
    kind: str,
    value: object,
    *,
    duration: float | None = None,
    confidence: float | None = None,
) -> Observation:
    return Observation(
        timestamp_seconds=timestamp,
        duration_seconds=duration,
        observer=observer,
        type=kind,
        value=value,
        confidence=confidence,
    )


def feature_timeline(
    tmp_path: Path,
    observations: list[Observation],
    *,
    duration: float = 30.0,
) -> FeatureTimeline:
    results = [
        ObserverResult(
            observer="combined",
            observations=observations,
            metadata={"duration_seconds": duration},
        )
    ]
    return FeatureTimeline(
        media_path=tmp_path / "media.mp4",
        audio_path=tmp_path / "audio.wav",
        timeline_path=tmp_path / "media.mp4.feature-timeline.json",
        timeline=FeatureAggregator().aggregate(results),
    )


def test_generator_combines_supported_signals_into_candidate(tmp_path: Path) -> None:
    loudness = observation(
        10.0, "audio", "loudness", {"loudness_dbfs": -10.0}
    )
    silence = observation(
        8.0, "audio", "silence", {"loudness_dbfs": -80.0}, duration=2.0
    )
    speech = observation(
        10.0,
        "whisper",
        "speech",
        {"text": "A strong moment", "speaker": "speaker-1"},
        duration=3.0,
        confidence=0.9,
    )
    intensity = observation(
        10.5,
        "audio",
        "speaking_intensity",
        {"intensity": 0.8},
        duration=1.0,
    )
    source = feature_timeline(
        tmp_path, [loudness, silence, speech, intensity], duration=30.0
    )

    candidates = CandidateGenerator().generate(source)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.start_seconds == 8.0
    assert candidate.end_seconds == 16.0
    assert candidate.metadata["start_time"] == 8.0
    assert candidate.metadata["end_time"] == 16.0
    assert candidate.metadata["confidence"] == 1.0
    assert candidate.source_signals == [
        "audio_loudness",
        "silence_buildup",
        "whisper_speech",
        "speaking_intensity",
    ]
    assert "Whisper speech" in candidate.reason
    assert "silence buildup" in candidate.reason
    contributing = candidate.metadata["contributing_observations"]
    assert isinstance(contributing, list)
    assert {id(item) for item in contributing} == {
        id(loudness),
        id(silence),
        id(speech),
        id(intensity),
    }


def test_generator_merges_nearby_events_and_separates_distant_events(
    tmp_path: Path,
) -> None:
    config = CandidateGenerationConfig(
        minimum_candidate_confidence=0.1,
        merge_gap_seconds=2.0,
        pre_roll_seconds=0.0,
        post_roll_seconds=0.0,
        minimum_clip_seconds=1.0,
        maximum_clip_seconds=10.0,
    )
    source = feature_timeline(
        tmp_path,
        [
            observation(
                5.0,
                "whisper",
                "speech",
                {"text": "first"},
                duration=1.0,
                confidence=0.8,
            ),
            observation(
                7.0,
                "audio",
                "speaking_intensity",
                {"intensity": 0.8},
                duration=1.0,
            ),
            observation(
                20.0,
                "whisper",
                "speech",
                {"text": "second"},
                duration=1.0,
                confidence=0.8,
            ),
        ],
    )

    candidates = CandidateGenerator(config).generate(source)

    assert [(item.start_seconds, item.end_seconds) for item in candidates] == [
        (5.0, 8.0),
        (20.0, 21.0),
    ]
    assert candidates[0].source_signals == [
        "whisper_speech",
        "speaking_intensity",
    ]


def test_generator_filters_events_below_configured_thresholds(tmp_path: Path) -> None:
    config = CandidateGenerationConfig(
        minimum_speech_confidence=0.8,
        loudness_threshold_dbfs=-20.0,
        minimum_silence_seconds=2.0,
        speaking_intensity_threshold=0.7,
    )
    source = feature_timeline(
        tmp_path,
        [
            observation(
                1.0,
                "whisper",
                "speech",
                {"text": "quiet"},
                confidence=0.5,
            ),
            observation(1.0, "audio", "loudness", {"loudness_dbfs": -40.0}),
            observation(1.0, "audio", "silence", {}, duration=1.0),
            observation(
                1.0,
                "audio",
                "speaking_intensity",
                {"intensity": 0.4},
                duration=0.5,
            ),
        ],
    )

    assert CandidateGenerator(config).generate(source) == []


def test_silence_buildup_anchors_candidate_at_end_of_silence(tmp_path: Path) -> None:
    config = CandidateGenerationConfig(
        minimum_candidate_confidence=0.1,
        pre_roll_seconds=1.0,
        post_roll_seconds=2.0,
        minimum_clip_seconds=3.0,
        maximum_clip_seconds=10.0,
    )
    silence = observation(4.0, "audio", "silence", {}, duration=2.0)

    candidate = CandidateGenerator(config).generate(
        feature_timeline(tmp_path, [silence])
    )[0]

    assert (candidate.start_seconds, candidate.end_seconds) == (5.0, 8.0)
    assert candidate.source_signals == ["silence_buildup"]


def test_generator_clamps_and_expands_windows_to_media_bounds(tmp_path: Path) -> None:
    config = CandidateGenerationConfig(
        minimum_candidate_confidence=0.1,
        pre_roll_seconds=1.0,
        post_roll_seconds=1.0,
        minimum_clip_seconds=8.0,
        maximum_clip_seconds=12.0,
    )
    speech = observation(
        1.0,
        "whisper",
        "speech",
        {"text": "opening"},
        duration=1.0,
        confidence=1.0,
    )

    candidate = CandidateGenerator(config).generate(
        feature_timeline(tmp_path, [speech], duration=6.0)
    )[0]

    assert (candidate.start_seconds, candidate.end_seconds) == (0.0, 6.0)


def test_generator_is_deterministic_for_observation_input_order(tmp_path: Path) -> None:
    items = [
        observation(
            10.0,
            "whisper",
            "speech",
            {"text": "deterministic"},
            duration=2.0,
            confidence=0.9,
        ),
        observation(
            10.0,
            "audio",
            "speaking_intensity",
            {"intensity": 0.8},
            duration=1.0,
        ),
    ]
    generator = CandidateGenerator()

    first = generator.generate(feature_timeline(tmp_path, items))
    second = generator.generate(feature_timeline(tmp_path, list(reversed(items))))

    assert first[0].start_seconds == second[0].start_seconds
    assert first[0].end_seconds == second[0].end_seconds
    assert first[0].source_signals == second[0].source_signals
    assert first[0].reason == second[0].reason
    assert first[0].metadata["confidence"] == second[0].metadata["confidence"]
    assert first[0].metadata == second[0].metadata


def test_dense_supporting_audio_does_not_expand_to_maximum(tmp_path: Path) -> None:
    items = [
        observation(float(index), "audio", "speaking_intensity", {"intensity": 0.8}, duration=1.0)
        for index in range(10, 66)
    ]
    candidates = CandidateGenerator().generate(
        feature_timeline(tmp_path, items, duration=100.0)
    )
    candidate = candidates[0]

    assert all(item.end_seconds - item.start_seconds < 60.0 for item in candidates)
    assert (candidate.start_seconds, candidate.end_seconds) == (20.5, 55.5)
    assert candidate.metadata["original_cluster_start"] == 10.0
    assert candidate.metadata["original_cluster_end"] == 65.0
    assert candidate.metadata["boundary_refinement"] == "supporting_event_anchor"


def test_weak_supporting_events_cannot_extend_strong_speech_anchor(tmp_path: Path) -> None:
    speech = observation(30.0, "whisper", "speech", {"text": "anchor"}, duration=5.0, confidence=0.9)
    supporting = [
        observation(float(index), "audio", "speaking_intensity", {"intensity": 0.4}, duration=1.0)
        for index in range(10, 66)
    ]
    candidate = CandidateGenerator().generate(feature_timeline(tmp_path, [*supporting, speech], duration=100.0))[0]

    assert (candidate.start_seconds, candidate.end_seconds) == (28.0, 38.0)
    assert candidate.metadata["anchor_core_start"] == 30.0
    assert candidate.metadata["anchor_core_end"] == 35.0


def test_long_whisper_segment_is_framed_deterministically(tmp_path: Path) -> None:
    speech = observation(20.0, "whisper", "speech", {"text": "long anchor"}, duration=30.0, confidence=0.75)
    trailing = [
        observation(50.0 + index, "audio", "speaking_intensity", {"intensity": 0.5}, duration=1.0)
        for index in range(20)
    ]
    candidate = CandidateGenerator().generate(feature_timeline(tmp_path, [speech, *trailing], duration=100.0))[0]

    assert (candidate.start_seconds, candidate.end_seconds) == (18.0, 53.0)
    assert candidate.metadata["boundary_refinement"] == "strongest_local_contribution_core"


def test_silence_end_supports_anchor_without_retaining_full_silence(tmp_path: Path) -> None:
    silence = observation(5.0, "audio", "silence", {}, duration=20.0)
    speech = observation(25.0, "whisper", "speech", {"text": "after silence"}, duration=3.0, confidence=0.9)
    trailing = [
        observation(float(index), "audio", "speaking_intensity", {"intensity": 0.4}, duration=1.0)
        for index in range(28, 61)
    ]
    candidate = CandidateGenerator().generate(feature_timeline(tmp_path, [silence, speech, *trailing], duration=80.0))[0]

    assert (candidate.start_seconds, candidate.end_seconds) == (23.0, 31.0)
    assert silence in candidate.metadata["contributing_observations"]
    assert candidate.start_seconds > silence.timestamp_seconds


def test_sustained_high_signal_may_exceed_anchor_target(tmp_path: Path) -> None:
    sustained = [
        observation(start, "whisper", "speech", {"text": f"strong {start}"}, duration=15.0, confidence=1.0)
        for start in (10.0, 25.0, 40.0)
    ]
    candidate = CandidateGenerator().generate(feature_timeline(tmp_path, sustained, duration=80.0))[0]

    assert (candidate.start_seconds, candidate.end_seconds) == (8.0, 58.0)
    assert candidate.end_seconds - candidate.start_seconds > 35.0
    assert candidate.metadata["boundary_refinement"] == "sustained_high_signal_core"


def test_retained_observations_intersect_refined_window(tmp_path: Path) -> None:
    outside = observation(10.0, "audio", "speaking_intensity", {"intensity": 0.4}, duration=1.0)
    anchor = observation(40.0, "whisper", "speech", {"text": "anchor"}, duration=4.0, confidence=0.9)
    bridge = [
        observation(float(index), "audio", "speaking_intensity", {"intensity": 0.4}, duration=1.0)
        for index in range(11, 40)
    ]
    candidate = CandidateGenerator().generate(feature_timeline(tmp_path, [outside, *bridge, anchor], duration=80.0))[0]

    retained = candidate.metadata["contributing_observations"]
    assert outside not in retained
    assert all(
        item.timestamp_seconds + (item.duration_seconds or 0.0) >= candidate.start_seconds
        and item.timestamp_seconds <= candidate.end_seconds
        for item in retained
    )


def test_equal_supporting_concentrations_prefer_central_then_earliest(tmp_path: Path) -> None:
    items = [
        observation(float(timestamp), "audio", "speaking_intensity", {"intensity": 0.8}, duration=1.0)
        for timestamp in (*range(10, 16), *range(35, 41), *range(60, 66))
    ]
    bridge = [
        observation(float(timestamp), "audio", "speaking_intensity", {"intensity": 0.4}, duration=1.0)
        for timestamp in (*range(16, 35), *range(41, 60))
    ]
    config = CandidateGenerationConfig(maximum_clip_seconds=90.0, anchor_core_seconds=10.0)

    candidate = CandidateGenerator(config).generate(
        feature_timeline(tmp_path, [*items, *bridge], duration=90.0)
    )[0]

    assert candidate.metadata["anchor_core_start"] == 32.5
    assert candidate.metadata["anchor_core_end"] == 42.5


def test_short_strong_event_beats_one_long_weak_whisper_segment(tmp_path: Path) -> None:
    long_weak = observation(
        5.0, "whisper", "speech", {"text": "long weak"}, duration=35.0, confidence=0.51
    )
    short_strong = observation(
        42.0, "whisper", "speech", {"text": "short strong"}, duration=2.0, confidence=0.95
    )
    bridge = [
        observation(float(timestamp), "audio", "speaking_intensity", {"intensity": 0.4}, duration=1.0)
        for timestamp in (40, 41)
    ]
    config = CandidateGenerationConfig(anchor_core_seconds=10.0)

    candidate = CandidateGenerator(config).generate(
        feature_timeline(tmp_path, [long_weak, *bridge, short_strong], duration=70.0)
    )[0]

    assert (candidate.metadata["anchor_core_start"], candidate.metadata["anchor_core_end"]) == (42.0, 44.0)
    assert candidate.metadata["boundary_refinement"] == "strongest_local_contribution_core"


def test_retained_metadata_and_confidence_are_internally_consistent(tmp_path: Path) -> None:
    outside = observation(10.0, "audio", "speaking_intensity", {"intensity": 1.0}, duration=1.0)
    bridge = [
        observation(float(timestamp), "audio", "speaking_intensity", {"intensity": 0.4}, duration=1.0)
        for timestamp in range(11, 40)
    ]
    anchor = observation(40.0, "whisper", "speech", {"text": "anchor"}, duration=2.0, confidence=0.8)

    candidate = CandidateGenerator().generate(
        feature_timeline(tmp_path, [outside, *bridge, anchor], duration=70.0)
    )[0]
    contributions = candidate.metadata["signal_contributions"]
    retained = candidate.metadata["contributing_observations"]

    assert outside not in retained
    assert candidate.source_signals == list(dict.fromkeys(item["signal"] for item in contributions))
    assert candidate.metadata["confidence"] == round(
        min(1.0, sum(item["contribution"] for item in contributions)), 6
    )
    assert candidate.metadata["original_cluster_confidence"] > candidate.metadata["confidence"]


@pytest.mark.parametrize(
    ("timestamp", "duration", "media_duration", "expected"),
    [
        (0.0, 1.0, 100.0, (0.0, 8.0)),
        (98.0, 2.0, 100.0, (92.0, 100.0)),
    ],
)
def test_refinement_and_minimum_expansion_preserve_media_edge_anchor(
    tmp_path: Path,
    timestamp: float,
    duration: float,
    media_duration: float,
    expected: tuple[float, float],
) -> None:
    speech = observation(
        timestamp, "whisper", "speech", {"text": "edge"}, duration=duration, confidence=1.0
    )
    support_range = range(1, 36) if timestamp == 0.0 else range(64, 98)
    supporting = [
        observation(float(second), "audio", "speaking_intensity", {"intensity": 0.4}, duration=1.0)
        for second in support_range
    ]

    candidate = CandidateGenerator().generate(
        feature_timeline(tmp_path, [speech, *supporting], duration=media_duration)
    )[0]

    assert (candidate.start_seconds, candidate.end_seconds) == expected
    assert candidate.start_seconds <= timestamp <= candidate.end_seconds
    assert candidate.metadata["boundary_refinement"] == "strongest_local_contribution_core"


def test_injected_event_must_explicitly_declare_boundary_role(tmp_path: Path) -> None:
    class CustomHeuristic:
        def detect(self, item: Observation) -> CandidateEvent | None:
            if item.type == "routine":
                return CandidateEvent(
                    item.timestamp_seconds,
                    item.timestamp_seconds + (item.duration_seconds or 0.0),
                    "custom_routine",
                    1.0,
                    10.0,
                    item,
                )
            if item.type == "anchor":
                return CandidateEvent(
                    item.timestamp_seconds,
                    item.timestamp_seconds + (item.duration_seconds or 0.0),
                    "custom_anchor",
                    0.8,
                    0.5,
                    item,
                    boundary_role=EventBoundaryRole.DRIVING,
                    sustained_strength=0.8,
                )
            return None

    routine = observation(5.0, "custom", "routine", {}, duration=35.0)
    anchor = observation(42.0, "custom", "anchor", {}, duration=2.0)
    candidate = CandidateGenerator(heuristics=[CustomHeuristic()]).generate(
        feature_timeline(tmp_path, [routine, anchor], duration=70.0)
    )[0]

    assert candidate.metadata["anchor_core_start"] == 42.0
    assert candidate.metadata["anchor_core_end"] == 44.0


@pytest.mark.parametrize(
    ("confidence", "expected_reason"),
    [
        (0.499, "strongest_local_contribution_core"),
        (0.501, "sustained_high_signal_core"),
    ],
)
def test_sustained_chain_uses_normalized_strength_threshold(
    tmp_path: Path,
    confidence: float,
    expected_reason: str,
) -> None:
    events = [
        observation(start, "whisper", "speech", {"text": str(start)}, duration=16.0, confidence=confidence)
        for start in (5.0, 21.0, 37.0)
    ]

    candidate = CandidateGenerator().generate(
        feature_timeline(tmp_path, events, duration=70.0)
    )[0]

    assert candidate.metadata["boundary_refinement"] == expected_reason


def test_dense_anchor_search_has_quadratic_operation_bound(tmp_path: Path) -> None:
    class CountingGenerator(CandidateGenerator):
        role_checks = 0

        @staticmethod
        def _is_boundary_driving(event: CandidateEvent) -> bool:
            CountingGenerator.role_checks += 1
            return CandidateGenerator._is_boundary_driving(event)

    events = [
        observation(index / 10.0, "whisper", "speech", {"text": str(index)}, duration=0.2, confidence=0.49)
        for index in range(300)
    ]
    generator = CountingGenerator()

    assert generator.generate(feature_timeline(tmp_path, events, duration=40.0))
    assert CountingGenerator.role_checks <= len(events) ** 2 + len(events) * 5


def test_short_cluster_preserves_legacy_boundaries_signals_and_confidence(tmp_path: Path) -> None:
    speech = observation(10.0, "whisper", "speech", {"text": "short"}, duration=2.0, confidence=0.9)
    intensity = observation(11.0, "audio", "speaking_intensity", {"intensity": 0.8}, duration=1.0)

    candidate = CandidateGenerator().generate(
        feature_timeline(tmp_path, [speech, intensity], duration=30.0)
    )[0]

    assert (candidate.start_seconds, candidate.end_seconds) == (7.5, 15.5)
    assert candidate.source_signals == ["whisper_speech", "speaking_intensity"]
    assert candidate.metadata["confidence"] == 0.82
    assert candidate.metadata["boundary_refinement"] == "cluster_within_anchor_target"


def test_generator_accepts_injected_heuristics(tmp_path: Path) -> None:
    class CustomHeuristic:
        def detect(self, item: Observation) -> CandidateEvent | None:
            if item.type != "custom":
                return None
            return CandidateEvent(
                start_seconds=item.timestamp_seconds,
                end_seconds=item.timestamp_seconds,
                signal="custom_signal",
                strength=1.0,
                weight=1.0,
                observation=item,
            )

    config = CandidateGenerationConfig(
        pre_roll_seconds=0.0,
        post_roll_seconds=1.0,
        minimum_clip_seconds=1.0,
    )
    custom = observation(3.0, "custom", "custom", True)

    candidate = CandidateGenerator(config, heuristics=[CustomHeuristic()]).generate(
        feature_timeline(tmp_path, [custom])
    )[0]

    assert candidate.source_signals == ["custom_signal"]
    assert "custom signal" in candidate.reason


def test_generator_accepts_explicitly_empty_heuristic_set(tmp_path: Path) -> None:
    speech = observation(
        1.0,
        "whisper",
        "speech",
        {"text": "ignored"},
        confidence=1.0,
    )

    candidates = CandidateGenerator(heuristics=[]).generate(
        feature_timeline(tmp_path, [speech])
    )

    assert candidates == []


def test_generator_rejects_invalid_window_configuration() -> None:
    with pytest.raises(CandidateGenerationError, match="shorter"):
        CandidateGenerator(
            CandidateGenerationConfig(
                minimum_clip_seconds=20.0,
                maximum_clip_seconds=10.0,
            )
        )


class FixtureSpeechBackend:
    def transcribe(
        self,
        audio_path: Path,
        config: WhisperObserverConfig,
    ) -> TranscriptionResult:
        with wave.open(str(audio_path), "rb") as audio:
            duration = audio.getnframes() / audio.getframerate()
        return TranscriptionResult(
            segments=[
                TranscriptionSegment(
                    start_seconds=0.0,
                    end_seconds=duration,
                    text="candidate fixture speech",
                    confidence=0.95,
                )
            ],
            text="candidate fixture speech",
            language="en",
        )


def write_media_fixture(path: Path) -> None:
    sample_rate_hz = 8_000
    samples = [
        int(10_000 * math.sin(2 * math.pi * 440 * index / sample_rate_hz))
        for index in range(2_000)
    ]
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate_hz)
        audio.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
def test_candidate_generator_offline_pipeline_integration(tmp_path: Path) -> None:
    media_path = tmp_path / "candidate-fixture.wav"
    write_media_fixture(media_path)
    engine = ObserverEngine(
        ObserverRegistry(
            observers=[
                AudioObserver(),
                WhisperObserver(backend=FixtureSpeechBackend()),
            ]
        )
    )
    pipeline = PipelineOrchestrator(
        audio_extractor=FFmpegAudioExtractor(
            FFmpegAudioExtractorConfig(output_dir=tmp_path / "audio")
        ),
        observer_engine=engine,
    )

    timeline = pipeline.analyze(media_path)
    candidates = CandidateGenerator().generate(timeline)

    assert timeline.timeline_path.is_file()
    assert candidates
    candidate = candidates[0]
    assert candidate.source_video_path == media_path
    assert candidate.start_seconds == 0.0
    assert candidate.end_seconds == pytest.approx(0.25)
    assert "whisper_speech" in candidate.source_signals
    assert candidate.metadata["confidence"] >= 0.45
    contributing = candidate.metadata["contributing_observations"]
    assert any(item.observer == "whisper" for item in contributing)
