import math
import shutil
import struct
import wave
from pathlib import Path

import pytest

from audio_observer import AudioObserver, FFmpegAudioExtractor, FFmpegAudioExtractorConfig
from candidate_generation import CandidateGenerator
from candidate_scoring import (
    CandidateScorer,
    CandidateScoringConfig,
    CandidateScoringError,
    ComponentScore,
    default_weights,
)
from core import ClipCandidate, Observation
from observers import ObserverEngine, ObserverRegistry
from pipeline import PipelineOrchestrator
from whisper_observer import (
    TranscriptionResult,
    TranscriptionSegment,
    WhisperObserver,
    WhisperObserverConfig,
)


def observed(
    observer: str,
    kind: str,
    value: object,
    *,
    duration: float | None = None,
    confidence: float | None = None,
) -> Observation:
    return Observation(
        timestamp_seconds=1.0,
        duration_seconds=duration,
        observer=observer,
        type=kind,
        value=value,
        confidence=confidence,
    )


def candidate(
    tmp_path: Path,
    start: float,
    observations: list[Observation],
    *,
    reason: str = "candidate",
) -> ClipCandidate:
    return ClipCandidate(
        source_video_path=tmp_path / "media.mp4",
        start_seconds=start,
        end_seconds=start + 10.0,
        reason=reason,
        metadata={"contributing_observations": observations},
    )


def high_signal_observations() -> list[Observation]:
    return [
        observed(
            "whisper",
            "speech",
            {"text": "WOW!!"},
            confidence=0.9,
            duration=1.0,
        ),
        observed(
            "audio", "speaking_intensity", {"intensity": 0.8}, duration=0.5
        ),
        observed("audio", "peak", {"amplitude": 0.95}),
        observed("audio", "silence", {"loudness_dbfs": -80.0}, duration=2.0),
    ]


def test_scorer_combines_all_explainable_components(tmp_path: Path) -> None:
    clip = candidate(tmp_path, 0.0, high_signal_observations())

    result = CandidateScorer().score([clip])[0]

    assert result.candidate is clip
    assert result.overall_score == pytest.approx(0.883167, abs=0.000001)
    assert result.score_components == {
        "speech_excitement": 0.2655,
        "speaking_intensity": 0.16,
        "loudness_peaks": 0.171,
        "silence_buildup": 0.12,
        "supporting_observations": 0.066667,
        "observation_diversity": 0.1,
    }
    assert sum(result.score_components.values()) == pytest.approx(
        result.overall_score
    )
    assert result.passed_threshold is True
    assert result.rationale is not None
    assert "speech excitement: 0.885 raw x 0.300 weight" in result.rationale
    assert "4 supporting observations" in result.rationale
    assert "4 distinct observer/type families" in result.rationale


def test_scorer_ranks_highest_score_first(tmp_path: Path) -> None:
    low = candidate(
        tmp_path,
        0.0,
        [observed("whisper", "speech", {"text": "hello"}, confidence=0.5)],
        reason="low",
    )
    high = candidate(tmp_path, 20.0, high_signal_observations(), reason="high")

    results = CandidateScorer().score([low, high])

    assert [result.candidate.reason for result in results] == ["high", "low"]
    assert results[0].overall_score > results[1].overall_score
    assert results[1].passed_threshold is False


def test_scorer_uses_configured_weights(tmp_path: Path) -> None:
    weights = {name: 0.0 for name in default_weights()}
    weights["speaking_intensity"] = 2.0
    config = CandidateScoringConfig(weights=weights, passing_score=0.7)
    clip = candidate(
        tmp_path,
        0.0,
        [observed("audio", "speaking_intensity", {"intensity": 0.8})],
    )

    result = CandidateScorer(config).score([clip])[0]

    assert result.overall_score == 0.8
    assert result.score_components["speaking_intensity"] == 0.8
    assert all(
        contribution == 0.0
        for name, contribution in result.score_components.items()
        if name != "speaking_intensity"
    )
    assert result.passed_threshold is True


def test_speech_excitement_uses_confidence_punctuation_and_uppercase(
    tmp_path: Path,
) -> None:
    excited = candidate(
        tmp_path,
        0.0,
        [observed("whisper", "speech", {"text": "REALLY?!"}, confidence=0.8)],
    )
    plain = candidate(
        tmp_path,
        20.0,
        [observed("whisper", "speech", {"text": "really"}, confidence=0.8)],
    )
    weights = {name: 0.0 for name in default_weights()}
    weights["speech_excitement"] = 1.0

    scores = CandidateScorer(CandidateScoringConfig(weights=weights)).score(
        [plain, excited]
    )

    assert scores[0].candidate is excited
    assert scores[0].overall_score == 0.77
    assert scores[1].overall_score == 0.6


def test_loudness_uses_peak_amplitude_from_loudness_observation(
    tmp_path: Path,
) -> None:
    weights = {name: 0.0 for name in default_weights()}
    weights["loudness_peaks"] = 1.0
    clip = candidate(
        tmp_path,
        0.0,
        [
            observed(
                "audio",
                "loudness",
                {"loudness_dbfs": -8.0, "peak_amplitude": 0.87},
            )
        ],
    )

    result = CandidateScorer(CandidateScoringConfig(weights=weights)).score([clip])[0]

    assert result.overall_score == 0.87


def test_support_and_diversity_are_normalized_and_capped(tmp_path: Path) -> None:
    weights = {name: 0.0 for name in default_weights()}
    weights["supporting_observations"] = 1.0
    weights["observation_diversity"] = 1.0
    observations = [
        observed(f"observer-{index}", f"type-{index}", index)
        for index in range(8)
    ]

    result = CandidateScorer(CandidateScoringConfig(weights=weights)).score(
        [candidate(tmp_path, 0.0, observations)]
    )[0]

    assert result.score_components["supporting_observations"] == 0.5
    assert result.score_components["observation_diversity"] == 0.5
    assert result.overall_score == 1.0


def test_scorer_is_deterministic_and_uses_stable_tie_breakers(tmp_path: Path) -> None:
    first = candidate(tmp_path, 5.0, [], reason="later")
    second = candidate(tmp_path, 1.0, [], reason="earlier")
    scorer = CandidateScorer()

    forward = scorer.score([first, second])
    reverse = scorer.score([second, first])

    assert [item.candidate for item in forward] == [second, first]
    assert [item.candidate for item in reverse] == [second, first]
    assert [item.overall_score for item in forward] == [0.0, 0.0]


def test_scorer_does_not_modify_candidates_or_observations(tmp_path: Path) -> None:
    observations = high_signal_observations()
    clip = candidate(tmp_path, 0.0, observations)
    original_metadata = dict(clip.metadata)

    CandidateScorer().score([clip])

    assert clip.metadata == original_metadata
    assert clip.metadata["contributing_observations"] is observations


def test_scorer_supports_injected_heuristics(tmp_path: Path) -> None:
    class CustomHeuristic:
        name = "custom"

        def score(
            self,
            clip: ClipCandidate,
            observations: list[Observation],
            config: CandidateScoringConfig,
        ) -> ComponentScore:
            return ComponentScore(0.42, "custom deterministic signal")

    config = CandidateScoringConfig(weights={"custom": 3.0}, passing_score=0.4)

    result = CandidateScorer(config, heuristics=[CustomHeuristic()]).score(
        [candidate(tmp_path, 0.0, [])]
    )[0]

    assert result.overall_score == 0.42
    assert result.score_components == {"custom": 0.42}
    assert "custom deterministic signal" in (result.rationale or "")


def test_scorer_rejects_invalid_weights_and_heuristic_outputs(tmp_path: Path) -> None:
    with pytest.raises(CandidateScoringError, match="positive"):
        CandidateScorer(CandidateScoringConfig(weights={name: 0.0 for name in default_weights()}))

    class InvalidHeuristic:
        name = "invalid"

        def score(
            self,
            clip: ClipCandidate,
            observations: list[Observation],
            config: CandidateScoringConfig,
        ) -> ComponentScore:
            return ComponentScore(1.5, "invalid")

    scorer = CandidateScorer(
        CandidateScoringConfig(weights={"invalid": 1.0}),
        heuristics=[InvalidHeuristic()],
    )
    with pytest.raises(CandidateScoringError, match="outside"):
        scorer.score([candidate(tmp_path, 0.0, [])])


class OfflineSpeechBackend:
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
                    text="THIS IS EXCITING!",
                    confidence=0.96,
                )
            ],
            text="THIS IS EXCITING!",
            language="en",
        )


def write_offline_fixture(path: Path) -> None:
    sample_rate_hz = 8_000
    samples = [
        int(11_000 * math.sin(2 * math.pi * 440 * index / sample_rate_hz))
        for index in range(2_000)
    ]
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate_hz)
        audio.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
def test_candidate_scorer_offline_pipeline_integration(tmp_path: Path) -> None:
    media_path = tmp_path / "scoring-fixture.wav"
    write_offline_fixture(media_path)
    engine = ObserverEngine(
        ObserverRegistry(
            observers=[
                AudioObserver(),
                WhisperObserver(backend=OfflineSpeechBackend()),
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
    scores = CandidateScorer().score(candidates)

    assert timeline.timeline_path.is_file()
    assert candidates
    assert scores
    assert scores[0].candidate is candidates[0]
    assert scores[0].overall_score > 0.0
    assert scores[0].score_components["speech_excitement"] > 0.0
    assert scores[0].score_components["speaking_intensity"] > 0.0
    assert scores[0].rationale is not None
    assert "raw" in scores[0].rationale
