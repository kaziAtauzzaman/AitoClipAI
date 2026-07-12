from pathlib import Path

import pytest

from aggregation import FeatureAggregator
from captioning import (
    CaptionCue,
    CaptionGenerator,
    CaptionGeneratorConfig,
    InvalidCaptionSourceError,
    InvalidCaptionTimingError,
    SrtCaptionFormatter,
)
from core import ClipCandidate, ClipScore, FeatureTimeline, Observation, ObserverResult


def speech(
    start: float,
    duration: float | None,
    text: str,
    *,
    speaker: str | None = None,
    confidence: float | None = None,
) -> Observation:
    return Observation(
        timestamp_seconds=start,
        duration_seconds=duration,
        observer="whisper",
        type="speech",
        value={"text": text, "speaker": speaker},
        confidence=confidence,
        metadata={"segment": text},
    )


def timeline(tmp_path: Path, observations: list[Observation]) -> FeatureTimeline:
    result = ObserverResult(
        observer="whisper",
        observations=observations,
        metadata={"language": "en"},
    )
    return FeatureTimeline(
        media_path=tmp_path / "source.mp4",
        audio_path=tmp_path / "source.wav",
        timeline_path=tmp_path / "source.feature-timeline.json",
        timeline=FeatureAggregator().aggregate([result]),
    )


def score(
    tmp_path: Path,
    start: float = 10.0,
    end: float = 20.0,
    *,
    reason: str = "candidate",
) -> ClipScore:
    candidate = ClipCandidate(
        source_video_path=tmp_path / "source.mp4",
        start_seconds=start,
        end_seconds=end,
        reason=reason,
    )
    return ClipScore(candidate=candidate, overall_score=0.9)


def test_caption_generator_uses_complete_timeline_and_rebases_overlaps(
    tmp_path: Path,
) -> None:
    observations = [
        speech(8.0, 4.0, "starts before", speaker="A", confidence=0.9),
        speech(12.0, 3.0, "inside", speaker="B", confidence=0.8),
        speech(19.0, 3.0, "ends after", confidence=0.7),
        speech(21.0, 1.0, "outside"),
        Observation(
            timestamp_seconds=13.0,
            observer="audio",
            type="peak",
            value={"amplitude": 0.9},
        ),
    ]
    clip_score = score(tmp_path)
    generator = CaptionGenerator(
        CaptionGeneratorConfig(output_dir=tmp_path / "captions")
    )

    artifact = generator.generate(timeline(tmp_path, observations), [clip_score])[0]

    assert artifact.candidate is clip_score.candidate
    assert artifact.language == "en"
    assert artifact.path == tmp_path / "captions" / "source.captions-10000-20000.srt"
    assert [(cue.start_seconds, cue.end_seconds) for cue in artifact.cues] == [
        (0.0, 2.0),
        (2.0, 5.0),
        (9.0, 10.0),
    ]
    assert [cue.text for cue in artifact.cues] == [
        "[A] starts before",
        "[B] inside",
        "ends after",
    ]
    assert [cue.confidence for cue in artifact.cues] == [0.9, 0.8, 0.7]
    assert artifact.cues[0].metadata["source_start_seconds"] == 8.0
    assert artifact.cues[2].metadata["source_end_seconds"] == 22.0
    assert artifact.path.read_text(encoding="utf-8") == (
        "1\n00:00:00,000 --> 00:00:02,000\n[A] starts before\n\n"
        "2\n00:00:02,000 --> 00:00:05,000\n[B] inside\n\n"
        "3\n00:00:09,000 --> 00:00:10,000\nends after\n"
    )


def test_caption_generator_does_not_use_candidate_contributing_subset(
    tmp_path: Path,
) -> None:
    clip_score = score(tmp_path)
    clip_score.candidate.metadata["contributing_observations"] = []
    complete_speech = speech(11.0, 1.0, "from timeline")

    artifact = CaptionGenerator(
        CaptionGeneratorConfig(output_dir=tmp_path / "captions")
    ).generate(timeline(tmp_path, [complete_speech]), [clip_score])[0]

    assert [cue.text for cue in artifact.cues] == ["from timeline"]


def test_caption_generator_is_deterministic_for_input_order(tmp_path: Path) -> None:
    first = score(tmp_path, 10.0, 15.0, reason="first")
    second = score(tmp_path, 20.0, 25.0, reason="second")
    observations = [speech(11.0, 1.0, "one"), speech(21.0, 1.0, "two")]
    generator = CaptionGenerator(
        CaptionGeneratorConfig(output_dir=tmp_path / "captions")
    )

    forward = generator.generate(timeline(tmp_path, observations), [second, first])
    reverse = generator.generate(timeline(tmp_path, observations), [first, second])

    assert [item.path.name for item in forward] == [item.path.name for item in reverse]
    assert [item.candidate.reason for item in forward] == ["first", "second"]


def test_caption_generator_rejects_ambiguous_candidate_identity(
    tmp_path: Path,
) -> None:
    first = score(tmp_path, reason="first")
    duplicate = score(tmp_path, reason="duplicate")

    with pytest.raises(InvalidCaptionSourceError, match="same source path"):
        CaptionGenerator(
            CaptionGeneratorConfig(output_dir=tmp_path / "captions")
        ).generate(timeline(tmp_path, []), [first, duplicate])


def test_caption_generator_validates_speech_and_candidate_timing(
    tmp_path: Path,
) -> None:
    generator = CaptionGenerator(
        CaptionGeneratorConfig(output_dir=tmp_path / "captions")
    )
    with pytest.raises(InvalidCaptionTimingError, match="positive duration"):
        generator.generate(timeline(tmp_path, [speech(11.0, None, "invalid")]), [score(tmp_path)])

    with pytest.raises(InvalidCaptionTimingError, match="after"):
        generator.generate(timeline(tmp_path, []), [score(tmp_path, 10.0, 10.0)])

    mismatched_candidate = ClipCandidate(
        source_video_path=tmp_path / "different.mp4",
        start_seconds=10.0,
        end_seconds=20.0,
        reason="different source",
    )
    with pytest.raises(InvalidCaptionSourceError, match="does not match"):
        generator.generate(
            timeline(tmp_path, []),
            [ClipScore(candidate=mismatched_candidate, overall_score=0.9)],
        )


def test_caption_generator_preserves_existing_file_when_overwrite_is_disabled(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "captions"
    output_dir.mkdir()
    existing = output_dir / "source.captions-10000-20000.srt"
    existing.write_text("existing captions\n", encoding="utf-8")
    generator = CaptionGenerator(
        CaptionGeneratorConfig(output_dir=output_dir, overwrite_existing=False)
    )

    artifact = generator.generate(
        timeline(tmp_path, [speech(11.0, 1.0, "new")]),
        [score(tmp_path)],
    )[0]

    assert artifact.path == existing
    assert existing.read_text(encoding="utf-8") == "existing captions\n"


def test_srt_formatter_uses_millisecond_precision_and_normalizes_newlines() -> None:
    content = SrtCaptionFormatter().format(
        [
            CaptionCue(
                index=99,
                start_seconds=3661.2344,
                end_seconds=3662.3456,
                text="line one\r\nline two",
            )
        ]
    )

    assert content == (
        "1\n01:01:01,234 --> 01:01:02,346\nline one\nline two\n"
    )
