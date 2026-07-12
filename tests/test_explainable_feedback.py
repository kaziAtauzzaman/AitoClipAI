import copy
import json
from pathlib import Path

import pytest

from core import (
    AggregatedTimeline,
    ClipCandidate,
    ClipScore,
    FeatureTimeline,
    Observation,
    RenderJob,
    TimelineGroup,
)
from explainable_feedback import (
    ExplainableFeedbackConfig,
    ExplainableFeedbackGenerator,
    ExplainableFeedbackService,
    FeedbackAssociationError,
)
from pipeline import (
    MediaProbeResult,
    MediaStreamProbe,
    PipelineValidationReport,
    PrerecordedPipelineResult,
    RenderedArtifactValidation,
)


def pipeline_result(tmp_path: Path) -> PrerecordedPipelineResult:
    source = tmp_path / "source.mp4"
    direct = Observation(1.0, "whisper", "speech", "exciting", 1.0, 0.9)
    context = Observation(1.5, "audio", "volume_spike", 0.8, 0.1, 0.8)
    outside = Observation(8.0, "audio", "volume_spike", 0.5)
    timeline = FeatureTimeline(
        media_path=source,
        audio_path=tmp_path / "audio.wav",
        timeline_path=tmp_path / "timeline.json",
        timeline=AggregatedTimeline(
            groups=[
                TimelineGroup(item.timestamp_seconds, [item])
                for item in (direct, context, outside)
            ]
        ),
    )
    rendered = ClipCandidate(
        source,
        0.5,
        2.5,
        "speech plus loudness",
        ["speech", "volume_spike"],
        metadata={
            "confidence": 0.8,
            "contributing_observations": [direct],
            "signal_contributions": [{"signal": "speech", "value": 0.9}],
        },
    )
    passed = ClipCandidate(
        source,
        3.0,
        4.0,
        "passed but limited",
        metadata={"contributing_observations": []},
    )
    below = ClipCandidate(
        source,
        7.5,
        8.5,
        "below threshold",
        metadata={"contributing_observations": []},
    )
    scores = [
        ClipScore(
            rendered,
            0.8,
            {"speech_excitement": 0.3, "loudness_peaks": 0.2},
            "authoritative rendered rationale",
            True,
        ),
        ClipScore(passed, 0.7, {"speech_excitement": 0.25}, "passed rationale", True),
        ClipScore(below, 0.1, {"speech_excitement": 0.01}, "below rationale", False),
    ]
    output = tmp_path / "clip.mp4"
    output.write_bytes(b"playable")
    job = RenderJob(rendered, output, metadata={"rank": 1})
    video = MediaStreamProbe("video", "h264", 0.0, 2.0)
    audio = MediaStreamProbe("audio", "aac", 0.0, 2.0)
    artifact = RenderedArtifactValidation(
        output,
        output.stat().st_size,
        video,
        audio,
        2.0,
        {"exists": True, "video_stream": True, "audio_stream": True},
    )
    validation = PipelineValidationReport(
        "passed",
        "local",
        source,
        MediaProbeResult(source, "mp4", 9.0, [video, audio]),
        timeline.timeline_path,
        ["audio", "whisper"],
        ["audio", "whisper"],
        [],
        3,
        3,
        2,
        [artifact],
    )
    return PrerecordedPipelineResult(
        timeline,
        [rendered, passed, below],
        scores,
        scores[:2],
        [job],
        validation,
        tmp_path / "validation-report.json",
    )


def test_feedback_explains_every_score_and_preserves_authoritative_values(
    tmp_path: Path,
) -> None:
    result = pipeline_result(tmp_path)

    report = ExplainableFeedbackGenerator().generate(result)

    assert report.scored_candidate_count == len(result.scores) == 3
    assert [clip.selection_status for clip in report.clips] == [
        "rendered",
        "passed_not_rendered",
        "below_threshold",
    ]
    first = report.clips[0]
    assert [(item.signal, item.contribution) for item in first.score_contributions] == list(
        result.scores[0].score_components.items()
    )
    assert first.scorer_rationale == result.scores[0].rationale
    assert first.passed_threshold is True
    assert first.identity.resolved_source_path == (tmp_path / "source.mp4").resolve()
    assert first.identity.start_microseconds == 500_000
    assert first.identity.end_microseconds == 2_500_000


def test_feedback_includes_overlapping_evidence_and_marks_direct_contributors(
    tmp_path: Path,
) -> None:
    report = ExplainableFeedbackGenerator().generate(pipeline_result(tmp_path))

    evidence = report.clips[0].supporting_evidence
    assert [(item.observer, item.type) for item in evidence] == [
        ("whisper", "speech"),
        ("audio", "volume_spike"),
    ]
    assert [item.direct_candidate_contributor for item in evidence] == [True, False]
    assert report.clips[2].supporting_evidence[0].timestamp_seconds == 8.0


def test_render_feedback_contains_playable_validation_evidence(tmp_path: Path) -> None:
    report = ExplainableFeedbackGenerator().generate(pipeline_result(tmp_path))

    render = report.clips[0].render
    assert render is not None
    assert render.rank == 1
    assert render.video_codec == "h264"
    assert render.audio_codec == "aac"
    assert render.duration_seconds == 2.0
    assert render.size_bytes > 0
    assert all(render.checks.values())
    assert report.clips[1].render is None


def test_feedback_is_deterministic_and_does_not_mutate_inputs(tmp_path: Path) -> None:
    result = pipeline_result(tmp_path)
    original = copy.deepcopy(result)
    path = tmp_path / "reports" / "explainable-feedback.json"
    service = ExplainableFeedbackService(ExplainableFeedbackConfig(path))

    first_report, first_path = service.create(result)
    first_json = first_path.read_bytes()
    second_report, second_path = service.create(result)

    assert first_report == second_report
    assert first_json == second_path.read_bytes()
    assert result == original
    data = json.loads(first_json.decode("utf-8"))
    assert data["report_type"] == "explainable_heuristic_provenance"
    assert len(data["clips"]) == 3


def test_candidate_identity_rounds_boundaries_to_microseconds(tmp_path: Path) -> None:
    result = pipeline_result(tmp_path)
    candidate = result.scores[0].candidate
    replacement = ClipCandidate(
        candidate.source_video_path,
        0.0000005,
        1.0000005,
        candidate.reason,
        metadata={"contributing_observations": []},
    )
    score = ClipScore(replacement, 0.1, passed_threshold=False)
    result = PrerecordedPipelineResult(
        result.feature_timeline,
        [replacement],
        [score],
        [],
        [],
        PipelineValidationReport(
            "passed",
            "local",
            result.validation_report.source_path,
            result.validation_report.source_metadata,
            result.validation_report.timeline_path,
            [],
            [],
            [],
            1,
            1,
            0,
            [],
        ),
        result.report_path,
    )

    identity = ExplainableFeedbackGenerator().generate(result).clips[0].identity

    assert identity.start_microseconds == 1
    assert identity.end_microseconds == 1_000_001


def test_feedback_rejects_duplicate_scored_candidate_identity(tmp_path: Path) -> None:
    result = pipeline_result(tmp_path)
    result.scores.append(result.scores[0])

    with pytest.raises(FeedbackAssociationError, match="Duplicate scored"):
        ExplainableFeedbackGenerator().generate(result)


def test_feedback_rejects_duplicate_render_and_artifact_associations(
    tmp_path: Path,
) -> None:
    result = pipeline_result(tmp_path)
    result.render_jobs.append(result.render_jobs[0])
    with pytest.raises(FeedbackAssociationError, match="Duplicate render-job"):
        ExplainableFeedbackGenerator().generate(result)

    result = pipeline_result(tmp_path)
    result.validation_report.rendered_artifacts.append(
        result.validation_report.rendered_artifacts[0]
    )
    with pytest.raises(FeedbackAssociationError, match="Duplicate artifact"):
        ExplainableFeedbackGenerator().generate(result)


def test_feedback_rejects_unmatched_render_or_artifact(tmp_path: Path) -> None:
    result = pipeline_result(tmp_path)
    result.validation_report.rendered_artifacts.clear()
    with pytest.raises(FeedbackAssociationError, match="no artifact validation"):
        ExplainableFeedbackGenerator().generate(result)

    result = pipeline_result(tmp_path)
    result.render_jobs.clear()
    with pytest.raises(FeedbackAssociationError, match="does not match"):
        ExplainableFeedbackGenerator().generate(result)
