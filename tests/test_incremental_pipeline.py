from pathlib import Path
import subprocess

import pytest

from candidate_selection import CandidateSelector
from clip_rendering import ClipRenderer, ClipRendererConfig
from core import (
    AggregatedTimeline,
    ClipCandidate,
    ClipScore,
    FeatureTimeline,
    Observation,
    ObserverResult,
    RenderJob,
    TimelineGroup,
)
from pipeline import (
    candidate_fingerprint,
    CompletedTimelineReplayAdapter,
    CoordinatorLifecycle,
    IncrementalEOF,
    IncrementalPipelineConfig,
    IncrementalPrerecordedCoordinator,
    ObserverWatermarks,
    RenderLifecycleState,
)


class MarkerGenerator:
    maximum_backtrack_seconds = 5.0
    incremental_deterministic = True

    def generate(self, timeline: FeatureTimeline) -> list[ClipCandidate]:
        return [
            ClipCandidate(
                timeline.media_path,
                float(item.value["start"]),
                float(item.value["end"]),
                str(item.value["name"]),
                metadata={
                    "score": float(item.value["score"]),
                    "contributing_observations": [item],
                },
            )
            for result in timeline.timeline.observer_results
            for item in result.observations
        ]


class MarkerScorer:
    candidate_local_deterministic = True

    def score(self, candidates) -> list[ClipScore]:
        return sorted(
            [
                ClipScore(
                    candidate,
                    candidate.metadata["score"],
                    passed_threshold=True,
                )
                for candidate in candidates
            ],
            key=lambda item: -item.overall_score,
        )


class RecordingRenderer:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.calls: list[tuple[ClipScore, int]] = []

    def render_one(self, score: ClipScore, identity: int) -> RenderJob:
        self.calls.append((score, identity))
        return RenderJob(
            score.candidate,
            self.output_dir / f"clip-{identity:03d}.mp4",
            metadata={"rank": identity},
        )


def feature_timeline(
    tmp_path: Path,
    markers: list[dict],
    *,
    source_id: str = "fixture-source",
    observer: str = "fixture",
) -> FeatureTimeline:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    observations = [
        Observation(
            timestamp_seconds=float(marker["seen_at"]),
            duration_seconds=float(marker.get("duration", 0.0)),
            observer=observer,
            type="candidate",
            value=marker,
        )
        for marker in markers
    ]
    result = ObserverResult(observer=observer, observations=observations)
    return FeatureTimeline(
        media_path=source,
        audio_path=tmp_path / "audio.wav",
        timeline_path=tmp_path / "timeline.json",
        timeline=AggregatedTimeline(
            groups=[TimelineGroup(item.timestamp_seconds, [item]) for item in observations],
            observer_results=[result],
        ),
        metadata={"source_id": source_id},
    )


def coordinator(
    tmp_path: Path,
    renderer: RecordingRenderer,
    *,
    generator=None,
    scorer=None,
    required_observers=("fixture",),
):
    return IncrementalPrerecordedCoordinator(
        candidate_generator=generator or MarkerGenerator(),
        candidate_scorer=scorer or MarkerScorer(),
        candidate_selector=CandidateSelector(),
        clip_renderer=renderer,
        config=IncrementalPipelineConfig(required_observers=required_observers),
    )


def watermarks(value: float, observer: str = "fixture") -> ObserverWatermarks:
    return ObserverWatermarks({observer: value})


def eof(value: float, observer: str = "fixture") -> IncrementalEOF:
    return IncrementalEOF(value, watermarks(value, observer))


def test_coordinator_uses_only_explicit_observer_watermark(tmp_path: Path) -> None:
    timeline = feature_timeline(
        tmp_path,
        [{"seen_at": 1, "start": 0, "end": 5, "score": 0.8, "name": "one"}],
    )
    renderer = RecordingRenderer(tmp_path)
    pipeline = coordinator(tmp_path, renderer)

    assert pipeline.advance(timeline, watermarks(9.9)) == []
    assert pipeline.advance(timeline, watermarks(10))
    assert pipeline.watermark_seconds == 10


def test_replay_adapter_withholds_retroactive_observation(tmp_path: Path) -> None:
    timeline = feature_timeline(
        tmp_path,
        [
            {"seen_at": 5, "start": 0, "end": 10, "score": 0.6, "name": "weak"},
            {
                "seen_at": 0,
                "duration": 100,
                "start": 1,
                "end": 11,
                "score": 0.9,
                "name": "retroactive-strong",
            },
        ],
    )
    adapter = CompletedTimelineReplayAdapter()

    assert adapter.watermarks_at(timeline, 70).stable_through == {"fixture": 0}


class UnsafeGenerator(MarkerGenerator):
    maximum_backtrack_seconds = float("inf")


class GlobalStateScorer(MarkerScorer):
    candidate_local_deterministic = False


def test_unsafe_generator_backtracking_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        coordinator(tmp_path, RecordingRenderer(tmp_path), generator=UnsafeGenerator())


def test_global_state_scorer_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="candidate-local"):
        coordinator(tmp_path, RecordingRenderer(tmp_path), scorer=GlobalStateScorer())


def test_fingerprint_is_cross_platform_path_independent() -> None:
    windows = ClipCandidate(Path(r"C:\media\source.mp4"), 0, 10, "same")
    linux = ClipCandidate(Path("/media/source.mp4"), 0, 10, "same")

    assert candidate_fingerprint(windows, "youtube:abc") == candidate_fingerprint(
        linux, "youtube:abc"
    )


@pytest.mark.parametrize("bad", [object(), float("nan"), float("inf")])
def test_fingerprint_rejects_unsupported_or_nonfinite_metadata(bad: object) -> None:
    candidate = ClipCandidate(Path("source.mp4"), 0, 10, "bad", metadata={"bad": bad})

    with pytest.raises((TypeError, ValueError)):
        candidate_fingerprint(candidate, "stable-source")


class FailingOnceRenderer(RecordingRenderer):
    def __init__(self, output_dir: Path, error: BaseException) -> None:
        super().__init__(output_dir)
        self.error = error
        self.attempted_identities: list[int] = []

    def render_one(self, score: ClipScore, identity: int) -> RenderJob:
        self.attempted_identities.append(identity)
        if len(self.attempted_identities) == 1:
            raise self.error
        return super().render_one(score, identity)


def test_render_retry_reuses_identity_and_output(tmp_path: Path) -> None:
    timeline = feature_timeline(
        tmp_path,
        [{"seen_at": 1, "start": 0, "end": 5, "score": 0.8, "name": "retry"}],
    )
    renderer = FailingOnceRenderer(tmp_path, RuntimeError("failed"))
    pipeline = coordinator(tmp_path, renderer)

    with pytest.raises(RuntimeError):
        pipeline.advance(timeline, watermarks(10))
    jobs = pipeline.advance(timeline, watermarks(10))

    assert renderer.attempted_identities == [1, 1]
    assert [job.output_path.name for job in jobs] == ["clip-001.mp4"]


def test_interruption_recovers_rendering_as_retryable(tmp_path: Path) -> None:
    timeline = feature_timeline(
        tmp_path,
        [{"seen_at": 1, "start": 0, "end": 5, "score": 0.8, "name": "retry"}],
    )
    renderer = FailingOnceRenderer(tmp_path, KeyboardInterrupt())
    pipeline = coordinator(tmp_path, renderer)
    score = MarkerScorer().score(MarkerGenerator().generate(timeline))[0]

    with pytest.raises(KeyboardInterrupt):
        pipeline.advance(timeline, watermarks(10))
    assert pipeline.render_state(score) is RenderLifecycleState.FAILED
    assert pipeline.advance(timeline, watermarks(10))
    assert pipeline.render_state(score) is RenderLifecycleState.RENDERED


def test_coordinator_lifecycle_is_explicitly_single_use(tmp_path: Path) -> None:
    timeline = feature_timeline(tmp_path, [])
    pipeline = coordinator(tmp_path, RecordingRenderer(tmp_path))

    pipeline.flush(timeline, eof(1))
    assert pipeline.lifecycle is CoordinatorLifecycle.FLUSHED
    with pytest.raises(RuntimeError, match="already been flushed"):
        pipeline.advance(timeline, watermarks(1))
    with pytest.raises(RuntimeError, match="already been flushed"):
        pipeline.flush(timeline, eof(1))


def test_coordinator_rejects_second_source(tmp_path: Path) -> None:
    first = feature_timeline(tmp_path, [], source_id="first")
    second = feature_timeline(tmp_path, [], source_id="second")
    pipeline = coordinator(tmp_path, RecordingRenderer(tmp_path))

    pipeline.advance(first, watermarks(0))
    with pytest.raises(RuntimeError, match="single-use"):
        pipeline.advance(second, watermarks(0))


def test_eof_waits_for_every_required_observer(tmp_path: Path) -> None:
    timeline = feature_timeline(tmp_path, [])
    pipeline = coordinator(
        tmp_path,
        RecordingRenderer(tmp_path),
        required_observers=("fixture", "other"),
    )

    with pytest.raises(ValueError, match="Missing required observer"):
        pipeline.flush(timeline, IncrementalEOF(10, ObserverWatermarks({"fixture": 10})))
    with pytest.raises(ValueError, match="final media duration"):
        pipeline.flush(
            timeline,
            IncrementalEOF(10, ObserverWatermarks({"fixture": 10, "other": 9})),
        )


def test_later_stronger_overlap_and_batch_selection_match(tmp_path: Path) -> None:
    timeline = feature_timeline(
        tmp_path,
        [
            {"seen_at": 1, "start": 0, "end": 10, "score": 0.6, "name": "weak"},
            {"seen_at": 2, "start": 1, "end": 11, "score": 0.9, "name": "strong"},
            {"seen_at": 3, "start": 20, "end": 25, "score": 0.7, "name": "later"},
        ],
    )
    renderer = RecordingRenderer(tmp_path)
    pipeline = coordinator(tmp_path, renderer)
    pipeline.advance(timeline, watermarks(30))
    result = pipeline.flush(timeline, eof(30))
    scores = MarkerScorer().score(MarkerGenerator().generate(timeline))
    batch = CandidateSelector().select(scores)

    assert [call[0].candidate.reason for call in renderer.calls] == ["strong", "later"]
    assert {item.candidate.reason for item in result.selected_scores} == {
        item.candidate.reason for item in batch.selected
    }
    assert {item.score.candidate.reason for item in result.suppressed} == {
        item.score.candidate.reason for item in batch.suppressed
    }


class CreatingRunner:
    def run(self, command):
        command = list(command)
        Path(command[-1]).write_bytes(b"clip")
        return subprocess.CompletedProcess(command, 0, "", "")


def test_batch_renderer_limit_remains_isolated(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    renderer = ClipRenderer(
        ClipRendererConfig(output_dir=tmp_path / "clips", maximum_clips=1),
        runner=CreatingRunner(),
        executable_locator=lambda _: "ffmpeg",
    )
    scores = [
        ClipScore(ClipCandidate(source, 0, 5, "first"), 0.8),
        ClipScore(ClipCandidate(source, 10, 15, "second"), 0.7),
    ]

    assert len(renderer.render(scores)) == 1
