from pathlib import Path

import pytest

from aggregation import FeatureAggregator
from candidate_generation import CandidateGenerator
from candidate_scoring import CandidateScorer
from candidate_selection import CandidateSelector
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
    IncrementalEOF,
    IncrementalPipelineConfig,
    IncrementalPrerecordedCoordinator,
    ObserverWatermarks,
    ObserverDeltaIdentity,
)


class MarkerGenerator:
    maximum_backtrack_seconds = 5.0
    maximum_competition_seconds = 5.0
    incremental_deterministic = True

    def generate(self, timeline):
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


class TrackingScorer:
    candidate_local_deterministic = True

    def __init__(self):
        self.calls = 0
        self.candidates = 0

    def score(self, candidates):
        values = list(candidates)
        self.calls += 1
        self.candidates += len(values)
        return [
            ClipScore(
                item,
                float(item.metadata["score"]),
                passed_threshold=True,
            )
            for item in values
        ]


class Renderer:
    def __init__(self, root: Path):
        self.root = root
        self.calls = []

    def render_one(self, score, identity):
        self.calls.append((score, identity))
        return RenderJob(
            score.candidate,
            self.root / f"clip-{identity:03d}.mp4",
            metadata={"rank": identity},
        )


def delta_timeline(tmp_path, observations, observer="fixture", metadata=None):
    result = ObserverResult(observer, list(observations), dict(metadata or {}))
    return FeatureTimeline(
        media_path=tmp_path / "source.mp4",
        audio_path=tmp_path / "audio.wav",
        timeline_path=tmp_path / "timeline.json",
        timeline=FeatureAggregator().aggregate([result]),
        metadata={"source_id": "state-cache-fixture"},
    )


def marker(seen, start, end, score, name):
    return Observation(
        float(seen),
        "fixture",
        "candidate",
        {"start": start, "end": end, "score": score, "name": name},
    )


def marker_coordinator(tmp_path, scorer=None):
    return IncrementalPrerecordedCoordinator(
        MarkerGenerator(),
        scorer or TrackingScorer(),
        CandidateSelector(),
        Renderer(tmp_path),
        IncrementalPipelineConfig(required_observers=("fixture",)),
    )


def delta_identity(observer, sequence, eof=False):
    return ObserverDeltaIdentity(
        "state-cache-fixture",
        "incremental:state-cache-fixture",
        observer,
        sequence,
        eof,
    )


def audio_coordinator(tmp_path, scorer=None):
    return IncrementalPrerecordedCoordinator(
        MarkerGenerator(),
        scorer or TrackingScorer(),
        CandidateSelector(),
        Renderer(tmp_path),
        IncrementalPipelineConfig(required_observers=("audio",)),
    )


def audio_diagnostic(kind, timestamp, duration, name="audio-window"):
    return Observation(
        timestamp,
        "audio",
        kind,
        {
            "start": timestamp,
            "end": timestamp + min(duration, 0.5),
            "score": 0.8,
            "name": name,
        },
        duration_seconds=duration,
    )


def audio_identity(sequence, eof=False):
    return ObserverDeltaIdentity(
        "state-cache-fixture",
        "incremental:state-cache-fixture",
        "audio",
        sequence,
        eof,
    )


def audio_metadata(frames):
    return {"incremental_frames_processed": frames, "sample_rate_hz": 10}


def test_audio_diagnostic_interval_may_extend_beyond_stable_watermark(tmp_path):
    coordinator = audio_coordinator(tmp_path)
    intensity = audio_diagnostic("speaking_intensity", 8.5, 1.0)

    coordinator.advance_delta(
        delta_timeline(
            tmp_path, [intensity], observer="audio", metadata=audio_metadata(100)
        ),
        ObserverWatermarks({"audio": 9.0}),
        audio_identity(0),
    )

    assert coordinator.state_metrics.active_observations == 1


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({"sample_rate_hz": 10}, "incremental_frames_processed"),
        ({"incremental_frames_processed": 100}, "sample_rate_hz"),
        (
            {"incremental_frames_processed": True, "sample_rate_hz": 10},
            "incremental_frames_processed",
        ),
        (
            {"incremental_frames_processed": 100, "sample_rate_hz": False},
            "sample_rate_hz",
        ),
        (
            {"incremental_frames_processed": "100", "sample_rate_hz": 10},
            "incremental_frames_processed",
        ),
        (
            {"incremental_frames_processed": 100.0, "sample_rate_hz": 10},
            "incremental_frames_processed",
        ),
        (
            {"incremental_frames_processed": float("nan"), "sample_rate_hz": 10},
            "incremental_frames_processed",
        ),
        (
            {"incremental_frames_processed": float("inf"), "sample_rate_hz": 10},
            "incremental_frames_processed",
        ),
        (
            {"incremental_frames_processed": 100, "sample_rate_hz": "10"},
            "sample_rate_hz",
        ),
        (
            {"incremental_frames_processed": 100, "sample_rate_hz": 10 + 0j},
            "sample_rate_hz",
        ),
        (
            {"incremental_frames_processed": 100, "sample_rate_hz": float("nan")},
            "sample_rate_hz",
        ),
        (
            {"incremental_frames_processed": 100, "sample_rate_hz": float("inf")},
            "sample_rate_hz",
        ),
        (
            {"incremental_frames_processed": 0, "sample_rate_hz": 10},
            "incremental_frames_processed",
        ),
        (
            {"incremental_frames_processed": -1, "sample_rate_hz": 10},
            "incremental_frames_processed",
        ),
        (
            {"incremental_frames_processed": 100, "sample_rate_hz": 0},
            "sample_rate_hz",
        ),
        (
            {"incremental_frames_processed": 100, "sample_rate_hz": -10},
            "sample_rate_hz",
        ),
    ],
)
def test_strict_audio_diagnostic_requires_valid_frame_metadata(
    tmp_path, metadata, message
):
    coordinator = audio_coordinator(tmp_path)
    intensity = audio_diagnostic("speaking_intensity", 8.5, 1.0)

    with pytest.raises(ValueError, match=message):
        coordinator.advance_delta(
            delta_timeline(
                tmp_path, [intensity], observer="audio", metadata=metadata
            ),
            ObserverWatermarks({"audio": 9.0}),
            audio_identity(0),
        )


def test_audio_closed_silence_starting_at_watermark_is_stable(tmp_path):
    coordinator = audio_coordinator(tmp_path)
    silence = audio_diagnostic("silence", 9.0, 1.0, "closed-silence")

    coordinator.advance_delta(
        delta_timeline(
            tmp_path, [silence], observer="audio", metadata=audio_metadata(100)
        ),
        ObserverWatermarks({"audio": 9.0}),
        audio_identity(0),
    )

    assert coordinator.state_metrics.active_observations == 1


@pytest.mark.parametrize(
    ("timestamp", "duration", "message"),
    [
        (9.1, 0.5, "stable watermark"),
        (8.5, 2.0, "processed frames"),
    ],
)
def test_future_audio_diagnostic_observation_is_rejected(
    tmp_path, timestamp, duration, message
):
    coordinator = audio_coordinator(tmp_path)
    future = audio_diagnostic("speaking_intensity", timestamp, duration)

    with pytest.raises(ValueError, match=message):
        coordinator.advance_delta(
            delta_timeline(
                tmp_path, [future], observer="audio", metadata=audio_metadata(100)
            ),
            ObserverWatermarks({"audio": 9.0}),
            audio_identity(0),
        )


def test_whisper_segment_must_end_within_stable_watermark(tmp_path):
    coordinator = IncrementalPrerecordedCoordinator(
        MarkerGenerator(), TrackingScorer(), CandidateSelector(), Renderer(tmp_path),
        IncrementalPipelineConfig(required_observers=("whisper",)),
    )
    segment = Observation(
        8.5,
        "whisper",
        "speech",
        {"start": 8.5, "end": 9.0, "score": 0.8, "name": "speech"},
        duration_seconds=1.0,
    )

    with pytest.raises(ValueError, match="stable watermark"):
        coordinator.advance_delta(
            delta_timeline(tmp_path, [segment], observer="whisper"),
            ObserverWatermarks({"whisper": 9.0}),
            ObserverDeltaIdentity(
                "state-cache-fixture",
                "incremental:state-cache-fixture",
                "whisper",
                0,
            ),
        )


def test_audio_boundary_delta_retry_is_idempotent(tmp_path):
    scorer = TrackingScorer()
    coordinator = audio_coordinator(tmp_path, scorer)
    intensity = audio_diagnostic("speaking_intensity", 8.5, 1.0)
    timeline = delta_timeline(
        tmp_path, [intensity], observer="audio", metadata=audio_metadata(100)
    )
    identity = audio_identity(0)

    coordinator.advance_delta(
        timeline, ObserverWatermarks({"audio": 9.0}), identity
    )
    assert coordinator.advance_delta(
        timeline, ObserverWatermarks({"audio": 9.0}), identity
    ) == []

    assert scorer.candidates == 1
    assert coordinator.state_metrics.active_observations == 1


def test_audio_diagnostic_end_is_retained_then_evicted_as_context(tmp_path):
    coordinator = audio_coordinator(tmp_path)
    intensity = audio_diagnostic("speaking_intensity", 8.5, 1.0)
    coordinator.advance_delta(
        delta_timeline(
            tmp_path, [intensity], observer="audio", metadata=audio_metadata(100)
        ),
        ObserverWatermarks({"audio": 9.0}),
        audio_identity(0),
    )
    assert coordinator.state_metrics.active_observations == 1

    coordinator.advance_delta(
        delta_timeline(
            tmp_path, [], observer="audio", metadata=audio_metadata(300)
        ),
        ObserverWatermarks({"audio": 30.0}),
        audio_identity(1),
    )

    assert coordinator.state_metrics.active_observations == 0


def test_pending_score_and_fingerprint_cache_hits_return_same_score(tmp_path):
    scorer = TrackingScorer()
    coordinator = marker_coordinator(tmp_path, scorer)
    first = marker(10, 10, 20, 0.8, "pending")

    coordinator.advance_delta(delta_timeline(tmp_path, [first]), ObserverWatermarks({"fixture": 20.0}))
    cached = coordinator.result.scores[0]
    fingerprint_count = coordinator.state_metrics.score_fingerprints
    coordinator.advance_delta(delta_timeline(tmp_path, []), ObserverWatermarks({"fixture": 21.0}))

    assert coordinator.result.scores[0] is cached
    assert scorer.candidates == 1
    assert coordinator.state_metrics.score_fingerprints == fingerprint_count


def test_finalized_candidate_is_never_rescored_or_refingerprinted(tmp_path):
    scorer = TrackingScorer()
    coordinator = marker_coordinator(tmp_path, scorer)
    value = marker(1, 0, 5, 0.8, "final")

    jobs = coordinator.advance_delta(delta_timeline(tmp_path, [value]), ObserverWatermarks({"fixture": 10.0}))
    assert len(jobs) == 1
    after_finalization = coordinator.state_metrics
    for watermark in (11.0, 12.0, 13.0):
        coordinator.advance_delta(delta_timeline(tmp_path, []), ObserverWatermarks({"fixture": watermark}))

    final = coordinator.state_metrics
    assert scorer.candidates == 1
    assert final.scored_candidates == after_finalization.scored_candidates
    assert final.score_fingerprints == after_finalization.score_fingerprints
    assert final.candidate_fingerprints == after_finalization.candidate_fingerprints


def test_retroactive_delta_inside_horizon_changes_overlap_winner(tmp_path):
    scorer = TrackingScorer()
    coordinator = marker_coordinator(tmp_path, scorer)
    weak = marker(5, 0, 10, 0.6, "weak")
    strong = marker(7, 1, 11, 0.9, "strong")

    assert coordinator.advance_delta(delta_timeline(tmp_path, [weak]), ObserverWatermarks({"fixture": 7.0})) == []
    assert coordinator.advance_delta(delta_timeline(tmp_path, [strong]), ObserverWatermarks({"fixture": 12.0})) == []
    jobs = coordinator.advance_delta(delta_timeline(tmp_path, []), ObserverWatermarks({"fixture": 16.0}))

    assert [job.candidate.reason for job in jobs] == ["strong"]
    assert [item.score.candidate.reason for item in coordinator.result.suppressed] == ["weak"]


def test_unsafe_overlap_group_keeps_earlier_member_until_group_is_final(tmp_path):
    coordinator = marker_coordinator(tmp_path)
    weak = marker(5, 0, 10, 0.6, "weak")
    strong = marker(9, 2, 14, 0.9, "strong")

    assert coordinator.advance_delta(
        delta_timeline(tmp_path, [weak, strong]),
        ObserverWatermarks({"fixture": 15.0}),
    ) == []
    jobs = coordinator.advance_delta(
        delta_timeline(tmp_path, []),
        ObserverWatermarks({"fixture": 19.0}),
    )

    assert [job.candidate.reason for job in jobs] == ["strong"]
    assert [item.score.candidate.reason for item in coordinator.result.suppressed] == ["weak"]


def test_delta_older_than_retained_revision_horizon_is_rejected(tmp_path):
    coordinator = marker_coordinator(tmp_path)
    coordinator.advance_delta(delta_timeline(tmp_path, []), ObserverWatermarks({"fixture": 30.0}))

    with pytest.raises(ValueError, match="older than the active revision horizon"):
        coordinator.advance_delta(
            delta_timeline(tmp_path, [marker(1, 0, 5, 0.8, "too-old")]),
            ObserverWatermarks({"fixture": 31.0}),
        )


def test_explicit_delta_identity_is_idempotent_and_rejects_sequence_misuse(tmp_path):
    scorer = TrackingScorer()
    coordinator = marker_coordinator(tmp_path, scorer)
    value = marker(1, 1, 3, 0.8, "once")
    timeline = delta_timeline(tmp_path, [value])
    identity = delta_identity("fixture", 0)

    coordinator.advance_delta(
        timeline, ObserverWatermarks({"fixture": 2.0}), identity
    )
    assert coordinator.advance_delta(
        timeline, ObserverWatermarks({"fixture": 2.0}), identity
    ) == []
    assert scorer.candidates == 1

    with pytest.raises(ValueError, match="must be 1"):
        coordinator.advance_delta(
            delta_timeline(tmp_path, []),
            ObserverWatermarks({"fixture": 3.0}),
            delta_identity("fixture", 2),
        )
    with pytest.raises(ValueError, match="backwards|regress"):
        coordinator.advance_delta(
            delta_timeline(tmp_path, []),
            ObserverWatermarks({"fixture": 1.0}),
            delta_identity("fixture", 1),
        )

    stalled = marker_coordinator(tmp_path)
    with pytest.raises(ValueError, match="must advance"):
        stalled.advance_delta(
            delta_timeline(tmp_path, []),
            ObserverWatermarks({"fixture": 0.0}),
            delta_identity("fixture", 0),
        )


def test_explicit_delta_rejects_unexpected_observer_and_repeated_observation(tmp_path):
    coordinator = marker_coordinator(tmp_path)
    value = marker(1, 1, 3, 0.8, "once")
    with pytest.raises(ValueError, match="Unexpected"):
        coordinator.advance_delta(
            delta_timeline(tmp_path, [value], observer="other"),
            ObserverWatermarks({"fixture": 2.0}),
            delta_identity("other", 0),
        )
    foreign = Observation(1.0, "other", "candidate", value.value)
    with pytest.raises(ValueError, match="owned"):
        coordinator.advance_delta(
            delta_timeline(tmp_path, [foreign]),
            ObserverWatermarks({"fixture": 2.0}),
            delta_identity("fixture", 0),
        )
    coordinator.advance_delta(
        delta_timeline(tmp_path, [value]),
        ObserverWatermarks({"fixture": 2.0}),
        delta_identity("fixture", 0),
    )
    with pytest.raises(ValueError, match="already accepted"):
        coordinator.advance_delta(
            delta_timeline(tmp_path, [value]),
            ObserverWatermarks({"fixture": 3.0}),
            delta_identity("fixture", 1),
        )


@pytest.mark.parametrize("failure", [RuntimeError("render failed"), KeyboardInterrupt()])
def test_retry_after_post_acceptance_failure_does_not_duplicate_input(tmp_path, failure):
    class FailsOnce(Renderer):
        def __init__(self, root):
            super().__init__(root)
            self.failed = False

        def render_one(self, score, identity):
            if not self.failed:
                self.failed = True
                raise failure
            return super().render_one(score, identity)

    scorer = TrackingScorer()
    renderer = FailsOnce(tmp_path)
    coordinator = IncrementalPrerecordedCoordinator(
        MarkerGenerator(), scorer, CandidateSelector(), renderer,
        IncrementalPipelineConfig(required_observers=("fixture",)),
    )
    timeline = delta_timeline(tmp_path, [marker(1, 0, 2, 0.8, "retry")])
    identity = delta_identity("fixture", 0)
    with pytest.raises(type(failure)):
        coordinator.advance_delta(
            timeline, ObserverWatermarks({"fixture": 7.0}), identity
        )
    jobs = coordinator.advance_delta(
        timeline, ObserverWatermarks({"fixture": 7.0}), identity
    )
    assert len(jobs) == 1
    assert scorer.candidates == 1
    assert len(coordinator.result.selected_scores) == 1


def test_finalized_observations_are_excluded_from_later_generator_calls(tmp_path):
    class RecordingGenerator(MarkerGenerator):
        def __init__(self):
            self.seen_names = []

        def generate(self, timeline):
            self.seen_names.append([
                item.value["name"]
                for result in timeline.timeline.observer_results
                for item in result.observations
            ])
            return super().generate(timeline)

    generator = RecordingGenerator()
    coordinator = IncrementalPrerecordedCoordinator(
        generator, TrackingScorer(), CandidateSelector(), Renderer(tmp_path),
        IncrementalPipelineConfig(required_observers=("fixture",)),
    )
    coordinator.advance_delta(
        delta_timeline(tmp_path, [marker(1, 0, 2, 0.8, "finalized")]),
        ObserverWatermarks({"fixture": 7.0}),
        delta_identity("fixture", 0),
    )
    coordinator.advance_delta(
        delta_timeline(tmp_path, [marker(20, 20, 22, 0.8, "new")]),
        ObserverWatermarks({"fixture": 27.0}),
        delta_identity("fixture", 1),
    )
    assert generator.seen_names == [["finalized"], ["new"]]


def test_continuous_overlap_chain_exceeding_declared_bound_is_rejected(tmp_path):
    coordinator = marker_coordinator(tmp_path)
    chain = [
        marker(1, 0, 10, 0.9, "chain-1"),
        marker(2, 2, 12, 0.8, "chain-2"),
        marker(3, 4, 14, 0.7, "chain-3"),
        marker(4, 6, 16, 0.6, "chain-4"),
    ]
    with pytest.raises(RuntimeError, match="exceeded the declared finite bound"):
        coordinator.advance_delta(
            delta_timeline(tmp_path, chain),
            ObserverWatermarks({"fixture": 5.0}),
            delta_identity("fixture", 0),
        )
    assert coordinator.state_metrics.active_observations == len(chain)
    assert coordinator.state_metrics.peak_unresolved_group_size == len(chain)


def test_eof_delta_retry_is_exactly_once_after_render_failure(tmp_path):
    class FailsOnce(Renderer):
        def __init__(self, root):
            super().__init__(root)
            self.failed = False

        def render_one(self, score, identity):
            if not self.failed:
                self.failed = True
                raise RuntimeError("render failed")
            return super().render_one(score, identity)

    scorer = TrackingScorer()
    coordinator = IncrementalPrerecordedCoordinator(
        MarkerGenerator(), scorer, CandidateSelector(), FailsOnce(tmp_path),
        IncrementalPipelineConfig(required_observers=("fixture",)),
    )
    timeline = delta_timeline(
        tmp_path, [marker(1, 0, 2, 0.8, "eof-retry")]
    )
    eof = IncrementalEOF(3.0, ObserverWatermarks({"fixture": 3.0}))
    with pytest.raises(RuntimeError, match="render failed"):
        coordinator.flush_delta(timeline, eof)
    result = coordinator.flush_delta(timeline, eof)

    assert scorer.candidates == 1
    assert len(result.selected_scores) == 1
    assert len(result.render_jobs) == 1
    with pytest.raises(RuntimeError, match="already been flushed"):
        coordinator.flush_delta(timeline, eof)


@pytest.mark.parametrize("after_ingest", [False, True])
def test_interruption_around_ingestion_receipt_commit_is_recoverable(
    tmp_path, monkeypatch, after_ingest
):
    scorer = TrackingScorer()
    coordinator = marker_coordinator(tmp_path, scorer)
    timeline = delta_timeline(
        tmp_path, [marker(1, 0, 2, 0.8, "transaction")]
    )
    identity = delta_identity("fixture", 0)
    original = coordinator._ingest_delta
    interrupted = False

    def interrupt_once(*args, **kwargs):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            if after_ingest:
                original(*args, **kwargs)
            raise KeyboardInterrupt()
        return original(*args, **kwargs)

    monkeypatch.setattr(coordinator, "_ingest_delta", interrupt_once)
    with pytest.raises(KeyboardInterrupt):
        coordinator.advance_delta(
            timeline, ObserverWatermarks({"fixture": 7.0}), identity
        )
    jobs = coordinator.advance_delta(
        timeline, ObserverWatermarks({"fixture": 7.0}), identity
    )
    assert len(jobs) == 1
    assert scorer.candidates == 1
    assert len(coordinator.result.selected_scores) == 1


def test_interruption_after_successful_render_does_not_duplicate_job_or_report(tmp_path, monkeypatch):
    scorer = TrackingScorer()
    renderer = Renderer(tmp_path)
    coordinator = IncrementalPrerecordedCoordinator(
        MarkerGenerator(), scorer, CandidateSelector(), renderer,
        IncrementalPipelineConfig(required_observers=("fixture",)),
    )
    timeline = delta_timeline(
        tmp_path, [marker(1, 0, 2, 0.8, "post-render")]
    )
    identity = delta_identity("fixture", 0)
    original = coordinator._render_plan
    interrupted = False

    def interrupt_after(plan):
        nonlocal interrupted
        jobs = original(plan)
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt()
        return jobs

    monkeypatch.setattr(coordinator, "_render_plan", interrupt_after)
    with pytest.raises(KeyboardInterrupt):
        coordinator.advance_delta(
            timeline, ObserverWatermarks({"fixture": 7.0}), identity
        )
    coordinator.advance_delta(
        timeline, ObserverWatermarks({"fixture": 7.0}), identity
    )
    result = coordinator.result
    assert len(renderer.calls) == 1
    assert len(result.selected_scores) == 1
    assert len(result.render_jobs) == 1
    assert len(result.scores) == 1


def speech(timestamp, text):
    return Observation(
        float(timestamp),
        "whisper",
        "speech",
        {"text": text, "speaker": None},
        duration_seconds=4.0,
        confidence=0.9,
    )


def complete_timeline(tmp_path, observations, duration):
    results = [
        ObserverResult("audio", [], {"duration_seconds": duration}),
        ObserverResult("whisper", list(observations), {"duration_seconds": duration}),
    ]
    return FeatureTimeline(
        media_path=tmp_path / "source.mp4",
        audio_path=tmp_path / "audio.wav",
        timeline_path=tmp_path / "timeline.json",
        timeline=FeatureAggregator().aggregate(results),
        metadata={"source_id": "state-cache-real-components"},
    )


def test_rolling_boundary_and_eof_match_complete_batch_decisions(tmp_path):
    observations = [
        speech(55, "first half of a boundary moment"),
        speech(58, "second half of a boundary moment"),
        speech(180, "a separate later moment"),
    ]
    duration = 300.0
    full = complete_timeline(tmp_path, observations, duration)
    generator = CandidateGenerator()
    scorer = CandidateScorer()
    selector = CandidateSelector()
    batch_scores = scorer.score(generator.generate(full))
    batch = selector.select([item for item in batch_scores if item.passed_threshold])
    renderer = Renderer(tmp_path)
    incremental = IncrementalPrerecordedCoordinator(
        generator,
        scorer,
        selector,
        renderer,
        IncrementalPipelineConfig(required_observers=("audio", "whisper")),
    )
    watermarks = {"audio": 0.0, "whisper": 0.0}
    for item in observations:
        watermarks = {"audio": item.timestamp_seconds + 4.0, "whisper": item.timestamp_seconds + 4.0}
        incremental.advance_delta(
            delta_timeline(
                tmp_path,
                [item],
                "whisper",
                {"duration_seconds": duration},
            ),
            ObserverWatermarks(watermarks),
        )
    result = incremental.flush_delta(
        delta_timeline(tmp_path, [], "whisper", {"duration_seconds": duration}),
        IncrementalEOF(duration, ObserverWatermarks({"audio": duration, "whisper": duration})),
    )

    def identity(score):
        return (
            score.candidate.start_seconds,
            score.candidate.end_seconds,
            score.overall_score,
            score.candidate.reason,
        )

    assert [identity(item) for item in result.scores] == [identity(item) for item in batch_scores]
    assert [identity(item) for item in result.selected_scores] == [identity(item) for item in batch.selected]
    assert [item.reason for item in result.suppressed] == [item.reason for item in batch.suppressed]


@pytest.mark.parametrize("count", [1000, 2000, 4000])
def test_synthetic_work_and_active_state_scale_linearly(tmp_path, count):
    scorer = TrackingScorer()
    coordinator = marker_coordinator(tmp_path, scorer)
    for index in range(count):
        timestamp = float(index * 20 + 1)
        coordinator.advance_delta(
            delta_timeline(
                tmp_path,
                [marker(timestamp, timestamp, timestamp + 2, 0.8, f"m-{index}")],
            ),
            ObserverWatermarks({"fixture": timestamp + 7}),
        )
    coordinator.flush_delta(
        delta_timeline(tmp_path, []),
        IncrementalEOF(float(count * 20 + 20), ObserverWatermarks({"fixture": float(count * 20 + 20)})),
    )
    metrics = coordinator.state_metrics

    assert metrics.scored_candidates == count
    assert metrics.score_fingerprints == count
    assert metrics.peak_active_observations <= 2
    assert len(coordinator.result.selected_scores) == count
