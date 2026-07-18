from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from aggregation import FeatureAggregator
from candidate_generation import (
    CandidateFamilyId,
    CandidateGenerationAdvance,
    CandidateGenerationCheckpoint,
    CandidateGenerationConfig,
    CandidateGenerator,
    ClosedCandidateFamily,
)
from candidate_scoring import CandidateScorer, CandidateScoringConfig
from candidate_selection import (
    CandidateSelectionConfig,
    CandidateSelectionResult,
    CandidateSelector,
)
from core import (
    AggregatedTimeline,
    ClipCandidate,
    ClipScore,
    FeatureTimeline,
    Observation,
    ObserverResult,
    RenderJob,
    SelectionPriorityContract,
    TimelineGroup,
)
from pipeline import (
    IncrementalEOF,
    IncrementalPipelineConfig,
    IncrementalPrerecordedCoordinator,
    ObserverWatermarks,
    ObserverDeltaIdentity,
    RenderLifecycleState,
)
from pipeline.incremental import candidate_fingerprint, score_fingerprint
from whisper_observer import finalized_speech_segment_identity
from incremental_generator_support import DeltaClosedFamilyGeneratorMixin


class MarkerGenerator(DeltaClosedFamilyGeneratorMixin):
    maximum_backtrack_seconds = 5.0
    maximum_competition_seconds = 5.0
    incremental_deterministic = True

    @staticmethod
    def revision_start_seconds(candidate):
        return candidate.start_seconds

    @classmethod
    def revision_stable_after_seconds(cls, candidate):
        return candidate.end_seconds

    @staticmethod
    def revision_partition_seconds(candidate):
        return candidate.end_seconds

    @staticmethod
    def earliest_unresolved_cluster_start_seconds(timeline, stable):
        return None

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
    selection_priority_contract = SelectionPriorityContract()

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
        self.completions = {}

    def render_one(self, score, identity):
        self.calls.append((score, identity))
        job = RenderJob(
            score.candidate,
            self.root / f"clip-{identity:03d}.mp4",
            metadata={"rank": identity},
        )
        self.completions[identity] = job
        return job

    def recover_render(self, score, identity):
        job = self.completions.get(identity)
        if job is not None and job.candidate != score.candidate:
            raise RuntimeError("Render identity changed candidate ownership.")
        return job


def delta_timeline(tmp_path, observations, observer="fixture", metadata=None):
    result = ObserverResult(observer, list(observations), dict(metadata or {}))
    return FeatureTimeline(
        media_path=tmp_path / "source.mp4",
        audio_path=tmp_path / "audio.wav",
        timeline_path=tmp_path / "timeline.json",
        timeline=FeatureAggregator().aggregate([result]),
        metadata={"source_id": "state-cache-fixture"},
    )


def multi_delta_timeline(tmp_path, results):
    return FeatureTimeline(
        media_path=tmp_path / "source.mp4",
        audio_path=tmp_path / "audio.wav",
        timeline_path=tmp_path / "timeline.json",
        timeline=FeatureAggregator().aggregate(list(results)),
        metadata={"source_id": "state-cache-fixture"},
    )


def marker(seen, start, end, score, name):
    return Observation(
        float(seen),
        "fixture",
        "candidate",
        {"start": start, "end": end, "score": score, "name": name},
    )


def observer_marker(observer, seen, start, end, score, name):
    return Observation(
        float(seen),
        observer,
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


def audio_metadata(frames, finalized_peaks=()):
    metadata = {"incremental_frames_processed": frames, "sample_rate_hz": 10}
    if finalized_peaks:
        metadata["finalized_peak_timestamps_seconds"] = tuple(finalized_peaks)
    return metadata


def audio_peak(timestamp):
    return Observation(
        timestamp,
        "audio",
        "peak",
        {
            "amplitude": 1.0,
            "start": timestamp,
            "end": timestamp + 0.1,
            "score": 0.8,
            "name": "finalized-peak",
        },
    )


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

    assert len(coordinator.result.scores) == 1


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

    assert len(coordinator.result.scores) == 1
    assert coordinator.result.scores[0].candidate.reason == "closed-silence"


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


def test_unknown_whisper_observation_must_end_within_stable_watermark(tmp_path):
    coordinator = IncrementalPrerecordedCoordinator(
        MarkerGenerator(), TrackingScorer(), CandidateSelector(), Renderer(tmp_path),
        IncrementalPipelineConfig(required_observers=("whisper",)),
    )
    segment = Observation(
        8.5,
        "whisper",
        "unknown",
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
    assert len(coordinator.result.scores) == 1


def test_audio_diagnostic_end_is_retained_then_evicted_as_context(tmp_path):
    coordinator = IncrementalPrerecordedCoordinator(
        CandidateGenerator(),
        CandidateScorer(),
        CandidateSelector(),
        Renderer(tmp_path),
        IncrementalPipelineConfig(required_observers=("audio",)),
    )
    intensity = Observation(
        8.5,
        "audio",
        "speaking_intensity",
        {"intensity": 0.8, "loudness_dbfs": -18.0},
        duration_seconds=1.0,
    )
    coordinator.advance_delta(
        delta_timeline(
            tmp_path, [intensity], observer="audio", metadata=audio_metadata(100)
        ),
        ObserverWatermarks({"audio": 9.0}),
        audio_identity(0),
    )
    assert coordinator.state_metrics.active_observations == 1
    assert coordinator._generation_checkpoint.retained_observation_count == 1
    retained = coordinator._generation_checkpoint._open_events[0].observation
    assert retained.timestamp_seconds == 8.5
    assert retained.duration_seconds == 1.0

    coordinator.advance_delta(
        delta_timeline(
            tmp_path, [], observer="audio", metadata=audio_metadata(700)
        ),
        ObserverWatermarks({"audio": 70.0}),
        audio_identity(1),
    )

    assert coordinator.state_metrics.active_observations == 0
    assert coordinator.result.scores == []
    assert coordinator._generation_checkpoint.retained_observation_count == 0
    assert coordinator._generation_checkpoint.next_family_ordinal == 1


def test_finalized_audio_peak_uses_processed_frontier_not_watermark(tmp_path):
    coordinator = audio_coordinator(tmp_path)
    peak = audio_peak(84.6530625)

    coordinator.advance_delta(
        delta_timeline(
            tmp_path,
            [peak],
            observer="audio",
            metadata=audio_metadata(850, [peak.timestamp_seconds]),
        ),
        ObserverWatermarks({"audio": 84.0}),
        audio_identity(0),
    )

    assert len(coordinator.result.scores) == 1
    assert coordinator.result.scores[0].candidate.reason == "finalized-peak"


def test_audio_peak_beyond_processed_frontier_is_rejected(tmp_path):
    coordinator = audio_coordinator(tmp_path)
    peak = audio_peak(85.1)

    with pytest.raises(ValueError, match="processed frames"):
        coordinator.advance_delta(
            delta_timeline(
                tmp_path,
                [peak],
                observer="audio",
                metadata=audio_metadata(850, [peak.timestamp_seconds]),
            ),
            ObserverWatermarks({"audio": 84.0}),
            audio_identity(0),
        )


@pytest.mark.parametrize(
    "metadata",
    [
        audio_metadata(850),
        {"incremental_frames_processed": 850, "sample_rate_hz": 10,
         "finalized_peak_timestamps_seconds": [True]},
        {"incremental_frames_processed": 850, "sample_rate_hz": 10,
         "finalized_peak_timestamps_seconds": [float("nan")]},
    ],
)
def test_unfinalized_or_malformed_audio_peak_is_rejected(tmp_path, metadata):
    coordinator = audio_coordinator(tmp_path)
    peak = audio_peak(84.6530625)

    with pytest.raises(ValueError, match="finalized|Finalized"):
        coordinator.advance_delta(
            delta_timeline(tmp_path, [peak], observer="audio", metadata=metadata),
            ObserverWatermarks({"audio": 84.0}),
            audio_identity(0),
        )


def test_finalized_audio_peak_delta_retry_is_idempotent(tmp_path):
    coordinator = audio_coordinator(tmp_path)
    peak = audio_peak(84.6530625)
    timeline = delta_timeline(
        tmp_path,
        [peak],
        observer="audio",
        metadata=audio_metadata(850, [peak.timestamp_seconds]),
    )
    identity = audio_identity(0)

    coordinator.advance_delta(
        timeline, ObserverWatermarks({"audio": 84.0}), identity
    )
    assert coordinator.advance_delta(
        timeline, ObserverWatermarks({"audio": 84.0}), identity
    ) == []
    assert len(coordinator.result.scores) == 1


def test_duplicate_finalized_peak_provenance_is_rejected(tmp_path):
    coordinator = audio_coordinator(tmp_path)
    peak = audio_peak(84.6530625)

    with pytest.raises(ValueError, match="timestamps must be unique"):
        coordinator.advance_delta(
            delta_timeline(
                tmp_path,
                [peak],
                observer="audio",
                metadata=audio_metadata(
                    850, [peak.timestamp_seconds, peak.timestamp_seconds]
                ),
            ),
            ObserverWatermarks({"audio": 84.0}),
            audio_identity(0),
        )


@pytest.mark.parametrize("different_payload", [False, True])
def test_duplicate_peak_observation_timestamps_are_rejected(
    tmp_path, different_payload
):
    coordinator = audio_coordinator(tmp_path)
    first = audio_peak(84.6530625)
    second = audio_peak(84.6530625)
    if different_payload:
        second = Observation(
            second.timestamp_seconds,
            second.observer,
            second.type,
            {**second.value, "amplitude": 0.95},
        )

    with pytest.raises(ValueError, match="observations must have unique timestamps"):
        coordinator.advance_delta(
            delta_timeline(
                tmp_path,
                [first, second],
                observer="audio",
                metadata=audio_metadata(850, [first.timestamp_seconds]),
            ),
            ObserverWatermarks({"audio": 84.0}),
            audio_identity(0),
        )


@pytest.mark.parametrize(
    "declared",
    [
        [84.6530625],
        [84.6530625, 84.8, 84.9],
    ],
)
def test_missing_or_extra_finalized_peak_provenance_is_rejected(
    tmp_path, declared
):
    coordinator = audio_coordinator(tmp_path)
    peaks = [audio_peak(84.6530625), audio_peak(84.8)]

    with pytest.raises(ValueError, match="exactly match"):
        coordinator.advance_delta(
            delta_timeline(
                tmp_path,
                peaks,
                observer="audio",
                metadata=audio_metadata(850, declared),
            ),
            ObserverWatermarks({"audio": 84.0}),
            audio_identity(0),
        )


def test_unique_finalized_peak_timestamps_match_one_to_one(tmp_path):
    coordinator = audio_coordinator(tmp_path)
    peaks = [audio_peak(84.6530625), audio_peak(84.8)]

    coordinator.advance_delta(
        delta_timeline(
            tmp_path,
            peaks,
            observer="audio",
            metadata=audio_metadata(
                850, [item.timestamp_seconds for item in reversed(peaks)]
            ),
        ),
        ObserverWatermarks({"audio": 84.0}),
        audio_identity(0),
    )

    assert len(coordinator.result.scores) == 2


def test_eof_finalized_peaks_use_same_uniqueness_contract(tmp_path):
    coordinator = audio_coordinator(tmp_path)
    peak = audio_peak(84.8)
    timeline = delta_timeline(
        tmp_path,
        [peak],
        observer="audio",
        metadata=audio_metadata(850, [peak.timestamp_seconds, peak.timestamp_seconds]),
    )

    with pytest.raises(ValueError, match="timestamps must be unique"):
        coordinator.flush_delta(
            timeline,
            IncrementalEOF(85.0, ObserverWatermarks({"audio": 85.0})),
            (audio_identity(0, eof=True),),
        )


def whisper_coordinator(tmp_path):
    return IncrementalPrerecordedCoordinator(
        MarkerGenerator(), TrackingScorer(), CandidateSelector(), Renderer(tmp_path),
        IncrementalPipelineConfig(required_observers=("whisper",)),
    )


def whisper_speech(start, end, text="finalized speech"):
    return Observation(
        start,
        "whisper",
        "speech",
        {
            "text": text,
            "speaker": None,
            "start": start,
            "end": end,
            "score": 0.8,
            "name": text,
        },
        duration_seconds=end - start,
        confidence=0.9,
    )


def whisper_metadata(frames, speech=()):
    return {
        "incremental_frames_processed": frames,
        "sample_rate_hz": 10,
        "finalized_speech_segment_identities": tuple(
            finalized_speech_segment_identity(item) for item in speech
        ),
    }


def whisper_identity(sequence, eof=False):
    return ObserverDeltaIdentity(
        "state-cache-fixture",
        "incremental:state-cache-fixture",
        "whisper",
        sequence,
        eof,
    )


def test_finalized_whisper_speech_may_extend_beyond_held_watermark(tmp_path):
    coordinator = whisper_coordinator(tmp_path)
    segment = whisper_speech(153.0, 155.0)
    coordinator.advance_delta(
        delta_timeline(
            tmp_path, [segment], observer="whisper",
            metadata=whisper_metadata(1600, [segment]),
        ),
        ObserverWatermarks({"whisper": 150.0}),
        whisper_identity(0),
    )
    assert len(coordinator.result.scores) == 1
    assert coordinator.result.scores[0].candidate.reason == "finalized speech"


def test_undeclared_whisper_speech_beyond_watermark_is_rejected(tmp_path):
    coordinator = whisper_coordinator(tmp_path)
    segment = whisper_speech(153.0, 155.0)
    with pytest.raises(ValueError, match="provenance"):
        coordinator.advance_delta(
            delta_timeline(
                tmp_path, [segment], observer="whisper",
                metadata={"incremental_frames_processed": 1600, "sample_rate_hz": 10},
            ),
            ObserverWatermarks({"whisper": 150.0}),
            whisper_identity(0),
        )


def test_finalized_whisper_speech_beyond_processed_frames_is_rejected(tmp_path):
    coordinator = whisper_coordinator(tmp_path)
    segment = whisper_speech(159.0, 161.0)
    with pytest.raises(ValueError, match="processed frames"):
        coordinator.advance_delta(
            delta_timeline(
                tmp_path, [segment], observer="whisper",
                metadata=whisper_metadata(1600, [segment]),
            ),
            ObserverWatermarks({"whisper": 150.0}),
            whisper_identity(0),
        )


@pytest.mark.parametrize(
    "declared",
    [
        [],
        ["0" * 64],
        ["0" * 64, "1" * 64],
        ["not-a-sha256"],
        [True],
        ["0" * 64, "0" * 64],
    ],
)
def test_invalid_finalized_whisper_provenance_is_rejected(tmp_path, declared):
    coordinator = whisper_coordinator(tmp_path)
    segment = whisper_speech(153.0, 155.0)
    metadata = {
        "incremental_frames_processed": 1600,
        "sample_rate_hz": 10,
        "finalized_speech_segment_identities": declared,
    }
    with pytest.raises(ValueError, match="Finalized|finalized|provenance"):
        coordinator.advance_delta(
            delta_timeline(tmp_path, [segment], observer="whisper", metadata=metadata),
            ObserverWatermarks({"whisper": 150.0}),
            whisper_identity(0),
        )


def test_distinct_same_time_finalized_whisper_segments_are_deterministic(tmp_path):
    coordinator = whisper_coordinator(tmp_path)
    segments = [
        whisper_speech(153.0, 155.0, "first wording"),
        whisper_speech(153.0, 155.0, "second wording"),
    ]
    coordinator.advance_delta(
        delta_timeline(
            tmp_path, segments, observer="whisper",
            metadata=whisper_metadata(1600, list(reversed(segments))),
        ),
        ObserverWatermarks({"whisper": 150.0}),
        whisper_identity(0),
    )
    assert len(coordinator.result.scores) == 2
    assert {
        item.candidate.reason for item in coordinator.result.scores
    } == {"first wording", "second wording"}


def test_finalized_whisper_receipt_retry_and_provenance_hashing(tmp_path):
    coordinator = whisper_coordinator(tmp_path)
    segments = [
        whisper_speech(153.0, 155.0, "first wording"),
        whisper_speech(156.0, 157.0, "second wording"),
    ]
    identity = whisper_identity(0)
    first = delta_timeline(
        tmp_path, segments, observer="whisper",
        metadata=whisper_metadata(1600, segments),
    )
    coordinator.advance_delta(first, ObserverWatermarks({"whisper": 150.0}), identity)
    assert coordinator.advance_delta(
        first, ObserverWatermarks({"whisper": 150.0}), identity
    ) == []
    changed = delta_timeline(
        tmp_path, segments, observer="whisper",
        metadata=whisper_metadata(1600, list(reversed(segments))),
    )
    with pytest.raises(ValueError, match="different content"):
        coordinator.advance_delta(
            changed, ObserverWatermarks({"whisper": 150.0}), identity
        )


def test_eof_finalized_whisper_speech_uses_same_contract(tmp_path):
    coordinator = whisper_coordinator(tmp_path)
    segment = whisper_speech(153.0, 155.0)
    result = coordinator.flush_delta(
        delta_timeline(
            tmp_path, [segment], observer="whisper",
            metadata=whisper_metadata(1600, [segment]),
        ),
        IncrementalEOF(160.0, ObserverWatermarks({"whisper": 160.0})),
        (whisper_identity(0, eof=True),),
    )
    assert len(result.selected_scores) == 1


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


def test_duplicate_boundaries_with_different_metadata_remain_distinct_families(
    tmp_path,
):
    coordinator = marker_coordinator(tmp_path)
    coordinator.advance_delta(
        delta_timeline(
            tmp_path,
            [
                marker(1, 0, 5, 0.9, "strong-metadata"),
                marker(2, 0, 5, 0.8, "weak-metadata"),
            ],
        ),
        ObserverWatermarks({"fixture": 10.0}),
    )

    result = coordinator.result
    assert len(result.scores) == 2
    assert len(result.selected_scores) == 1
    assert len(result.suppressed) == 1
    assert {
        item.candidate.reason for item in result.scores
    } == {"strong-metadata", "weak-metadata"}
    assert len(
        {
            candidate_fingerprint(item.candidate, "state-cache-fixture")
            for item in result.scores
        }
    ) == 2


def test_retroactive_delta_inside_horizon_changes_overlap_winner(tmp_path):
    scorer = TrackingScorer()
    coordinator = marker_coordinator(tmp_path, scorer)
    weak = marker(5, 0, 10, 0.6, "weak")
    strong = marker(7, 1, 11, 0.9, "strong")

    assert coordinator.advance_delta(delta_timeline(tmp_path, [weak]), ObserverWatermarks({"fixture": 6.0})) == []
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


def test_delta_behind_committed_observer_frontier_is_rejected_without_raw_horizon(
    tmp_path,
):
    coordinator = marker_coordinator(tmp_path)
    coordinator.advance_delta(delta_timeline(tmp_path, []), ObserverWatermarks({"fixture": 30.0}))

    with pytest.raises(ValueError, match="accepted observer frontier"):
        coordinator.advance_delta(
            delta_timeline(tmp_path, [marker(1, 0, 5, 0.8, "too-late")]),
            ObserverWatermarks({"fixture": 31.0}),
        )
    assert coordinator.watermark_seconds == 30.0


def test_distinct_evidence_ending_at_committed_frontier_cannot_reopen_family(
    tmp_path,
):
    coordinator = marker_coordinator(tmp_path)
    first = marker(1, 0, 4, 0.8, "first-at-frontier")
    same_end = marker(1, 0, 5, 0.9, "distinct-at-same-frontier")

    coordinator.advance_delta(
        delta_timeline(tmp_path, [first]),
        ObserverWatermarks({"fixture": 1.0}),
        delta_identity("fixture", 0),
    )
    with pytest.raises(ValueError, match="accepted observer frontier"):
        coordinator.advance_delta(
            delta_timeline(tmp_path, [same_end]),
            ObserverWatermarks({"fixture": 2.0}),
            delta_identity("fixture", 1),
        )

    assert [item.candidate.reason for item in coordinator.result.scores] == [
        "first-at-frontier"
    ]
    assert len(coordinator._finalized_scores) == 0


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


def competition_point(index):
    timestamp = 10.0 + 2.1 * index
    return Observation(
        timestamp,
        "whisper",
        "speech",
        {"text": "HIGH CONFIDENCE!", "speaker": None},
        duration_seconds=0.0,
        confidence=1.0,
    )


def improving_competition_points(count):
    return [
        replace(
            competition_point(index),
            confidence=0.80 + 0.19 * index / max(1, count - 1),
        )
        for index in range(count)
    ]


def real_competition_oracle(tmp_path, observations, duration, priority=None):
    priority = priority or SelectionPriorityContract()
    timeline = real_observer_timeline(tmp_path, observations, duration)
    generator = CandidateGenerator()
    scorer = CandidateScorer(
        CandidateScoringConfig(
            passing_score=0.0,
            selection_priority=priority,
        )
    )
    selector = CandidateSelector(
        CandidateSelectionConfig(selection_priority=priority)
    )
    candidates = generator.generate(timeline)
    scores = scorer.score(candidates)
    return timeline, candidates, scores, selector.select(scores)


def real_competition_coordinator(tmp_path, priority=None):
    priority = priority or SelectionPriorityContract()
    return IncrementalPrerecordedCoordinator(
        CandidateGenerator(),
        CandidateScorer(
            CandidateScoringConfig(
                passing_score=0.0,
                selection_priority=priority,
            )
        ),
        CandidateSelector(
            CandidateSelectionConfig(selection_priority=priority)
        ),
        Renderer(tmp_path),
        IncrementalPipelineConfig(required_observers=("whisper",)),
    )


def test_real_thirty_cluster_component_over_sixty_seconds_matches_batch(tmp_path):
    observations = [competition_point(index) for index in range(30)]
    duration = 200.0
    _, candidates, scores, batch = real_competition_oracle(
        tmp_path, observations, duration
    )
    assert len(candidates) == 30
    assert candidates[-1].end_seconds - candidates[0].start_seconds > 60.0
    selector = CandidateSelector()
    assert all(
        selector.competes(first, second)
        for first, second in zip(candidates, candidates[1:])
    )

    coordinator = real_competition_coordinator(tmp_path)
    last = observations[-1].timestamp_seconds
    assert coordinator.advance_delta(
        delta_timeline(
            tmp_path,
            observations,
            "whisper",
            whisper_metadata(round(last * 10) + 1, observations),
        ),
        ObserverWatermarks({"whisper": last}),
        whisper_identity(0),
    ) == []
    jobs = coordinator.advance_delta(
        delta_timeline(
            tmp_path, [], "whisper", whisper_metadata(round(duration * 10))
        ),
        ObserverWatermarks({"whisper": duration}),
        whisper_identity(1),
    )
    result = coordinator.flush_delta(
        delta_timeline(
            tmp_path, [], "whisper", whisper_metadata(round(duration * 10))
        ),
        IncrementalEOF(duration, ObserverWatermarks({"whisper": duration})),
        (whisper_identity(2, eof=True),),
    )

    expected = [
        candidate_fingerprint(item.candidate, "state-cache-fixture")
        for item in batch.selected
    ]
    assert [
        candidate_fingerprint(item.candidate, "state-cache-fixture")
        for item in result.selected_scores
    ] == expected
    assert result.scores == scores
    assert result.selected_scores == batch.selected
    assert [
        candidate_fingerprint(item.candidate, "state-cache-fixture")
        for item in jobs
    ] == expected
    assert coordinator.state_metrics.active_scores == 0


def test_real_long_competition_chain_is_locally_finalized_with_bounded_state(
    tmp_path,
):
    observations = [competition_point(index) for index in range(180)]
    duration = observations[-1].timestamp_seconds + 140.0
    _, candidates, _, batch = real_competition_oracle(
        tmp_path, observations, duration
    )
    assert candidates[-1].end_seconds - candidates[0].start_seconds > 300.0
    coordinator = real_competition_coordinator(tmp_path)
    assert coordinator._direct_competition_span_seconds == 60.0
    assert coordinator._maximum_selection_ownership_span_seconds == (
        60.0
        * SelectionPriorityContract().maximum_strictly_improving_chain_length
    )
    peak_active = 0
    rendered_before_eof = 0
    for sequence, observation in enumerate(observations):
        frontier = observation.timestamp_seconds
        rendered_before_eof += len(
            coordinator.advance_delta(
                delta_timeline(
                    tmp_path,
                    [observation],
                    "whisper",
                    whisper_metadata(round(frontier * 10) + 1, [observation]),
                ),
                ObserverWatermarks({"whisper": frontier}),
                whisper_identity(sequence),
            )
        )
        peak_active = max(peak_active, coordinator.state_metrics.active_scores)

    result = coordinator.flush_delta(
        delta_timeline(
            tmp_path, [], "whisper", whisper_metadata(round(duration * 10))
        ),
        IncrementalEOF(duration, ObserverWatermarks({"whisper": duration})),
        (whisper_identity(len(observations), eof=True),),
    )
    expected = [
        candidate_fingerprint(item.candidate, "state-cache-fixture")
        for item in batch.selected
    ]
    assert rendered_before_eof > 0
    assert [
        candidate_fingerprint(item.candidate, "state-cache-fixture")
        for item in result.selected_scores
    ] == expected
    assert peak_active < 40
    assert coordinator.state_metrics.active_scores == 0


def test_real_competition_component_closes_after_true_noncompeting_gap(tmp_path):
    chain = [competition_point(index) for index in range(30)]
    gap = Observation(
        200.0,
        "whisper",
        "speech",
        {"text": "HIGH CONFIDENCE!", "speaker": None},
        duration_seconds=0.0,
        confidence=1.0,
    )
    duration = 300.0
    _, _, _, batch = real_competition_oracle(tmp_path, [*chain, gap], duration)
    coordinator = real_competition_coordinator(tmp_path)
    last = chain[-1].timestamp_seconds
    coordinator.advance_delta(
        delta_timeline(
            tmp_path,
            chain,
            "whisper",
            whisper_metadata(round(last * 10) + 1, chain),
        ),
        ObserverWatermarks({"whisper": last}),
        whisper_identity(0),
    )
    pre_eof = coordinator.advance_delta(
        delta_timeline(
            tmp_path,
            [gap],
            "whisper",
            whisper_metadata(2000, [gap]),
        ),
        ObserverWatermarks({"whisper": 200.0}),
        whisper_identity(1),
    )
    assert len(pre_eof) == len(batch.selected) - 1

    result = coordinator.flush_delta(
        delta_timeline(tmp_path, [], "whisper", whisper_metadata(3000)),
        IncrementalEOF(300.0, ObserverWatermarks({"whisper": 300.0})),
        (whisper_identity(2, eof=True),),
    )
    assert [
        candidate_fingerprint(item.candidate, "state-cache-fixture")
        for item in result.selected_scores
    ] == [
        candidate_fingerprint(item.candidate, "state-cache-fixture")
        for item in batch.selected
    ]


@pytest.mark.parametrize("count", [60, 100, 180])
def test_real_improving_priority_chains_match_batch_without_duration_timeout(
    tmp_path,
    count,
):
    observations = improving_competition_points(count)
    duration = observations[-1].timestamp_seconds + 140.0
    _, candidates, scores, batch = real_competition_oracle(
        tmp_path,
        observations,
        duration,
    )
    chronological_scores = {
        candidate_fingerprint(item.candidate, "state-cache-fixture"): (
            item.overall_score
        )
        for item in scores
    }
    assert len(candidates) == count
    assert all(
        chronological_scores[
            candidate_fingerprint(first, "state-cache-fixture")
        ]
        < chronological_scores[
            candidate_fingerprint(second, "state-cache-fixture")
        ]
        for first, second in zip(candidates, candidates[1:])
    )

    coordinator = real_competition_coordinator(tmp_path)
    peak_active = 0
    for sequence, observation in enumerate(observations):
        frontier = observation.timestamp_seconds
        coordinator.advance_delta(
            delta_timeline(
                tmp_path,
                [observation],
                "whisper",
                whisper_metadata(round(frontier * 10) + 1, [observation]),
            ),
            ObserverWatermarks({"whisper": frontier}),
            whisper_identity(sequence),
        )
        peak_active = max(peak_active, coordinator.state_metrics.active_scores)
    jobs = coordinator.advance_delta(
        delta_timeline(
            tmp_path,
            [],
            "whisper",
            whisper_metadata(round(duration * 10)),
        ),
        ObserverWatermarks({"whisper": duration}),
        whisper_identity(count),
    )
    result = coordinator.flush_delta(
        delta_timeline(
            tmp_path,
            [],
            "whisper",
            whisper_metadata(round(duration * 10)),
        ),
        IncrementalEOF(duration, ObserverWatermarks({"whisper": duration})),
        (whisper_identity(count + 1, eof=True),),
    )
    expected = [
        candidate_fingerprint(item.candidate, "state-cache-fixture")
        for item in batch.selected
    ]
    assert [
        candidate_fingerprint(item.candidate, "state-cache-fixture")
        for item in result.selected_scores
    ] == expected
    assert result.scores == scores
    assert result.selected_scores == batch.selected
    rendered = [
        candidate_fingerprint(item.candidate, "state-cache-fixture")
        for item in jobs
    ]
    assert len(rendered) == len(expected)
    assert sorted(rendered) == sorted(expected)
    assert peak_active <= SelectionPriorityContract().rank_count
    assert coordinator.state_metrics.active_scores == 0


def test_real_chain_longer_than_configured_priority_alphabet_repeats_boundedly(
    tmp_path,
):
    priority = SelectionPriorityContract(score_decimal_places=1)
    count = 60
    observations = improving_competition_points(count)
    duration = observations[-1].timestamp_seconds + 140.0
    _, _, scores, batch = real_competition_oracle(
        tmp_path,
        observations,
        duration,
        priority,
    )
    coordinator = real_competition_coordinator(tmp_path, priority)
    peak_active = 0
    for sequence, observation in enumerate(observations):
        frontier = observation.timestamp_seconds
        coordinator.advance_delta(
            delta_timeline(
                tmp_path,
                [observation],
                "whisper",
                whisper_metadata(round(frontier * 10) + 1, [observation]),
            ),
            ObserverWatermarks({"whisper": frontier}),
            whisper_identity(sequence),
        )
        peak_active = max(peak_active, coordinator.state_metrics.active_scores)
    coordinator.advance_delta(
        delta_timeline(
            tmp_path,
            [],
            "whisper",
            whisper_metadata(round(duration * 10)),
        ),
        ObserverWatermarks({"whisper": duration}),
        whisper_identity(count),
    )
    result = coordinator.flush_delta(
        delta_timeline(tmp_path, [], "whisper", whisper_metadata(round(duration * 10))),
        IncrementalEOF(duration, ObserverWatermarks({"whisper": duration})),
        (whisper_identity(count + 1, eof=True),),
    )

    assert count > priority.rank_count
    assert [
        candidate_fingerprint(item.candidate, "state-cache-fixture")
        for item in result.selected_scores
    ] == [
        candidate_fingerprint(item.candidate, "state-cache-fixture")
        for item in batch.selected
    ]
    assert result.scores == scores
    assert result.selected_scores == batch.selected
    assert peak_active <= priority.rank_count + 4


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
    retry = coordinator.flush_delta(timeline, eof)
    assert retry == result
    assert scorer.candidates == 1
    assert len(retry.render_jobs) == 1


@pytest.mark.parametrize("failure_step", ["observer_proposal", "generator_proposal"])
def test_interruption_around_ingestion_receipt_commit_is_recoverable(
    tmp_path, monkeypatch, failure_step
):
    generator = CountingGenerator()
    scorer = TrackingScorer()
    coordinator = transaction_coordinator(
        tmp_path, generator, scorer, Renderer(tmp_path)
    )
    timeline = delta_timeline(
        tmp_path, [marker(1, 0, 2, 0.8, "transaction")]
    )
    identity = delta_identity("fixture", 0)
    coordinator._activate(timeline)
    before = transaction_state_snapshot(coordinator, generator)
    interrupted = False

    def interrupt_once(step):
        nonlocal interrupted
        if step == failure_step and not interrupted:
            interrupted = True
            raise KeyboardInterrupt()

    monkeypatch.setattr(coordinator, "_transaction_preparation_step", interrupt_once)
    with pytest.raises(KeyboardInterrupt):
        coordinator.advance_delta(
            timeline, ObserverWatermarks({"fixture": 7.0}), identity
        )
    assert transaction_state_snapshot(coordinator, generator) == before

    monkeypatch.setattr(
        coordinator, "_transaction_preparation_step", lambda step: None
    )
    jobs = coordinator.advance_delta(
        timeline, ObserverWatermarks({"fixture": 7.0}), identity
    )
    assert len(jobs) == 1
    assert generator.advance_calls == 1
    assert generator.finalize_calls == 0
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


class CountingGenerator(MarkerGenerator):
    def __init__(self):
        self.advance_calls = 0
        self.finalize_calls = 0

    def advance_incremental(
        self,
        checkpoint,
        observations,
        stable_through_seconds,
        observer_frontiers=None,
    ):
        self.advance_calls += 1
        return super().advance_incremental(
            checkpoint,
            observations,
            stable_through_seconds,
            observer_frontiers,
        )

    def finalize_incremental(
        self,
        checkpoint,
        observations,
        media_duration_seconds,
    ):
        self.finalize_calls += 1
        return super().finalize_incremental(
            checkpoint,
            observations,
            media_duration_seconds,
        )


def transaction_timeline(tmp_path):
    return delta_timeline(
        tmp_path,
        [
            marker(1, 0, 2, 0.81, "first"),
            marker(10, 10, 12, 0.82, "second"),
            marker(20, 20, 22, 0.83, "third"),
        ],
    )


def transaction_coordinator(tmp_path, generator, scorer, renderer):
    return IncrementalPrerecordedCoordinator(
        generator,
        scorer,
        CandidateSelector(),
        renderer,
        IncrementalPipelineConfig(required_observers=("fixture",)),
    )


def transaction_advance(coordinator, timeline):
    return coordinator.advance_delta(
        timeline,
        ObserverWatermarks({"fixture": 30.0}),
        delta_identity("fixture", 0),
    )


def transaction_state_snapshot(coordinator, generator):
    """Snapshot every authoritative value covered by the publication swap."""

    state = coordinator._decision_state
    result = coordinator.result
    return {
        "decision_state_identity": id(state),
        "generation_checkpoint": state.generation_checkpoint,
        "observer_watermarks": tuple(state.observer_watermarks.items()),
        "observer_frontiers": tuple(state.observer_acceptance_frontiers.items()),
        "observer_frames": tuple(state.observer_frames.items()),
        "observer_sequences": tuple(state.accepted_delta_sequences.items()),
        "recent_observation_ids": tuple(
            (observer, tuple(identities.items()))
            for observer, identities in state.recent_observation_ids.items()
        ),
        "snapshot_observation_ids": state.snapshot_observation_ids,
        "committed_snapshot": (
            state.committed_snapshot_payload,
            state.committed_snapshot_render_start_index,
            state.committed_snapshot_render_end_index,
        ),
        "render_identity": state.render_identity,
        "render_identities": tuple(state.render_identities.items()),
        "render_families": tuple(
            state.render_family_ids_by_fingerprint.items()
        ),
        "selected_scores": state.selected_scores,
        "suppressions": state.suppressions,
        "finalized_scores": tuple(state.finalized_scores.items()),
        "active_scores": state.active_scores,
        "pending_suppressions": state.pending_suppressions,
        "render_plan": state.pending_render_plan,
        "committed_receipt": (
            state.committed_receipt_key,
            state.committed_receipt_payload,
        ),
        "finalized_family_ids": state.finalized_family_ids,
        "immutable_score_fingerprints": tuple(
            state.immutable_score_fingerprints.items()
        ),
        "score_family_ids": tuple(state.score_family_ids.items()),
        "cumulative_metrics": (
            state.generation_passes,
            state.scored_candidates,
            state.candidate_fingerprints,
            state.score_fingerprints,
            state.peak_active_observations,
            state.peak_active_scores,
            state.peak_unresolved_group_size,
        ),
        "result": (
            tuple(result.scores),
            tuple(result.selected_scores),
            tuple(result.suppressed),
            tuple(result.render_jobs),
        ),
        "render_states": tuple(coordinator._render_states.items()),
        "render_completions": tuple(
            coordinator._authoritative_state.render_completions.items()
        ),
        "currently_rendering": frozenset(coordinator._currently_rendering),
        "completed_delta_receipts": tuple(
            coordinator._completed_delta_receipts.items()
        ),
        "completed_eof_fingerprint": coordinator._completed_eof_fingerprint,
        "lifecycle": coordinator.lifecycle,
        "generator": generator.incremental_state_snapshot(),
    }


def test_failure_before_generation_retries_one_generator_transition(
    tmp_path, monkeypatch
):
    generator = CountingGenerator()
    scorer = TrackingScorer()
    renderer = Renderer(tmp_path)
    coordinator = transaction_coordinator(tmp_path, generator, scorer, renderer)
    timeline = transaction_timeline(tmp_path)
    original = coordinator._advance_generation
    interrupted = False

    def interrupt_before_generation(*args, **kwargs):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt()
        return original(*args, **kwargs)

    monkeypatch.setattr(
        coordinator, "_advance_generation", interrupt_before_generation
    )
    with pytest.raises(KeyboardInterrupt):
        transaction_advance(coordinator, timeline)
    assert generator.advance_calls == 0
    assert coordinator._pending_delta.generation is None

    jobs = transaction_advance(coordinator, timeline)
    assert generator.advance_calls == 1
    assert scorer.candidates == 3
    assert [item.metadata["rank"] for item in jobs] == [1, 2, 3]


def test_failure_after_generation_reuses_saved_checkpoint_without_rescoring(
    tmp_path, monkeypatch
):
    generator = CountingGenerator()
    scorer = TrackingScorer()
    renderer = Renderer(tmp_path)
    coordinator = transaction_coordinator(tmp_path, generator, scorer, renderer)
    timeline = transaction_timeline(tmp_path)
    original = coordinator._scores_for_closed_families
    interrupted = False

    def interrupt_after_generation(*args, **kwargs):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt()
        return original(*args, **kwargs)

    monkeypatch.setattr(
        coordinator, "_scores_for_closed_families", interrupt_after_generation
    )
    with pytest.raises(KeyboardInterrupt):
        transaction_advance(coordinator, timeline)
    receipt = coordinator._pending_delta
    saved_generation = receipt.generation
    assert saved_generation is not None
    assert generator.advance_calls == 1
    assert scorer.candidates == 0

    transaction_advance(coordinator, timeline)
    assert generator.advance_calls == 1
    assert scorer.candidates == 3
    assert coordinator._generation_checkpoint == saved_generation.checkpoint


def test_failure_during_scoring_reuses_generation_and_is_deterministic(tmp_path):
    class FailingScorer(TrackingScorer):
        def __init__(self):
            super().__init__()
            self.attempts = 0

        def score(self, candidates):
            values = list(candidates)
            self.attempts += 1
            if self.attempts == 1:
                raise KeyboardInterrupt()
            return super().score(values)

    generator = CountingGenerator()
    scorer = FailingScorer()
    renderer = Renderer(tmp_path)
    coordinator = transaction_coordinator(tmp_path, generator, scorer, renderer)
    timeline = transaction_timeline(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        transaction_advance(coordinator, timeline)
    saved_generation = coordinator._pending_delta.generation
    assert saved_generation is not None

    transaction_advance(coordinator, timeline)
    assert generator.advance_calls == 1
    assert scorer.attempts == 2
    assert scorer.candidates == 3
    assert coordinator._generation_checkpoint == saved_generation.checkpoint


def test_failure_before_decision_commit_reuses_scores_and_family_lineage(
    tmp_path, monkeypatch
):
    generator = CountingGenerator()
    scorer = TrackingScorer()
    renderer = Renderer(tmp_path)
    coordinator = transaction_coordinator(tmp_path, generator, scorer, renderer)
    timeline = transaction_timeline(tmp_path)
    original = coordinator._publish_generation_decision
    interrupted = False

    def interrupt_before_commit(*args, **kwargs):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt()
        return original(*args, **kwargs)

    monkeypatch.setattr(
        coordinator, "_publish_generation_decision", interrupt_before_commit
    )
    with pytest.raises(KeyboardInterrupt):
        transaction_advance(coordinator, timeline)
    receipt = coordinator._pending_delta
    family_ids = tuple(item.family_id for item in receipt.scores)
    score_fingerprints = tuple(
        score_fingerprint(item.score, "state-cache-fixture")
        for item in receipt.scores
    )
    assert receipt.prepared_decisions is not None

    transaction_advance(coordinator, timeline)
    assert generator.advance_calls == 1
    assert scorer.candidates == 3
    assert set(coordinator._finalized_scores) == set(family_ids)
    assert tuple(
        score_fingerprint(item, "state-cache-fixture")
        for item in coordinator.result.scores
    ) == score_fingerprints


@pytest.mark.parametrize(
    "failure_step",
    [
        "checkpoint",
        "render_identities",
        "selection_results",
        "score_state",
        "metrics",
        "render_plan",
    ],
)
def test_failure_during_decision_preparation_publishes_nothing(
    tmp_path, monkeypatch, failure_step
):
    generator = CountingGenerator()
    scorer = TrackingScorer()
    renderer = Renderer(tmp_path)
    coordinator = transaction_coordinator(tmp_path, generator, scorer, renderer)
    timeline = transaction_timeline(tmp_path)
    coordinator._activate(timeline)
    published = coordinator._decision_state
    before = transaction_state_snapshot(coordinator, generator)

    def interrupt(step):
        if step == failure_step:
            raise KeyboardInterrupt()

    monkeypatch.setattr(coordinator, "_decision_preparation_step", interrupt)

    with pytest.raises(KeyboardInterrupt):
        transaction_advance(coordinator, timeline)
    receipt = coordinator._pending_delta
    assert receipt is not None
    assert receipt.prepared_decisions is not None
    assert transaction_state_snapshot(coordinator, generator) == before
    assert coordinator._decision_state is published
    assert coordinator._generation_checkpoint is published.generation_checkpoint
    assert coordinator.watermark_seconds == published.watermark_seconds == 0.0
    assert dict(coordinator._render_identities) == {}
    assert dict(coordinator._finalized_scores) == {}
    assert published.active_scores == ()
    assert published.pending_render_plan == ()
    assert published.generation_passes == 0
    assert published.scored_candidates == 0
    assert published.candidate_fingerprints == 0
    assert published.score_fingerprints == 0
    assert coordinator.state_metrics.active_observations == 3
    assert coordinator.state_metrics.active_scores == 3

    monkeypatch.setattr(coordinator, "_decision_preparation_step", lambda step: None)

    transaction_advance(coordinator, timeline)
    assert generator.advance_calls == 1
    assert scorer.candidates == 3
    assert list(coordinator._render_identities.values()) == [1, 2, 3]
    assert [identity for _, identity in renderer.calls] == [1, 2, 3]
    assert len(coordinator.result.selected_scores) == 3
    assert coordinator._decision_state.pending_render_plan == ()


@pytest.mark.parametrize(
    "failure_step",
    [
        "observer_proposal",
        "generator_proposal",
        "before_commit",
        "former_generator_commit_gap",
        "after_state_construction",
    ],
)
def test_cross_component_transaction_failure_has_no_partial_publication(
    tmp_path, monkeypatch, failure_step
):
    generator = CountingGenerator()
    scorer = TrackingScorer()
    renderer = Renderer(tmp_path)
    coordinator = transaction_coordinator(tmp_path, generator, scorer, renderer)
    timeline = transaction_timeline(tmp_path)
    coordinator._activate(timeline)
    before = transaction_state_snapshot(coordinator, generator)

    def interrupt(step):
        if step == failure_step:
            raise KeyboardInterrupt()

    monkeypatch.setattr(coordinator, "_transaction_preparation_step", interrupt)
    with pytest.raises(KeyboardInterrupt):
        transaction_advance(coordinator, timeline)

    assert transaction_state_snapshot(coordinator, generator) == before
    assert generator.advance_calls == (
        0 if failure_step == "observer_proposal" else 1
    )
    assert scorer.candidates == (
        3
        if failure_step
        in {"before_commit", "former_generator_commit_gap", "after_state_construction"}
        else 0
    )
    assert renderer.calls == []

    receipt = coordinator._pending_delta
    saved_generation = None if receipt is None else receipt.generation
    saved_scores = () if receipt is None else tuple(receipt.scores)
    monkeypatch.setattr(
        coordinator, "_transaction_preparation_step", lambda step: None
    )
    jobs = transaction_advance(coordinator, timeline)

    assert generator.advance_calls == 1
    assert scorer.candidates == 3
    assert [item.metadata["rank"] for item in jobs] == [1, 2, 3]
    assert [identity for _, identity in renderer.calls] == [1, 2, 3]
    assert len(coordinator.result.selected_scores) == 3
    assert len(coordinator.result.render_jobs) == 3
    if saved_generation is not None:
        assert coordinator._generation_checkpoint is saved_generation.checkpoint
    if saved_scores:
        assert tuple(coordinator.result.scores) == tuple(
            item.score for item in saved_scores
        )


def test_delta_interruption_immediately_after_authoritative_swap_replays_commit(
    tmp_path, monkeypatch
):
    generator = CountingGenerator()
    scorer = TrackingScorer()
    renderer = Renderer(tmp_path)
    coordinator = transaction_coordinator(tmp_path, generator, scorer, renderer)
    timeline = transaction_timeline(tmp_path)
    interrupted = False

    def interrupt(step):
        nonlocal interrupted
        if step == "after_commit" and not interrupted:
            interrupted = True
            raise KeyboardInterrupt()

    monkeypatch.setattr(coordinator, "_transaction_publication_step", interrupt)
    with pytest.raises(KeyboardInterrupt):
        transaction_advance(coordinator, timeline)

    published = coordinator._decision_state
    published_metrics = coordinator.state_metrics
    assert published.committed_receipt_key == coordinator._identity_key(
        delta_identity("fixture", 0)
    )
    assert generator.advance_calls == 1
    assert scorer.candidates == 3
    assert renderer.calls == []

    monkeypatch.setattr(
        coordinator, "_transaction_publication_step", lambda step: None
    )
    jobs = transaction_advance(coordinator, timeline)
    assert [item.metadata["rank"] for item in jobs] == [1, 2, 3]
    assert generator.advance_calls == 1
    assert scorer.candidates == 3
    assert [identity for _, identity in renderer.calls] == [1, 2, 3]
    assert coordinator.state_metrics.generation_passes == (
        published_metrics.generation_passes
    )
    assert coordinator.state_metrics.scored_candidates == (
        published_metrics.scored_candidates
    )

    replay = transaction_advance(coordinator, timeline)
    assert replay == jobs
    assert [identity for _, identity in renderer.calls] == [1, 2, 3]
    assert coordinator._pending_delta is None
    coordinator.advance_delta(
        delta_timeline(tmp_path, []),
        ObserverWatermarks({"fixture": 31.0}),
        delta_identity("fixture", 1),
    )
    assert coordinator._accepted_delta_sequences["fixture"] == 1
    assert coordinator._pending_delta is None


@pytest.mark.parametrize("boundary", ["before", "after"])
def test_delta_completion_ledger_and_pending_clear_publish_together(
    tmp_path, monkeypatch, boundary
):
    generator = CountingGenerator()
    scorer = TrackingScorer()
    renderer = Renderer(tmp_path)
    coordinator = transaction_coordinator(tmp_path, generator, scorer, renderer)
    timeline = transaction_timeline(tmp_path)
    interrupted = False

    def interrupt(step):
        nonlocal interrupted
        expected = "delta_completion" if boundary == "before" else "after_delta_completion"
        if step == expected and not interrupted:
            interrupted = True
            raise KeyboardInterrupt()

    hook = (
        "_completion_preparation_step"
        if boundary == "before"
        else "_completion_publication_step"
    )
    monkeypatch.setattr(coordinator, hook, interrupt)
    with pytest.raises(KeyboardInterrupt):
        transaction_advance(coordinator, timeline)

    key = coordinator._identity_key(delta_identity("fixture", 0))
    if boundary == "before":
        assert coordinator._pending_delta is not None
        assert key not in coordinator._completed_delta_receipts
    else:
        assert coordinator._pending_delta is None
        assert key in coordinator._completed_delta_receipts
    assert [identity for _, identity in renderer.calls] == [1, 2, 3]

    monkeypatch.setattr(coordinator, hook, lambda step: None)
    jobs = transaction_advance(coordinator, timeline)
    assert [item.metadata["rank"] for item in jobs] == [1, 2, 3]
    assert generator.advance_calls == 1
    assert scorer.candidates == 3
    assert [identity for _, identity in renderer.calls] == [1, 2, 3]
    assert coordinator._pending_delta is None
    assert coordinator._decision_state.pending_render_plan == ()
    assert key in coordinator._completed_delta_receipts

    coordinator.advance_delta(
        delta_timeline(tmp_path, []),
        ObserverWatermarks({"fixture": 31.0}),
        delta_identity("fixture", 1),
    )
    assert coordinator._accepted_delta_sequences["fixture"] == 1
    assert coordinator._pending_delta is None
    assert generator.advance_calls == 2
    assert scorer.candidates == 3
    assert len(coordinator.result.render_jobs) == 3


def test_implicit_delta_retry_recognizes_post_completion_publication(
    tmp_path, monkeypatch
):
    generator = CountingGenerator()
    scorer = TrackingScorer()
    renderer = Renderer(tmp_path)
    coordinator = transaction_coordinator(tmp_path, generator, scorer, renderer)
    timeline = delta_timeline(
        tmp_path, [marker(1, 0, 2, 0.8, "implicit delta")]
    )
    interrupted = False

    def interrupt(step):
        nonlocal interrupted
        if step == "after_delta_completion" and not interrupted:
            interrupted = True
            raise KeyboardInterrupt()

    monkeypatch.setattr(coordinator, "_completion_publication_step", interrupt)
    with pytest.raises(KeyboardInterrupt):
        coordinator.advance_delta(
            timeline,
            ObserverWatermarks({"fixture": 7.0}),
        )

    assert coordinator._pending_delta is None
    assert [identity for _, identity in renderer.calls] == [1]
    monkeypatch.setattr(
        coordinator, "_completion_publication_step", lambda step: None
    )
    jobs = coordinator.advance_delta(
        timeline,
        ObserverWatermarks({"fixture": 7.0}),
    )
    assert len(jobs) == 1
    assert generator.advance_calls == 1
    assert scorer.candidates == 1
    assert [identity for _, identity in renderer.calls] == [1]


def test_completed_delta_retry_survives_other_observer_watermark_progress(tmp_path):
    generator = CountingGenerator()
    scorer = TrackingScorer()
    renderer = Renderer(tmp_path)
    coordinator = IncrementalPrerecordedCoordinator(
        generator,
        scorer,
        CandidateSelector(),
        renderer,
        IncrementalPipelineConfig(required_observers=("audio", "whisper")),
    )
    audio_timeline = multi_delta_timeline(
        tmp_path,
        [
            ObserverResult(
                "audio",
                [observer_marker("audio", 1, 0, 2, 0.81, "audio")],
            )
        ],
    )
    audio_watermarks = ObserverWatermarks({"audio": 10.0, "whisper": 0.0})
    audio_identity = delta_identity("audio", 0)
    first_jobs = coordinator.advance_delta(
        audio_timeline,
        audio_watermarks,
        audio_identity,
    )
    assert first_jobs == []

    coordinator.advance_delta(
        multi_delta_timeline(
            tmp_path,
            [
                ObserverResult(
                    "whisper",
                    [
                        observer_marker(
                            "whisper", 20, 20, 22, 0.82, "whisper"
                        )
                    ],
                )
            ],
        ),
        ObserverWatermarks({"audio": 10.0, "whisper": 30.0}),
        delta_identity("whisper", 0),
    )
    assert coordinator.watermark_seconds == 10.0
    before_retry = transaction_state_snapshot(coordinator, generator)
    calls_before_retry = (
        generator.advance_calls,
        scorer.calls,
        len(renderer.calls),
    )

    assert coordinator.advance_delta(
        audio_timeline,
        audio_watermarks,
        audio_identity,
    ) == first_jobs
    assert transaction_state_snapshot(coordinator, generator) == before_retry
    assert (
        generator.advance_calls,
        scorer.calls,
        len(renderer.calls),
    ) == calls_before_retry


@pytest.mark.parametrize(
    "failure_step",
    ["after_render_one", "after_completion_construction", "after_render_completion"],
)
def test_render_completion_status_and_job_are_one_retryable_publication(
    tmp_path, monkeypatch, failure_step
):
    generator = CountingGenerator()
    scorer = TrackingScorer()
    renderer = Renderer(tmp_path)
    coordinator = transaction_coordinator(tmp_path, generator, scorer, renderer)
    timeline = transaction_timeline(tmp_path)
    interrupted = False

    def interrupt(step):
        nonlocal interrupted
        if step == failure_step and not interrupted:
            interrupted = True
            raise KeyboardInterrupt()

    monkeypatch.setattr(coordinator, "_render_publication_step", interrupt)
    with pytest.raises(KeyboardInterrupt):
        transaction_advance(coordinator, timeline)

    assert [identity for _, identity in renderer.calls] == [1]
    assert len(coordinator.result.render_jobs) == 1
    assert list(coordinator._render_states.values()) == [
        RenderLifecycleState.RENDERED
    ]

    monkeypatch.setattr(coordinator, "_render_publication_step", lambda step: None)
    jobs = transaction_advance(coordinator, timeline)
    assert [item.metadata["rank"] for item in jobs] == [1, 2, 3]
    assert [identity for _, identity in renderer.calls] == [1, 2, 3]
    assert len(coordinator.result.render_jobs) == 3
    assert generator.advance_calls == 1
    assert scorer.candidates == 3


class DeepCopyScorer(TrackingScorer):
    def __init__(self):
        super().__init__()
        self.input_candidate_ids = []
        self.output_candidate_ids = []

    def score(self, candidates):
        values = list(candidates)
        self.input_candidate_ids.extend(id(item) for item in values)
        output = deepcopy(super().score(values))
        self.output_candidate_ids.extend(id(item.candidate) for item in output)
        return output


class DeepCopySelector:
    selection_priority_contract = SelectionPriorityContract()

    def __init__(self):
        self.delegate = CandidateSelector()
        self.input_score_ids = []
        self.output_score_ids = []

    def select(self, scores):
        values = list(scores)
        self.input_score_ids.extend(id(item) for item in values)
        output = deepcopy(self.delegate.select(values))
        self.output_score_ids.extend(id(item) for item in output.selected)
        self.output_score_ids.extend(id(item.score) for item in output.suppressed)
        return output


def test_value_equivalent_scorer_and_selector_dtos_preserve_ownership(tmp_path):
    generator = CountingGenerator()
    scorer = DeepCopyScorer()
    selector = DeepCopySelector()
    renderer = Renderer(tmp_path)
    coordinator = IncrementalPrerecordedCoordinator(
        generator,
        scorer,
        selector,
        renderer,
        IncrementalPipelineConfig(required_observers=("fixture",)),
    )
    jobs = transaction_advance(coordinator, transaction_timeline(tmp_path))

    assert len(jobs) == 3
    assert set(scorer.input_candidate_ids).isdisjoint(scorer.output_candidate_ids)
    assert set(selector.input_score_ids).isdisjoint(selector.output_score_ids)
    assert len(coordinator.result.scores) == 3
    assert len(coordinator.result.selected_scores) == 3
    assert [identity for _, identity in renderer.calls] == [1, 2, 3]


def _score_with_other_source(score, source_path):
    candidate = score.candidate
    moved = ClipCandidate(
        source_path,
        candidate.start_seconds,
        candidate.end_seconds,
        candidate.reason,
        source_signals=deepcopy(candidate.source_signals),
        title=candidate.title,
        metadata=deepcopy(candidate.metadata),
    )
    return ClipScore(
        moved,
        score.overall_score,
        score_components=deepcopy(score.score_components),
        rationale=score.rationale,
        passed_threshold=score.passed_threshold,
    )


def test_value_equivalent_scorer_cannot_change_source_path(tmp_path):
    class WrongSourceScorer(TrackingScorer):
        def score(self, candidates):
            return [
                _score_with_other_source(item, tmp_path / "other-source.mp4")
                for item in super().score(candidates)
            ]

    coordinator = IncrementalPrerecordedCoordinator(
        MarkerGenerator(),
        WrongSourceScorer(),
        CandidateSelector(),
        Renderer(tmp_path),
        IncrementalPipelineConfig(required_observers=("fixture",)),
    )
    with pytest.raises(RuntimeError, match="closed-family ownership"):
        coordinator.advance_delta(
            delta_timeline(tmp_path, [marker(1, 0, 2, 0.8, "wrong source")]),
            ObserverWatermarks({"fixture": 30.0}),
            delta_identity("fixture", 0),
        )


def test_value_equivalent_selector_cannot_change_source_path(tmp_path):
    class WrongSourceSelector(DeepCopySelector):
        def select(self, scores):
            output = super().select(scores)
            if not output.selected:
                return output
            return CandidateSelectionResult(
                selected=[
                    _score_with_other_source(
                        output.selected[0], tmp_path / "other-source.mp4"
                    ),
                    *output.selected[1:],
                ],
                suppressed=output.suppressed,
            )

    coordinator = IncrementalPrerecordedCoordinator(
        MarkerGenerator(),
        TrackingScorer(),
        WrongSourceSelector(),
        Renderer(tmp_path),
        IncrementalPipelineConfig(required_observers=("fixture",)),
    )
    with pytest.raises(RuntimeError, match="score ownership"):
        coordinator.advance_delta(
            delta_timeline(tmp_path, [marker(1, 0, 2, 0.8, "wrong source")]),
            ObserverWatermarks({"fixture": 30.0}),
            delta_identity("fixture", 0),
        )


def test_duplicate_equal_candidates_keep_occurrence_family_ownership(tmp_path):
    class DuplicateEqualGenerator(MarkerGenerator):
        def generate(self, timeline):
            count = sum(
                len(result.observations)
                for result in timeline.timeline.observer_results
            )
            candidate = ClipCandidate(
                timeline.media_path,
                0.0,
                8.0,
                "value-identical duplicate",
                metadata={"score": 0.8},
            )
            return [deepcopy(candidate) for _ in range(count)]

    scorer = DeepCopyScorer()
    selector = DeepCopySelector()
    renderer = Renderer(tmp_path)
    coordinator = IncrementalPrerecordedCoordinator(
        DuplicateEqualGenerator(),
        scorer,
        selector,
        renderer,
        IncrementalPipelineConfig(required_observers=("fixture",)),
    )
    observations = [
        marker(1, 0, 1, 0.8, "first occurrence"),
        marker(2, 10, 11, 0.8, "second occurrence"),
    ]
    jobs = coordinator.advance_delta(
        delta_timeline(tmp_path, observations),
        ObserverWatermarks({"fixture": 30.0}),
        delta_identity("fixture", 0),
    )

    fingerprints = [
        candidate_fingerprint(item.candidate, "state-cache-fixture")
        for item in coordinator.result.scores
    ]
    assert len(fingerprints) == 2
    assert len(set(fingerprints)) == 1
    shared_score_fingerprint = score_fingerprint(
        coordinator.result.scores[0], "state-cache-fixture"
    )
    assert len(
        coordinator._decision_state.score_family_ids[shared_score_fingerprint]
    ) == 2
    assert len(coordinator.result.selected_scores) == 1
    assert len(coordinator.result.suppressed) == 1
    assert len(jobs) == len(renderer.calls) == 1


def test_selector_reordered_value_equivalent_winners_are_rejected(tmp_path):
    class ReorderingSelector(DeepCopySelector):
        def select(self, scores):
            output = super().select(scores)
            if len(output.selected) > 1 and output.suppressed:
                return CandidateSelectionResult(
                    selected=list(reversed(output.selected)),
                    suppressed=output.suppressed,
                )
            return output

    coordinator = IncrementalPrerecordedCoordinator(
        MarkerGenerator(),
        DeepCopyScorer(),
        ReorderingSelector(),
        Renderer(tmp_path),
        IncrementalPipelineConfig(required_observers=("fixture",)),
    )
    values = [
        marker(1, 0.0, 10.0, 0.9, "first winner"),
        marker(2, 7.0, 17.0, 0.8, "second winner"),
        marker(3, 3.5, 13.5, 0.7, "shared loser"),
    ]
    with pytest.raises(RuntimeError, match="selection ownership"):
        coordinator.advance_delta(
            delta_timeline(tmp_path, values),
            ObserverWatermarks({"fixture": 30.0}),
            delta_identity("fixture", 0),
        )


@pytest.mark.parametrize("mode", ["missing", "additional"])
def test_scorer_missing_or_additional_value_output_is_rejected(tmp_path, mode):
    class InvalidCardinalityScorer(TrackingScorer):
        def score(self, candidates):
            output = super().score(candidates)
            return output[:-1] if mode == "missing" else [*output, deepcopy(output[0])]

    coordinator = transaction_coordinator(
        tmp_path,
        CountingGenerator(),
        InvalidCardinalityScorer(),
        Renderer(tmp_path),
    )
    with pytest.raises(RuntimeError, match="one score per closed family"):
        transaction_advance(coordinator, transaction_timeline(tmp_path))


@pytest.mark.parametrize("mode", ["missing", "additional"])
def test_selector_missing_or_additional_value_output_is_rejected(tmp_path, mode):
    class InvalidCardinalitySelector(DeepCopySelector):
        def select(self, scores):
            values = list(scores)
            if len(values) == 1:
                selected = [] if mode == "missing" else [deepcopy(values[0])] * 2
                return CandidateSelectionResult(selected=selected, suppressed=[])
            return super().select(values)

    coordinator = IncrementalPrerecordedCoordinator(
        MarkerGenerator(),
        TrackingScorer(),
        InvalidCardinalitySelector(),
        Renderer(tmp_path),
        IncrementalPipelineConfig(required_observers=("fixture",)),
    )
    with pytest.raises(RuntimeError, match="selection ownership"):
        coordinator.advance_delta(
            delta_timeline(tmp_path, [marker(1, 0, 2, 0.8, "invalid")]),
            ObserverWatermarks({"fixture": 30.0}),
            delta_identity("fixture", 0),
        )


@pytest.mark.parametrize("priority", [1, float("nan"), float("inf")])
def test_online_competition_requires_a_finite_float_priority_domain(
    tmp_path, priority
):
    class InvalidPriorityScorer(TrackingScorer):
        def score(self, candidates):
            values = list(candidates)
            return [
                ClipScore(values[0], priority, passed_threshold=True)
            ]

    coordinator = IncrementalPrerecordedCoordinator(
        MarkerGenerator(),
        InvalidPriorityScorer(),
        CandidateSelector(),
        Renderer(tmp_path),
        IncrementalPipelineConfig(required_observers=("fixture",)),
    )
    with pytest.raises(RuntimeError, match="finite float priorities"):
        coordinator.advance_delta(
            delta_timeline(tmp_path, [marker(1, 0, 2, 0.8, "invalid priority")]),
            ObserverWatermarks({"fixture": 30.0}),
            delta_identity("fixture", 0),
        )


def test_coordinator_rejects_mismatched_finite_priority_contracts(tmp_path) -> None:
    with pytest.raises(ValueError, match="selection-priority contracts differ"):
        IncrementalPrerecordedCoordinator(
            CandidateGenerator(),
            CandidateScorer(
                CandidateScoringConfig(
                    selection_priority=SelectionPriorityContract(1)
                )
            ),
            CandidateSelector(),
            Renderer(tmp_path),
            IncrementalPipelineConfig(required_observers=("whisper",)),
        )


def eof_transaction_fixture(tmp_path):
    generator = CountingGenerator()
    scorer = TrackingScorer()
    renderer = Renderer(tmp_path)
    coordinator = IncrementalPrerecordedCoordinator(
        generator,
        scorer,
        CandidateSelector(),
        renderer,
        IncrementalPipelineConfig(required_observers=("audio", "whisper")),
    )
    timeline = multi_delta_timeline(
        tmp_path,
        [
            ObserverResult(
                "audio",
                [observer_marker("audio", 1, 0, 1, 0.81, "audio-eof")],
            ),
            ObserverResult(
                "whisper",
                [observer_marker("whisper", 11, 10, 11, 0.82, "whisper-eof")],
            ),
        ],
    )
    eof = IncrementalEOF(
        20.0,
        ObserverWatermarks({"audio": 20.0, "whisper": 20.0}),
    )
    identities = (
        ObserverDeltaIdentity(
            "state-cache-fixture",
            "incremental:state-cache-fixture",
            "audio",
            0,
            True,
        ),
        ObserverDeltaIdentity(
            "state-cache-fixture",
            "incremental:state-cache-fixture",
            "whisper",
            0,
            True,
        ),
    )
    return generator, scorer, renderer, coordinator, timeline, eof, identities


def test_multi_observer_eof_validation_failure_does_not_accept_first_observer(
    tmp_path,
):
    (
        generator,
        scorer,
        renderer,
        coordinator,
        timeline,
        eof,
        identities,
    ) = eof_transaction_fixture(tmp_path)
    coordinator._activate(timeline)
    before = transaction_state_snapshot(coordinator, generator)
    invalid_identities = (
        identities[0],
        ObserverDeltaIdentity(
            identities[1].source_id,
            identities[1].session_id,
            identities[1].observer,
            1,
            True,
        ),
    )

    with pytest.raises(ValueError, match="must be 0"):
        coordinator.flush_delta(timeline, eof, invalid_identities)

    assert transaction_state_snapshot(coordinator, generator) == before
    assert coordinator._pending_eof is None
    assert generator.advance_calls == 0
    assert scorer.candidates == 0
    assert renderer.calls == []

    result = coordinator.flush_delta(timeline, eof, identities)
    assert generator.advance_calls == 0
    assert generator.finalize_calls == 1
    assert scorer.candidates == 2
    assert [identity for _, identity in renderer.calls] == [1, 2]
    assert len(result.scores) == len(result.selected_scores) == 2
    assert len(result.render_jobs) == 2


@pytest.mark.parametrize(
    "failure_step",
    [
        "eof_observer:audio",
        "eof_observer:whisper",
        "observer_proposal",
        "generator_proposal",
        "before_commit",
        "former_generator_commit_gap",
        "after_state_construction",
    ],
)
def test_multi_observer_eof_transaction_failure_publishes_nothing(
    tmp_path, monkeypatch, failure_step
):
    (
        generator,
        scorer,
        renderer,
        coordinator,
        timeline,
        eof,
        identities,
    ) = eof_transaction_fixture(tmp_path)
    coordinator._activate(timeline)
    before = transaction_state_snapshot(coordinator, generator)

    def interrupt(step):
        if step == failure_step:
            raise KeyboardInterrupt()

    monkeypatch.setattr(coordinator, "_transaction_preparation_step", interrupt)
    with pytest.raises(KeyboardInterrupt):
        coordinator.flush_delta(timeline, eof, identities)

    assert transaction_state_snapshot(coordinator, generator) == before
    assert generator.finalize_calls == (
        1
        if failure_step
        in {
            "generator_proposal",
            "before_commit",
            "former_generator_commit_gap",
            "after_state_construction",
        }
        else 0
    )
    assert scorer.candidates == (
        2
        if failure_step
        in {"before_commit", "former_generator_commit_gap", "after_state_construction"}
        else 0
    )
    assert renderer.calls == []

    receipt = coordinator._pending_eof
    saved_generation = None if receipt is None else receipt.generation
    saved_scores = () if receipt is None else tuple(receipt.scores)
    monkeypatch.setattr(
        coordinator, "_transaction_preparation_step", lambda step: None
    )
    result = coordinator.flush_delta(timeline, eof, identities)

    assert generator.advance_calls == 0
    assert generator.finalize_calls == 1
    assert scorer.candidates == 2
    assert [identity for _, identity in renderer.calls] == [1, 2]
    assert len(result.scores) == len(result.selected_scores) == 2
    assert len(result.render_jobs) == 2
    if saved_generation is not None:
        assert coordinator._generation_checkpoint is saved_generation.checkpoint
    if saved_scores:
        assert tuple(result.scores) == tuple(item.score for item in saved_scores)


def test_eof_interruption_immediately_after_authoritative_swap_replays_commit(
    tmp_path, monkeypatch
):
    (
        generator,
        scorer,
        renderer,
        coordinator,
        timeline,
        eof,
        identities,
    ) = eof_transaction_fixture(tmp_path)
    interrupted = False

    def interrupt(step):
        nonlocal interrupted
        if step == "after_commit" and not interrupted:
            interrupted = True
            raise KeyboardInterrupt()

    monkeypatch.setattr(coordinator, "_transaction_publication_step", interrupt)
    with pytest.raises(KeyboardInterrupt):
        coordinator.flush_delta(timeline, eof, identities)

    published_metrics = coordinator.state_metrics
    assert generator.finalize_calls == 1
    assert scorer.candidates == 2
    assert renderer.calls == []
    assert coordinator._decision_state.committed_receipt_key == (
        coordinator._identity_key(coordinator._pending_eof.identity)
    )

    monkeypatch.setattr(
        coordinator, "_transaction_publication_step", lambda step: None
    )
    result = coordinator.flush_delta(timeline, eof, identities)
    assert generator.finalize_calls == 1
    assert scorer.candidates == 2
    assert [identity for _, identity in renderer.calls] == [1, 2]
    assert coordinator.state_metrics.generation_passes == (
        published_metrics.generation_passes
    )
    assert coordinator.state_metrics.scored_candidates == (
        published_metrics.scored_candidates
    )
    retry = coordinator.flush_delta(timeline, eof, identities)
    assert retry == result
    assert [identity for _, identity in renderer.calls] == [1, 2]


@pytest.mark.parametrize("boundary", ["before", "after"])
def test_eof_completion_and_pending_clear_publish_together(
    tmp_path, monkeypatch, boundary
):
    (
        generator,
        scorer,
        renderer,
        coordinator,
        timeline,
        eof,
        identities,
    ) = eof_transaction_fixture(tmp_path)
    interrupted = False

    def interrupt(step):
        nonlocal interrupted
        expected = "eof_completion" if boundary == "before" else "after_eof_completion"
        if step == expected and not interrupted:
            interrupted = True
            raise KeyboardInterrupt()

    hook = (
        "_completion_preparation_step"
        if boundary == "before"
        else "_completion_publication_step"
    )
    monkeypatch.setattr(coordinator, hook, interrupt)
    with pytest.raises(KeyboardInterrupt):
        coordinator.flush_delta(timeline, eof, identities)

    if boundary == "before":
        assert coordinator._pending_eof is not None
        assert coordinator.lifecycle.value == "active"
        assert coordinator._completed_eof_fingerprint is None
    else:
        assert coordinator._pending_eof is None
        assert coordinator.lifecycle.value == "flushed"
        assert coordinator._completed_eof_fingerprint is not None
    assert [identity for _, identity in renderer.calls] == [1, 2]

    monkeypatch.setattr(coordinator, hook, lambda step: None)
    result = coordinator.flush_delta(timeline, eof, identities)
    assert generator.finalize_calls == 1
    assert scorer.candidates == 2
    assert [identity for _, identity in renderer.calls] == [1, 2]
    assert coordinator._pending_eof is None
    assert coordinator._decision_state.pending_render_plan == ()
    assert coordinator.lifecycle.value == "flushed"
    assert len(result.render_jobs) == 2


def test_implicit_eof_identities_survive_interruption_before_completion_swap(
    tmp_path, monkeypatch
):
    generator = CountingGenerator()
    scorer = TrackingScorer()
    renderer = Renderer(tmp_path)
    coordinator = transaction_coordinator(tmp_path, generator, scorer, renderer)
    timeline = delta_timeline(
        tmp_path, [marker(1, 0, 2, 0.8, "implicit eof")]
    )
    eof = IncrementalEOF(20.0, ObserverWatermarks({"fixture": 20.0}))
    interrupted = False

    def interrupt(step):
        nonlocal interrupted
        if step == "eof_completion" and not interrupted:
            interrupted = True
            raise KeyboardInterrupt()

    monkeypatch.setattr(coordinator, "_completion_preparation_step", interrupt)
    with pytest.raises(KeyboardInterrupt):
        coordinator.flush_delta(timeline, eof)

    receipt = coordinator._pending_eof
    assert receipt is not None
    assert len(receipt.completed_inputs) == 1
    assert [identity for _, identity in renderer.calls] == [1]

    monkeypatch.setattr(
        coordinator, "_completion_preparation_step", lambda step: None
    )
    result = coordinator.flush_delta(timeline, eof)
    assert len(result.render_jobs) == 1
    assert generator.finalize_calls == 1
    assert scorer.candidates == 1
    assert [identity for _, identity in renderer.calls] == [1]
    assert coordinator._pending_eof is None
    assert receipt.completed_inputs == []


def test_completed_eof_receipt_releases_large_ignored_observation_payload(
    tmp_path, monkeypatch
):
    observations = [
        Observation(float(index), "fixture", "ignored", {"index": index})
        for index in range(1000)
    ]
    timeline = delta_timeline(tmp_path, observations)
    eof = IncrementalEOF(1000.0, ObserverWatermarks({"fixture": 1000.0}))
    identity = delta_identity("fixture", 0, eof=True)
    coordinator = IncrementalPrerecordedCoordinator(
        CandidateGenerator(),
        CandidateScorer(),
        CandidateSelector(),
        Renderer(tmp_path),
        IncrementalPipelineConfig(required_observers=("fixture",)),
    )

    def interrupt(step):
        if step == "render_plan":
            raise KeyboardInterrupt()

    monkeypatch.setattr(coordinator, "_decision_preparation_step", interrupt)
    with pytest.raises(KeyboardInterrupt):
        coordinator.flush_delta(timeline, eof, (identity,))

    receipt = coordinator._pending_eof
    assert receipt is not None
    assert len(receipt.observations) == 1000
    assert receipt.generation is not None
    assert coordinator.state_metrics.active_observations == 1000
    assert coordinator.state_metrics.peak_active_observations == 1000
    assert coordinator.lifecycle.value == "active"

    monkeypatch.setattr(coordinator, "_decision_preparation_step", lambda step: None)
    result = coordinator.flush_delta(timeline, eof, (identity,))

    assert result.scores == []
    assert result.render_jobs == []
    assert receipt.observations == ()
    assert receipt.generation is None
    assert receipt.scores == []
    assert receipt.finalized_scores == []
    assert receipt.next_active_scores == []
    assert receipt.safe_scores == []
    assert receipt.prepared_decisions is None
    assert receipt.pending_renders == ()
    assert receipt.completed_inputs == []
    assert coordinator._pending_eof is None
    assert coordinator._completed_eof_fingerprint == receipt.payload
    assert coordinator.state_metrics.active_observations == 0
    assert coordinator.state_metrics.active_scores == 0
    assert coordinator.state_metrics.peak_active_observations == 1000
    assert coordinator._decision_state.pending_render_plan == ()
    assert coordinator.lifecycle.value == "flushed"


@pytest.mark.parametrize("failure_identity", [1, 2, 3])
def test_failure_at_each_render_position_preserves_entire_saved_plan(
    tmp_path, failure_identity
):
    class PositionalFailureRenderer(Renderer):
        def __init__(self, root):
            super().__init__(root)
            self.attempts = []
            self.failed = False

        def render_one(self, score, identity):
            self.attempts.append(identity)
            if identity == failure_identity and not self.failed:
                self.failed = True
                raise RuntimeError(f"failed render {identity}")
            return super().render_one(score, identity)

    generator = CountingGenerator()
    scorer = TrackingScorer()
    renderer = PositionalFailureRenderer(tmp_path)
    coordinator = transaction_coordinator(tmp_path, generator, scorer, renderer)
    timeline = transaction_timeline(tmp_path)
    with pytest.raises(RuntimeError, match="failed render"):
        transaction_advance(coordinator, timeline)

    receipt = coordinator._pending_delta
    generation = receipt.generation
    assert generation is not None
    checkpoint = generation.checkpoint
    family_ids = tuple(item.family_id for item in receipt.scores)
    candidate_fingerprints = tuple(
        candidate_fingerprint(item.score.candidate, "state-cache-fixture")
        for item in receipt.scores
    )
    score_fingerprints = tuple(
        score_fingerprint(item.score, "state-cache-fixture")
        for item in receipt.scores
    )
    plan = tuple(
        (item.family_id, fingerprint, identity)
        for item, fingerprint, identity in receipt.pending_renders
    )
    assert [identity for _, _, identity in plan] == [1, 2, 3]
    with pytest.raises(TypeError):
        receipt.pending_renders[0] = receipt.pending_renders[0]  # type: ignore[index]

    transaction_advance(coordinator, timeline)
    expected_attempts = [*range(1, failure_identity + 1), *range(failure_identity, 4)]
    assert renderer.attempts == expected_attempts
    assert [identity for _, identity in renderer.calls] == [1, 2, 3]
    assert generator.advance_calls == 1
    assert scorer.candidates == 3
    assert coordinator._generation_checkpoint == checkpoint
    assert set(coordinator._finalized_scores) == set(family_ids)
    assert tuple(
        candidate_fingerprint(item.candidate, "state-cache-fixture")
        for item in coordinator.result.scores
    ) == candidate_fingerprints
    assert tuple(
        score_fingerprint(item, "state-cache-fixture")
        for item in coordinator.result.scores
    ) == score_fingerprints
    assert dict(coordinator._render_identities) == {
        family_id: identity for family_id, _, identity in plan
    }
    assert len(coordinator.result.selected_scores) == 3
    assert len(coordinator.result.render_jobs) == 3


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


def real_observer_timeline(tmp_path, observations, duration):
    results = [
        ObserverResult(
            "audio",
            [item for item in observations if item.observer == "audio"],
            {"duration_seconds": duration},
        ),
        ObserverResult(
            "whisper",
            [item for item in observations if item.observer == "whisper"],
            {"duration_seconds": duration},
        ),
    ]
    return FeatureTimeline(
        media_path=tmp_path / "source.mp4",
        audio_path=tmp_path / "audio.wav",
        timeline_path=tmp_path / "timeline.json",
        timeline=FeatureAggregator().aggregate(results),
        metadata={"source_id": "state-cache-fixture"},
    )


def test_real_generator_accepts_late_stable_overlap_before_family_closure(
    tmp_path,
):
    first = whisper_speech(0.0, 2.0, "first span")
    audio = Observation(
        0.5,
        "audio",
        "speaking_intensity",
        {"intensity": 0.8, "loudness_dbfs": -18.0},
        duration_seconds=2.0,
    )
    late = whisper_speech(1.0, 3.0, "legally late overlap")
    all_observations = [first, audio, late]
    batch = CandidateGenerator().generate(
        real_observer_timeline(tmp_path, all_observations, 90.0)
    )
    renderer = Renderer(tmp_path)
    coordinator = IncrementalPrerecordedCoordinator(
        CandidateGenerator(),
        CandidateScorer(CandidateScoringConfig(passing_score=0.0)),
        CandidateSelector(),
        renderer,
        IncrementalPipelineConfig(required_observers=("audio", "whisper")),
    )
    watermarks = ObserverWatermarks({"audio": 200.0, "whisper": 200.0})
    deltas = [
        (
            delta_timeline(
                tmp_path,
                [first],
                "whisper",
                whisper_metadata(20, [first]),
            ),
            whisper_identity(0),
        ),
        (
            delta_timeline(
                tmp_path,
                [audio],
                "audio",
                audio_metadata(25),
            ),
            audio_identity(0),
        ),
        (
            delta_timeline(
                tmp_path,
                [late],
                "whisper",
                whisper_metadata(30, [late]),
            ),
            whisper_identity(1),
        ),
        (
            delta_timeline(tmp_path, [], "audio", audio_metadata(2000)),
            audio_identity(1),
        ),
    ]
    for index, (delta, identity) in enumerate(deltas):
        assert coordinator.advance_delta(delta, watermarks, identity) == []
        assert renderer.calls == []
        if index == 0:
            assert coordinator.watermark_seconds == 200.0
            assert coordinator._generation_checkpoint.observer_frontiers == (
                ("audio", 0.0),
                ("whisper", 2.0),
            )
            assert coordinator.advance_delta(delta, watermarks, identity) == []
            assert coordinator._generation_checkpoint.retained_observation_count == 1

    jobs = coordinator.advance_delta(
        delta_timeline(tmp_path, [], "whisper", whisper_metadata(2000)),
        watermarks,
        whisper_identity(2),
    )

    assert len(batch) == len(jobs) == 1
    assert jobs[0].candidate == batch[0]
    assert len(renderer.calls) == 1
    assert coordinator._generation_checkpoint.observer_frontiers == (
        ("audio", 200.0),
        ("whisper", 200.0),
    )
    assert coordinator._generation_checkpoint.retained_observation_count == 0
    assert coordinator.state_metrics.peak_active_observations <= 6

    with pytest.raises(ValueError, match="accepted observer frontier"):
        coordinator.advance_delta(
            delta_timeline(
                tmp_path,
                [first],
                "whisper",
                whisper_metadata(2001, [first]),
            ),
            watermarks,
            whisper_identity(3),
        )
    assert len(renderer.calls) == 1

    result = coordinator.flush_delta(
        delta_timeline(tmp_path, [], "whisper", whisper_metadata(2000)),
        IncrementalEOF(
            90.0,
            watermarks,
        ),
    )
    assert [item.candidate for item in result.scores] == batch
    assert len(result.render_jobs) == 1


def test_audio_diagnostic_end_advances_behind_high_global_watermark(tmp_path):
    first = Observation(
        0.0,
        "audio",
        "speaking_intensity",
        {"intensity": 0.8, "loudness_dbfs": -18.0},
        duration_seconds=2.0,
    )
    late = Observation(
        1.0,
        "audio",
        "speaking_intensity",
        {"intensity": 0.9, "loudness_dbfs": -17.0},
        duration_seconds=2.0,
    )
    batch = CandidateGenerator().generate(
        real_observer_timeline(tmp_path, [first, late], 90.0)
    )
    renderer = Renderer(tmp_path)
    coordinator = IncrementalPrerecordedCoordinator(
        CandidateGenerator(),
        CandidateScorer(CandidateScoringConfig(passing_score=0.0)),
        CandidateSelector(),
        renderer,
        IncrementalPipelineConfig(required_observers=("audio",)),
    )
    watermarks = ObserverWatermarks({"audio": 200.0})

    assert coordinator.advance_delta(
        delta_timeline(tmp_path, [first], "audio", audio_metadata(20)),
        watermarks,
        audio_identity(0),
    ) == []
    assert coordinator._generation_checkpoint.observer_frontiers == (("audio", 2.0),)
    assert coordinator._generation_checkpoint.retained_observation_count == 1
    assert coordinator.advance_delta(
        delta_timeline(tmp_path, [late], "audio", audio_metadata(30)),
        watermarks,
        audio_identity(1),
    ) == []
    assert coordinator._generation_checkpoint.observer_frontiers == (("audio", 3.0),)
    assert coordinator._generation_checkpoint.retained_observation_count == 2

    jobs = coordinator.advance_delta(
        delta_timeline(tmp_path, [], "audio", audio_metadata(2000)),
        watermarks,
        audio_identity(2),
    )
    assert len(batch) == len(jobs) == 1
    assert jobs[0].candidate == batch[0]
    assert coordinator._generation_checkpoint.retained_observation_count == 0
    assert coordinator.state_metrics.active_observations == 0
    assert coordinator.state_metrics.peak_active_observations <= 4


def test_coordinator_eof_accepts_frontier_beyond_duration_and_clamps_candidates(
    tmp_path,
):
    audio = Observation(
        99.0,
        "audio",
        "speaking_intensity",
        {"intensity": 0.8, "loudness_dbfs": -18.0},
        duration_seconds=1.0,
    )
    whisper = whisper_speech(94.0, 100.0, "tail segment")
    observations = [audio, whisper]
    batch = CandidateGenerator().generate(
        real_observer_timeline(tmp_path, observations, 100.0)
    )
    renderer = Renderer(tmp_path)
    coordinator = IncrementalPrerecordedCoordinator(
        CandidateGenerator(),
        CandidateScorer(CandidateScoringConfig(passing_score=0.0)),
        CandidateSelector(),
        renderer,
        IncrementalPipelineConfig(required_observers=("audio", "whisper")),
    )
    watermarks = ObserverWatermarks({"audio": 101.0, "whisper": 101.0})
    assert coordinator.advance_delta(
        delta_timeline(tmp_path, [audio], "audio", audio_metadata(1010)),
        watermarks,
        audio_identity(0),
    ) == []
    assert coordinator.advance_delta(
        delta_timeline(
            tmp_path,
            [whisper],
            "whisper",
            whisper_metadata(1010, [whisper]),
        ),
        watermarks,
        whisper_identity(0),
    ) == []
    assert coordinator._generation_checkpoint.stable_through_seconds == 101.0
    assert coordinator._generation_checkpoint.observer_frontiers == (
        ("audio", 101.0),
        ("whisper", 101.0),
    )

    result = coordinator.flush_delta(
        delta_timeline(tmp_path, [], "audio", audio_metadata(1010)),
        IncrementalEOF(100.0, watermarks),
    )

    assert [item.candidate for item in result.scores] == batch
    assert all(item.candidate.end_seconds <= 100.0 for item in result.scores)
    assert len(result.selected_scores) == len(result.render_jobs) == 1


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
                    {
                        "duration_seconds": duration,
                        "incremental_frames_processed": round(
                            (item.timestamp_seconds + 4.0) * 10
                        ),
                        "sample_rate_hz": 10,
                        "finalized_speech_segment_identities": (
                            finalized_speech_segment_identity(item),
                        ),
                    },
                ),
            ObserverWatermarks(watermarks),
        )
    result = incremental.flush_delta(
        delta_timeline(
            tmp_path,
            [],
            "whisper",
            {
                "duration_seconds": duration,
                "incremental_frames_processed": round(duration * 10),
                "sample_rate_hz": 10,
            },
        ),
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


def intensity(timestamp, value=0.8):
    return Observation(
        float(timestamp),
        "audio",
        "speaking_intensity",
        {"intensity": value, "loudness_dbfs": -18.0},
        duration_seconds=1.0,
    )


def test_real_generator_keeps_evolving_cluster_revision_mutable(tmp_path):
    observations = [intensity(index * 0.5, 0.8) for index in range(360)]
    duration = 200.0
    full = FeatureTimeline(
        media_path=tmp_path / "source.mp4",
        audio_path=tmp_path / "audio.wav",
        timeline_path=tmp_path / "timeline.json",
        timeline=FeatureAggregator().aggregate(
            [ObserverResult("audio", observations, {"duration_seconds": duration})]
        ),
        metadata={"source_id": "state-cache-dense-revisions"},
    )
    generator = CandidateGenerator()
    scorer = CandidateScorer()
    selector = CandidateSelector()
    batch_scores = scorer.score(generator.generate(full))
    batch = selector.select(item for item in batch_scores if item.passed_threshold)
    renderer = Renderer(tmp_path)
    coordinator = IncrementalPrerecordedCoordinator(
        generator,
        scorer,
        selector,
        renderer,
        IncrementalPipelineConfig(required_observers=("audio",)),
    )

    for offset in range(0, len(observations), 10):
        batch_observations = observations[offset : offset + 10]
        frontier = batch_observations[-1].timestamp_seconds + 1.0
        coordinator.advance_delta(
            delta_timeline(
                tmp_path,
                batch_observations,
                "audio",
                {
                    "duration_seconds": duration,
                    "incremental_frames_processed": round(frontier * 10),
                    "sample_rate_hz": 10,
                },
            ),
            ObserverWatermarks({"audio": batch_observations[-1].timestamp_seconds}),
        )
    result = coordinator.flush_delta(
        delta_timeline(
            tmp_path,
            [],
            "audio",
            {
                "duration_seconds": duration,
                "incremental_frames_processed": round(duration * 10),
                "sample_rate_hz": 10,
            },
        ),
        IncrementalEOF(duration, ObserverWatermarks({"audio": duration})),
    )

    identity = lambda score: (
        score.candidate.start_seconds,
        score.candidate.end_seconds,
        score.overall_score,
    )
    assert [identity(item) for item in result.scores] == [
        identity(item) for item in batch_scores
    ], (
        [
            (
                item.candidate.metadata.get("original_cluster_start"),
                item.candidate.metadata.get("original_cluster_end"),
            )
            for item in result.scores
        ],
        [
            (
                item.candidate.metadata.get("original_cluster_start"),
                item.candidate.metadata.get("original_cluster_end"),
            )
            for item in batch_scores
        ],
    )
    assert [identity(item) for item in result.selected_scores] == [
        identity(item) for item in batch.selected
    ]
    assert len(renderer.calls) == len(batch.selected)
    assert coordinator.state_metrics.peak_active_observations < 260


def test_candidate_revision_contract_covers_boundary_anchor_and_score_changes(tmp_path):
    generator = CandidateGenerator()
    early = [intensity(index * 0.5, 0.55) for index in range(90)]
    later = [
        Observation(
            40.0 + index * 0.5,
            "whisper",
            "speech",
            {"text": f"strong reaction {index}", "speaker": None},
            duration_seconds=2.0,
            confidence=0.99,
        )
        for index in range(3)
    ]
    prefix = complete_timeline(tmp_path, early, 100.0)
    revised = complete_timeline(tmp_path, [*early, *later], 100.0)
    first = generator.generate(prefix)[0]
    final = generator.generate(revised)[0]

    assert (first.start_seconds, first.end_seconds) != (
        final.start_seconds,
        final.end_seconds,
    )
    assert first.metadata["contributing_observations"] != final.metadata[
        "contributing_observations"
    ]
    assert generator.revision_start_seconds(first) == 0.0
    assert generator.revision_stable_after_seconds(first) == 60.0


def test_below_confidence_overlapping_cluster_survives_until_it_becomes_candidate(
    tmp_path,
):
    first_cluster = [intensity(index * 0.5, 0.8) for index in range(109)]
    early_next_cluster = Observation(
        54.5,
        "audio",
        "speaking_intensity",
        {"intensity": 0.8, "loudness_dbfs": -18.0},
        duration_seconds=50.0,
    )
    completing_evidence = intensity(106.5, 0.8)
    observations = [*first_cluster, early_next_cluster, completing_evidence]
    duration = 180.0
    generator = CandidateGenerator(
        CandidateGenerationConfig(minimum_candidate_confidence=0.5)
    )
    scorer = CandidateScorer(CandidateScoringConfig(passing_score=0.0))
    selector = CandidateSelector()
    full = FeatureTimeline(
        media_path=tmp_path / "source.mp4",
        audio_path=tmp_path / "audio.wav",
        timeline_path=tmp_path / "timeline.json",
        timeline=FeatureAggregator().aggregate(
            [ObserverResult("audio", observations, {"duration_seconds": duration})]
        ),
        metadata={"source_id": "state-cache-fixture"},
    )
    batch_scores = scorer.score(generator.generate(full))
    batch = selector.select(item for item in batch_scores if item.passed_threshold)
    renderer = Renderer(tmp_path)
    coordinator = IncrementalPrerecordedCoordinator(
        generator,
        scorer,
        selector,
        renderer,
        IncrementalPipelineConfig(required_observers=("audio",)),
    )

    coordinator.advance_delta(
        delta_timeline(
            tmp_path,
            [*first_cluster, early_next_cluster],
            "audio",
            {
                "duration_seconds": duration,
                "incremental_frames_processed": 1060,
                "sample_rate_hz": 10,
            },
        ),
        ObserverWatermarks({"audio": 106.0}),
    )
    checkpoint = coordinator._generation_checkpoint
    assert checkpoint is not None
    assert any(
        event.observation == early_next_cluster for event in checkpoint._open_events
    )

    coordinator.advance_delta(
        delta_timeline(
            tmp_path,
            [completing_evidence],
            "audio",
            {
                "duration_seconds": duration,
                "incremental_frames_processed": 1075,
                "sample_rate_hz": 10,
            },
        ),
        ObserverWatermarks({"audio": 107.5}),
    )
    result = coordinator.flush_delta(
        delta_timeline(
            tmp_path,
            [],
            "audio",
            {
                "duration_seconds": duration,
                "incremental_frames_processed": 1800,
                "sample_rate_hz": 10,
            },
        ),
        IncrementalEOF(duration, ObserverWatermarks({"audio": duration})),
    )

    identity = lambda score: (
        score.candidate.start_seconds,
        score.candidate.end_seconds,
        score.overall_score,
    )
    assert [identity(item) for item in result.selected_scores] == [
        identity(item) for item in batch.selected
    ]
    assert len(renderer.calls) == len(batch.selected) == 2


@pytest.mark.parametrize(
    "malformation", ["family", "checkpoint", "source", "frontier", "owner"]
)
def test_malformed_or_inconsistent_checkpoint_lineage_is_rejected(
    tmp_path, malformation
):
    class InvalidCheckpointGenerator(MarkerGenerator):
        def advance_incremental(
            self,
            checkpoint,
            observations,
            stable_through_seconds,
            observer_frontiers=None,
        ):
            output = super().advance_incremental(
                checkpoint,
                observations,
                stable_through_seconds,
                observer_frontiers,
            )
            if malformation == "family":
                family = output.closed_families[0]
                return CandidateGenerationAdvance(
                    output.checkpoint,
                    (
                        ClosedCandidateFamily(
                            CandidateFamilyId(checkpoint.source_id, 7),
                            family.candidate,
                        ),
                    ),
                )
            if malformation == "checkpoint":
                next_checkpoint = output.checkpoint
                assert next_checkpoint is not None
                return CandidateGenerationAdvance(
                    CandidateGenerationCheckpoint(
                        next_checkpoint._owner_token,
                        next_checkpoint.source_id,
                        next_checkpoint.media_path,
                        next_checkpoint.stable_through_seconds,
                        next_checkpoint.next_family_ordinal + 1,
                        next_checkpoint._open_events,
                    ),
                    output.closed_families,
                )
            next_checkpoint = output.checkpoint
            assert next_checkpoint is not None
            if malformation == "frontier":
                return CandidateGenerationAdvance(
                    CandidateGenerationCheckpoint(
                        next_checkpoint._owner_token,
                        next_checkpoint.source_id,
                        next_checkpoint.media_path,
                        next_checkpoint.stable_through_seconds + 1.0,
                        next_checkpoint.next_family_ordinal,
                        next_checkpoint._open_events,
                    ),
                    output.closed_families,
                )
            if malformation == "owner":
                return CandidateGenerationAdvance(
                    CandidateGenerationCheckpoint(
                        object(),
                        next_checkpoint.source_id,
                        next_checkpoint.media_path,
                        next_checkpoint.stable_through_seconds,
                        next_checkpoint.next_family_ordinal,
                        next_checkpoint._open_events,
                    ),
                    output.closed_families,
                )
            return CandidateGenerationAdvance(
                CandidateGenerationCheckpoint(
                    next_checkpoint._owner_token,
                    "another-source",
                    next_checkpoint.media_path,
                    next_checkpoint.stable_through_seconds,
                    next_checkpoint.next_family_ordinal,
                    next_checkpoint._open_events,
                ),
                output.closed_families,
            )

    coordinator = IncrementalPrerecordedCoordinator(
        InvalidCheckpointGenerator(),
        TrackingScorer(),
        CandidateSelector(),
        Renderer(tmp_path),
        IncrementalPipelineConfig(required_observers=("fixture",)),
    )
    with pytest.raises(
        RuntimeError, match="lineage|source ownership|accepted frontier"
    ):
        coordinator.advance_delta(
            delta_timeline(tmp_path, [marker(1, 0, 2, 0.8, "invalid")]),
            ObserverWatermarks({"fixture": 10.0}),
        )


def test_nonempty_initial_generator_checkpoint_is_rejected(tmp_path):
    class InvalidStartGenerator(MarkerGenerator):
        def start_incremental(
            self, *, source_id, media_path, required_observers=()
        ):
            checkpoint = super().start_incremental(
                source_id=source_id,
                media_path=media_path,
                required_observers=required_observers,
            )
            return CandidateGenerationCheckpoint(
                checkpoint._owner_token,
                checkpoint.source_id,
                checkpoint.media_path,
                1.0,
                1,
                (),
            )

    coordinator = IncrementalPrerecordedCoordinator(
        InvalidStartGenerator(),
        TrackingScorer(),
        CandidateSelector(),
        Renderer(tmp_path),
        IncrementalPipelineConfig(required_observers=("fixture",)),
    )
    with pytest.raises(RuntimeError, match="initial checkpoint was not empty"):
        coordinator.advance_delta(
            delta_timeline(tmp_path, []),
            ObserverWatermarks({"fixture": 1.0}),
        )


def test_validation_04_replacement_boundaries_never_render_as_old_revisions(tmp_path):
    class Validation04RevisionGenerator:
        maximum_backtrack_seconds = 70.0
        maximum_competition_seconds = 0.0
        incremental_deterministic = True

        def __init__(self):
            self._owner = object()
            self._committed = None
            self._pending = None
            self._publication = None

        @staticmethod
        def earliest_future_candidate_start_seconds(checkpoint):
            if checkpoint.next_family_ordinal >= 2:
                return float("inf")
            frontier = min(value for _, value in checkpoint._observer_frontiers)
            return frontier - 70.0

        def start_incremental(
            self, *, source_id, media_path, required_observers=()
        ):
            observers = tuple(required_observers)
            checkpoint = CandidateGenerationCheckpoint(
                self._owner,
                source_id,
                Path(media_path),
                0.0,
                0,
                (),
                observers,
                tuple(sorted((observer, 0.0) for observer in observers)),
            )
            self._committed = checkpoint
            return checkpoint

        def bind_incremental_publication(self, checkpoint, committed_checkpoint):
            assert checkpoint is self._committed
            assert callable(committed_checkpoint)
            self._publication = committed_checkpoint

        def _published_checkpoint(self):
            if self._publication is None:
                return self._committed
            return self._publication()

        def advance_incremental(
            self,
            checkpoint,
            observations,
            stable_through_seconds,
            observer_frontiers=None,
        ):
            assert checkpoint is self._published_checkpoint()
            frontiers = tuple(sorted((observer_frontiers or {}).items()))
            if stable_through_seconds < 170.0:
                output = CandidateGenerationAdvance(
                    CandidateGenerationCheckpoint(
                        self._owner,
                        checkpoint.source_id,
                        checkpoint.media_path,
                        stable_through_seconds,
                        checkpoint.next_family_ordinal,
                        (),
                        checkpoint._required_observers,
                        frontiers,
                    ),
                    (),
                )
            else:
                families = tuple(
                    ClosedCandidateFamily(
                        CandidateFamilyId(
                            checkpoint.source_id,
                            checkpoint.next_family_ordinal + index,
                        ),
                        ClipCandidate(
                            checkpoint.media_path,
                            start,
                            end,
                            f"revision-{index}",
                            metadata={"score": 0.8},
                        ),
                    )
                    for index, (start, end) in enumerate(
                        [(73.0, 108.0), (102.6035, 135.0)]
                    )
                )
                output = CandidateGenerationAdvance(
                    CandidateGenerationCheckpoint(
                        self._owner,
                        checkpoint.source_id,
                        checkpoint.media_path,
                        stable_through_seconds,
                        checkpoint.next_family_ordinal + len(families),
                        (),
                        checkpoint._required_observers,
                        frontiers,
                    ),
                    families,
                )
            if self._publication is None:
                self._pending = (checkpoint, output)
            return output

        def finalize_incremental(self, checkpoint, observations, media_duration_seconds):
            assert checkpoint is self._published_checkpoint()
            output = CandidateGenerationAdvance(None, ())
            if self._publication is None:
                self._pending = (checkpoint, output)
            return output

        def commit_incremental(self, checkpoint, advance):
            assert self._publication is None
            assert self._pending == (checkpoint, advance)
            self._committed = advance.checkpoint
            self._pending = None

        @staticmethod
        def generate(timeline):
            raise AssertionError("Production continuation must not replay batch history.")

        @staticmethod
        def revision_start_seconds(candidate):
            raise AssertionError("Legacy revision contracts must not be called.")

        @staticmethod
        def revision_partition_seconds(candidate):
            raise AssertionError("Legacy revision contracts must not be called.")

        @staticmethod
        def revision_stable_after_seconds(candidate):
            raise AssertionError("Legacy revision contracts must not be called.")

        @staticmethod
        def earliest_unresolved_cluster_start_seconds(timeline, stable):
            raise AssertionError("Legacy revision contracts must not be called.")

    initial = Observation(150.0, "fixture", "phase", {"phase": "initial"})
    final = Observation(165.0, "fixture", "phase", {"phase": "final"})
    renderer = Renderer(tmp_path)
    coordinator = IncrementalPrerecordedCoordinator(
        Validation04RevisionGenerator(),
        TrackingScorer(),
        CandidateSelector(),
        renderer,
        IncrementalPipelineConfig(required_observers=("fixture",)),
    )

    assert coordinator.advance_delta(
        delta_timeline(tmp_path, [initial]),
        ObserverWatermarks({"fixture": 160.0}),
    ) == []
    assert renderer.calls == []
    jobs = coordinator.advance_delta(
        delta_timeline(tmp_path, [final]),
        ObserverWatermarks({"fixture": 170.0}),
    )

    assert [
        (job.candidate.start_seconds, job.candidate.end_seconds) for job in jobs
    ] == [(73.0, 108.0), (102.6035, 135.0)]
    assert all(
        (score.candidate.start_seconds, score.candidate.end_seconds)
        not in {(85.225, 119.0), (123.0, 156.0)}
        for score in coordinator.result.selected_scores
    )


def test_retained_evidence_can_revise_score_without_moving_candidate_boundary(tmp_path):
    generator = CandidateGenerator()
    scorer = CandidateScorer()
    anchor = speech(10.0, "complete reaction")
    support = intensity(12.0, 0.95)
    prefix = complete_timeline(tmp_path, [anchor], 80.0)
    revised = complete_timeline(tmp_path, [anchor, support], 80.0)
    first = scorer.score(generator.generate(prefix))[0]
    final = scorer.score(generator.generate(revised))[0]

    assert (
        first.candidate.start_seconds,
        first.candidate.end_seconds,
    ) == (
        final.candidate.start_seconds,
        final.candidate.end_seconds,
    )
    assert first.candidate.source_signals != final.candidate.source_signals
    assert first.candidate.metadata["confidence"] != final.candidate.metadata["confidence"]
    assert first.overall_score != final.overall_score


def test_overlapping_whisper_revisions_render_only_final_identity(tmp_path):
    observations = [
        speech(10.0, "first provisional wording"),
        speech(12.0, "reconciled wording"),
        speech(14.0, "final complete wording"),
    ]
    duration = 90.0
    full = complete_timeline(tmp_path, observations, duration)
    generator = CandidateGenerator()
    scorer = CandidateScorer(CandidateScoringConfig(passing_score=0.0))
    selector = CandidateSelector()
    batch_scores = scorer.score(generator.generate(full))
    batch = selector.select(item for item in batch_scores if item.passed_threshold)
    renderer = Renderer(tmp_path)
    coordinator = IncrementalPrerecordedCoordinator(
        generator,
        scorer,
        selector,
        renderer,
        IncrementalPipelineConfig(required_observers=("whisper",)),
    )

    for sequence, item in enumerate(observations):
        frontier = item.timestamp_seconds + (item.duration_seconds or 0.0)
        coordinator.advance_delta(
            delta_timeline(
                tmp_path,
                [item],
                "whisper",
                {
                    "duration_seconds": duration,
                    "incremental_frames_processed": round(frontier * 10),
                    "sample_rate_hz": 10,
                    "finalized_speech_segment_identities": (
                        finalized_speech_segment_identity(item),
                    ),
                },
            ),
            ObserverWatermarks({"whisper": frontier}),
            ObserverDeltaIdentity(
                "state-cache-fixture",
                "incremental:state-cache-fixture",
                "whisper",
                sequence,
                False,
            ),
        )
        assert renderer.calls == []
    result = coordinator.flush_delta(
        delta_timeline(
            tmp_path,
            [],
            "whisper",
            {
                "duration_seconds": duration,
                "incremental_frames_processed": round(duration * 10),
                "sample_rate_hz": 10,
            },
        ),
        IncrementalEOF(duration, ObserverWatermarks({"whisper": duration})),
    )

    identity = lambda score: (
        score.candidate.start_seconds,
        score.candidate.end_seconds,
        score.overall_score,
    )
    assert [identity(item) for item in result.selected_scores] == [
        identity(item) for item in batch.selected
    ]
    assert len(renderer.calls) == 1


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
