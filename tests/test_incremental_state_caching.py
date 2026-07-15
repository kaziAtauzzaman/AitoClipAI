from pathlib import Path

import pytest

from aggregation import FeatureAggregator
from candidate_generation import CandidateGenerationConfig, CandidateGenerator
from candidate_scoring import CandidateScorer, CandidateScoringConfig
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
from whisper_observer import finalized_speech_segment_identity


class MarkerGenerator:
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

    assert coordinator.state_metrics.active_observations == 1


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
    assert coordinator.state_metrics.active_observations == 1


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

    assert coordinator.state_metrics.active_observations == 2


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
    assert coordinator.state_metrics.active_observations == 1


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
    assert coordinator.state_metrics.active_observations == 2


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
    assert coordinator._immutable_through == 54.5
    assert early_next_cluster in coordinator._active_observations["audio"]

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
    ("start", "partition", "stable_after"),
    [
        (-1.0, 2.0, 2.0),
        (0.0, -1.0, 2.0),
        (0.0, 3.0, 2.0),
        (2.0, 1.0, 3.0),
        (0.0, float("nan"), 2.0),
        (0.0, 2.0, float("inf")),
        ("bad", 2.0, 2.0),
    ],
)
def test_malformed_or_inconsistent_revision_contract_is_rejected(
    tmp_path, start, partition, stable_after
):
    class InvalidRevisionGenerator(MarkerGenerator):
        @staticmethod
        def revision_start_seconds(candidate):
            return start

        @staticmethod
        def revision_partition_seconds(candidate):
            return partition

        @staticmethod
        def revision_stable_after_seconds(candidate):
            return stable_after

    coordinator = IncrementalPrerecordedCoordinator(
        InvalidRevisionGenerator(),
        TrackingScorer(),
        CandidateSelector(),
        Renderer(tmp_path),
        IncrementalPipelineConfig(required_observers=("fixture",)),
    )
    with pytest.raises(ValueError, match="revision contract"):
        coordinator.advance_delta(
            delta_timeline(tmp_path, [marker(1, 0, 2, 0.8, "invalid")]),
            ObserverWatermarks({"fixture": 10.0}),
        )


def test_validation_04_replacement_boundaries_never_render_as_old_revisions(tmp_path):
    class Validation04RevisionGenerator:
        maximum_backtrack_seconds = 70.0
        maximum_competition_seconds = 0.0
        incremental_deterministic = True

        @staticmethod
        def generate(timeline):
            observations = [
                item
                for result in timeline.timeline.observer_results
                for item in result.observations
            ]
            final = any(item.value.get("phase") == "final" for item in observations)
            boundaries = (
                [(73.0, 108.0), (102.6035, 135.0)]
                if final
                else [(85.225, 119.0), (123.0, 156.0)]
            )
            return [
                ClipCandidate(
                    timeline.media_path,
                    start,
                    end,
                    f"revision-{index}",
                    metadata={
                        "score": 0.8,
                        "revision_start": 50.0 if index == 0 else 100.0,
                        "revision_partition": 105.0 if index == 0 else 153.0,
                        "contributing_observations": observations,
                    },
                )
                for index, (start, end) in enumerate(boundaries)
            ]

        @staticmethod
        def revision_start_seconds(candidate):
            return candidate.metadata["revision_start"]

        @staticmethod
        def revision_partition_seconds(candidate):
            return candidate.metadata["revision_partition"]

        @staticmethod
        def revision_stable_after_seconds(candidate):
            return 170.0

        @staticmethod
        def earliest_unresolved_cluster_start_seconds(timeline, stable):
            return 50.0 if stable < 170.0 else None

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
