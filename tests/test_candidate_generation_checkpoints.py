from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from aggregation import FeatureAggregator
from candidate_generation import (
    CandidateFamilyId,
    CandidateGenerationAdvance,
    CandidateGenerationConfig,
    CandidateGenerator,
    CandidateGenerationCheckpoint,
)
from candidate_scoring import CandidateScorer
from core import FeatureTimeline, Observation, ObserverResult
from pipeline.incremental import candidate_fingerprint, score_fingerprint


SOURCE_ID = "checkpoint-fixture"


def intensity(
    timestamp: float,
    strength: float = 0.8,
    duration: float = 1.0,
) -> Observation:
    return Observation(
        timestamp,
        "audio",
        "speaking_intensity",
        {"intensity": strength, "loudness_dbfs": -18.0},
        duration_seconds=duration,
    )


def speech(
    timestamp: float,
    duration: float = 2.0,
    confidence: float = 0.95,
) -> Observation:
    return Observation(
        timestamp,
        "whisper",
        "speech",
        {"text": f"speech at {timestamp}", "speaker": None},
        duration_seconds=duration,
        confidence=confidence,
    )


def timeline(
    tmp_path: Path,
    observations: list[Observation],
    duration: float,
) -> FeatureTimeline:
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
        metadata={"source_id": SOURCE_ID},
    )


def incremental_candidates(
    generator: CandidateGenerator,
    tmp_path: Path,
    deltas: list[tuple[list[Observation], float]],
    duration: float,
):
    checkpoint = generator.start_incremental(
        source_id=SOURCE_ID,
        media_path=tmp_path / "source.mp4",
    )
    closed = []
    for observations, stable in deltas:
        advance = generator.advance_incremental(checkpoint, observations, stable)
        assert advance.checkpoint is not None
        generator.commit_incremental(checkpoint, advance)
        checkpoint = advance.checkpoint
        closed.extend(advance.closed_families)
    final = generator.finalize_incremental(checkpoint, (), duration)
    assert final.checkpoint is None
    generator.commit_incremental(checkpoint, final)
    closed.extend(final.closed_families)
    return closed


@pytest.mark.parametrize(
    "cut_points",
    [
        (80, 160),
        (1, 121, 241),
        (40, 80, 120, 160, 200),
    ],
)
def test_checkpoint_chunkings_and_unchanged_watermarks_equal_batch(
    tmp_path: Path,
    cut_points: tuple[int, ...],
) -> None:
    observations = [intensity(index * 0.5) for index in range(240)]
    observations.extend([speech(25.0, 30.0), speech(75.0, 30.0)])
    observations.sort(key=lambda item: (item.timestamp_seconds, item.observer, item.type))
    duration = 180.0
    generator = CandidateGenerator()
    batch = generator.generate(timeline(tmp_path, observations, duration))

    boundaries = (0, *cut_points, len(observations))
    chunks = [
        observations[start:end]
        for start, end in zip(boundaries, boundaries[1:])
    ]
    closed = incremental_candidates(
        generator,
        tmp_path,
        [
            (chunk, 120.0 if index == len(chunks) - 1 else 0.0)
            for index, chunk in enumerate(chunks)
        ],
        duration,
    )

    assert [item.candidate for item in closed if item.candidate is not None] == batch
    assert [item.family_id.ordinal for item in closed] == list(range(len(closed)))


def test_equal_position_observer_interleavings_equal_batch(tmp_path: Path) -> None:
    first = Observation(
        10.0,
        "whisper",
        "speech",
        {"text": "first equally strong segment", "speaker": None},
        duration_seconds=2.0,
        confidence=0.95,
        metadata={"segment": 1},
    )
    second = Observation(
        10.0,
        "audio",
        "speaking_intensity",
        {"intensity": 0.8, "loudness_dbfs": -18.0},
        duration_seconds=2.0,
        metadata={"segment": 2},
    )
    generator = CandidateGenerator()
    batch = generator.generate(timeline(tmp_path, [first, second], 90.0))

    closed = incremental_candidates(
        generator,
        tmp_path,
        [([second], 10.0), ([first], 10.0)],
        90.0,
    )

    assert [item.candidate for item in closed if item.candidate is not None] == batch


@pytest.mark.parametrize(
    "chunks",
    [
        ((0, 2),),
        ((0, 1), (1, 2)),
        ((0, 1), (1, 1), (1, 2)),
    ],
)
def test_equal_event_keys_preserve_batch_input_order_across_legal_chunkings(
    tmp_path: Path,
    chunks: tuple[tuple[int, int], ...],
) -> None:
    first = Observation(
        10.0,
        "audio",
        "speaking_intensity",
        {"intensity": 0.8, "loudness_dbfs": -18.0, "payload": "first"},
        duration_seconds=2.0,
        metadata={"delivery": 1, "nested": ["first"]},
    )
    second = Observation(
        10.0,
        "audio",
        "speaking_intensity",
        {"intensity": 0.8, "loudness_dbfs": -18.0, "payload": "second"},
        duration_seconds=2.0,
        metadata={"delivery": 2, "nested": ["second"]},
    )
    observations = [first, second]
    batch = CandidateGenerator().generate(timeline(tmp_path, observations, 90.0))
    reversed_batch = CandidateGenerator().generate(
        timeline(tmp_path, list(reversed(observations)), 90.0)
    )
    assert len(batch) == len(reversed_batch) == 1
    assert [
        item.value["payload"]
        for item in batch[0].metadata["contributing_observations"]
    ] == ["first", "second"]
    assert [
        item.value["payload"]
        for item in reversed_batch[0].metadata["contributing_observations"]
    ] == ["second", "first"]
    assert candidate_fingerprint(batch[0], SOURCE_ID) != candidate_fingerprint(
        reversed_batch[0], SOURCE_ID
    )
    assert candidate_fingerprint(batch[0], SOURCE_ID) == (
        "233ce3fbfa5625a6da0b30c18410bafec11ee89bf1cc16e3e5cd43e2ea44e612"
    )
    assert score_fingerprint(CandidateScorer().score(batch)[0], SOURCE_ID) == (
        "c0d8bf794e22060d7263c600ecc7842004a224aac1c0e4eeaa6c65ffbadab9ce"
    )

    closed = incremental_candidates(
        CandidateGenerator(),
        tmp_path,
        [
            (observations[start:end], 0.0)
            for start, end in chunks
        ],
        90.0,
    )
    incremental = [
        item.candidate for item in closed if item.candidate is not None
    ]
    assert incremental == batch
    assert [candidate_fingerprint(item, SOURCE_ID) for item in incremental] == [
        candidate_fingerprint(item, SOURCE_ID) for item in batch
    ]
    assert [
        score_fingerprint(item, SOURCE_ID)
        for item in CandidateScorer().score(incremental)
    ] == [
        score_fingerprint(item, SOURCE_ID)
        for item in CandidateScorer().score(batch)
    ]


def test_required_observer_frontiers_prevent_legal_late_overlap_reopening(
    tmp_path: Path,
) -> None:
    first = speech(0.0, duration=2.0)
    interleaved_audio = intensity(0.5, duration=2.0)
    late = speech(1.0, duration=2.0)
    observations = [first, interleaved_audio, late]
    batch = CandidateGenerator().generate(timeline(tmp_path, observations, 90.0))
    generator = CandidateGenerator()
    checkpoint = generator.start_incremental(
        source_id=SOURCE_ID,
        media_path=tmp_path / "source.mp4",
        required_observers=("audio", "whisper"),
    )

    steps = [
        ([first], {"audio": 0.0, "whisper": 2.0}),
        ([interleaved_audio], {"audio": 2.5, "whisper": 2.0}),
        ([late], {"audio": 2.5, "whisper": 3.0}),
        ([], {"audio": 60.0, "whisper": 3.0}),
    ]
    for delta, frontiers in steps:
        advance = generator.advance_incremental(
            checkpoint,
            delta,
            60.0,
            frontiers,
        )
        assert advance.checkpoint is not None
        assert advance.closed_families == ()
        generator.commit_incremental(checkpoint, advance)
        checkpoint = advance.checkpoint

    closed = generator.advance_incremental(
        checkpoint,
        (),
        60.0,
        {"audio": 60.0, "whisper": 60.0},
    )
    assert closed.checkpoint is not None
    assert len(closed.closed_families) == 1
    assert closed.closed_families[0].candidate == batch[0]
    generator.commit_incremental(checkpoint, closed)

    no_reopen = generator.advance_incremental(
        closed.checkpoint,
        (),
        61.0,
        {"audio": 61.0, "whisper": 61.0},
    )
    assert no_reopen.closed_families == ()
    generator.commit_incremental(closed.checkpoint, no_reopen)


def test_long_whisper_event_end_frontier_preserves_late_overlapping_segment(
    tmp_path: Path,
) -> None:
    first = speech(0.0, duration=50.0)
    late = speech(10.0, duration=45.0)
    batch = CandidateGenerator().generate(timeline(tmp_path, [first, late], 100.0))
    generator = CandidateGenerator()
    checkpoint = generator.start_incremental(
        source_id=SOURCE_ID,
        media_path=tmp_path / "source.mp4",
        required_observers=("whisper",),
    )

    first_advance = generator.advance_incremental(
        checkpoint,
        [first],
        100.0,
        {"whisper": 50.0},
    )
    assert first_advance.checkpoint is not None
    assert first_advance.closed_families == ()
    generator.commit_incremental(checkpoint, first_advance)

    late_advance = generator.advance_incremental(
        first_advance.checkpoint,
        [late],
        100.0,
        {"whisper": 55.0},
    )
    assert late_advance.checkpoint is not None
    assert late_advance.closed_families == ()
    assert late_advance.checkpoint.retained_observation_count == 2
    generator.commit_incremental(first_advance.checkpoint, late_advance)

    closed = generator.advance_incremental(
        late_advance.checkpoint,
        (),
        100.0,
        {"whisper": 60.0},
    )
    assert len(closed.closed_families) == 1
    assert closed.closed_families[0].candidate == batch[0]
    generator.commit_incremental(late_advance.checkpoint, closed)


def test_eof_allows_observer_frontiers_beyond_clamped_media_duration(
    tmp_path: Path,
) -> None:
    observations = [speech(94.0, duration=6.0), intensity(99.0)]
    batch = CandidateGenerator().generate(timeline(tmp_path, observations, 100.0))
    generator = CandidateGenerator()
    checkpoint = generator.start_incremental(
        source_id=SOURCE_ID,
        media_path=tmp_path / "source.mp4",
        required_observers=("audio", "whisper"),
    )
    advance = generator.advance_incremental(
        checkpoint,
        observations,
        101.0,
        {"audio": 101.0, "whisper": 101.0},
    )
    assert advance.checkpoint is not None
    assert advance.checkpoint.stable_through_seconds == 101.0
    assert advance.checkpoint.observer_frontiers == (
        ("audio", 101.0),
        ("whisper", 101.0),
    )
    assert advance.closed_families == ()
    generator.commit_incremental(checkpoint, advance)

    final = generator.finalize_incremental(advance.checkpoint, (), 100.0)
    assert final.checkpoint is None
    assert [
        family.candidate
        for family in final.closed_families
        if family.candidate is not None
    ] == batch
    assert all(
        family.candidate is None or family.candidate.end_seconds <= 100.0
        for family in final.closed_families
    )
    generator.commit_incremental(advance.checkpoint, final)


def test_checkpoint_is_immutable_owned_and_retry_deterministic(tmp_path: Path) -> None:
    generator = CandidateGenerator()
    checkpoint = generator.start_incremental(
        source_id=SOURCE_ID,
        media_path=tmp_path / "source.mp4",
    )
    observations = [speech(0.0), speech(1.0)]

    first = generator.advance_incremental(checkpoint, observations, 60.0)
    second = generator.advance_incremental(checkpoint, observations, 60.0)

    assert first == second
    assert first is second
    with pytest.raises(ValueError, match="different uncommitted transition"):
        generator.advance_incremental(checkpoint, (), 60.0)
    assert checkpoint.retained_observation_count == 0
    assert first.checkpoint is not None
    with pytest.raises(ValueError, match="stale|not yet committed"):
        generator.advance_incremental(first.checkpoint, (), 60.0)
    with pytest.raises(FrozenInstanceError):
        checkpoint.stable_through_seconds = 1.0  # type: ignore[misc]
    with pytest.raises(ValueError, match="originating generator"):
        CandidateGenerator().advance_incremental(checkpoint, (), 60.0)
    generator.commit_incremental(checkpoint, first)
    with pytest.raises(ValueError, match="active uncommitted"):
        generator.commit_incremental(checkpoint, first)
    with pytest.raises(ValueError, match="stale"):
        generator.advance_incremental(checkpoint, observations, 60.0)

    forged_source = CandidateGenerationCheckpoint(
        first.checkpoint._owner_token,
        "another-source",
        first.checkpoint.media_path,
        first.checkpoint.stable_through_seconds,
        first.checkpoint.next_family_ordinal,
        first.checkpoint._open_events,
        first.checkpoint._required_observers,
        first.checkpoint._observer_frontiers,
    )
    with pytest.raises(ValueError, match="another source"):
        generator.advance_incremental(forged_source, (), 60.0)

    original_config = generator._config
    generator._config = CandidateGenerationConfig(merge_gap_seconds=3.0)
    try:
        with pytest.raises(ValueError, match="another configuration"):
            generator.advance_incremental(first.checkpoint, (), 60.0)
    finally:
        generator._config = original_config

    final = generator.finalize_incremental(first.checkpoint, (), 90.0)
    assert generator.finalize_incremental(first.checkpoint, (), 90.0) is final
    generator.commit_incremental(first.checkpoint, final)
    with pytest.raises(ValueError, match="finalized"):
        generator.finalize_incremental(first.checkpoint, (), 90.0)
    with pytest.raises(ValueError, match="finalized"):
        generator.advance_incremental(first.checkpoint, (), 90.0)


def test_checkpoint_snapshots_mutable_observation_payloads(tmp_path: Path) -> None:
    config = CandidateGenerationConfig(minimum_candidate_confidence=0.0)
    generator = CandidateGenerator(config)
    checkpoint = generator.start_incremental(
        source_id=SOURCE_ID,
        media_path=tmp_path / "source.mp4",
    )
    observation = intensity(10.0)
    observation.metadata["nested"] = ["original"]
    observation.metadata["nested_mapping"] = {"labels": ["original"]}
    observation.metadata["tuple"] = ("preserved",)
    batch_observation = intensity(10.0)
    batch_observation.metadata["nested"] = ["original"]
    batch_observation.metadata["nested_mapping"] = {"labels": ["original"]}
    batch_observation.metadata["tuple"] = ("preserved",)
    batch = CandidateGenerator(config).generate(
        timeline(tmp_path, [batch_observation], 90.0)
    )

    advance = generator.advance_incremental(checkpoint, [observation], 10.0)
    assert advance.checkpoint is not None
    retained = advance.checkpoint._open_events[0].observation
    observation.value["intensity"] = 0.0
    observation.metadata["nested"].append("mutated")
    observation.metadata["nested_mapping"]["labels"].append("mutated")

    assert retained.value["intensity"] == 0.8
    assert tuple(retained.metadata["nested"]) == ("original",)
    assert tuple(retained.metadata["nested_mapping"]["labels"]) == ("original",)
    assert not isinstance(retained.value, dict)
    assert not isinstance(retained.metadata["nested"], list)
    with pytest.raises(TypeError):
        retained.value["intensity"] = 0.1
    with pytest.raises(TypeError):
        dict.__setitem__(retained.value, "intensity", 0.1)
    with pytest.raises(AttributeError):
        retained.value.update({"intensity": 0.1})
    with pytest.raises(TypeError):
        list.__setitem__(retained.metadata["nested"], 0, "forbidden")
    with pytest.raises(TypeError):
        dict.__setitem__(
            retained.metadata["nested_mapping"],
            "labels",
            (),
        )
    with pytest.raises(AttributeError):
        retained.metadata["nested"].append("forbidden")

    generator.commit_incremental(checkpoint, advance)
    final = generator.finalize_incremental(advance.checkpoint, (), 90.0)
    assert [
        family.candidate
        for family in final.closed_families
        if family.candidate is not None
    ] == batch
    contributing = final.closed_families[0].candidate.metadata[
        "contributing_observations"
    ][0]
    assert contributing.metadata["nested"] == ["original"]
    assert contributing.metadata["nested_mapping"] == {"labels": ["original"]}
    assert contributing.metadata["tuple"] == ("preserved",)
    generator.commit_incremental(advance.checkpoint, final)


def test_closed_family_never_reopens_or_reemits(tmp_path: Path) -> None:
    generator = CandidateGenerator()
    checkpoint = generator.start_incremental(
        source_id=SOURCE_ID,
        media_path=tmp_path / "source.mp4",
    )
    first = generator.advance_incremental(
        checkpoint,
        [speech(0.0), speech(1.0)],
        60.0,
    )
    assert first.checkpoint is not None
    assert len(first.closed_families) == 1
    generator.commit_incremental(checkpoint, first)

    later = generator.advance_incremental(first.checkpoint, [speech(75.0)], 75.0)
    assert later.checkpoint is not None
    assert later.closed_families == ()
    generator.commit_incremental(first.checkpoint, later)
    final = generator.finalize_incremental(later.checkpoint, (), 120.0)
    generator.commit_incremental(later.checkpoint, final)

    assert [item.family_id.ordinal for item in first.closed_families] == [0]
    assert [item.family_id.ordinal for item in final.closed_families] == [1]


def test_closed_candidate_window_may_cross_next_open_family_start(
    tmp_path: Path,
) -> None:
    generator = CandidateGenerator()
    observations = [speech(10.0, 1.0), speech(14.0, 1.0)]
    batch = generator.generate(timeline(tmp_path, observations, 90.0))
    checkpoint = generator.start_incremental(
        source_id=SOURCE_ID,
        media_path=tmp_path / "source.mp4",
    )

    first = generator.advance_incremental(checkpoint, observations, 70.0)

    assert first.checkpoint is not None
    assert [item.family_id.ordinal for item in first.closed_families] == [0]
    closed_candidate = first.closed_families[0].candidate
    assert closed_candidate is not None
    assert closed_candidate.end_seconds > 14.0
    assert first.checkpoint.next_family_ordinal == 1
    assert any(
        event.observation == observations[1]
        for event in first.checkpoint._open_events
    )

    generator.commit_incremental(checkpoint, first)
    second = generator.advance_incremental(first.checkpoint, (), 74.0)
    assert second.checkpoint is not None
    assert [item.family_id.ordinal for item in second.closed_families] == [1]
    assert [
        item.candidate
        for item in (*first.closed_families, *second.closed_families)
        if item.candidate is not None
    ] == batch
    generator.commit_incremental(first.checkpoint, second)


def test_below_threshold_family_advances_lineage(tmp_path: Path) -> None:
    generator = CandidateGenerator(
        CandidateGenerationConfig(minimum_candidate_confidence=0.9)
    )
    checkpoint = generator.start_incremental(
        source_id=SOURCE_ID,
        media_path=tmp_path / "source.mp4",
    )
    observations = [intensity(0.0, 0.4), speech(70.0), speech(71.0)]

    advance = generator.advance_incremental(checkpoint, observations, 130.0)

    assert advance.checkpoint is not None
    assert [item.family_id.ordinal for item in advance.closed_families] == [0, 1]
    assert advance.closed_families[0].candidate is None
    assert advance.closed_families[1].candidate is not None
    assert advance.checkpoint.next_family_ordinal == 2
    generator.commit_incremental(checkpoint, advance)


def test_long_evolving_cluster_preserves_non_window_evidence(tmp_path: Path) -> None:
    observations = [intensity(index * 0.5, 0.8) for index in range(110)]
    observations.extend([speech(40.0, 2.0, 0.99), speech(42.0, 2.0, 0.99)])
    duration = 100.0
    generator = CandidateGenerator()
    batch = generator.generate(timeline(tmp_path, observations, duration))
    checkpoint = generator.start_incremental(
        source_id=SOURCE_ID,
        media_path=tmp_path / "source.mp4",
    )

    early = generator.advance_incremental(checkpoint, observations[:80], 39.5)
    assert early.checkpoint is not None
    assert early.closed_families == ()
    assert early.checkpoint.retained_observation_count == 80
    generator.commit_incremental(checkpoint, early)
    late = generator.advance_incremental(early.checkpoint, observations[80:], 60.0)
    assert late.checkpoint is not None
    generator.commit_incremental(early.checkpoint, late)
    final = generator.finalize_incremental(late.checkpoint, (), duration)

    assert [
        item.candidate
        for item in (*late.closed_families, *final.closed_families)
        if item.candidate is not None
    ] == batch
    generator.commit_incremental(late.checkpoint, final)


def test_eof_closes_tail_with_authoritative_duration(tmp_path: Path) -> None:
    observations = [speech(94.0, duration=6.0), intensity(99.0)]
    generator = CandidateGenerator()
    batch = generator.generate(timeline(tmp_path, observations, 100.0))
    checkpoint = generator.start_incremental(
        source_id=SOURCE_ID,
        media_path=tmp_path / "source.mp4",
    )
    advance = generator.advance_incremental(checkpoint, observations[:1], 99.0)
    assert advance.checkpoint is not None
    assert advance.closed_families == ()
    generator.commit_incremental(checkpoint, advance)

    final = generator.finalize_incremental(
        advance.checkpoint, observations[1:], 100.0
    )

    assert final.checkpoint is None
    assert [
        item.candidate
        for item in final.closed_families
        if item.candidate is not None
    ] == batch
    assert final.closed_families[0].candidate is not None
    assert final.closed_families[0].candidate.end_seconds == 100.0
    generator.commit_incremental(advance.checkpoint, final)


def test_checkpoint_contract_rejects_invalid_resume_values(tmp_path: Path) -> None:
    generator = CandidateGenerator()
    checkpoint: CandidateGenerationCheckpoint = generator.start_incremental(
        source_id=SOURCE_ID,
        media_path=tmp_path / "source.mp4",
    )
    advanced = generator.advance_incremental(checkpoint, (), 10.0)
    assert advanced.checkpoint is not None
    generator.commit_incremental(checkpoint, advanced)

    with pytest.raises(ValueError, match="move backwards"):
        generator.advance_incremental(advanced.checkpoint, (), 9.0)
    with pytest.raises(ValueError, match="finite and non-negative"):
        generator.finalize_incremental(advanced.checkpoint, (), float("nan"))
    final = generator.finalize_incremental(advanced.checkpoint, (), 9.0)
    assert final.checkpoint is None
    generator.commit_incremental(advanced.checkpoint, final)


@pytest.mark.parametrize("ordinal", [True, -1, 1.5])
def test_family_lineage_requires_non_negative_integer_ordinal(ordinal) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        CandidateFamilyId(SOURCE_ID, ordinal)


def test_checkpoint_collections_are_runtime_immutable_contracts(tmp_path: Path) -> None:
    generator = CandidateGenerator()
    checkpoint = generator.start_incremental(
        source_id=SOURCE_ID,
        media_path=tmp_path / "source.mp4",
    )
    with pytest.raises(TypeError, match="immutable tuple"):
        CandidateGenerationCheckpoint(
            checkpoint._owner_token,
            checkpoint.source_id,
            checkpoint.media_path,
            checkpoint.stable_through_seconds,
            checkpoint.next_family_ordinal,
            [],  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="immutable tuple"):
        CandidateGenerationAdvance(checkpoint, [])  # type: ignore[arg-type]
