import json
from dataclasses import asdict
from pathlib import Path

import pytest

from core import ClipCandidate, ClipScore, Observation
from decision_engine import (
    EditorialStrengthError,
    EditorialStrengthEvaluator,
    InsufficientEditorialEvidenceError,
)
from decision_engine.editorial_strength import candidate_identity


def audio(at, kind, loudness, duration=1.0):
    value = {"loudness_dbfs": loudness}
    if kind == "speaking_intensity":
        value["intensity"] = 0.8
    return Observation(at, "audio", kind, value, duration_seconds=duration)


def speech(at, text, *, compression=2.0, duration=1.0):
    return Observation(
        at,
        "whisper",
        "speech",
        {"text": text},
        duration_seconds=duration,
        metadata={"compression_ratio": compression},
    )


def scored(observations, *, start=0.0, end=10.0, reason="candidate", path=None):
    candidate = ClipCandidate(
        path or Path("source.mp4"),
        start,
        end,
        reason,
        metadata={"contributing_observations": list(observations)},
    )
    return ClipScore(candidate, 0.6, {"base": 0.6}, "unchanged", True)


def evaluate(observations, **kwargs):
    return EditorialStrengthEvaluator().evaluate_one(
        scored(observations, **kwargs), "source-1"
    )


def test_all_v1_components_and_formula_are_explicit():
    result = evaluate(
        [
            audio(0.0, "silence", -60.0, 2.0),
            audio(2.5, "speaking_intensity", -20.0, 2.0),
            speech(3.0, "A complete useful sentence with clear setup and conclusion!"),
        ]
    )

    raw = result.raw_evidence
    components = result.normalized_components
    assert result.formula_version == "editorial_strength_v1"
    assert raw.silence_coverage == pytest.approx(0.2)
    assert raw.speaking_intensity_coverage == pytest.approx(0.2)
    assert raw.total_audio_coverage == pytest.approx(0.4)
    assert raw.audio_evidence_coverage == pytest.approx(0.8)
    assert raw.maximum_silence_to_reaction_rise_db == 40.0
    assert raw.local_loudness_baseline_dbfs == -40.0
    assert raw.local_loudness_p80_dbfs == -20.0
    assert raw.loudness_population_stddev_db == 20.0
    assert components.silence_to_reaction_rise == 1.0
    assert components.local_baseline_contrast == 1.0
    assert components.audio_variability == 1.0
    assert components.meaningful_transition == pytest.approx(0.68)
    assert components.speech_information_completeness == 1.0
    assert result.penalties.muted_or_extremely_silent == 0.0
    assert result.penalties.sustained_routine_intensity == pytest.approx(0.064)
    assert result.penalties.low_information_speech == 0.0
    assert result.editorial_score == pytest.approx(0.80488)


def test_sparse_audio_evidence_reduces_transition_reliability():
    sparse = evaluate(
        [
            audio(0.0, "silence", -60.0, 0.5),
            audio(0.5, "speaking_intensity", -20.0, 0.5),
        ]
    )
    covered = evaluate(
        [
            audio(0.0, "silence", -60.0, 2.5),
            audio(2.5, "speaking_intensity", -20.0, 2.5),
        ]
    )
    assert sparse.raw_evidence.audio_evidence_coverage == pytest.approx(0.2)
    assert covered.raw_evidence.audio_evidence_coverage == 1.0
    assert covered.editorial_score > sparse.editorial_score


def test_cross_type_audio_overlap_is_unioned_once_for_total_coverage():
    result = evaluate(
        [
            audio(0.0, "silence", -60.0, 3.0),
            audio(2.0, "speaking_intensity", -20.0, 3.0),
        ],
        end=20.0,
    )
    assert result.raw_evidence.silence_coverage == pytest.approx(0.15)
    assert result.raw_evidence.speaking_intensity_coverage == pytest.approx(0.15)
    assert result.raw_evidence.total_audio_coverage == pytest.approx(0.25)
    assert result.raw_evidence.audio_evidence_coverage == pytest.approx(0.5)


def test_insufficient_evidence_is_explicit_instead_of_scored():
    with pytest.raises(
        InsufficientEditorialEvidenceError, match="minimum retained"
    ):
        evaluate([audio(0.0, "silence", -60.0, 0.1)])


def test_sustained_routine_intensity_and_muting_are_penalized():
    routine = evaluate([audio(0.0, "speaking_intensity", -20.0, 8.0)])
    muted = evaluate([audio(0.0, "silence", -70.0, 9.0)])
    assert routine.penalties.sustained_routine_intensity == pytest.approx(0.8)
    assert muted.penalties.muted_or_extremely_silent == 1.0
    assert muted.editorial_score == 0.0


def test_quiet_setup_is_distinguished_from_muted_media():
    muted = evaluate([audio(0.0, "silence", -70.0, 9.0)])
    quiet_conversation = evaluate(
        [
            audio(0.0, "silence", -55.0, 7.0),
            speech(0.0, "A calm conversational introduction continues."),
        ]
    )
    quiet_reaction = evaluate(
        [
            audio(0.0, "silence", -60.0, 7.0),
            audio(7.1, "speaking_intensity", -20.0, 2.0),
            speech(7.1, "That reaction changes everything!"),
        ]
    )
    assert muted.penalties.muted_or_extremely_silent == 1.0
    assert quiet_conversation.penalties.muted_or_extremely_silent < 0.01
    assert quiet_reaction.penalties.muted_or_extremely_silent < 0.1
    assert quiet_reaction.editorial_score > muted.editorial_score


def test_complete_speech_beats_fragmented_repetitive_speech():
    complete = evaluate(
        [speech(0.0, "This has a clear setup and conclusion!", compression=2.0)]
    )
    repetitive = evaluate([speech(0.0, "go " * 100, compression=30.0)])
    assert complete.normalized_components.speech_information_completeness == 1.0
    assert repetitive.normalized_components.speech_information_completeness == 0.0
    assert complete.editorial_score > repetitive.editorial_score


@pytest.mark.parametrize(
    "item,match",
    [
        (audio(0.0, "silence", float("nan")), "finite"),
        (audio(0.0, "speaking_intensity", "loud"), "numeric"),
        (speech(0.0, "words", compression=float("inf")), "finite"),
        (speech(0.0, "words", compression=object()), "numeric"),
    ],
)
def test_invalid_observation_metadata_is_rejected(item, match):
    with pytest.raises(EditorialStrengthError, match=match):
        evaluate([item])


def test_candidate_identity_is_portable_semantic_and_strict():
    first = scored([speech(0.0, "Same.")], path=Path(r"C:\media\source.mp4"))
    second = scored([speech(0.0, "Same.")], path=Path("/media/source.mp4"))
    assert candidate_identity(first.candidate, "source") == candidate_identity(
        second.candidate, "source"
    )
    unsupported = scored([], reason="bad")
    unsupported.candidate.metadata["path"] = Path("machine-specific")
    with pytest.raises(EditorialStrengthError, match="paths"):
        candidate_identity(unsupported.candidate, "source")
    nonstring = scored([])
    nonstring.candidate.metadata[1] = "bad"
    with pytest.raises(EditorialStrengthError, match="string keys"):
        candidate_identity(nonstring.candidate, "source")


def test_results_serialize_deterministically_and_batch_incremental_match():
    values = [
        scored(
            [
                audio(0.0, "silence", -60.0),
                audio(1.0, "speaking_intensity", -20.0),
            ],
            reason="later",
        ),
        scored([speech(0.0, "A complete sentence!")], reason="earlier"),
    ]
    evaluator = EditorialStrengthEvaluator()
    batch = evaluator.evaluate(values, "source")
    incremental = [evaluator.evaluate_one(item, "source") for item in reversed(values)]
    incremental.sort(key=lambda item: item.candidate_identity)
    assert evaluator.candidate_local_deterministic is True
    assert evaluator.incremental_compatible is True
    assert {item.candidate_identity: item for item in batch} == {
        item.candidate_identity: item for item in incremental
    }
    first = json.dumps(
        [asdict(item) for item in batch], sort_keys=True, separators=(",", ":")
    )
    second = json.dumps(
        [asdict(item) for item in evaluator.evaluate(values, "source")],
        sort_keys=True,
        separators=(",", ":"),
    )
    assert first == second


def test_ranking_regressions_transition_candidates_beat_routine_candidates():
    clip_1 = evaluate(
        [
            audio(0.0, "silence", -55.0, 2.0),
            audio(2.2, "speaking_intensity", -28.0, 2.0),
            speech(2.0, "A complete arrival and reaction!"),
        ],
        reason="clip-1",
    )
    clip_14 = evaluate(
        [
            audio(0.0, "silence", -58.0, 2.0),
            audio(2.1, "speaking_intensity", -24.0, 2.0),
            speech(2.0, "reaction " * 30, compression=20.0),
        ],
        reason="clip-14",
    )
    clip_20 = evaluate(
        [
            audio(0.0, "silence", -56.0, 1.0),
            audio(1.1, "speaking_intensity", -27.0, 5.0),
            speech(1.0, "routine " * 30, compression=30.0),
        ],
        reason="clip-20",
    )
    clip_4 = evaluate([audio(0.0, "speaking_intensity", -25.0, 4.0)], reason="clip-4")
    clip_13 = evaluate(
        [
            audio(0.0, "silence", -50.0, 0.5),
            audio(0.6, "speaking_intensity", -28.0, 0.5),
        ],
        reason="clip-13",
    )
    clip_23 = evaluate(
        [
            audio(0.0, "speaking_intensity", -24.0, 8.0),
            speech(0.0, "you " * 100, compression=25.0),
        ],
        reason="clip-23",
    )

    assert clip_1.editorial_score > clip_4.editorial_score
    assert clip_14.editorial_score > clip_13.editorial_score
    assert clip_20.editorial_score > clip_23.editorial_score


def test_evaluation_does_not_mutate_base_scores_or_decisions():
    base = scored([speech(0.0, "A valid sentence!")])
    before = (base.overall_score, dict(base.score_components), base.passed_threshold)
    EditorialStrengthEvaluator().evaluate([base], "source")
    assert (base.overall_score, base.score_components, base.passed_threshold) == before
