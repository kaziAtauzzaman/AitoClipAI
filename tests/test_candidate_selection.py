from pathlib import Path

import pytest

from candidate_selection import (
    CandidateSelectionConfig,
    CandidateSelectionError,
    CandidateSelector,
)
from core import ClipCandidate, ClipScore, SelectionPriorityContract


def score(
    source: Path,
    start: float,
    end: float,
    value: float,
    reason: str = "candidate",
) -> ClipScore:
    return ClipScore(
        candidate=ClipCandidate(source, start, end, reason),
        overall_score=value,
        passed_threshold=True,
    )


def test_selection_priority_contract_is_the_exact_scorer_alphabet() -> None:
    contract = SelectionPriorityContract()

    assert contract.score_decimal_places == 6
    assert contract.scale == 1_000_000
    assert contract.rank_count == 1_000_001
    assert contract.maximum_strictly_improving_chain_length == 1_000_001
    assert contract.normalize(0.0) == 0
    assert contract.normalize(0.123456) == 123_456
    assert contract.normalize(1.0) == 1_000_000
    assert contract.identity == (
        "selection-priority-v1",
        6,
        0,
        1_000_000,
        "decimal-half-even",
        "stable-input-order",
    )


def test_selection_priority_rejects_precision_outside_scoring_semantics() -> None:
    with pytest.raises(ValueError, match="between zero"):
        SelectionPriorityContract(score_decimal_places=7)
    with pytest.raises(ValueError, match="between zero"):
        SelectionPriorityContract(score_decimal_places=-1)
    with pytest.raises(TypeError, match="integer"):
        SelectionPriorityContract(score_decimal_places=True)


def test_reduced_priority_alphabet_preserves_multiplicity_and_stable_ties(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    priority = SelectionPriorityContract(score_decimal_places=1)
    first = score(source, 0.0, 1.0, 0.81, "first")
    second = score(source, 2.0, 3.0, 0.84, "second")
    duplicate = score(source, 4.0, 5.0, 0.84, "duplicate")

    result = CandidateSelector(
        CandidateSelectionConfig(selection_priority=priority)
    ).select([first, second, duplicate])

    assert result.selected == [first, second, duplicate]
    assert result.suppressed == []


def test_selector_suppresses_weaker_substantial_overlap(tmp_path: Path) -> None:
    stronger = score(tmp_path / "source.mp4", 10.0, 20.0, 0.9, "stronger")
    weaker = score(tmp_path / "source.mp4", 12.0, 18.0, 0.7, "weaker")

    result = CandidateSelector().select([weaker, stronger])

    assert result.selected == [stronger]
    assert len(result.suppressed) == 1
    suppression = result.suppressed[0]
    assert suppression.score is weaker
    assert suppression.retained_score is stronger
    assert suppression.overlap_seconds == 6.0
    assert suppression.overlap_ratio == 1.0
    assert "stronger candidate" in suppression.reason


def test_selector_preserves_non_overlapping_and_minor_overlaps(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    strongest = score(source, 0.0, 10.0, 0.9)
    minor_overlap = score(source, 8.0, 18.0, 0.8)
    adjacent = score(source, 18.0, 25.0, 0.7)

    result = CandidateSelector().select([adjacent, minor_overlap, strongest])

    assert result.selected == [strongest, minor_overlap, adjacent]
    assert result.suppressed == []


def test_selector_does_not_suppress_windows_from_different_sources(
    tmp_path: Path,
) -> None:
    first = score(tmp_path / "first.mp4", 0.0, 10.0, 0.9)
    second = score(tmp_path / "second.mp4", 0.0, 10.0, 0.8)

    result = CandidateSelector().select([second, first])

    assert result.selected == [first, second]
    assert result.suppressed == []


def test_selector_uses_shorter_window_overlap_ratio(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    long = score(source, 0.0, 20.0, 0.9)
    partly_contained = score(source, 10.0, 20.0, 0.8)

    result = CandidateSelector().select([partly_contained, long])

    assert result.selected == [long]
    assert result.suppressed[0].overlap_ratio == 1.0


def test_selector_honors_exact_configured_boundaries(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    stronger = score(source, 0.0, 10.0, 0.9)
    exact_ratio = score(source, 3.5, 13.5, 0.8)
    exact_minimum = score(source, 9.0, 11.0, 0.7)
    selector = CandidateSelector(
        CandidateSelectionConfig(
            overlap_ratio_threshold=0.65,
            minimum_overlap_seconds=1.0,
        )
    )

    result = selector.select([exact_minimum, exact_ratio, stronger])

    assert result.selected == [stronger, exact_minimum]
    assert result.suppressed[0].score is exact_ratio
    assert result.suppressed[0].overlap_seconds == 6.5
    assert result.suppressed[0].overlap_ratio == 0.65


def test_selector_preserves_equal_rank_input_order(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    earlier = score(source, 0.0, 10.0, 0.8, "earlier")
    later = score(source, 1.0, 9.0, 0.8, "later")

    first = CandidateSelector().select([later, earlier])
    second = CandidateSelector().select([earlier, later])

    assert first.selected == [later]
    assert first.suppressed[0].score is earlier
    assert second.selected == [earlier]
    assert second.suppressed[0].score is later


def test_selector_does_not_mutate_scores_or_input(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    stronger = score(source, 0.0, 10.0, 0.9)
    weaker = score(source, 1.0, 9.0, 0.8)
    scores = [weaker, stronger]
    original = list(scores)

    CandidateSelector().select(scores)

    assert scores == original
    assert stronger.overall_score == 0.9
    assert weaker.passed_threshold is True


@pytest.mark.parametrize(
    "config",
    [
        CandidateSelectionConfig(overlap_ratio_threshold=0.0),
        CandidateSelectionConfig(overlap_ratio_threshold=-0.1),
        CandidateSelectionConfig(overlap_ratio_threshold=1.1),
        CandidateSelectionConfig(minimum_overlap_seconds=0.0),
        CandidateSelectionConfig(minimum_overlap_seconds=-0.1),
    ],
)
def test_selector_rejects_invalid_configuration(
    config: CandidateSelectionConfig,
) -> None:
    with pytest.raises(CandidateSelectionError):
        CandidateSelector(config)


@pytest.mark.parametrize(
    ("start", "end"),
    [(-1.0, 2.0), (2.0, 2.0), (3.0, 2.0)],
)
def test_selector_rejects_invalid_candidate_windows(
    tmp_path: Path,
    start: float,
    end: float,
) -> None:
    invalid = score(tmp_path / "source.mp4", start, end, 0.8)

    with pytest.raises(CandidateSelectionError):
        CandidateSelector().select([invalid])
