"""Deterministic suppression of substantially overlapping scored candidates."""

from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

from candidate_selection.config import CandidateSelectionConfig
from candidate_selection.contracts import (
    CandidateSelectionResult,
    SuppressedCandidate,
)
from candidate_selection.errors import CandidateSelectionError
from core import ClipCandidate, ClipScore, SelectionPriorityContract


class CandidateSelector:
    """Keep strongest scores while suppressing weaker substantial overlaps."""

    def __init__(self, config: CandidateSelectionConfig | None = None) -> None:
        self._config = config or CandidateSelectionConfig()
        self._validate_config()

    @property
    def selection_priority_contract(self) -> SelectionPriorityContract:
        """Return the finite priority alphabet used by greedy selection."""

        return self._config.selection_priority

    def select(self, scores: Iterable[ClipScore]) -> CandidateSelectionResult:
        """Return deterministic render selections without changing input scores."""

        try:
            ranked = sorted(
                scores,
                key=lambda score: self._config.selection_priority.ordering_key(
                    score.overall_score
                ),
            )
        except ValueError as exc:
            raise CandidateSelectionError(str(exc)) from exc
        selected: list[ClipScore] = []
        suppressed: list[SuppressedCandidate] = []
        for score in ranked:
            self._validate_candidate(score.candidate)
            retained = self._substantial_overlap(score, selected)
            if retained is None:
                selected.append(score)
                continue
            retained_score, overlap_seconds, overlap_ratio = retained
            suppressed.append(
                SuppressedCandidate(
                    score=score,
                    retained_score=retained_score,
                    overlap_seconds=overlap_seconds,
                    overlap_ratio=overlap_ratio,
                    reason=(
                        "Suppressed because a stronger candidate overlaps "
                        f"{overlap_seconds:.3f}s ({overlap_ratio:.3f} of the "
                        "shorter window)."
                    ),
                )
            )
        return CandidateSelectionResult(selected=selected, suppressed=suppressed)

    def competes(self, first: ClipCandidate, second: ClipCandidate) -> bool:
        """Return whether two candidates participate in the same suppression decision."""

        overlap_seconds, overlap_ratio = _overlap(first, second)
        return (
            overlap_seconds >= self._config.minimum_overlap_seconds
            and overlap_ratio >= self._config.overlap_ratio_threshold
        )

    def _substantial_overlap(
        self,
        score: ClipScore,
        selected: list[ClipScore],
    ) -> tuple[ClipScore, float, float] | None:
        for retained in selected:
            overlap_seconds, overlap_ratio = _overlap(
                score.candidate,
                retained.candidate,
            )
            if (
                overlap_seconds >= self._config.minimum_overlap_seconds
                and overlap_ratio >= self._config.overlap_ratio_threshold
            ):
                return retained, overlap_seconds, overlap_ratio
        return None

    def _validate_config(self) -> None:
        config = self._config
        if not isinstance(config.selection_priority, SelectionPriorityContract):
            raise CandidateSelectionError(
                "Selection priority must be a SelectionPriorityContract."
            )
        if not 0.0 < config.overlap_ratio_threshold <= 1.0:
            raise CandidateSelectionError(
                "Overlap ratio threshold must be greater than 0 and at most 1."
            )
        if config.minimum_overlap_seconds <= 0:
            raise CandidateSelectionError(
                "Minimum overlap seconds must be positive."
            )

    @staticmethod
    def _validate_candidate(candidate: ClipCandidate) -> None:
        if candidate.start_seconds < 0:
            raise CandidateSelectionError("Candidate start time cannot be negative.")
        if candidate.end_seconds <= candidate.start_seconds:
            raise CandidateSelectionError(
                "Candidate end time must be after its start time."
            )


def _overlap(first: ClipCandidate, second: ClipCandidate) -> tuple[float, float]:
    if _resolved_source_path(first.source_video_path) != _resolved_source_path(
        second.source_video_path
    ):
        return 0.0, 0.0
    overlap_seconds = max(
        0.0,
        min(first.end_seconds, second.end_seconds)
        - max(first.start_seconds, second.start_seconds),
    )
    shorter_duration = min(
        first.end_seconds - first.start_seconds,
        second.end_seconds - second.start_seconds,
    )
    return overlap_seconds, overlap_seconds / shorter_duration


@lru_cache(maxsize=256)
def _resolved_source_path(path: Path) -> Path:
    """Resolve a repeated source identity once for overlap-heavy streams."""

    return path.resolve(strict=False)
