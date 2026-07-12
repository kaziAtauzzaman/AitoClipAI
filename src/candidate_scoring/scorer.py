"""Deterministic explainable candidate scoring and ranking."""

from typing import Iterable

from candidate_scoring.config import CandidateScoringConfig
from candidate_scoring.errors import CandidateScoringError
from candidate_scoring.heuristics import (
    ScoringHeuristic,
    candidate_observations,
    default_heuristics,
)
from core import ClipCandidate, ClipScore


class CandidateScorer:
    """Score and rank clip candidates using weighted explainable heuristics."""

    def __init__(
        self,
        config: CandidateScoringConfig | None = None,
        heuristics: Iterable[ScoringHeuristic] | None = None,
    ) -> None:
        self._config = config or CandidateScoringConfig()
        self._heuristics = list(
            default_heuristics() if heuristics is None else heuristics
        )
        self._validate_config()

    def score(self, candidates: Iterable[ClipCandidate]) -> list[ClipScore]:
        """Return candidates sorted from highest to lowest overall score."""

        scored = [self._score_candidate(candidate) for candidate in candidates]
        return sorted(
            scored,
            key=lambda result: (
                -result.overall_score,
                result.candidate.start_seconds,
                result.candidate.end_seconds,
                result.candidate.reason,
                str(result.candidate.source_video_path),
            ),
        )

    def _score_candidate(self, candidate: ClipCandidate) -> ClipScore:
        observations = candidate_observations(candidate)
        total_weight = sum(
            self._config.weights[heuristic.name] for heuristic in self._heuristics
        )
        components: dict[str, float] = {}
        explanations: list[str] = []

        for heuristic in self._heuristics:
            result = heuristic.score(candidate, observations, self._config)
            if not 0.0 <= result.value <= 1.0:
                raise CandidateScoringError(
                    f"Heuristic {heuristic.name!r} returned a score outside 0..1."
                )
            weight = self._config.weights[heuristic.name]
            contribution = result.value * weight / total_weight
            components[heuristic.name] = round(contribution, 6)
            explanations.append(
                f"{heuristic.name.replace('_', ' ')}: {result.value:.3f} raw "
                f"x {weight:.3f} weight / {total_weight:.3f} total = "
                f"{contribution:.3f} ({result.detail})"
            )

        overall = round(sum(components.values()), 6)
        rationale = f"Overall {overall:.3f}. " + "; ".join(explanations) + "."
        return ClipScore(
            candidate=candidate,
            overall_score=overall,
            score_components=components,
            rationale=rationale,
            passed_threshold=overall >= self._config.passing_score,
        )

    def _validate_config(self) -> None:
        config = self._config
        if not 0.0 <= config.passing_score <= 1.0:
            raise CandidateScoringError("Passing score must be between 0 and 1.")
        if config.supporting_observation_reference <= 0:
            raise CandidateScoringError(
                "Supporting observation reference must be positive."
            )
        if config.observation_diversity_reference <= 0:
            raise CandidateScoringError(
                "Observation diversity reference must be positive."
            )
        if config.silence_reference_seconds <= 0:
            raise CandidateScoringError("Silence reference must be positive.")
        missing = [
            heuristic.name
            for heuristic in self._heuristics
            if heuristic.name not in config.weights
        ]
        if missing:
            raise CandidateScoringError(
                f"Missing configured weights for: {', '.join(missing)}."
            )
        active_weights = [config.weights[item.name] for item in self._heuristics]
        if any(weight < 0 for weight in active_weights):
            raise CandidateScoringError("Scoring weights cannot be negative.")
        if not active_weights or sum(active_weights) <= 0:
            raise CandidateScoringError("At least one positive scoring weight is required.")
