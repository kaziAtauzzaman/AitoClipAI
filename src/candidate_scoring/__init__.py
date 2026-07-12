"""Explainable deterministic clip-candidate scoring."""

from candidate_scoring.config import CandidateScoringConfig, default_weights
from candidate_scoring.errors import CandidateScoringError
from candidate_scoring.heuristics import (
    ComponentScore,
    LoudnessPeakScoreHeuristic,
    ObservationDiversityScoreHeuristic,
    ScoringHeuristic,
    SilenceBuildupScoreHeuristic,
    SpeakingIntensityScoreHeuristic,
    SpeechExcitementHeuristic,
    SupportingObservationScoreHeuristic,
)
from candidate_scoring.scorer import CandidateScorer

__all__ = [
    "CandidateScorer",
    "CandidateScoringConfig",
    "CandidateScoringError",
    "ComponentScore",
    "LoudnessPeakScoreHeuristic",
    "ObservationDiversityScoreHeuristic",
    "ScoringHeuristic",
    "SilenceBuildupScoreHeuristic",
    "SpeakingIntensityScoreHeuristic",
    "SpeechExcitementHeuristic",
    "SupportingObservationScoreHeuristic",
    "default_weights",
]
