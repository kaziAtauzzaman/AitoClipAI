"""Deterministic feature-timeline candidate generation."""

from candidate_generation.config import CandidateGenerationConfig
from candidate_generation.contracts import (
    CandidateFamilyId,
    CandidateGenerationAdvance,
    CandidateGenerationCheckpoint,
    ClosedCandidateFamily,
    IncrementalCandidateGenerator,
)
from candidate_generation.errors import CandidateGenerationError
from candidate_generation.generator import CandidateGenerator
from candidate_generation.heuristics import (
    AudioLoudnessHeuristic,
    CandidateEvent,
    CandidateHeuristic,
    EventBoundaryRole,
    SilenceBuildupHeuristic,
    SpeakingIntensityHeuristic,
    WhisperSpeechHeuristic,
)

__all__ = [
    "AudioLoudnessHeuristic",
    "CandidateEvent",
    "CandidateFamilyId",
    "CandidateGenerationAdvance",
    "CandidateGenerationCheckpoint",
    "CandidateGenerationConfig",
    "CandidateGenerationError",
    "CandidateGenerator",
    "CandidateHeuristic",
    "ClosedCandidateFamily",
    "EventBoundaryRole",
    "IncrementalCandidateGenerator",
    "SilenceBuildupHeuristic",
    "SpeakingIntensityHeuristic",
    "WhisperSpeechHeuristic",
]
