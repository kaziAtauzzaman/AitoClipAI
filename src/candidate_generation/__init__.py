"""Deterministic feature-timeline candidate generation."""

from candidate_generation.config import CandidateGenerationConfig
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
    "CandidateGenerationConfig",
    "CandidateGenerationError",
    "CandidateGenerator",
    "CandidateHeuristic",
    "EventBoundaryRole",
    "SilenceBuildupHeuristic",
    "SpeakingIntensityHeuristic",
    "WhisperSpeechHeuristic",
]
