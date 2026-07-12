"""Configuration for explainable candidate scoring."""

from dataclasses import dataclass, field


def default_weights() -> dict[str, float]:
    """Return independent default signal weights."""

    return {
        "speech_excitement": 0.30,
        "speaking_intensity": 0.20,
        "loudness_peaks": 0.18,
        "silence_buildup": 0.12,
        "supporting_observations": 0.10,
        "observation_diversity": 0.10,
    }


@dataclass(frozen=True, slots=True)
class CandidateScoringConfig:
    """Weights, normalizers, and threshold for candidate scoring."""

    weights: dict[str, float] = field(default_factory=default_weights)
    passing_score: float = 0.50
    supporting_observation_reference: int = 6
    observation_diversity_reference: int = 4
    silence_reference_seconds: float = 2.0
    speech_confidence_factor: float = 0.75
    exclamation_boost: float = 0.08
    question_boost: float = 0.04
    maximum_punctuation_boost: float = 0.20
    uppercase_boost: float = 0.05
