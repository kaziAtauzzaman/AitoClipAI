"""Configuration for candidate-local Editorial Strength v1 diagnostics."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EditorialStrengthConfig:
    """Stable v1 normalization constants and formula weights."""

    formula_version: str = "editorial_strength_v1"
    reaction_max_gap_seconds: float = 2.0
    audio_coverage_reference: float = 0.50
    minimum_audio_evidence_coverage: float = 0.20
    minimum_speech_tokens: int = 3
    reaction_rise_floor_db: float = 10.0
    reaction_rise_range_db: float = 30.0
    baseline_contrast_range_db: float = 20.0
    variability_floor_db: float = 4.0
    variability_range_db: float = 14.0
    sustained_intensity_discount: float = 0.75
    unique_token_floor: float = 0.02
    unique_token_range: float = 0.18
    compression_ratio_floor: float = 2.0
    compression_ratio_range: float = 18.0
    repeated_run_floor: float = 0.15
    repeated_run_range: float = 0.55
    muted_silence_floor: float = 0.60
    muted_silence_range: float = 0.30
    rise_weight: float = 0.35
    baseline_weight: float = 0.10
    variability_weight: float = 0.10
    transition_weight: float = 0.25
    speech_weight: float = 0.20
    muted_penalty_weight: float = 0.60
    routine_penalty_weight: float = 0.08
    low_information_penalty_weight: float = 0.08
