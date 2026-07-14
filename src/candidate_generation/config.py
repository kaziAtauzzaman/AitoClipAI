"""Configurable candidate-generation heuristics and window behavior."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CandidateGenerationConfig:
    """Thresholds, weights, and clip-window settings for candidate generation."""

    speech_weight: float = 0.60
    loudness_weight: float = 0.30
    peak_weight: float = 0.25
    silence_weight: float = 0.25
    speaking_intensity_weight: float = 0.35
    minimum_candidate_confidence: float = 0.45
    minimum_speech_confidence: float = 0.0
    loudness_threshold_dbfs: float = -30.0
    peak_threshold: float = 0.85
    minimum_silence_seconds: float = 0.75
    silence_reference_seconds: float = 2.0
    speaking_intensity_threshold: float = 0.35
    merge_gap_seconds: float = 2.0
    pre_roll_seconds: float = 2.0
    post_roll_seconds: float = 3.0
    minimum_clip_seconds: float = 8.0
    maximum_clip_seconds: float = 60.0
    anchor_core_seconds: float = 30.0
    sustained_event_contribution: float = 0.80
