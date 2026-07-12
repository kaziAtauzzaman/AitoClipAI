"""Configuration for deterministic post-scoring candidate selection."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CandidateSelectionConfig:
    """Rules for suppressing substantially overlapping passing candidates."""

    overlap_ratio_threshold: float = 0.65
    minimum_overlap_seconds: float = 1.0
