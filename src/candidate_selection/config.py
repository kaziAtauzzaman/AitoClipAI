"""Configuration for deterministic post-scoring candidate selection."""

from dataclasses import dataclass, field

from core import SelectionPriorityContract


@dataclass(frozen=True, slots=True)
class CandidateSelectionConfig:
    """Rules for suppressing substantially overlapping passing candidates."""

    overlap_ratio_threshold: float = 0.65
    minimum_overlap_seconds: float = 1.0
    selection_priority: SelectionPriorityContract = field(
        default_factory=SelectionPriorityContract
    )
