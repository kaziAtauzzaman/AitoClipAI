"""Deterministic post-scoring candidate selection."""

from candidate_selection.config import CandidateSelectionConfig
from candidate_selection.contracts import (
    CandidateSelectionResult,
    SuppressedCandidate,
)
from candidate_selection.errors import CandidateSelectionError
from candidate_selection.selector import CandidateSelector

__all__ = [
    "CandidateSelectionConfig",
    "CandidateSelectionError",
    "CandidateSelectionResult",
    "CandidateSelector",
    "SuppressedCandidate",
]
