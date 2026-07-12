"""Package-local contracts for candidate selection decisions."""

from dataclasses import dataclass, field

from core import ClipScore


@dataclass(frozen=True, slots=True)
class SuppressedCandidate:
    """A weaker score suppressed by a substantially overlapping stronger score."""

    score: ClipScore
    retained_score: ClipScore
    overlap_seconds: float
    overlap_ratio: float
    reason: str


@dataclass(frozen=True, slots=True)
class CandidateSelectionResult:
    """Selected render inputs and their deterministic suppression provenance."""

    selected: list[ClipScore] = field(default_factory=list)
    suppressed: list[SuppressedCandidate] = field(default_factory=list)
