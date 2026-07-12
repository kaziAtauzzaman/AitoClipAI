"""Package-local contracts for explainable heuristic feedback."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class CandidateIdentity:
    """Stable identity for a candidate window in one source."""

    resolved_source_path: Path
    start_microseconds: int
    end_microseconds: int


@dataclass(frozen=True, slots=True)
class ScoreContribution:
    """One authoritative weighted contribution emitted by the scorer."""

    signal: str
    contribution: float


@dataclass(frozen=True, slots=True)
class ObserverEvidence:
    """One timeline observation overlapping a scored candidate."""

    timestamp_seconds: float
    duration_seconds: float | None
    observer: str
    type: str
    value: Any
    confidence: float | None
    metadata: dict[str, Any] = field(default_factory=dict)
    direct_candidate_contributor: bool = False


@dataclass(frozen=True, slots=True)
class RenderFeedback:
    """Optional rendered artifact and playback-validation provenance."""

    output_path: Path
    rank: Any
    duration_seconds: float
    size_bytes: int
    video_codec: str | None
    audio_codec: str | None
    checks: dict[str, bool] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ClipFeedback:
    """Trace one scored candidate from observations through optional rendering."""

    identity: CandidateIdentity
    selection_status: str
    candidate_reason: str
    candidate_confidence: Any
    source_signals: list[str]
    candidate_signal_contributions: list[Any]
    overall_score: float
    passed_threshold: bool | None
    score_contributions: list[ScoreContribution]
    scorer_rationale: str | None
    supporting_evidence: list[ObserverEvidence]
    render: RenderFeedback | None = None


@dataclass(frozen=True, slots=True)
class ExplainableFeedbackReport:
    """Aggregate deterministic provenance report for every scored candidate."""

    schema_version: str
    report_type: str
    source_path: Path
    timeline_path: Path
    scored_candidate_count: int
    rendered_clip_count: int
    clips: list[ClipFeedback]
