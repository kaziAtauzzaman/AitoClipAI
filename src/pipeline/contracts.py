"""Pipeline-local contracts for prerecorded runs and validation reports."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core import ClipCandidate, ClipScore, FeatureTimeline, RenderJob


@dataclass(frozen=True, slots=True)
class MediaStreamProbe:
    """Normalized FFprobe information for one media stream."""

    codec_type: str
    codec_name: str | None
    start_seconds: float | None
    duration_seconds: float | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MediaProbeResult:
    """Normalized FFprobe result for one source or rendered artifact."""

    path: Path
    format_name: str | None
    duration_seconds: float | None
    streams: list[MediaStreamProbe] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RenderedArtifactValidation:
    """Successful playback-oriented validation for one rendered clip."""

    path: Path
    size_bytes: int
    video_stream: MediaStreamProbe
    audio_stream: MediaStreamProbe
    duration_seconds: float
    checks: dict[str, bool] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PipelineValidationReport:
    """Deterministic machine-readable result of Pipeline Validation 0.1."""

    status: str
    source_type: str
    source_path: Path
    source_metadata: MediaProbeResult
    timeline_path: Path
    required_observers: list[str]
    observed_observers: list[str]
    observer_failures: list[dict[str, Any]]
    candidate_count: int
    score_count: int
    passing_score_count: int
    rendered_artifacts: list[RenderedArtifactValidation]
    checks: dict[str, bool] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PrerecordedPipelineResult:
    """Complete orchestration result without expanding shared core contracts."""

    feature_timeline: FeatureTimeline
    candidates: list[ClipCandidate]
    scores: list[ClipScore]
    selected_scores: list[ClipScore]
    render_jobs: list[RenderJob]
    validation_report: PipelineValidationReport
    report_path: Path
