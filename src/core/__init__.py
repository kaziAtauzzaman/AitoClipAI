"""Shared data contracts for the AitoClipAI pipeline."""

from core.contracts import (
    AggregatedTimeline,
    AggregatedFeatures,
    AudioFeatures,
    ClipCandidate,
    ClipScore,
    DownloadResult,
    FeatureTimeline,
    FeatureTimelineFailure,
    OCRFeatures,
    Observation,
    ObserverResult,
    RenderJob,
    SpeechFeatures,
    TimelineGroup,
    UploadJob,
    VisionFeatures,
)
from core.selection_priority import (
    CANDIDATE_SCORE_DECIMAL_PLACES,
    DEFAULT_SELECTION_PRIORITY_CONTRACT,
    SelectionPriorityContract,
)

__all__ = [
    "AggregatedTimeline",
    "AggregatedFeatures",
    "AudioFeatures",
    "CANDIDATE_SCORE_DECIMAL_PLACES",
    "ClipCandidate",
    "ClipScore",
    "DownloadResult",
    "FeatureTimeline",
    "FeatureTimelineFailure",
    "OCRFeatures",
    "Observation",
    "ObserverResult",
    "RenderJob",
    "SpeechFeatures",
    "SelectionPriorityContract",
    "TimelineGroup",
    "UploadJob",
    "VisionFeatures",
    "DEFAULT_SELECTION_PRIORITY_CONTRACT",
]
