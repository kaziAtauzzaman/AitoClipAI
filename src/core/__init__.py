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

__all__ = [
    "AggregatedTimeline",
    "AggregatedFeatures",
    "AudioFeatures",
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
    "TimelineGroup",
    "UploadJob",
    "VisionFeatures",
]
