"""Deterministic caption generation and SRT persistence."""

from captioning.config import CaptionGeneratorConfig
from captioning.contracts import (
    CaptionArtifact,
    CaptionCue,
    CandidateCaptionIdentity,
    candidate_caption_identity,
)
from captioning.errors import (
    CaptionGenerationError,
    CaptionPersistenceError,
    InvalidCaptionSourceError,
    InvalidCaptionTimingError,
)
from captioning.generator import CaptionGenerator
from captioning.srt import (
    CaptionFormatter,
    CaptionWriter,
    FileCaptionWriter,
    SrtCaptionFormatter,
)

__all__ = [
    "CandidateCaptionIdentity",
    "CaptionArtifact",
    "CaptionCue",
    "CaptionFormatter",
    "CaptionGenerationError",
    "CaptionGenerator",
    "CaptionGeneratorConfig",
    "CaptionPersistenceError",
    "CaptionWriter",
    "FileCaptionWriter",
    "InvalidCaptionSourceError",
    "InvalidCaptionTimingError",
    "SrtCaptionFormatter",
    "candidate_caption_identity",
]
