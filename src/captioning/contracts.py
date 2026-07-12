"""Captioning-package contracts for generated subtitle artifacts."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core import ClipCandidate


@dataclass(frozen=True, slots=True)
class CandidateCaptionIdentity:
    """Deterministic source and time identity used to associate captions."""

    source_path: str
    start_microseconds: int
    end_microseconds: int


@dataclass(frozen=True, slots=True)
class CaptionCue:
    """One clip-relative subtitle cue derived from a speech observation."""

    index: int
    start_seconds: float
    end_seconds: float
    text: str
    speaker: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CaptionArtifact:
    """Persisted SRT track associated with one clip candidate."""

    candidate: ClipCandidate
    path: Path
    cues: list[CaptionCue] = field(default_factory=list)
    language: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def identity(self) -> CandidateCaptionIdentity:
        return candidate_caption_identity(self.candidate)


def candidate_caption_identity(candidate: ClipCandidate) -> CandidateCaptionIdentity:
    """Build a stable caption association key without hashing mutable metadata."""

    return CandidateCaptionIdentity(
        source_path=str(Path(candidate.source_video_path).resolve()),
        start_microseconds=round(candidate.start_seconds * 1_000_000),
        end_microseconds=round(candidate.end_seconds * 1_000_000),
    )
