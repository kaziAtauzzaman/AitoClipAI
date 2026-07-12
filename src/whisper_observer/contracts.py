"""Normalized transcription backend contracts."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from whisper_observer.config import WhisperObserverConfig


@dataclass(frozen=True, slots=True)
class TranscriptionSegment:
    """One timestamped speech segment returned by a transcription backend."""

    start_seconds: float
    end_seconds: float
    text: str
    speaker: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """Normalized result returned by a transcription backend."""

    segments: list[TranscriptionSegment] = field(default_factory=list)
    text: str = ""
    language: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class TranscriptionBackend(Protocol):
    """Transcribe extracted audio into normalized timestamped segments."""

    def transcribe(
        self,
        audio_path: Path,
        config: WhisperObserverConfig,
    ) -> TranscriptionResult:
        """Transcribe one local WAV artifact."""
