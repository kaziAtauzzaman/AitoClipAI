"""Normalized transcription backend contracts."""

from dataclasses import dataclass, field
from enum import Enum
import math
from pathlib import Path
from typing import Any, Protocol

from core import Observation
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


@dataclass(frozen=True, slots=True)
class IncrementalWhisperAudioChunk:
    """Transport-neutral chronological PCM chunk with an explicit stable edge."""

    pcm_bytes: bytes
    sample_rate_hz: int
    channels: int
    sample_width_bytes: int
    start_frame: int
    end_frame: int
    stable_through_frame: int

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0 or self.channels <= 0:
            raise ValueError("Incremental Whisper PCM format must be positive.")
        if self.sample_width_bytes not in {1, 2, 4}:
            raise ValueError("Incremental Whisper PCM sample width is unsupported.")
        if self.start_frame < 0 or self.end_frame < self.start_frame:
            raise ValueError("Incremental Whisper chunk boundaries are invalid.")
        if not self.start_frame <= self.stable_through_frame <= self.end_frame:
            raise ValueError("Incremental Whisper stable edge must be inside its chunk.")
        expected_bytes = (
            (self.end_frame - self.start_frame)
            * self.channels
            * self.sample_width_bytes
        )
        if len(self.pcm_bytes) != expected_bytes:
            raise ValueError("Incremental Whisper PCM length does not match its frames.")


@dataclass(frozen=True, slots=True)
class IncrementalWhisperEOF:
    """Authoritative final PCM frontier for an incremental Whisper source."""

    final_frame: int
    sample_rate_hz: int

    def __post_init__(self) -> None:
        if self.final_frame < 0 or self.sample_rate_hz <= 0:
            raise ValueError("Incremental Whisper EOF values are invalid.")


class IncrementalWhisperLifecycle(str, Enum):
    NEW = "new"
    ACTIVE = "active"
    FLUSHED = "flushed"
    CLOSED = "closed"


class SegmentReconciliationPolicy(Protocol):
    """Deterministically identify and select overlap segment revisions."""

    def reconcile(
        self,
        existing: TranscriptionSegment,
        candidate: TranscriptionSegment,
        *,
        timestamp_tolerance_seconds: float,
        similarity_threshold: float,
    ) -> TranscriptionSegment | None:
        """Return the selected representation, or None when segments differ."""


class TranscriptionBackend(Protocol):
    """Transcribe extracted audio into normalized timestamped segments."""

    def transcribe(
        self,
        audio_path: Path,
        config: WhisperObserverConfig,
    ) -> TranscriptionResult:
        """Transcribe one local WAV artifact."""


class IncrementalTranscriptionSession(Protocol):
    """One loaded model reused for every chunk in a source session."""

    def transcribe(
        self,
        audio_path: Path,
        initial_prompt: str | None,
    ) -> TranscriptionResult:
        """Transcribe one chronological WAV chunk."""

    def close(self) -> None:
        """Release session-specific model resources when applicable."""


class IncrementalTranscriptionBackend(Protocol):
    """Load exactly one reusable model handle for an incremental session."""

    def open_incremental_session(
        self,
        config: WhisperObserverConfig,
    ) -> IncrementalTranscriptionSession:
        """Return a loaded model session for one source."""


@dataclass(frozen=True, slots=True)
class IncrementalWhisperBatch:
    """New stable speech observations and the confirmed Whisper watermark."""

    observer: str
    observations: tuple[Observation, ...]
    watermark_seconds: float
    frames_processed: int
    eof: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.observer.strip():
            raise ValueError("Incremental Whisper batches require an observer name.")
        if not math.isfinite(self.watermark_seconds) or self.watermark_seconds < 0:
            raise ValueError("Whisper watermark must be finite and non-negative.")
        if self.frames_processed < 0:
            raise ValueError("Processed Whisper frame count cannot be negative.")
