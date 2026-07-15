"""Normalized transcription backend contracts."""

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
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
    """New stable speech observations and the confirmed Whisper watermark.

    Every emitted speech observation has left provisional reconciliation state.
    Its semantic identity is declared in batch metadata even when another
    provisional segment holds the global watermark behind it.
    """

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


def finalized_speech_segment_identity(observation: Observation) -> str:
    """Return a portable semantic identity for one finalized speech observation."""

    if observation.observer != "whisper" or observation.type != "speech":
        raise ValueError("Finalized speech identities require Whisper speech.")
    payload = {
        "observer": observation.observer,
        "type": observation.type,
        "timestamp_seconds": observation.timestamp_seconds,
        "duration_seconds": observation.duration_seconds,
        "value": observation.value,
        "confidence": observation.confidence,
        "metadata": observation.metadata,
    }
    encoded = json.dumps(
        _speech_identity_value(payload), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _speech_identity_value(value: object) -> object:
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Finalized speech metadata requires string keys.")
        return {key: _speech_identity_value(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_speech_identity_value(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Finalized speech identity values must be finite.")
        return {"$float": "0" if value == 0 else format(value, ".17g")}
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(
        f"Unsupported finalized speech identity value: {type(value).__name__}."
    )
