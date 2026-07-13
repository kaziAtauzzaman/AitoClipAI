"""Audio observer data contracts."""

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any

from core import Observation


@dataclass(frozen=True, slots=True)
class AudioSource:
    """Resolved audio artifact for analysis."""

    path: Path
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AudioData:
    """Loaded mono audio samples normalized to -1.0 through 1.0."""

    samples: tuple[float, ...]
    sample_rate_hz: int
    channels: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        """Total audio duration in seconds."""

        if self.sample_rate_hz <= 0:
            return 0.0
        return len(self.samples) / self.sample_rate_hz


@dataclass(frozen=True, slots=True)
class IncrementalAudioBatch:
    """New immutable observations and the observer-confirmed stable prefix.

    Observations are emitted once their own values can no longer change. The
    watermark may be earlier than an emitted observation's end when an open
    overlapping analysis window, silence segment, or peak competition remains.
    """

    observer: str
    observations: tuple[Observation, ...]
    watermark_seconds: float
    frames_processed: int
    eof: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.observer.strip():
            raise ValueError("Incremental audio batches require an observer name.")
        if (
            not math.isfinite(self.watermark_seconds)
            or self.watermark_seconds < 0
        ):
            raise ValueError("Audio watermark must be finite and non-negative.")
        if self.frames_processed < 0:
            raise ValueError("Processed audio frame count cannot be negative.")
