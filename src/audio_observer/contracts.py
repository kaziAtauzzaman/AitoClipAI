"""Audio observer data contracts."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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
