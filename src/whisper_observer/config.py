"""Configurable Whisper transcription settings."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class WhisperObserverConfig:
    """Runtime configuration for Whisper transcription and observation."""

    observer_name: str = "whisper"
    order: int = 200
    model_name: str = "base"
    language: str | None = None
    task: str = "transcribe"
    device: str | None = None
    deterministic: bool = True
    options: dict[str, Any] = field(default_factory=dict)
