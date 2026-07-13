"""Configurable Whisper transcription settings."""

from dataclasses import dataclass, field
import math
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


@dataclass(frozen=True, slots=True)
class IncrementalWhisperObserverConfig:
    """Chunking and bounded-context policy for incremental transcription."""

    chunk_seconds: float = 30.0
    overlap_seconds: float = 5.0
    deduplication_tolerance_seconds: float = 0.75
    reconciliation_similarity_threshold: float = 0.70
    prompt_max_characters: int = 1_000
    analysis: WhisperObserverConfig = WhisperObserverConfig()

    def __post_init__(self) -> None:
        values = (
            self.chunk_seconds,
            self.overlap_seconds,
            self.deduplication_tolerance_seconds,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError(
                "Incremental Whisper durations must be finite and non-negative."
            )
        if self.chunk_seconds <= 0:
            raise ValueError("Incremental Whisper chunk duration must be positive.")
        if self.overlap_seconds >= self.chunk_seconds:
            raise ValueError(
                "Incremental Whisper overlap must be shorter than its chunk."
            )
        if self.prompt_max_characters < 0:
            raise ValueError("Incremental Whisper prompt limit cannot be negative.")
        if not 0.0 <= self.reconciliation_similarity_threshold <= 1.0:
            raise ValueError(
                "Incremental Whisper reconciliation similarity must be between 0 and 1."
            )
