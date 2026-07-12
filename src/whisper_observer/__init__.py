"""Whisper speech observer package."""

from whisper_observer.backend import OpenAIWhisperBackend
from whisper_observer.config import WhisperObserverConfig
from whisper_observer.contracts import (
    TranscriptionBackend,
    TranscriptionResult,
    TranscriptionSegment,
)
from whisper_observer.errors import (
    InvalidTranscriptionError,
    TranscriptionError,
    WhisperUnavailableError,
)
from whisper_observer.observer import WhisperObserver

__all__ = [
    "InvalidTranscriptionError",
    "OpenAIWhisperBackend",
    "TranscriptionBackend",
    "TranscriptionError",
    "TranscriptionResult",
    "TranscriptionSegment",
    "WhisperObserver",
    "WhisperObserverConfig",
    "WhisperUnavailableError",
]
