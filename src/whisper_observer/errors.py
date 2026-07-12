"""Whisper observer exceptions."""


class TranscriptionError(Exception):
    """Base error for expected transcription failures."""


class WhisperUnavailableError(TranscriptionError):
    """Raised when the optional Whisper runtime is unavailable."""


class InvalidTranscriptionError(TranscriptionError):
    """Raised when a backend returns an invalid transcription result."""
