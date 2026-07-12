"""Caption generation exceptions."""


class CaptionGenerationError(Exception):
    """Base error for expected caption generation failures."""


class InvalidCaptionSourceError(CaptionGenerationError):
    """Raised when timeline speech or candidate association is ambiguous."""


class InvalidCaptionTimingError(CaptionGenerationError):
    """Raised when source speech or candidate timing is invalid."""


class CaptionPersistenceError(CaptionGenerationError):
    """Raised when an SRT caption artifact cannot be persisted."""
