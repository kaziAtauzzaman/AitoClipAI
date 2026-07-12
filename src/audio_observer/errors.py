"""Audio observer exceptions."""


class AudioObserverError(Exception):
    """Raised when audio extraction, loading, or analysis cannot continue."""


class FFmpegNotFoundError(AudioObserverError):
    """Raised when the configured FFmpeg executable is unavailable."""


class AudioExtractionError(AudioObserverError):
    """Raised when FFmpeg cannot produce the requested PCM WAV artifact."""
