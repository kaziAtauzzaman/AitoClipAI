"""Clip rendering exceptions."""


class ClipRenderingError(Exception):
    """Base error for expected clip rendering failures."""


class RenderingFFmpegNotFoundError(ClipRenderingError):
    """Raised when the configured FFmpeg executable cannot be found."""


class InvalidRenderInputError(ClipRenderingError):
    """Raised when a score or candidate cannot be rendered safely."""


class SubtitleRenderingError(ClipRenderingError):
    """Raised when caption association or FFmpeg subtitle rendering fails."""
