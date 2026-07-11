"""Custom exceptions raised by the downloader package."""


class DownloaderError(Exception):
    """Base exception for expected downloader failures."""


class MetadataExtractionError(DownloaderError):
    """Raised when metadata cannot be extracted from a URL."""


class DownloadError(DownloaderError):
    """Raised when a video cannot be downloaded or persisted."""
