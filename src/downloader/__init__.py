"""Downloader package for the AitoClipAI input stage."""

from downloader.config import DownloaderConfig
from core import DownloadResult
from downloader.downloader import VideoDownloader
from downloader.errors import DownloadError, DownloaderError, MetadataExtractionError
from downloader.metadata import MetadataExtractor, VideoMetadata

__all__ = [
    "DownloadError",
    "DownloadResult",
    "DownloaderConfig",
    "DownloaderError",
    "MetadataExtractionError",
    "MetadataExtractor",
    "VideoDownloader",
    "VideoMetadata",
]
