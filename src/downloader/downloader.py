"""Download orchestration for source videos."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from downloader.config import DownloaderConfig
from downloader.errors import DownloadError
from downloader.metadata import MetadataWriter, VideoMetadata
from downloader.yt_dlp_client import YtDlpClient


class DownloadClient(Protocol):
    """Protocol for clients that can download media and return metadata."""

    def extract_info(self, url: str, *, download: bool) -> dict[str, Any]:
        """Download media when requested and return raw metadata."""


@dataclass(frozen=True, slots=True)
class DownloadResult:
    """Result returned after a successful source video download.

    Attributes:
        video_path: Path to the downloaded video file.
        metadata_path: Path to the JSON metadata file beside the video.
        metadata: Normalized metadata for the downloaded video.
    """

    video_path: Path
    metadata_path: Path
    metadata: VideoMetadata


class VideoDownloader:
    """Coordinates video download and metadata persistence."""

    def __init__(
        self,
        config: DownloaderConfig | None = None,
        client: DownloadClient | None = None,
        metadata_writer: MetadataWriter | None = None,
    ) -> None:
        """Initialize the downloader with injectable collaborators."""

        self._config = config or DownloaderConfig()
        self._client = client or YtDlpClient(self._config)
        self._metadata_writer = metadata_writer or MetadataWriter(self._config)

    def download(self, url: str) -> DownloadResult:
        """Download a video URL and write metadata beside the media file.

        Args:
            url: Video URL supported by the configured client.

        Returns:
            DownloadResult containing video path, metadata path, and metadata.

        Raises:
            DownloadError: If the URL is empty, media path cannot be resolved,
                metadata cannot be normalized, or JSON metadata cannot be saved.
        """

        if not url.strip():
            raise DownloadError("A non-empty URL is required.")

        self._config.ensure_directories()
        info = self._client.extract_info(url, download=True)
        metadata = VideoMetadata.from_yt_dlp(info)
        video_path = self._resolve_downloaded_path(info)

        try:
            metadata_path = self._metadata_writer.write(video_path, metadata)
        except OSError as exc:
            raise DownloadError(f"Failed to write metadata JSON: {exc}") from exc

        return DownloadResult(
            video_path=video_path,
            metadata_path=metadata_path,
            metadata=metadata,
        )

    def _resolve_downloaded_path(self, info: dict[str, Any]) -> Path:
        """Resolve the downloaded media path from a yt-dlp information payload."""

        direct_path = info.get("filepath") or info.get("_filename")
        if direct_path:
            return Path(str(direct_path))

        requested_downloads = info.get("requested_downloads")
        if isinstance(requested_downloads, list):
            for item in requested_downloads:
                if isinstance(item, dict):
                    item_path = item.get("filepath") or item.get("_filename")
                    if item_path:
                        return Path(str(item_path))

        raise DownloadError("Unable to resolve downloaded video path from yt-dlp.")
