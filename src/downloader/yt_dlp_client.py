"""Small adapter around yt-dlp.

Keeping yt-dlp behind this adapter makes it easier to add platform-specific
clients or replace implementation details without changing the downloader
or metadata extraction services.
"""

from typing import Any

from downloader.config import DownloaderConfig
from downloader.errors import DownloadError, MetadataExtractionError


class YtDlpClient:
    """Adapter that executes yt-dlp operations for a configured downloader."""

    def __init__(self, config: DownloaderConfig) -> None:
        """Initialize the client with downloader configuration."""

        self._config = config

    def extract_info(self, url: str, *, download: bool) -> dict[str, Any]:
        """Extract metadata and optionally download media for a URL.

        Args:
            url: Video or livestream archive URL supported by yt-dlp.
            download: When true, yt-dlp downloads the media while extracting.

        Returns:
            Raw yt-dlp information dictionary.

        Raises:
            MetadataExtractionError: If metadata extraction fails.
            DownloadError: If a requested download fails.
        """

        try:
            import yt_dlp
        except ImportError as exc:
            message = "yt-dlp is not installed. Install dependencies first."
            if download:
                raise DownloadError(message) from exc
            raise MetadataExtractionError(message) from exc

        options = self._build_options()

        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                result = ydl.extract_info(url, download=download)
        except yt_dlp.utils.DownloadError as exc:
            if download:
                raise DownloadError(f"Failed to download media: {exc}") from exc
            raise MetadataExtractionError(f"Failed to extract metadata: {exc}") from exc

        if not isinstance(result, dict):
            message = "yt-dlp returned an unexpected response type."
            if download:
                raise DownloadError(message)
            raise MetadataExtractionError(message)

        return result

    def _build_options(self) -> dict[str, Any]:
        """Build yt-dlp options from the downloader configuration."""

        return {
            "noplaylist": True,
            "outtmpl": self._config.output_template(),
            "overwrites": self._config.overwrite_existing,
            "quiet": True,
            "no_warnings": True,
        }
