"""Configuration objects for downloader behavior."""

from dataclasses import dataclass
from pathlib import Path


DEFAULT_DOWNLOADS_DIR = Path("data") / "downloads"


@dataclass(frozen=True, slots=True)
class DownloaderConfig:
    """Runtime configuration for video downloads.

    Attributes:
        downloads_dir: Directory where downloaded source videos are stored.
        metadata_suffix: Suffix appended to a video path when writing metadata.
        filename_template: yt-dlp output template used for downloaded files.
        overwrite_existing: Whether yt-dlp may overwrite an existing media file.
    """

    downloads_dir: Path = DEFAULT_DOWNLOADS_DIR
    metadata_suffix: str = ".metadata.json"
    filename_template: str = "%(extractor_key)s/%(id)s/%(title).200B.%(ext)s"
    overwrite_existing: bool = False

    def output_template(self) -> str:
        """Return the full yt-dlp output template for this configuration."""

        return str(self.downloads_dir / self.filename_template)

    def ensure_directories(self) -> None:
        """Create required download directories if they do not already exist."""

        self.downloads_dir.mkdir(parents=True, exist_ok=True)
