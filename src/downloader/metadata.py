"""Metadata extraction and serialization helpers."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Protocol

from downloader.config import DownloaderConfig
from downloader.errors import MetadataExtractionError
from downloader.sanitization import (
    JsonMetadataSanitizer,
    JsonValue,
    MetadataSanitizer,
)


class MetadataClient(Protocol):
    """Protocol for clients that can extract media metadata."""

    def extract_info(self, url: str, *, download: bool) -> dict[str, Any]:
        """Return raw metadata for a URL."""


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """Normalized metadata for a downloadable video.

    Attributes:
        id: Platform-specific video identifier.
        title: Human-readable video title.
        uploader: Channel, creator, or account name when available.
        duration: Video duration in seconds when available.
        webpage_url: Canonical web URL from the provider.
        extractor: yt-dlp extractor name for the source platform.
        raw: Full yt-dlp metadata payload for future pipeline stages.
    """

    id: str
    title: str
    uploader: str | None
    duration: int | float | None
    webpage_url: str
    extractor: str
    raw: dict[str, Any]

    @classmethod
    def from_yt_dlp(cls, info: dict[str, Any]) -> "VideoMetadata":
        """Create normalized metadata from a yt-dlp information dictionary."""

        video_id = info.get("id")
        title = info.get("title")
        webpage_url = info.get("webpage_url") or info.get("original_url")
        extractor = info.get("extractor_key") or info.get("extractor")

        if not video_id or not title or not webpage_url or not extractor:
            raise MetadataExtractionError(
                "Metadata is missing required id, title, URL, or extractor fields."
            )

        return cls(
            id=str(video_id),
            title=str(title),
            uploader=_optional_str(info.get("uploader") or info.get("channel")),
            duration=_optional_number(info.get("duration")),
            webpage_url=str(webpage_url),
            extractor=str(extractor),
            raw=info,
        )

    def to_dict(
        self,
        sanitizer: MetadataSanitizer | None = None,
    ) -> dict[str, JsonValue]:
        """Return metadata as a JSON-serializable dictionary."""

        json_sanitizer = sanitizer or JsonMetadataSanitizer()
        return {
            "id": self.id,
            "title": self.title,
            "uploader": self.uploader,
            "duration": self.duration,
            "webpage_url": self.webpage_url,
            "extractor": self.extractor,
            "raw": json_sanitizer.sanitize(self.raw),
        }


class MetadataExtractor:
    """Service responsible only for metadata extraction."""

    def __init__(self, client: MetadataClient) -> None:
        """Initialize the extractor with a metadata-capable client."""

        self._client = client

    def extract(self, url: str) -> VideoMetadata:
        """Extract normalized metadata for a URL without downloading media."""

        info = self._client.extract_info(url, download=False)
        return VideoMetadata.from_yt_dlp(info)


class MetadataWriter:
    """Persist metadata beside downloaded videos."""

    def __init__(
        self,
        config: DownloaderConfig,
        sanitizer: MetadataSanitizer | None = None,
    ) -> None:
        """Initialize the writer with downloader configuration."""

        self._config = config
        self._sanitizer = sanitizer or JsonMetadataSanitizer()

    def metadata_path_for(self, video_path: Path) -> Path:
        """Return the JSON metadata path for a downloaded video path."""

        return video_path.with_name(f"{video_path.name}{self._config.metadata_suffix}")

    def write(self, video_path: Path, metadata: VideoMetadata) -> Path:
        """Write metadata JSON beside a downloaded video and return its path."""

        metadata_path = self.metadata_path_for(video_path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(
                metadata.to_dict(self._sanitizer),
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return metadata_path


def _optional_str(value: Any) -> str | None:
    """Return a string value or None for missing optional metadata."""

    if value is None:
        return None
    return str(value)


def _optional_number(value: Any) -> int | float | None:
    """Return a numeric value or None for missing optional metadata."""

    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return value
    return None
