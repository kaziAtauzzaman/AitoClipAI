from pathlib import Path
from typing import Any

import pytest

from downloader.config import DownloaderConfig
from downloader.downloader import VideoDownloader
from downloader.errors import DownloadError, MetadataExtractionError
from downloader.metadata import MetadataExtractor, MetadataWriter, VideoMetadata


class FakeClient:
    def __init__(self, info: dict[str, Any]) -> None:
        self.info = info
        self.calls: list[tuple[str, bool]] = []

    def extract_info(self, url: str, *, download: bool) -> dict[str, Any]:
        self.calls.append((url, download))
        return self.info


def metadata_payload(video_path: Path | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": "abc123",
        "title": "Example Video",
        "uploader": "Example Channel",
        "duration": 42,
        "webpage_url": "https://example.test/watch/abc123",
        "extractor_key": "Example",
    }
    if video_path is not None:
        payload["requested_downloads"] = [{"filepath": str(video_path)}]
    return payload


def test_metadata_extractor_normalizes_yt_dlp_payload() -> None:
    client = FakeClient(metadata_payload())
    extractor = MetadataExtractor(client)

    metadata = extractor.extract("https://example.test/watch/abc123")

    assert metadata.id == "abc123"
    assert metadata.title == "Example Video"
    assert metadata.uploader == "Example Channel"
    assert metadata.duration == 42
    assert metadata.extractor == "Example"
    assert client.calls == [("https://example.test/watch/abc123", False)]


def test_video_downloader_writes_metadata_beside_download(tmp_path: Path) -> None:
    video_path = tmp_path / "downloads" / "Example" / "abc123.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.write_text("media placeholder", encoding="utf-8")

    config = DownloaderConfig(downloads_dir=tmp_path / "downloads")
    client = FakeClient(metadata_payload(video_path))
    downloader = VideoDownloader(config=config, client=client)

    result = downloader.download("https://example.test/watch/abc123")

    assert result.video_path == video_path
    assert result.metadata_path == video_path.with_name("abc123.mp4.metadata.json")
    assert result.metadata_path.exists()
    assert '"id": "abc123"' in result.metadata_path.read_text(encoding="utf-8")
    assert client.calls == [("https://example.test/watch/abc123", True)]


def test_video_downloader_rejects_empty_url(tmp_path: Path) -> None:
    downloader = VideoDownloader(config=DownloaderConfig(downloads_dir=tmp_path))

    with pytest.raises(DownloadError, match="non-empty URL"):
        downloader.download("  ")


def test_video_downloader_requires_resolvable_download_path(tmp_path: Path) -> None:
    downloader = VideoDownloader(
        config=DownloaderConfig(downloads_dir=tmp_path),
        client=FakeClient(metadata_payload()),
    )

    with pytest.raises(DownloadError, match="Unable to resolve"):
        downloader.download("https://example.test/watch/abc123")


def test_video_metadata_requires_core_fields() -> None:
    incomplete = {"id": "abc123", "title": "Missing URL"}

    with pytest.raises(MetadataExtractionError, match="missing required"):
        VideoMetadata.from_yt_dlp(incomplete)


def test_metadata_writer_uses_configured_suffix(tmp_path: Path) -> None:
    writer = MetadataWriter(
        DownloaderConfig(downloads_dir=tmp_path, metadata_suffix=".json")
    )
    metadata = VideoMetadata.from_yt_dlp(metadata_payload())
    video_path = tmp_path / "video.mp4"

    metadata_path = writer.write(video_path, metadata)

    assert metadata_path == tmp_path / "video.mp4.json"
    assert metadata_path.exists()
