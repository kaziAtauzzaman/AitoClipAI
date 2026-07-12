from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
import threading
from typing import Any

import pytest

from core import DownloadResult as CoreDownloadResult
from downloader import DownloadResult as PublicDownloadResult
from downloader.config import DownloaderConfig
from downloader.downloader import VideoDownloader
from downloader.errors import DownloadError, MetadataExtractionError
from downloader.metadata import MetadataExtractor, MetadataWriter, VideoMetadata
from downloader.sanitization import JsonMetadataSanitizer


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

    assert isinstance(result, CoreDownloadResult)
    assert PublicDownloadResult is CoreDownloadResult
    assert result.source_url == "https://example.test/watch/abc123"
    assert result.provider == "Example"
    assert result.media_id == "abc123"
    assert result.title == "Example Video"
    assert result.duration_seconds == 42.0
    assert result.metadata["id"] == "abc123"
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


class ExampleValue(Enum):
    READY = "ready"


class RuntimeOnlyObject:
    pass


def test_metadata_sanitizer_handles_non_copyable_runtime_objects() -> None:
    lock = threading.Lock()
    raw = {
        "lock": lock,
        "nested": [{"runtime": RuntimeOnlyObject()}],
    }

    sanitized = JsonMetadataSanitizer().sanitize(raw)

    assert sanitized == {
        "lock": {"__aitoclipai_unsupported_type__": "_thread.lock"},
        "nested": [
            {
                "runtime": {
                    "__aitoclipai_unsupported_type__": (
                        f"{__name__}.RuntimeOnlyObject"
                    )
                }
            }
        ],
    }
    assert raw["lock"] is lock
    assert isinstance(raw["nested"], list)
    assert isinstance(raw["nested"][0]["runtime"], RuntimeOnlyObject)


def test_metadata_sanitizer_detects_cycles_without_mutating_payload() -> None:
    raw: dict[str, Any] = {"id": "cyclic"}
    raw["self"] = raw

    sanitized = JsonMetadataSanitizer().sanitize(raw)

    assert sanitized == {
        "id": "cyclic",
        "self": {"__aitoclipai_cycle__": True},
    }
    assert raw["self"] is raw


def test_metadata_sanitizer_preserves_useful_values_deterministically() -> None:
    raw = {
        "path": Path("media/video.mp4"),
        "created": datetime(2026, 7, 12, 1, 2, 3, tzinfo=timezone.utc),
        "status": ExampleValue.READY,
        "binary": b"metadata",
        "unordered": {"second", "first"},
        "non_finite": float("nan"),
    }
    sanitizer = JsonMetadataSanitizer()

    first = sanitizer.sanitize(raw)
    second = sanitizer.sanitize(raw)

    assert first == second
    assert json.dumps(first, allow_nan=False, sort_keys=True) == json.dumps(
        second,
        allow_nan=False,
        sort_keys=True,
    )
    assert first == {
        "path": str(Path("media/video.mp4")),
        "created": "2026-07-12T01:02:03+00:00",
        "status": "ready",
        "binary": {"__aitoclipai_bytes_base64__": "bWV0YWRhdGE="},
        "unordered": ["first", "second"],
        "non_finite": {"__aitoclipai_non_finite_float__": "nan"},
    }


def test_metadata_writer_persists_runtime_objects_as_stable_markers(
    tmp_path: Path,
) -> None:
    payload = metadata_payload()
    payload["runtime_lock"] = threading.Lock()
    metadata = VideoMetadata.from_yt_dlp(payload)
    writer = MetadataWriter(DownloaderConfig(downloads_dir=tmp_path))

    first_path = writer.write(tmp_path / "video.mp4", metadata)
    first_bytes = first_path.read_bytes()
    second_path = writer.write(tmp_path / "video.mp4", metadata)

    persisted = json.loads(second_path.read_text(encoding="utf-8"))
    assert first_bytes == second_path.read_bytes()
    assert persisted["id"] == "abc123"
    assert persisted["title"] == "Example Video"
    assert persisted["raw"]["runtime_lock"] == {
        "__aitoclipai_unsupported_type__": "_thread.lock"
    }
    assert payload["runtime_lock"].locked() is False


def test_video_downloader_returns_sanitized_metadata_for_runtime_objects(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"media")
    payload = metadata_payload(video_path)
    payload["runtime_lock"] = threading.Lock()
    downloader = VideoDownloader(
        config=DownloaderConfig(downloads_dir=tmp_path),
        client=FakeClient(payload),
    )

    result = downloader.download("https://example.test/watch/abc123")

    assert result.metadata["raw"]["runtime_lock"] == {
        "__aitoclipai_unsupported_type__": "_thread.lock"
    }
    assert result.metadata_path.is_file()
