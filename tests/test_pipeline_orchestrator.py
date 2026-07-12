import json
import math
import shutil
import struct
import wave
from pathlib import Path

import pytest

from audio_observer import FFmpegAudioExtractor, FFmpegAudioExtractorConfig
from core import DownloadResult, FeatureTimeline
from pipeline import PipelineOrchestrator


def write_tiny_media_fixture(path: Path) -> None:
    """Write a small local PCM fixture that FFmpeg can process offline."""

    sample_rate_hz = 8_000
    samples = [
        int(8_000 * math.sin(2 * math.pi * 440 * index / sample_rate_hz))
        for index in range(1_600)
    ]
    with wave.open(str(path), "wb") as fixture:
        fixture.setnchannels(1)
        fixture.setsampwidth(2)
        fixture.setframerate(sample_rate_hz)
        fixture.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))


class LocalFixtureDownloader:
    def __init__(self, fixture_path: Path, download_path: Path) -> None:
        self.fixture_path = fixture_path
        self.download_path = download_path
        self.urls: list[str] = []

    def download(self, url: str) -> DownloadResult:
        self.urls.append(url)
        self.download_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.fixture_path, self.download_path)
        metadata_path = self.download_path.with_name(
            f"{self.download_path.name}.metadata.json"
        )
        metadata_path.write_text('{"id": "fixture"}', encoding="utf-8")
        return DownloadResult(
            source_url=url,
            video_path=self.download_path,
            metadata_path=metadata_path,
            provider="Fixture",
            media_id="fixture",
            title="Local Fixture",
            duration_seconds=0.2,
        )


def make_orchestrator(
    tmp_path: Path,
    downloader: LocalFixtureDownloader | None = None,
) -> PipelineOrchestrator:
    return PipelineOrchestrator(
        downloader=downloader,
        audio_extractor=FFmpegAudioExtractor(
            FFmpegAudioExtractorConfig(
                sample_rate_hz=16_000,
                channels=1,
                output_dir=tmp_path / "audio",
            )
        ),
    )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
def test_pipeline_analyzes_local_media_and_persists_timeline(tmp_path: Path) -> None:
    media_path = tmp_path / "local-fixture.wav"
    write_tiny_media_fixture(media_path)

    timeline = make_orchestrator(tmp_path).analyze(media_path)

    assert isinstance(timeline, FeatureTimeline)
    assert timeline.media_path == media_path
    assert timeline.audio_path == tmp_path / "audio" / "local-fixture.16000hz.1ch.wav"
    assert timeline.timeline_path == media_path.with_name(
        "local-fixture.wav.feature-timeline.json"
    )
    assert timeline.source_url is None
    assert timeline.download is None
    assert timeline.failures == []
    assert timeline.metadata == {"input_type": "local", "observer_count": 1}
    assert [result.observer for result in timeline.timeline.observer_results] == [
        "audio"
    ]
    assert timeline.timeline.groups

    persisted = json.loads(timeline.timeline_path.read_text(encoding="utf-8"))
    assert persisted["media_path"] == str(media_path)
    assert persisted["audio_path"] == str(timeline.audio_path)
    assert persisted["timeline"]["observer_results"][0]["observer"] == "audio"
    assert persisted["timeline"]["groups"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
def test_pipeline_downloads_url_then_analyzes_local_fixture(tmp_path: Path) -> None:
    fixture_path = tmp_path / "source-fixture.wav"
    write_tiny_media_fixture(fixture_path)
    downloaded_path = tmp_path / "downloads" / "downloaded-fixture.wav"
    downloader = LocalFixtureDownloader(fixture_path, downloaded_path)
    url = "https://www.youtube.com/watch?v=fixture"

    timeline = make_orchestrator(tmp_path, downloader=downloader).analyze(url)

    assert downloader.urls == [url]
    assert timeline.media_path == downloaded_path
    assert timeline.source_url == url
    assert timeline.download is not None
    assert timeline.download.provider == "Fixture"
    assert timeline.metadata == {"input_type": "download", "observer_count": 1}
    assert timeline.audio_path.is_file()
    assert timeline.timeline_path == downloaded_path.with_name(
        "downloaded-fixture.wav.feature-timeline.json"
    )
    assert timeline.timeline_path.is_file()
    assert timeline.failures == []
    assert {observation.type for observation in timeline.timeline.groups[0].observations}
    persisted = json.loads(timeline.timeline_path.read_text(encoding="utf-8"))
    assert persisted["download"]["media_id"] == "fixture"
