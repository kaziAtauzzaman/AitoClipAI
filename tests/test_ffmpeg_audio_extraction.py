import shutil
import subprocess
import wave
from pathlib import Path
from typing import Sequence

import pytest

from audio_observer import (
    AudioExtractionError,
    FFmpegAudioExtractor,
    FFmpegAudioExtractorConfig,
    FFmpegNotFoundError,
    WavAudioLoader,
)
from observers import ObserverContext


class FakeRunner:
    def __init__(
        self,
        *,
        returncode: int = 0,
        stderr: str = "",
        create_output: bool = True,
    ) -> None:
        self.returncode = returncode
        self.stderr = stderr
        self.create_output = create_output
        self.commands: list[list[str]] = []

    def run(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        captured = list(command)
        self.commands.append(captured)
        if self.create_output:
            Path(captured[-1]).write_bytes(b"mock wav")
        return subprocess.CompletedProcess(
            captured,
            self.returncode,
            stdout="",
            stderr=self.stderr,
        )


def test_ffmpeg_extractor_builds_deterministic_pcm_command(tmp_path: Path) -> None:
    media_path = tmp_path / "downloaded video.mp4"
    media_path.write_bytes(b"media")
    runner = FakeRunner()
    config = FFmpegAudioExtractorConfig(
        sample_rate_hz=22_050,
        channels=2,
        output_dir=tmp_path / "audio",
        overwrite_existing=True,
    )
    extractor = FFmpegAudioExtractor(
        config=config,
        runner=runner,
        executable_locator=lambda binary: "/tools/ffmpeg",
    )

    source = extractor.extract(ObserverContext(source_path=media_path))

    expected_output = tmp_path / "audio" / "downloaded video.22050hz.2ch.wav"
    assert source.path == expected_output
    assert source.metadata == {
        "source": "ffmpeg",
        "input_path": str(media_path),
        "sample_rate_hz": 22_050,
        "channels": 2,
    }
    assert runner.commands == [
        [
            "/tools/ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(media_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "22050",
            "-ac",
            "2",
            str(expected_output),
        ]
    ]


def test_ffmpeg_extractor_reuses_output_when_overwrite_is_disabled(
    tmp_path: Path,
) -> None:
    media_path = tmp_path / "video.mp4"
    media_path.write_bytes(b"media")
    output_dir = tmp_path / "audio"
    output_dir.mkdir()
    output_path = output_dir / "video.16000hz.1ch.wav"
    output_path.write_bytes(b"existing")
    runner = FakeRunner()
    extractor = FFmpegAudioExtractor(
        config=FFmpegAudioExtractorConfig(output_dir=output_dir),
        runner=runner,
        executable_locator=lambda binary: "ffmpeg",
    )

    source = extractor.extract(ObserverContext(source_path=media_path))

    assert source.path == output_path
    assert output_path.read_bytes() == b"existing"
    assert runner.commands == []


def test_ffmpeg_extractor_reports_missing_executable(tmp_path: Path) -> None:
    media_path = tmp_path / "video.mp4"
    media_path.write_bytes(b"media")
    extractor = FFmpegAudioExtractor(
        config=FFmpegAudioExtractorConfig(output_dir=tmp_path / "audio"),
        runner=FakeRunner(),
        executable_locator=lambda binary: None,
    )

    with pytest.raises(FFmpegNotFoundError, match="was not found"):
        extractor.extract(ObserverContext(source_path=media_path))


def test_ffmpeg_extractor_reports_command_failure(tmp_path: Path) -> None:
    media_path = tmp_path / "video.mp4"
    media_path.write_bytes(b"media")
    extractor = FFmpegAudioExtractor(
        config=FFmpegAudioExtractorConfig(output_dir=tmp_path / "audio"),
        runner=FakeRunner(
            returncode=1,
            stderr="input contains no audio stream",
            create_output=False,
        ),
        executable_locator=lambda binary: "ffmpeg",
    )

    with pytest.raises(AudioExtractionError, match="no audio stream"):
        extractor.extract(ObserverContext(source_path=media_path))


def test_ffmpeg_extractor_requires_created_output(tmp_path: Path) -> None:
    media_path = tmp_path / "video.mp4"
    media_path.write_bytes(b"media")
    extractor = FFmpegAudioExtractor(
        config=FFmpegAudioExtractorConfig(output_dir=tmp_path / "audio"),
        runner=FakeRunner(create_output=False),
        executable_locator=lambda binary: "ffmpeg",
    )

    with pytest.raises(AudioExtractionError, match="without creating"):
        extractor.extract(ObserverContext(source_path=media_path))


def test_ffmpeg_extractor_validates_configuration(tmp_path: Path) -> None:
    media_path = tmp_path / "video.mp4"
    media_path.write_bytes(b"media")
    extractor = FFmpegAudioExtractor(
        config=FFmpegAudioExtractorConfig(
            sample_rate_hz=0,
            output_dir=tmp_path / "audio",
        ),
        runner=FakeRunner(),
        executable_locator=lambda binary: "ffmpeg",
    )

    with pytest.raises(AudioExtractionError, match="sample rate must be positive"):
        extractor.extract(ObserverContext(source_path=media_path))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
def test_ffmpeg_extractor_integration_produces_loadable_pcm_wav(
    tmp_path: Path,
) -> None:
    fixture_path = tmp_path / "tiny-fixture.wav"
    with wave.open(str(fixture_path), "wb") as fixture:
        fixture.setnchannels(1)
        fixture.setsampwidth(2)
        fixture.setframerate(8_000)
        fixture.writeframes(b"\x00\x00" * 800)

    extractor = FFmpegAudioExtractor(
        FFmpegAudioExtractorConfig(
            sample_rate_hz=16_000,
            channels=2,
            output_dir=tmp_path / "extracted",
        )
    )

    source = extractor.extract(ObserverContext(source_path=fixture_path))
    audio = WavAudioLoader().load(source)

    assert source.path.name == "tiny-fixture.16000hz.2ch.wav"
    assert audio.sample_rate_hz == 16_000
    assert audio.channels == 2
    assert audio.duration_seconds == pytest.approx(0.1, abs=0.001)
