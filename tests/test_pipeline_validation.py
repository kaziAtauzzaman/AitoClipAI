import json
import subprocess
from pathlib import Path
from typing import Sequence

import pytest

from core import ClipCandidate, RenderJob
from pipeline import (
    ArtifactValidationConfig,
    ArtifactValidationError,
    ArtifactValidator,
    FFprobeMediaProbe,
    MediaProbeError,
    MediaProbeResult,
    MediaStreamProbe,
)


class FakeProbeRunner:
    def __init__(self, payload: object, *, returncode: int = 0, stderr: str = "") -> None:
        self.payload = payload
        self.returncode = returncode
        self.stderr = stderr
        self.commands: list[list[str]] = []

    def run(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        captured = list(command)
        self.commands.append(captured)
        return subprocess.CompletedProcess(
            captured,
            self.returncode,
            stdout=json.dumps(self.payload),
            stderr=self.stderr,
        )


class FakeMediaProbe:
    def __init__(self, result: MediaProbeResult) -> None:
        self.result = result
        self.paths: list[Path] = []

    def probe(self, path: Path) -> MediaProbeResult:
        self.paths.append(path)
        return self.result


def render_job(path: Path) -> RenderJob:
    return RenderJob(
        candidate=ClipCandidate(
            source_video_path=path.with_name("source.mp4"),
            start_seconds=0.0,
            end_seconds=1.0,
            reason="validation",
        ),
        output_path=path,
    )


def probe_result(
    path: Path,
    *,
    video_start: float = 0.0,
    audio_start: float = 0.0,
    video_duration: float = 1.0,
    audio_duration: float = 1.0,
    include_video: bool = True,
    include_audio: bool = True,
) -> MediaProbeResult:
    streams: list[MediaStreamProbe] = []
    if include_video:
        streams.append(
            MediaStreamProbe("video", "h264", video_start, video_duration)
        )
    if include_audio:
        streams.append(MediaStreamProbe("audio", "aac", audio_start, audio_duration))
    return MediaProbeResult(
        path=path,
        format_name="mov,mp4",
        duration_seconds=max(video_duration, audio_duration),
        streams=streams,
    )


def test_ffprobe_adapter_normalizes_json_and_command(tmp_path: Path) -> None:
    media_path = tmp_path / "clip.mp4"
    media_path.write_bytes(b"media")
    runner = FakeProbeRunner(
        {
            "streams": [
                {
                    "index": 0,
                    "codec_type": "video",
                    "codec_name": "h264",
                    "start_time": "0.000000",
                    "duration": "1.250000",
                },
                {
                    "index": 1,
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "start_time": "0.000000",
                    "duration": "1.248000",
                },
            ],
            "format": {"format_name": "mov,mp4", "duration": "1.250000"},
        }
    )
    probe = FFprobeMediaProbe(
        runner=runner,
        executable_locator=lambda binary: "/tools/ffprobe",
    )

    result = probe.probe(media_path)

    assert result.duration_seconds == 1.25
    assert [stream.codec_type for stream in result.streams] == ["video", "audio"]
    assert result.streams[1].duration_seconds == 1.248
    assert runner.commands[0][0] == "/tools/ffprobe"
    assert runner.commands[0][-1] == str(media_path)


def test_ffprobe_adapter_preserves_diagnostics(tmp_path: Path) -> None:
    media_path = tmp_path / "clip.mp4"
    media_path.write_bytes(b"media")
    probe = FFprobeMediaProbe(
        runner=FakeProbeRunner({}, returncode=1, stderr="invalid media payload"),
        executable_locator=lambda binary: "ffprobe",
    )

    with pytest.raises(MediaProbeError, match="invalid media payload"):
        probe.probe(media_path)


def test_artifact_validator_accepts_playable_synchronized_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "clip.mp4"
    output.write_bytes(b"nonempty")
    probe = FakeMediaProbe(probe_result(output, audio_duration=0.96))
    validator = ArtifactValidator(
        probe=probe,
        config=ArtifactValidationConfig(
            maximum_duration_difference_seconds=0.05
        ),
    )

    result = validator.validate_jobs([render_job(output)])[0]

    assert result.path == output
    assert result.size_bytes == len(b"nonempty")
    assert all(result.checks.values())


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (lambda path: probe_result(path, include_video=False), "no video stream"),
        (lambda path: probe_result(path, include_audio=False), "no audio stream"),
        (lambda path: probe_result(path, video_start=0.2), "does not start near zero"),
        (
            lambda path: probe_result(path, video_duration=1.0, audio_duration=0.5),
            "difference exceeds tolerance",
        ),
    ],
)
def test_artifact_validator_rejects_invalid_streams(
    tmp_path: Path,
    result,
    message: str,
) -> None:
    output = tmp_path / "clip.mp4"
    output.write_bytes(b"nonempty")
    validator = ArtifactValidator(probe=FakeMediaProbe(result(output)))

    with pytest.raises(ArtifactValidationError, match=message):
        validator.validate_jobs([render_job(output)])


def test_artifact_validator_requires_existing_nonempty_output(tmp_path: Path) -> None:
    missing = tmp_path / "missing.mp4"
    validator = ArtifactValidator(probe=FakeMediaProbe(probe_result(missing)))
    with pytest.raises(ArtifactValidationError, match="does not exist"):
        validator.validate_jobs([render_job(missing)])

    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    with pytest.raises(ArtifactValidationError, match="empty"):
        validator.validate_jobs([render_job(empty)])


def test_artifact_validator_rejects_zero_stream_duration(tmp_path: Path) -> None:
    output = tmp_path / "clip.mp4"
    output.write_bytes(b"nonempty")
    result = probe_result(output, video_duration=0.0, audio_duration=1.0)
    result = MediaProbeResult(
        path=output,
        format_name="mp4",
        duration_seconds=1.0,
        streams=result.streams,
    )

    with pytest.raises(ArtifactValidationError, match="positive duration"):
        ArtifactValidator(probe=FakeMediaProbe(result)).validate_jobs(
            [render_job(output)]
        )
