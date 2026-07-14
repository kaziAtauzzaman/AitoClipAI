from array import array
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

import pytest

from clip_rendering import (
    ClipRenderer,
    ClipRendererConfig,
    ClipRenderingError,
    InvalidRenderInputError,
    RenderingFFmpegNotFoundError,
)
from core import ClipCandidate, ClipScore, RenderJob


class FakeRenderRunner:
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
            Path(captured[-1]).write_bytes(b"rendered fixture")
        return subprocess.CompletedProcess(
            captured,
            self.returncode,
            stdout="",
            stderr=self.stderr,
        )


def scored_candidate(
    source_path: Path,
    start: float,
    end: float,
    score: float,
    *,
    reason: str = "candidate",
) -> ClipScore:
    candidate = ClipCandidate(
        source_video_path=source_path,
        start_seconds=start,
        end_seconds=end,
        reason=reason,
    )
    return ClipScore(
        candidate=candidate,
        overall_score=score,
        score_components={"speech_excitement": score},
        rationale=f"score {score}",
        passed_threshold=True,
    )


def test_renderer_sorts_scores_and_renders_configured_top_candidates(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source video.mp4"
    source_path.write_bytes(b"source")
    runner = FakeRenderRunner()
    config = ClipRendererConfig(
        output_dir=tmp_path / "clips",
        maximum_clips=2,
        overwrite_existing=True,
    )
    renderer = ClipRenderer(
        config=config,
        runner=runner,
        executable_locator=lambda binary: "/tools/ffmpeg",
    )
    low = scored_candidate(source_path, 20.0, 25.0, 0.4, reason="low")
    high = scored_candidate(source_path, 1.25, 4.75, 0.9, reason="high")
    middle = scored_candidate(source_path, 10.0, 15.0, 0.7, reason="middle")

    jobs = renderer.render([low, high, middle])

    assert [job.candidate.reason for job in jobs] == ["high", "middle"]
    assert all(isinstance(job, RenderJob) for job in jobs)
    assert jobs[0].output_path.name == (
        "source video.clip-001-1250-4750-900000.mp4"
    )
    assert jobs[1].output_path.name == (
        "source video.clip-002-10000-15000-700000.mp4"
    )
    assert jobs[0].metadata["rank"] == 1
    assert jobs[0].metadata["overall_score"] == 0.9
    assert jobs[0].metadata["reused_existing"] is False
    assert len(runner.commands) == 2


def test_renderer_builds_synchronized_filter_and_configured_encoding_command(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"source")
    runner = FakeRenderRunner()
    config = ClipRendererConfig(
        output_dir=tmp_path / "clips",
        filename_template="{stem}-{rank}-{score}.{ext}",
        overwrite_existing=True,
        output_format="matroska",
        video_codec="libx265",
        audio_codec="libopus",
    )
    renderer = ClipRenderer(
        config=config,
        runner=runner,
        executable_locator=lambda binary: "ffmpeg.exe",
    )

    job = renderer.render([scored_candidate(source_path, 2.5, 7.75, 0.812345)])[0]

    assert job.output_path == tmp_path / "clips" / "source-1-0.812345.matroska"
    command = runner.commands[0]
    assert command[0] == "ffmpeg.exe"
    assert command.index("-ss") < command.index("-i")
    assert command[command.index("-ss") + 1] == "2.500000"
    assert command[command.index("-i") + 1] == str(source_path)
    assert command[command.index("-t") + 1] == "5.250000"
    assert command[command.index("-filter_complex") + 1] == (
        "[0:v:0]setpts=PTS-STARTPTS,trim=start=0:end=5.250000[v];"
        "[0:a:0]atrim=start=0:end=5.250000,aresample=async=1:first_pts=0[a]"
    )
    assert command[command.index("-c:v") + 1] == "libx265"
    assert command[command.index("-c:a") + 1] == "libopus"
    assert command[command.index("-f") + 1] == "matroska"
    assert "-shortest" in command
    assert Path(command[-1]).name == f".{job.output_path.name}.rendering"


def test_renderer_reuses_deterministic_output_when_overwrite_is_disabled(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"source")
    output_dir = tmp_path / "clips"
    output_dir.mkdir()
    output_path = output_dir / "source.clip-001-1000-2000-800000.mp4"
    output_path.write_bytes(b"existing")
    temporary_path = output_path.with_name(f".{output_path.name}.rendering")
    temporary_path.write_bytes(b"abandoned partial")
    runner = FakeRenderRunner()
    renderer = ClipRenderer(
        ClipRendererConfig(output_dir=output_dir),
        runner=runner,
        executable_locator=lambda binary: "ffmpeg",
    )

    job = renderer.render([scored_candidate(source_path, 1.0, 2.0, 0.8)])[0]

    assert job.output_path == output_path
    assert output_path.read_bytes() == b"existing"
    assert job.metadata["reused_existing"] is True
    assert runner.commands == []
    assert not temporary_path.exists()


def test_renderer_reports_missing_ffmpeg(tmp_path: Path) -> None:
    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"source")
    renderer = ClipRenderer(
        ClipRendererConfig(output_dir=tmp_path / "clips"),
        runner=FakeRenderRunner(),
        executable_locator=lambda binary: None,
    )

    with pytest.raises(RenderingFFmpegNotFoundError, match="was not found"):
        renderer.render([scored_candidate(source_path, 0.0, 1.0, 0.8)])


def test_renderer_accepts_empty_input_without_requiring_ffmpeg(tmp_path: Path) -> None:
    renderer = ClipRenderer(
        ClipRendererConfig(output_dir=tmp_path / "clips"),
        runner=FakeRenderRunner(),
        executable_locator=lambda binary: None,
    )

    assert renderer.render([]) == []


def test_renderer_reports_ffmpeg_failure(tmp_path: Path) -> None:
    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"source")
    renderer = ClipRenderer(
        ClipRendererConfig(output_dir=tmp_path / "clips"),
        runner=FakeRenderRunner(
            returncode=1,
            stderr="input has no video stream",
            create_output=False,
        ),
        executable_locator=lambda binary: "ffmpeg",
    )

    with pytest.raises(ClipRenderingError, match="no video stream"):
        renderer.render([scored_candidate(source_path, 0.0, 1.0, 0.8)])


def test_renderer_retry_reuses_identical_seek_command_and_output_path(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"source")
    runner = FakeRenderRunner(
        returncode=1,
        stderr="temporary encoder failure",
        create_output=True,
    )
    renderer = ClipRenderer(
        ClipRendererConfig(output_dir=tmp_path / "clips"),
        runner=runner,
        executable_locator=lambda binary: "ffmpeg",
    )
    clip_score = scored_candidate(source_path, 12.25, 17.75, 0.8)

    with pytest.raises(ClipRenderingError, match="temporary encoder failure"):
        renderer.render_one(clip_score, 7)
    first_command = runner.commands[0]
    temporary_path = Path(first_command[-1])
    final_path = tmp_path / "clips" / "source.clip-007-12250-17750-800000.mp4"
    assert temporary_path.name == f".{final_path.name}.rendering"
    assert not temporary_path.exists()
    assert not final_path.exists()
    runner.returncode = 0
    runner.stderr = ""
    runner.create_output = True

    job = renderer.render_one(clip_score, 7)

    assert runner.commands[1] == first_command
    assert job.output_path == final_path
    assert final_path.is_file()
    assert not temporary_path.exists()
    assert job.metadata["reused_existing"] is False


def test_renderer_interruption_cleans_temporary_output(tmp_path: Path) -> None:
    class InterruptingRunner:
        def run(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
            Path(command[-1]).write_bytes(b"partial")
            raise KeyboardInterrupt

    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"source")
    renderer = ClipRenderer(
        ClipRendererConfig(output_dir=tmp_path / "clips"),
        runner=InterruptingRunner(),
        executable_locator=lambda binary: "ffmpeg",
    )
    clip_score = scored_candidate(source_path, 2.0, 4.0, 0.8)

    with pytest.raises(KeyboardInterrupt):
        renderer.render_one(clip_score, 3)

    final_path = tmp_path / "clips" / "source.clip-003-2000-4000-800000.mp4"
    temporary_path = final_path.with_name(f".{final_path.name}.rendering")
    assert not final_path.exists()
    assert not temporary_path.exists()


def test_renderer_requires_ffmpeg_to_create_output(tmp_path: Path) -> None:
    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"source")
    renderer = ClipRenderer(
        ClipRendererConfig(output_dir=tmp_path / "clips"),
        runner=FakeRenderRunner(create_output=False),
        executable_locator=lambda binary: "ffmpeg",
    )

    with pytest.raises(ClipRenderingError, match="without creating"):
        renderer.render([scored_candidate(source_path, 0.0, 1.0, 0.8)])


def test_renderer_validates_source_window_and_filename_template(
    tmp_path: Path,
) -> None:
    renderer = ClipRenderer(
        ClipRendererConfig(output_dir=tmp_path / "clips"),
        runner=FakeRenderRunner(),
        executable_locator=lambda binary: "ffmpeg",
    )
    with pytest.raises(InvalidRenderInputError, match="does not exist"):
        renderer.render(
            [scored_candidate(tmp_path / "missing.mp4", 0.0, 1.0, 0.8)]
        )

    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"source")
    with pytest.raises(InvalidRenderInputError, match="after"):
        renderer.render([scored_candidate(source_path, 2.0, 1.0, 0.8)])

    invalid_template = ClipRenderer(
        ClipRendererConfig(
            output_dir=tmp_path / "clips",
            filename_template="../{stem}.{ext}",
        ),
        runner=FakeRenderRunner(),
        executable_locator=lambda binary: "ffmpeg",
    )
    with pytest.raises(InvalidRenderInputError, match="one non-empty filename"):
        invalid_template.render([scored_candidate(source_path, 0.0, 1.0, 0.8)])


def test_renderer_is_deterministic_for_tied_input_order(tmp_path: Path) -> None:
    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"source")
    config = ClipRendererConfig(
        output_dir=tmp_path / "clips",
        maximum_clips=None,
        overwrite_existing=True,
    )
    first = scored_candidate(source_path, 4.0, 5.0, 0.8, reason="later")
    second = scored_candidate(source_path, 1.0, 2.0, 0.8, reason="earlier")

    forward = ClipRenderer(
        config,
        runner=FakeRenderRunner(),
        executable_locator=lambda binary: "ffmpeg",
    ).render([first, second])
    reverse = ClipRenderer(
        config,
        runner=FakeRenderRunner(),
        executable_locator=lambda binary: "ffmpeg",
    ).render([second, first])

    assert [job.output_path.name for job in forward] == [
        job.output_path.name for job in reverse
    ]
    assert [job.candidate.reason for job in forward] == ["earlier", "later"]


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg and ffprobe are required",
)
def test_clip_renderer_offline_audio_video_sync_integration(tmp_path: Path) -> None:
    source_path = tmp_path / "fixture.mp4"
    fixture_command = [
        shutil.which("ffmpeg") or "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=160x120:rate=30:duration=2",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=48000:duration=2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(source_path),
    ]
    subprocess.run(fixture_command, check=True, capture_output=True, text=True)
    renderer = ClipRenderer(
        ClipRendererConfig(
            output_dir=tmp_path / "clips",
            overwrite_existing=True,
            maximum_clips=1,
        )
    )

    job = renderer.render([scored_candidate(source_path, 0.4, 1.4, 0.9)])[0]

    assert job.output_path.is_file()
    assert job.metadata["duration_seconds"] == pytest.approx(1.0)
    probe = subprocess.run(
        [
            shutil.which("ffprobe") or "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,start_time,duration",
            "-of",
            "json",
            str(job.output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(probe.stdout)["streams"]
    by_type = {stream["codec_type"]: stream for stream in streams}
    assert set(by_type) == {"video", "audio"}
    video_start = float(by_type["video"]["start_time"])
    audio_start = float(by_type["audio"]["start_time"])
    video_duration = float(by_type["video"]["duration"])
    audio_duration = float(by_type["audio"]["duration"])
    assert video_start == pytest.approx(0.0, abs=0.02)
    assert audio_start == pytest.approx(0.0, abs=0.02)
    assert video_duration == pytest.approx(1.0, abs=0.08)
    assert audio_duration == pytest.approx(1.0, abs=0.08)
    assert abs(video_duration - audio_duration) <= 0.05


def decoded_rms_dbfs(
    path: Path,
    *,
    start: float,
    end: float,
) -> float:
    """Decode a stable interior audio window and return its RMS level."""

    decoded = subprocess.run(
        [
            shutil.which("ffmpeg") or "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-af",
            f"atrim=start={start:.6f}:end={end:.6f},asetpts=PTS-STARTPTS",
            "-ac",
            "1",
            "-ar",
            "48000",
            "-f",
            "s16le",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    ).stdout
    samples = array("h")
    samples.frombytes(decoded)
    assert samples
    mean_square = sum(sample * sample for sample in samples) / len(samples)
    if mean_square == 0:
        return float("-inf")
    return 20.0 * math.log10(math.sqrt(mean_square) / 32768.0)


def probed_streams(path: Path) -> dict[str, dict[str, str]]:
    """Return normalized FFprobe stream data for a rendered fixture."""

    probe = subprocess.run(
        [
            shutil.which("ffprobe") or "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,start_time,duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        stream["codec_type"]: stream
        for stream in json.loads(probe.stdout)["streams"]
    }


def probed_format_duration(path: Path) -> float:
    probe = subprocess.run(
        [
            shutil.which("ffprobe") or "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(json.loads(probe.stdout)["format"]["duration"])


def decoded_rgb_frame(path: Path, timestamp: float) -> bytes:
    """Decode one small RGB frame for non-keyframe content comparison."""

    return subprocess.run(
        [
            shutil.which("ffmpeg") or "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{timestamp:.6f}",
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    ).stdout


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg and ffprobe are required",
)
def test_input_seek_preserves_non_keyframe_content_and_media_edges(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "seek-fixture.mp4"
    subprocess.run(
        [
            shutil.which("ffmpeg") or "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x120:rate=30:duration=3",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=523:sample_rate=48000:duration=3",
            "-c:v",
            "libx264",
            "-g",
            "90",
            "-keyint_min",
            "90",
            "-sc_threshold",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(source_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    windows = {
        "near-start": (0.01, 0.61),
        "non-keyframe": (1.37, 2.07),
        "near-eof": (2.45, 3.0),
    }
    jobs = ClipRenderer(
        ClipRendererConfig(
            output_dir=tmp_path / "clips",
            overwrite_existing=True,
            maximum_clips=None,
        )
    ).render(
        [
            scored_candidate(source_path, start, end, 0.9, reason=name)
            for name, (start, end) in windows.items()
        ]
    )
    by_name = {job.candidate.reason: job for job in jobs}

    for name, (start, end) in windows.items():
        streams = probed_streams(by_name[name].output_path)
        expected_duration = end - start
        assert float(streams["video"]["start_time"]) == pytest.approx(0.0, abs=0.02)
        assert float(streams["audio"]["start_time"]) == pytest.approx(0.0, abs=0.02)
        video_duration = float(streams["video"]["duration"])
        audio_duration = float(streams["audio"]["duration"])
        assert video_duration == pytest.approx(expected_duration, abs=0.08)
        assert audio_duration == pytest.approx(expected_duration, abs=0.08)
        assert abs(video_duration - audio_duration) <= 0.05

    source_frame = decoded_rgb_frame(source_path, 1.37)
    output_frame = decoded_rgb_frame(by_name["non-keyframe"].output_path, 0.0)
    assert len(source_frame) == len(output_frame) > 0
    mean_absolute_error = sum(
        abs(source - output)
        for source, output in zip(source_frame, output_frame, strict=True)
    ) / len(source_frame)
    assert mean_absolute_error < 12.0


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg and ffprobe are required",
)
def test_input_seek_preserves_vfr_and_delayed_audio_offset(tmp_path: Path) -> None:
    source_path = tmp_path / "vfr-delayed-audio.mp4"
    subprocess.run(
        [
            shutil.which("ffmpeg") or "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x120:rate=30:duration=1.5",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x120:rate=15:duration=1.5",
            "-itsoffset",
            "0.4",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=660:sample_rate=48000:duration=3",
            "-filter_complex",
            "[0:v:0][1:v:0]concat=n=2:v=1:a=0[v]",
            "-map",
            "[v]",
            "-map",
            "2:a:0",
            "-fps_mode",
            "vfr",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(source_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    source_streams = probed_streams(source_path)
    source_delay = float(source_streams["audio"]["start_time"]) - float(
        source_streams["video"]["start_time"]
    )
    assert source_delay == pytest.approx(0.4, abs=0.05)
    start, end = 0.2, 2.8
    job = ClipRenderer(
        ClipRendererConfig(
            output_dir=tmp_path / "clips",
            overwrite_existing=True,
        )
    ).render([scored_candidate(source_path, start, end, 0.9)])[0]

    streams = probed_streams(job.output_path)
    video_start = float(streams["video"]["start_time"])
    audio_start = float(streams["audio"]["start_time"])
    assert video_start == pytest.approx(0.0, abs=0.02)
    assert audio_start == pytest.approx(0.0, abs=0.02)
    expected_content_delay = source_delay - start
    assert expected_content_delay > 0.1
    assert decoded_rms_dbfs(
        job.output_path,
        start=0.02,
        end=expected_content_delay - 0.04,
    ) <= -60.0
    assert decoded_rms_dbfs(
        job.output_path,
        start=expected_content_delay + 0.05,
        end=expected_content_delay + 0.15,
    ) > -40.0
    assert float(streams["video"]["duration"]) == pytest.approx(
        end - start, abs=0.11
    )
    assert float(streams["audio"]["duration"]) == pytest.approx(
        end - start,
        abs=0.08,
    )
    presentation_duration = probed_format_duration(job.output_path)
    assert presentation_duration == pytest.approx(end - start, abs=0.08)
    assert abs(
        presentation_duration
        - (audio_start + float(streams["audio"]["duration"]))
    ) <= 0.08


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg and ffprobe are required",
)
def test_renderer_preserves_synthetic_audio_levels_silence_and_sync(
    tmp_path: Path,
) -> None:
    """Regression coverage for quiet, normal, loud, and silent source audio."""

    source_path = tmp_path / "audio-preservation-fixture.mp4"
    subprocess.run(
        [
            shutil.which("ffmpeg") or "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=160x120:rate=30:duration=8",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=48000:cl=mono:d=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=330:sample_rate=48000:duration=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=550:sample_rate=48000:duration=2",
            "-filter_complex",
            "[2:a]volume=0.08[quiet];"
            "[3:a]volume=1.0[normal];"
            "[4:a]volume=4.0[loud];"
            "[1:a][quiet][normal][loud]concat=n=4:v=0:a=1[a]",
            "-map",
            "0:v:0",
            "-map",
            "[a]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(source_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    windows = {
        "silence": (0.25, 1.75),
        "quiet": (2.25, 3.75),
        "normal": (4.25, 5.75),
        "loud": (6.25, 7.75),
    }
    scores = [
        scored_candidate(source_path, start, end, 0.9, reason=name)
        for name, (start, end) in windows.items()
    ]
    jobs = ClipRenderer(
        ClipRendererConfig(
            output_dir=tmp_path / "clips",
            overwrite_existing=True,
            maximum_clips=None,
        )
    ).render(scores)
    jobs_by_name = {job.candidate.reason: job for job in jobs}

    for name, (start, end) in windows.items():
        job = jobs_by_name[name]
        duration = end - start
        source_level = decoded_rms_dbfs(
            source_path,
            start=start + 0.05,
            end=end - 0.05,
        )
        output_level = decoded_rms_dbfs(
            job.output_path,
            start=0.05,
            end=duration - 0.05,
        )
        if name == "silence":
            assert source_level <= -60.0
            assert output_level <= -60.0
        else:
            assert source_level > -60.0
            assert output_level > -60.0
            assert output_level == pytest.approx(source_level, abs=0.75)

        streams = probed_streams(job.output_path)
        assert set(streams) == {"video", "audio"}
        video_start = float(streams["video"]["start_time"])
        audio_start = float(streams["audio"]["start_time"])
        video_duration = float(streams["video"]["duration"])
        audio_duration = float(streams["audio"]["duration"])
        assert video_start == pytest.approx(0.0, abs=0.02)
        assert audio_start == pytest.approx(0.0, abs=0.02)
        assert video_duration == pytest.approx(duration, abs=0.08)
        assert audio_duration == pytest.approx(duration, abs=0.08)
        assert abs(video_duration - audio_duration) <= 0.05
