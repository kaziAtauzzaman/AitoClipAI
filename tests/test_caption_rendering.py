import json
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

import pytest

from aggregation import FeatureAggregator
from captioning import CaptionArtifact, CaptionGenerator, CaptionGeneratorConfig
from clip_rendering import (
    ClipRenderer,
    ClipRendererConfig,
    IntelQSVUnavailableError,
    RendererBackend,
    SubtitleRenderingError,
    escape_subtitle_filter_path,
)
from core import (
    ClipCandidate,
    ClipScore,
    FeatureTimeline,
    Observation,
    ObserverResult,
)
from pipeline.validation import ArtifactValidator


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
            Path(captured[-1]).write_bytes(b"rendered")
        return subprocess.CompletedProcess(
            captured,
            self.returncode,
            stdout="",
            stderr=self.stderr,
        )


def score(source: Path, start: float = 1.0, end: float = 2.0) -> ClipScore:
    return ClipScore(
        candidate=ClipCandidate(
            source_video_path=source,
            start_seconds=start,
            end_seconds=end,
            reason="caption candidate",
        ),
        overall_score=0.9,
    )


def artifact(clip_score: ClipScore, path: Path) -> CaptionArtifact:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("1\n00:00:00,000 --> 00:00:00,500\nCaption\n", encoding="utf-8")
    return CaptionArtifact(candidate=clip_score.candidate, path=path)


def test_caption_free_rendering_remains_default_when_artifact_is_supplied(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    clip_score = score(source)
    captions = artifact(clip_score, tmp_path / "captions" / "clip.srt")
    runner = FakeRunner()
    renderer = ClipRenderer(
        ClipRendererConfig(output_dir=tmp_path / "clips", overwrite_existing=True),
        runner=runner,
        executable_locator=lambda binary: "ffmpeg",
    )

    job = renderer.render([clip_score], [captions])[0]

    assert job.captions_path is None
    assert job.metadata["subtitles_burned_in"] is False
    assert "subtitles=" not in runner.commands[0][runner.commands[0].index("-filter_complex") + 1]


def test_caption_renderer_adds_escaped_subtitle_filter_and_render_job_path(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    clip_score = score(source, 2.5, 4.5)
    captions = artifact(clip_score, tmp_path / "caption files" / "clip.srt")
    runner = FakeRunner()
    renderer = ClipRenderer(
        ClipRendererConfig(
            output_dir=tmp_path / "clips",
            overwrite_existing=True,
            burn_subtitles=True,
        ),
        runner=runner,
        executable_locator=lambda binary: "ffmpeg",
    )

    job = renderer.render([clip_score], [captions])[0]

    filter_graph = runner.commands[0][
        runner.commands[0].index("-filter_complex") + 1
    ]
    escaped = escape_subtitle_filter_path(captions.path)
    assert f"subtitles=filename='{escaped}':charenc='UTF-8'" in filter_graph
    assert filter_graph.startswith(
        "[0:v:0]setpts=PTS-STARTPTS,trim=start=0:end=2.000000,subtitles="
    )
    command = runner.commands[0]
    assert command.index("-ss") < command.index("-i")
    assert command[command.index("-ss") + 1] == "2.500000"
    assert command[command.index("-t") + 1] == "2.000000"
    assert job.captions_path == captions.path
    assert job.metadata["subtitles_burned_in"] is True


def test_subtitle_path_escaping_covers_windows_filter_characters() -> None:
    path = Path(r"C:\Media Files\captions\clip's,[final];.srt")

    escaped = escape_subtitle_filter_path(path)

    assert escaped == (
        r"C\:/Media Files/captions/clip\'s\,\[final\]\;.srt"
    )


def test_renderer_rejects_missing_or_ambiguous_caption_artifacts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    clip_score = score(source)
    renderer = ClipRenderer(
        ClipRendererConfig(output_dir=tmp_path / "clips", burn_subtitles=True),
        runner=FakeRunner(),
        executable_locator=lambda binary: "ffmpeg",
    )

    with pytest.raises(SubtitleRenderingError, match="No caption artifact"):
        renderer.render([clip_score])

    first = artifact(clip_score, tmp_path / "captions" / "first.srt")
    duplicate = artifact(clip_score, tmp_path / "captions" / "second.srt")
    with pytest.raises(SubtitleRenderingError, match="Multiple caption artifacts"):
        renderer.render([clip_score], [first, duplicate])


def test_renderer_rejects_missing_caption_file(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    clip_score = score(source)
    missing = CaptionArtifact(
        candidate=clip_score.candidate,
        path=tmp_path / "missing.srt",
    )
    renderer = ClipRenderer(
        ClipRendererConfig(output_dir=tmp_path / "clips", burn_subtitles=True),
        runner=FakeRunner(),
        executable_locator=lambda binary: "ffmpeg",
    )

    with pytest.raises(SubtitleRenderingError, match="does not exist"):
        renderer.render([clip_score], [missing])


def test_subtitle_rendering_failure_preserves_ffmpeg_diagnostics(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    clip_score = score(source)
    captions = artifact(clip_score, tmp_path / "captions" / "clip.srt")
    renderer = ClipRenderer(
        ClipRendererConfig(output_dir=tmp_path / "clips", burn_subtitles=True),
        runner=FakeRunner(
            returncode=1,
            stderr="No such filter: subtitles (libass unavailable)",
            create_output=False,
        ),
        executable_locator=lambda binary: "ffmpeg",
    )

    with pytest.raises(SubtitleRenderingError, match="libass unavailable"):
        renderer.render([clip_score], [captions])


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg and ffprobe are required",
)
def test_burned_subtitle_offline_ffmpeg_integration(tmp_path: Path) -> None:
    source = tmp_path / "source fixture.mp4"
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
            "color=c=black:size=320x180:rate=30:duration=2",
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
            str(source),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    clip_score = score(source, 0.4, 1.4)
    speech = Observation(
        timestamp_seconds=0.6,
        duration_seconds=0.6,
        observer="whisper",
        type="speech",
        value={"text": "Burned in caption", "speaker": None},
        confidence=0.95,
    )
    observer_result = ObserverResult(
        observer="whisper",
        observations=[speech],
        metadata={"language": "en"},
    )
    timeline = FeatureTimeline(
        media_path=source,
        audio_path=tmp_path / "source.wav",
        timeline_path=tmp_path / "timeline.json",
        timeline=FeatureAggregator().aggregate([observer_result]),
    )
    captions = CaptionGenerator(
        CaptionGeneratorConfig(output_dir=tmp_path / "caption files")
    ).generate(timeline, [clip_score])[0]
    plain_job = ClipRenderer(
        ClipRendererConfig(
            output_dir=tmp_path / "plain clips",
            overwrite_existing=True,
        )
    ).render([clip_score])[0]
    captioned_job = ClipRenderer(
        ClipRendererConfig(
            output_dir=tmp_path / "captioned clips",
            overwrite_existing=True,
            burn_subtitles=True,
        )
    ).render([clip_score], [captions])[0]

    assert captions.path.read_text(encoding="utf-8") == (
        "1\n00:00:00,200 --> 00:00:00,800\nBurned in caption\n"
    )
    assert captioned_job.captions_path == captions.path
    assert captioned_job.output_path.is_file()
    assert _frame_md5(plain_job.output_path, 0.4) != _frame_md5(
        captioned_job.output_path, 0.4
    )

    streams = _probe_streams(captioned_job.output_path)
    by_type = {stream["codec_type"]: stream for stream in streams}
    assert set(by_type) == {"video", "audio"}
    assert float(by_type["video"]["start_time"]) == pytest.approx(0.0, abs=0.02)
    assert float(by_type["audio"]["start_time"]) == pytest.approx(0.0, abs=0.02)
    video_duration = float(by_type["video"]["duration"])
    audio_duration = float(by_type["audio"]["duration"])
    assert video_duration == pytest.approx(1.0, abs=0.08)
    assert audio_duration == pytest.approx(1.0, abs=0.08)
    assert abs(video_duration - audio_duration) <= 0.05


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg and ffprobe are required",
)
def test_intel_qsv_burned_subtitle_hardware_integration(tmp_path: Path) -> None:
    source = tmp_path / "qsv source fixture.mp4"
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
            "color=c=black:size=320x180:rate=30:duration=2",
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
            str(source),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    clip_score = score(source, 0.4, 1.4)
    captions = artifact(clip_score, tmp_path / "captions" / "qsv.srt")
    config = ClipRendererConfig(
        output_dir=tmp_path / "qsv clips",
        overwrite_existing=True,
        burn_subtitles=True,
        renderer_backend=RendererBackend.INTEL_QSV,
    )
    renderer = ClipRenderer(config)
    try:
        captioned = renderer.render([clip_score], [captions])[0]
    except IntelQSVUnavailableError as exc:
        pytest.skip(f"Intel QSV hardware is unavailable: {exc}")

    plain = ClipRenderer(
        ClipRendererConfig(
            output_dir=tmp_path / "qsv plain clips",
            overwrite_existing=True,
            renderer_backend=RendererBackend.INTEL_QSV,
        )
    ).render([clip_score])[0]
    assert captioned.output_path.name.endswith(".intel_qsv.mp4")
    assert captioned.captions_path == captions.path
    assert _frame_md5(plain.output_path, 0.4) != _frame_md5(
        captioned.output_path, 0.4
    )
    streams = _probe_streams(captioned.output_path)
    by_type = {stream["codec_type"]: stream for stream in streams}
    assert float(by_type["video"]["start_time"]) == pytest.approx(0.0, abs=0.02)
    assert float(by_type["audio"]["start_time"]) == pytest.approx(0.0, abs=0.02)
    assert float(by_type["video"]["duration"]) == pytest.approx(1.0, abs=0.08)
    assert float(by_type["audio"]["duration"]) == pytest.approx(1.0, abs=0.08)
    validation = ArtifactValidator().validate_jobs([captioned])
    assert len(validation) == 1
    assert all(validation[0].checks.values())
    temporary = captioned.output_path.with_name(
        f".{captioned.output_path.name}.rendering"
    )
    assert not temporary.exists()


def _frame_md5(path: Path, timestamp: float) -> str:
    result = subprocess.run(
        [
            shutil.which("ffmpeg") or "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            str(timestamp),
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-f",
            "md5",
            "-",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _probe_streams(path: Path) -> list[dict[str, str]]:
    result = subprocess.run(
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
    return json.loads(result.stdout)["streams"]
