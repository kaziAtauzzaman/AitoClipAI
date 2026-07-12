"""FFmpeg-backed deterministic clip rendering service."""

import shutil
import subprocess
from pathlib import Path
from typing import Callable, Iterable, Protocol, Sequence

from clip_rendering.config import ClipRendererConfig
from clip_rendering.errors import (
    ClipRenderingError,
    InvalidRenderInputError,
    RenderingFFmpegNotFoundError,
)
from core import ClipScore, RenderJob


class RenderCommandRunner(Protocol):
    """Execute a rendering command and capture diagnostics."""

    def run(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        """Run a command without raising for a non-zero return code."""


class SubprocessRenderCommandRunner:
    """Render command runner backed by the standard subprocess module."""

    def run(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                list(command),
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            raise ClipRenderingError(f"Failed to execute FFmpeg: {exc}") from exc


class ClipRenderer:
    """Render the highest-scoring candidates from their original source media."""

    def __init__(
        self,
        config: ClipRendererConfig | None = None,
        runner: RenderCommandRunner | None = None,
        executable_locator: Callable[[str], str | None] | None = None,
    ) -> None:
        self._config = config or ClipRendererConfig()
        self._runner = runner or SubprocessRenderCommandRunner()
        self._executable_locator = executable_locator or shutil.which
        self._validate_config()

    def render(self, scores: Iterable[ClipScore]) -> list[RenderJob]:
        """Sort scores, render the configured top candidates, and return jobs."""

        ranked = sorted(scores, key=self._ranking_key)
        if self._config.maximum_clips is not None:
            ranked = ranked[: self._config.maximum_clips]
        if not ranked:
            return []

        ffmpeg_path = self._executable_locator(self._config.ffmpeg_binary)
        if ffmpeg_path is None:
            raise RenderingFFmpegNotFoundError(
                f"FFmpeg executable was not found: {self._config.ffmpeg_binary!r}."
            )

        try:
            self._config.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ClipRenderingError(
                f"Failed to create clip output directory: {exc}"
            ) from exc

        return [
            self._render_score(ffmpeg_path, score, rank)
            for rank, score in enumerate(ranked, start=1)
        ]

    def _render_score(
        self,
        ffmpeg_path: str,
        score: ClipScore,
        rank: int,
    ) -> RenderJob:
        candidate = score.candidate
        source_path = Path(candidate.source_video_path)
        if not source_path.is_file():
            raise InvalidRenderInputError(f"Source media does not exist: {source_path}")
        if candidate.start_seconds < 0:
            raise InvalidRenderInputError("Clip start time cannot be negative.")
        if candidate.end_seconds <= candidate.start_seconds:
            raise InvalidRenderInputError("Clip end time must be after its start time.")

        output_path = self._output_path(score, rank)
        if output_path.exists() and not self._config.overwrite_existing:
            return self._render_job(score, output_path, rank, reused=True)

        command = self._build_command(ffmpeg_path, score, output_path)
        result = self._runner.run(command)
        if result.returncode != 0:
            diagnostic = result.stderr.strip() or result.stdout.strip()
            detail = f": {diagnostic}" if diagnostic else ""
            raise ClipRenderingError(
                f"FFmpeg clip rendering failed with exit code {result.returncode}{detail}"
            )
        if not output_path.is_file():
            raise ClipRenderingError(
                f"FFmpeg completed without creating rendered clip: {output_path}"
            )
        return self._render_job(score, output_path, rank, reused=False)

    def _output_path(self, score: ClipScore, rank: int) -> Path:
        candidate = score.candidate
        extension = self._normalized_output_format()
        values = {
            "stem": candidate.source_video_path.stem,
            "rank": rank,
            "start_ms": round(candidate.start_seconds * 1000),
            "end_ms": round(candidate.end_seconds * 1000),
            "score": f"{score.overall_score:.6f}",
            "score_millionths": round(score.overall_score * 1_000_000),
            "ext": extension,
        }
        try:
            filename = self._config.filename_template.format(**values)
        except (KeyError, ValueError) as exc:
            raise InvalidRenderInputError(
                f"Invalid clip filename template: {exc}"
            ) from exc
        if not filename or Path(filename).name != filename:
            raise InvalidRenderInputError(
                "Clip filename template must produce one non-empty filename."
            )
        return self._config.output_dir / filename

    def _build_command(
        self,
        ffmpeg_path: str,
        score: ClipScore,
        output_path: Path,
    ) -> list[str]:
        candidate = score.candidate
        start = f"{candidate.start_seconds:.6f}"
        end = f"{candidate.end_seconds:.6f}"
        filter_graph = (
            f"[0:v:0]trim=start={start}:end={end},setpts=PTS-STARTPTS[v];"
            f"[0:a:0]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a]"
        )
        return [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y" if self._config.overwrite_existing else "-n",
            "-i",
            str(candidate.source_video_path),
            "-filter_complex",
            filter_graph,
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-c:v",
            self._config.video_codec,
            "-c:a",
            self._config.audio_codec,
            "-map_metadata",
            "-1",
            "-map_chapters",
            "-1",
            "-shortest",
            "-f",
            self._normalized_output_format(),
            str(output_path),
        ]

    def _render_job(
        self,
        score: ClipScore,
        output_path: Path,
        rank: int,
        *,
        reused: bool,
    ) -> RenderJob:
        candidate = score.candidate
        return RenderJob(
            candidate=candidate,
            output_path=output_path,
            metadata={
                "rank": rank,
                "overall_score": score.overall_score,
                "score_components": score.score_components,
                "score_rationale": score.rationale,
                "start_seconds": candidate.start_seconds,
                "end_seconds": candidate.end_seconds,
                "duration_seconds": candidate.end_seconds - candidate.start_seconds,
                "output_format": self._normalized_output_format(),
                "video_codec": self._config.video_codec,
                "audio_codec": self._config.audio_codec,
                "reused_existing": reused,
            },
        )

    def _ranking_key(self, score: ClipScore) -> tuple[float, float, float, str, str]:
        candidate = score.candidate
        return (
            -score.overall_score,
            candidate.start_seconds,
            candidate.end_seconds,
            candidate.reason,
            str(candidate.source_video_path),
        )

    def _validate_config(self) -> None:
        config = self._config
        if config.maximum_clips is not None and config.maximum_clips <= 0:
            raise InvalidRenderInputError("Maximum clips must be positive or None.")
        if not config.output_format.removeprefix("."):
            raise InvalidRenderInputError("Output format cannot be empty.")
        if not config.video_codec.strip() or not config.audio_codec.strip():
            raise InvalidRenderInputError("Video and audio codecs cannot be empty.")

    def _normalized_output_format(self) -> str:
        return self._config.output_format.removeprefix(".")
