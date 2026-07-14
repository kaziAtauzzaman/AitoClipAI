"""FFmpeg-backed deterministic clip rendering service."""

import shutil
import subprocess
import threading
from pathlib import Path
from typing import Callable, Iterable, Protocol, Sequence

from captioning import (
    CaptionArtifact,
    CandidateCaptionIdentity,
    candidate_caption_identity,
)
from clip_rendering.config import ClipRendererConfig, RendererBackend
from clip_rendering.errors import (
    ClipRenderingError,
    InvalidRenderInputError,
    IntelQSVUnavailableError,
    RenderingFFmpegNotFoundError,
    SubtitleRenderingError,
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

    _qsv_preflight_successes: set[str] = set()
    _qsv_preflight_lock = threading.Lock()

    def __init__(
        self,
        config: ClipRendererConfig | None = None,
        runner: RenderCommandRunner | None = None,
        executable_locator: Callable[[str], str | None] | None = None,
        qsv_capability_checker: Callable[[str], bool] | None = None,
    ) -> None:
        self._config = config or ClipRendererConfig()
        self._runner = runner or SubprocessRenderCommandRunner()
        self._executable_locator = executable_locator or shutil.which
        self._qsv_capability_checker = (
            qsv_capability_checker or self._runtime_qsv_preflight
        )
        self._validate_config()

    def render(
        self,
        scores: Iterable[ClipScore],
        caption_artifacts: Iterable[CaptionArtifact] | None = None,
    ) -> list[RenderJob]:
        """Sort scores, render the configured top candidates, and return jobs."""

        ranked = sorted(scores, key=self._ranking_key)
        if self._config.maximum_clips is not None:
            ranked = ranked[: self._config.maximum_clips]
        if not ranked:
            return []
        captions = self._caption_artifacts(caption_artifacts or [])

        ffmpeg_path = self._executable_locator(self._config.ffmpeg_binary)
        if ffmpeg_path is None:
            raise RenderingFFmpegNotFoundError(
                f"FFmpeg executable was not found: {self._config.ffmpeg_binary!r}."
            )
        self._require_requested_backend(ffmpeg_path)

        try:
            self._config.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ClipRenderingError(
                f"Failed to create clip output directory: {exc}"
            ) from exc

        return [
            self._render_score(
                ffmpeg_path,
                score,
                rank,
                self._caption_for_score(score, captions),
            )
            for rank, score in enumerate(ranked, start=1)
        ]

    def render_one(
        self,
        score: ClipScore,
        identity: int,
        caption_artifact: CaptionArtifact | None = None,
    ) -> RenderJob:
        """Render one finalized score with an explicit monotonic identity.

        This entry point intentionally bypasses ``maximum_clips`` and score
        reranking.  It is used by chronological incremental coordinators after
        selection has already finalized one winner.
        """

        if identity <= 0:
            raise InvalidRenderInputError("Render identity must be positive.")
        ffmpeg_path = self._executable_locator(self._config.ffmpeg_binary)
        if ffmpeg_path is None:
            raise RenderingFFmpegNotFoundError(
                f"FFmpeg executable was not found: {self._config.ffmpeg_binary!r}."
            )
        self._require_requested_backend(ffmpeg_path)
        try:
            self._config.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ClipRenderingError(
                f"Failed to create clip output directory: {exc}"
            ) from exc
        return self._render_score(ffmpeg_path, score, identity, caption_artifact)

    def _render_score(
        self,
        ffmpeg_path: str,
        score: ClipScore,
        rank: int,
        caption_artifact: CaptionArtifact | None,
    ) -> RenderJob:
        candidate = score.candidate
        source_path = Path(candidate.source_video_path)
        if not source_path.is_file():
            raise InvalidRenderInputError(f"Source media does not exist: {source_path}")
        if candidate.start_seconds < 0:
            raise InvalidRenderInputError("Clip start time cannot be negative.")
        if candidate.end_seconds <= candidate.start_seconds:
            raise InvalidRenderInputError("Clip end time must be after its start time.")
        if caption_artifact is not None and not caption_artifact.path.is_file():
            raise SubtitleRenderingError(
                f"Caption artifact does not exist: {caption_artifact.path}"
            )

        output_path = self._output_path(score, rank)
        temporary_path = self._temporary_output_path(output_path)
        self._remove_temporary_output(temporary_path)
        if output_path.exists() and not self._config.overwrite_existing:
            return self._render_job(
                score,
                output_path,
                rank,
                caption_artifact,
                reused=True,
            )

        try:
            command = self._build_command(
                ffmpeg_path,
                score,
                temporary_path,
                caption_artifact,
            )
            result = self._runner.run(command)
            if result.returncode != 0:
                diagnostic = result.stderr.strip() or result.stdout.strip()
                detail = f": {diagnostic}" if diagnostic else ""
                if (
                    self._config.renderer_backend is RendererBackend.INTEL_QSV
                    and self._is_qsv_initialization_failure(diagnostic)
                ):
                    raise IntelQSVUnavailableError(
                        "Intel QSV device, driver, session, or encoder "
                        f"initialization failed{detail}"
                    )
                error_type = (
                    SubtitleRenderingError
                    if caption_artifact is not None
                    and self._is_subtitle_failure(diagnostic)
                    else ClipRenderingError
                )
                raise error_type(
                    "FFmpeg clip rendering failed with exit code "
                    f"{result.returncode}{detail}"
                )
            if not temporary_path.is_file():
                raise ClipRenderingError(
                    "FFmpeg completed without creating rendered clip: "
                    f"{temporary_path}"
                )
            if output_path.exists() and not self._config.overwrite_existing:
                return self._render_job(
                    score,
                    output_path,
                    rank,
                    caption_artifact,
                    reused=True,
                )
            try:
                temporary_path.replace(output_path)
            except OSError as exc:
                raise ClipRenderingError(
                    f"Failed to promote rendered clip atomically: {exc}"
                ) from exc
        finally:
            self._remove_temporary_output(temporary_path)
        return self._render_job(
            score,
            output_path,
            rank,
            caption_artifact,
            reused=False,
        )

    @staticmethod
    def _temporary_output_path(output_path: Path) -> Path:
        return output_path.with_name(f".{output_path.name}.rendering")

    @staticmethod
    def _remove_temporary_output(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            raise ClipRenderingError(
                f"Failed to clean temporary render output: {exc}"
            ) from exc

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
        if self._config.renderer_backend is RendererBackend.INTEL_QSV:
            path = Path(filename)
            filename = f"{path.stem}.intel_qsv{path.suffix}"
        return self._config.output_dir / filename

    def _build_command(
        self,
        ffmpeg_path: str,
        score: ClipScore,
        output_path: Path,
        caption_artifact: CaptionArtifact | None,
    ) -> list[str]:
        candidate = score.candidate
        start = f"{candidate.start_seconds:.6f}"
        duration = f"{candidate.end_seconds - candidate.start_seconds:.6f}"
        # FFmpeg rebases the primary input seek onto one shared timeline. Video
        # is aligned to its first decoded frame for a zero-based output. Audio
        # remains on the shared timeline; first_pts=0 fills any intentional
        # positive delay with silence instead of collapsing the content offset.
        video_filters = f"setpts=PTS-STARTPTS,trim=start=0:end={duration}"
        if caption_artifact is not None:
            escaped_path = escape_subtitle_filter_path(caption_artifact.path)
            character_encoding = escape_subtitle_filter_value(
                self._config.subtitle_character_encoding
            )
            video_filters += (
                f",subtitles=filename='{escaped_path}':charenc='{character_encoding}'"
            )
        filter_graph = (
            f"[0:v:0]{video_filters}[v];"
            f"[0:a:0]atrim=start=0:end={duration},"
            "aresample=async=1:first_pts=0[a]"
        )
        video_codec = (
            "h264_qsv"
            if self._config.renderer_backend is RendererBackend.INTEL_QSV
            else self._config.video_codec
        )
        command = [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y" if self._config.overwrite_existing else "-n",
            # Input options apply to the next input only. Keep this seek grouped
            # with the primary source when future secondary inputs are added.
            "-ss",
            start,
            "-i",
            str(candidate.source_video_path),
            "-t",
            duration,
            "-filter_complex",
            filter_graph,
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-c:v",
            video_codec,
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
        if self._config.renderer_backend is RendererBackend.INTEL_QSV:
            codec_index = command.index("-c:v") + 2
            command[codec_index:codec_index] = [
                "-preset",
                "medium",
                "-global_quality",
                "23",
            ]
        return command

    def _render_job(
        self,
        score: ClipScore,
        output_path: Path,
        rank: int,
        caption_artifact: CaptionArtifact | None,
        *,
        reused: bool,
    ) -> RenderJob:
        candidate = score.candidate
        return RenderJob(
            candidate=candidate,
            output_path=output_path,
            captions_path=(
                caption_artifact.path if caption_artifact is not None else None
            ),
            metadata={
                "rank": rank,
                "overall_score": score.overall_score,
                "score_components": score.score_components,
                "score_rationale": score.rationale,
                "start_seconds": candidate.start_seconds,
                "end_seconds": candidate.end_seconds,
                "duration_seconds": candidate.end_seconds - candidate.start_seconds,
                "output_format": self._normalized_output_format(),
                "video_codec": (
                    "h264_qsv"
                    if self._config.renderer_backend is RendererBackend.INTEL_QSV
                    else self._config.video_codec
                ),
                "renderer_backend": self._config.renderer_backend.value,
                "audio_codec": self._config.audio_codec,
                "reused_existing": reused,
                "subtitles_burned_in": caption_artifact is not None,
            },
        )

    def _caption_artifacts(
        self,
        artifacts: Iterable[CaptionArtifact],
    ) -> dict[CandidateCaptionIdentity, CaptionArtifact]:
        indexed: dict[CandidateCaptionIdentity, CaptionArtifact] = {}
        for artifact in artifacts:
            identity = artifact.identity
            if identity in indexed:
                raise SubtitleRenderingError(
                    "Multiple caption artifacts share one source path and time window."
                )
            indexed[identity] = artifact
        return indexed

    def _caption_for_score(
        self,
        score: ClipScore,
        captions: dict[CandidateCaptionIdentity, CaptionArtifact],
    ) -> CaptionArtifact | None:
        if not self._config.burn_subtitles:
            return None
        identity = candidate_caption_identity(score.candidate)
        artifact = captions.get(identity)
        if artifact is None:
            raise SubtitleRenderingError(
                "No caption artifact matches the selected candidate source and window."
            )
        return artifact

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
        if not isinstance(config.renderer_backend, RendererBackend):
            raise InvalidRenderInputError(
                "Renderer backend must be a RendererBackend value."
            )
        if config.maximum_clips is not None and config.maximum_clips <= 0:
            raise InvalidRenderInputError("Maximum clips must be positive or None.")
        if not config.output_format.removeprefix("."):
            raise InvalidRenderInputError("Output format cannot be empty.")
        if not config.video_codec.strip() or not config.audio_codec.strip():
            raise InvalidRenderInputError("Video and audio codecs cannot be empty.")
        if not config.subtitle_character_encoding.strip():
            raise InvalidRenderInputError(
                "Subtitle character encoding cannot be empty."
            )

    def _require_requested_backend(self, ffmpeg_path: str) -> None:
        if (
            self._config.renderer_backend is RendererBackend.INTEL_QSV
            and not self._qsv_capability_checker(ffmpeg_path)
        ):
            raise IntelQSVUnavailableError(
                "Intel QSV device, driver, session, or encoder initialization failed."
            )

    @classmethod
    def _runtime_qsv_preflight(cls, ffmpeg_path: str) -> bool:
        cache_key = str(Path(ffmpeg_path).resolve(strict=False)).casefold()
        with cls._qsv_preflight_lock:
            if cache_key in cls._qsv_preflight_successes:
                return True
            command = [
                ffmpeg_path,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:size=64x64:rate=1:duration=1",
                "-frames:v",
                "1",
                "-an",
                "-c:v",
                "h264_qsv",
                "-preset",
                "medium",
                "-global_quality",
                "23",
                "-f",
                "null",
                "-",
            ]
            try:
                result = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                )
            except OSError as exc:
                raise ClipRenderingError(
                    f"Failed to execute Intel QSV runtime preflight: {exc}"
                ) from exc
            if result.returncode != 0:
                diagnostic = result.stderr.strip() or result.stdout.strip()
                if cls._is_qsv_initialization_failure(diagnostic):
                    return False
                detail = f": {diagnostic}" if diagnostic else ""
                raise ClipRenderingError(
                    "Intel QSV runtime preflight failed for a reason unrelated "
                    f"to device or encoder initialization{detail}"
                )
            cls._qsv_preflight_successes.add(cache_key)
            return True

    @staticmethod
    def _is_qsv_initialization_failure(diagnostic: str) -> bool:
        normalized = " ".join(diagnostic.casefold().split())
        return any(
            marker in normalized
            for marker in (
                "mfx",
                "quick sync",
                "qsv device",
                "qsv session",
                "h264_qsv encoder",
                "unknown encoder 'h264_qsv'",
                'unknown encoder "h264_qsv"',
                "no device available",
                "device creation failed",
                "device setup failed",
                "failed to initialise vaapi connection",
                "failed to initialize vaapi connection",
                "intel media driver",
            )
        )

    @staticmethod
    def _is_subtitle_failure(diagnostic: str) -> bool:
        normalized = diagnostic.casefold()
        return any(
            marker in normalized
            for marker in ("subtitle", "libass", "ass filter", "fontconfig")
        )

    def _normalized_output_format(self) -> str:
        return self._config.output_format.removeprefix(".")


def escape_subtitle_filter_path(path: Path) -> str:
    """Escape a subtitle filename for FFmpeg's filter-option parser."""

    normalized = str(path).replace("\\", "/")
    return escape_subtitle_filter_value(normalized)


def escape_subtitle_filter_value(value: str) -> str:
    """Escape characters significant inside a quoted FFmpeg filter value."""

    escaped = value.replace("\\", "\\\\")
    for character in (":", "'", ",", "[", "]", ";"):
        escaped = escaped.replace(character, f"\\{character}")
    return escaped
