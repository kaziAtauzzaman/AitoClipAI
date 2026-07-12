"""FFprobe-backed rendered-artifact validation and report persistence."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable, Protocol, Sequence

from core import FeatureTimeline, RenderJob
from pipeline.contracts import (
    MediaProbeResult,
    MediaStreamProbe,
    PipelineValidationReport,
    RenderedArtifactValidation,
)
from pipeline.errors import ArtifactValidationError, MediaProbeError, PipelineError


@dataclass(frozen=True, slots=True)
class ArtifactValidationConfig:
    """Playback validation tolerances and FFprobe executable settings."""

    ffprobe_binary: str = "ffprobe"
    maximum_start_offset_seconds: float = 0.05
    maximum_duration_difference_seconds: float = 0.08


class ProbeCommandRunner(Protocol):
    """Execute FFprobe and return captured diagnostics."""

    def run(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        """Run a command without raising for a non-zero result."""


class SubprocessProbeCommandRunner:
    """Probe command runner backed by the standard subprocess module."""

    def run(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                list(command),
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            raise MediaProbeError(f"Failed to execute FFprobe: {exc}") from exc


class MediaProbe(Protocol):
    """Inspect a local media file and return normalized stream metadata."""

    def probe(self, path: Path) -> MediaProbeResult:
        """Probe one media file."""


class FFprobeMediaProbe:
    """Dependency-injected adapter around FFprobe JSON output."""

    def __init__(
        self,
        config: ArtifactValidationConfig | None = None,
        runner: ProbeCommandRunner | None = None,
        executable_locator: Callable[[str], str | None] | None = None,
    ) -> None:
        self._config = config or ArtifactValidationConfig()
        self._runner = runner or SubprocessProbeCommandRunner()
        self._executable_locator = executable_locator or shutil.which

    def probe(self, path: Path) -> MediaProbeResult:
        ffprobe_path = self._executable_locator(self._config.ffprobe_binary)
        if ffprobe_path is None:
            raise MediaProbeError(
                f"FFprobe executable was not found: {self._config.ffprobe_binary!r}."
            )
        if not path.is_file():
            raise MediaProbeError(f"Media artifact does not exist: {path}")
        command = [
            ffprobe_path,
            "-v",
            "error",
            "-show_entries",
            "stream=index,codec_type,codec_name,start_time,duration:"
            "format=format_name,duration",
            "-of",
            "json",
            str(path),
        ]
        result = self._runner.run(command)
        if result.returncode != 0:
            diagnostic = result.stderr.strip() or result.stdout.strip()
            detail = f": {diagnostic}" if diagnostic else ""
            raise MediaProbeError(
                f"FFprobe failed with exit code {result.returncode}{detail}"
            )
        try:
            raw = json.loads(result.stdout)
        except (TypeError, json.JSONDecodeError) as exc:
            raise MediaProbeError(f"FFprobe returned invalid JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise MediaProbeError("FFprobe returned an unexpected JSON value.")
        raw_streams = raw.get("streams", [])
        if not isinstance(raw_streams, list):
            raise MediaProbeError("FFprobe streams must be a list.")
        streams = [self._stream(item) for item in raw_streams]
        raw_format = raw.get("format", {})
        format_data = raw_format if isinstance(raw_format, dict) else {}
        return MediaProbeResult(
            path=path,
            format_name=_optional_string(format_data.get("format_name")),
            duration_seconds=_optional_float(format_data.get("duration")),
            streams=streams,
        )

    def _stream(self, raw: object) -> MediaStreamProbe:
        if not isinstance(raw, dict):
            raise MediaProbeError("FFprobe returned an invalid stream entry.")
        codec_type = raw.get("codec_type")
        if not isinstance(codec_type, str) or not codec_type:
            raise MediaProbeError("FFprobe stream is missing its codec type.")
        return MediaStreamProbe(
            codec_type=codec_type,
            codec_name=_optional_string(raw.get("codec_name")),
            start_seconds=_optional_float(raw.get("start_time")),
            duration_seconds=_optional_float(raw.get("duration")),
            metadata={"index": raw.get("index")},
        )


class ArtifactValidator:
    """Enforce playback requirements for every rendered clip."""

    def __init__(
        self,
        probe: MediaProbe | None = None,
        config: ArtifactValidationConfig | None = None,
    ) -> None:
        self._config = config or ArtifactValidationConfig()
        self._probe = probe or FFprobeMediaProbe(self._config)
        self._validate_config()

    def probe_source(self, path: Path) -> MediaProbeResult:
        """Collect source metadata without fabricating a downloader result."""

        return self._probe.probe(path)

    def validate_jobs(
        self,
        jobs: list[RenderJob],
    ) -> list[RenderedArtifactValidation]:
        """Validate every rendered job or raise at the first invalid artifact."""

        return [self._validate_job(job) for job in jobs]

    def _validate_job(self, job: RenderJob) -> RenderedArtifactValidation:
        path = job.output_path
        if not path.is_file():
            raise ArtifactValidationError(f"Rendered clip does not exist: {path}")
        size = path.stat().st_size
        if size <= 0:
            raise ArtifactValidationError(f"Rendered clip is empty: {path}")
        probe = self._probe.probe(path)
        video = _one_stream(probe, "video")
        audio = _one_stream(probe, "audio")
        video_duration = _positive_duration(video, probe)
        audio_duration = _positive_duration(audio, probe)
        video_start = _required_start(video)
        audio_start = _required_start(audio)
        if abs(video_start) > self._config.maximum_start_offset_seconds:
            raise ArtifactValidationError(
                f"Video stream does not start near zero: {video_start:.6f}s"
            )
        if abs(audio_start) > self._config.maximum_start_offset_seconds:
            raise ArtifactValidationError(
                f"Audio stream does not start near zero: {audio_start:.6f}s"
            )
        difference = abs(video_duration - audio_duration)
        if difference > self._config.maximum_duration_difference_seconds:
            raise ArtifactValidationError(
                "Audio/video duration difference exceeds tolerance: "
                f"{difference:.6f}s"
            )
        return RenderedArtifactValidation(
            path=path,
            size_bytes=size,
            video_stream=video,
            audio_stream=audio,
            duration_seconds=max(video_duration, audio_duration),
            checks={
                "exists": True,
                "nonempty": True,
                "video_stream": True,
                "audio_stream": True,
                "positive_duration": True,
                "starts_near_zero": True,
                "durations_synchronized": True,
            },
        )

    def _validate_config(self) -> None:
        if self._config.maximum_start_offset_seconds < 0:
            raise ArtifactValidationError("Start offset tolerance cannot be negative.")
        if self._config.maximum_duration_difference_seconds < 0:
            raise ArtifactValidationError("Duration tolerance cannot be negative.")


class ValidationReportWriter(Protocol):
    """Persist one deterministic validation report."""

    def write(self, report: PipelineValidationReport) -> Path:
        """Write a report and return its path."""


class JsonValidationReportWriter:
    """Write a stable JSON validation report to a configured run path."""

    def __init__(self, report_path: Path) -> None:
        self._report_path = report_path

    def write(self, report: PipelineValidationReport) -> Path:
        try:
            self._report_path.parent.mkdir(parents=True, exist_ok=True)
            self._report_path.write_text(
                json.dumps(asdict(report), default=str, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except (OSError, TypeError, ValueError) as exc:
            raise PipelineError(f"Failed to write validation report: {exc}") from exc
        return self._report_path


def build_validation_report(
    *,
    timeline: FeatureTimeline,
    required_observers: list[str],
    candidates_count: int,
    scores_count: int,
    passing_scores_count: int,
    source_metadata: MediaProbeResult,
    rendered_artifacts: list[RenderedArtifactValidation],
) -> PipelineValidationReport:
    """Build the deterministic success report after all strict checks pass."""

    observed = [result.observer for result in timeline.timeline.observer_results]
    failures = [
        {
            "observer": failure.observer,
            "error_type": failure.error_type,
            "message": failure.message,
            "metadata": failure.metadata,
        }
        for failure in timeline.failures
    ]
    return PipelineValidationReport(
        status="passed",
        source_type=str(timeline.metadata.get("input_type", "unknown")),
        source_path=timeline.media_path,
        source_metadata=source_metadata,
        timeline_path=timeline.timeline_path,
        required_observers=required_observers,
        observed_observers=observed,
        observer_failures=failures,
        candidate_count=candidates_count,
        score_count=scores_count,
        passing_score_count=passing_scores_count,
        rendered_artifacts=rendered_artifacts,
        checks={
            "required_observers": True,
            "candidates_generated": True,
            "passing_scores": True,
            "rendered_artifacts": True,
        },
    )


def _one_stream(probe: MediaProbeResult, codec_type: str) -> MediaStreamProbe:
    matches = [stream for stream in probe.streams if stream.codec_type == codec_type]
    if not matches:
        raise ArtifactValidationError(
            f"Rendered clip has no {codec_type} stream: {probe.path}"
        )
    return matches[0]


def _positive_duration(
    stream: MediaStreamProbe,
    probe: MediaProbeResult,
) -> float:
    duration = (
        stream.duration_seconds
        if stream.duration_seconds is not None
        else probe.duration_seconds
    )
    if duration is None or duration <= 0:
        raise ArtifactValidationError(
            f"{stream.codec_type.capitalize()} stream has no positive duration."
        )
    return duration


def _required_start(stream: MediaStreamProbe) -> float:
    if stream.start_seconds is None:
        raise ArtifactValidationError(
            f"{stream.codec_type.capitalize()} stream has no start time."
        )
    return stream.start_seconds


def _optional_string(value: object) -> str | None:
    return str(value) if value is not None else None


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
