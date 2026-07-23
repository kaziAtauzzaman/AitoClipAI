"""UI boundary for running the existing prerecorded pipeline safely."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import json
import math
import os
from pathlib import Path
import re
import shutil
import threading
import traceback
from typing import Any, Callable, Protocol
from urllib.parse import urlparse
from uuid import uuid4


OPERATOR_RUNS_DIRECTORY = Path("data") / "runs" / "operator"
SUPPORTED_LOCAL_MEDIA_SUFFIXES = frozenset(
    {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm", ".wmv"}
)
_SENSITIVE_VALUE = re.compile(
    r"(?i)(access[_ -]?token|refresh[_ -]?token|client[_ -]?secret|"
    r"authorization)(\s*[:=]\s*)([^\s,;]+)"
)


class SourceKind(str, Enum):
    """Accepted operator source forms."""

    YOUTUBE = "youtube"
    LOCAL = "local"


class PipelineStage(str, Enum):
    """Real coarse-grained stages exposed to the operator UI."""

    RESOLVING_SOURCE = "Resolving source"
    READING_MEDIA = "Downloading or reading media"
    EXTRACTING_AUDIO = "Extracting audio"
    OBSERVING = "Observing"
    GENERATING_CANDIDATES = "Generating candidates"
    SELECTING_CANDIDATES = "Selecting candidates"
    RENDERING_CLIPS = "Rendering clips"
    COMPLETED = "Completed"
    FAILED = "Failed"


class SourceValidationError(ValueError):
    """Raised before a run when the source is unsupported or unavailable."""


class RunInProgressError(RuntimeError):
    """Raised when a second pipeline run is requested concurrently."""


@dataclass(frozen=True, slots=True)
class ValidatedSource:
    """Normalized source accepted by the existing pipeline entry point."""

    kind: SourceKind
    value: str | Path


@dataclass(frozen=True, slots=True)
class PipelineExecution:
    """Minimum successful result returned by an injected pipeline runner."""

    output_directory: Path
    rendered_clip_count: int
    report_path: Path | None = None
    rendered_clips: tuple[RenderedClipOutput, ...] = ()


@dataclass(frozen=True, slots=True)
class RenderedClipOutput:
    """Immutable rendered output ready for optional upload orchestration."""

    path: Path
    identity: str
    title: str
    description: str


@dataclass(frozen=True, slots=True)
class PipelineRunSuccess:
    """Immutable UI-facing successful run completion."""

    run_directory: Path
    log_path: Path
    output_directory: Path
    rendered_clip_count: int
    report_path: Path | None = None
    rendered_clips: tuple[RenderedClipOutput, ...] = ()


@dataclass(frozen=True, slots=True)
class PipelineRunFailure:
    """Immutable UI-facing failed run completion."""

    message: str
    run_directory: Path | None
    log_path: Path | None


StageCallback = Callable[[PipelineStage], None]
SuccessCallback = Callable[[PipelineRunSuccess], None]
FailureCallback = Callable[[PipelineRunFailure], None]


class PipelineRunner(Protocol):
    """Injectable boundary around one existing pipeline invocation."""

    def run(
        self,
        source: ValidatedSource,
        run_directory: Path,
        emit_stage: StageCallback,
    ) -> PipelineExecution:
        """Run the pipeline and return its output summary."""


def validate_source(source: str | Path) -> ValidatedSource:
    """Validate one YouTube URL or supported local video path."""

    raw = str(source).strip()
    if not raw:
        raise SourceValidationError("Enter a YouTube URL or local media path.")
    parsed = urlparse(raw)
    if parsed.scheme.lower() in {"http", "https"}:
        if parsed.username is not None or parsed.password is not None:
            raise SourceValidationError("Source URLs must not contain credentials.")
        hostname = (parsed.hostname or "").lower().rstrip(".")
        if not _is_youtube_hostname(hostname):
            raise SourceValidationError("Only YouTube URLs are supported remotely.")
        return ValidatedSource(SourceKind.YOUTUBE, raw)
    if "://" in raw:
        raise SourceValidationError("Only HTTP or HTTPS YouTube URLs are supported.")

    path = Path(raw).expanduser().resolve(strict=False)
    if not path.is_file():
        raise SourceValidationError(f"Local media file does not exist: {path}")
    if path.suffix.lower() not in SUPPORTED_LOCAL_MEDIA_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_LOCAL_MEDIA_SUFFIXES))
        raise SourceValidationError(
            f"Unsupported local media type {path.suffix or '(none)'}. "
            f"Supported types: {supported}."
        )
    return ValidatedSource(SourceKind.LOCAL, path)


def _is_youtube_hostname(hostname: str) -> bool:
    return (
        hostname == "youtu.be"
        or hostname == "youtube.com"
        or hostname.endswith(".youtube.com")
    )


def default_run_directory() -> Path:
    """Return one collision-resistant operator run directory path."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return OPERATOR_RUNS_DIRECTORY / f"{timestamp}-{uuid4().hex[:8]}"


class ProductionPipelineRunner:
    """Lazy source adapter for the exported production incremental pipeline."""

    def run(
        self,
        source: ValidatedSource,
        run_directory: Path,
        emit_stage: StageCallback,
    ) -> PipelineExecution:
        # Keep expensive media and model modules out of UI import/startup.
        from audio_observer import (
            FFmpegAudioExtractor,
            FFmpegAudioExtractorConfig,
            IncrementalWavAudioObserver,
        )
        from candidate_generation import CandidateGenerator
        from candidate_scoring import CandidateScorer
        from candidate_selection import CandidateSelector
        from clip_rendering import ClipRenderer, ClipRendererConfig
        from downloader import DownloaderConfig, VideoDownloader
        from observers import ObserverContext
        from pipeline import (
            ArtifactValidator,
            JsonProductionIncrementalReportWriter,
            ProductionIncrementalOrchestrator,
        )
        from whisper_observer import IncrementalWavWhisperObserver

        artifact_validator = ArtifactValidator()
        if source.kind is SourceKind.YOUTUBE:
            download = VideoDownloader(
                DownloaderConfig(
                    downloads_dir=run_directory / "downloads",
                    overwrite_existing=False,
                )
            ).download(str(source.value))
            source_video = download.video_path.resolve(strict=True)
        else:
            source_video = _stage_local_source(
                Path(source.value),
                run_directory / "source",
                artifact_validator,
            )

        emit_stage(PipelineStage.EXTRACTING_AUDIO)
        extracted_audio = FFmpegAudioExtractor(
            FFmpegAudioExtractorConfig(
                output_dir=run_directory / "audio",
                overwrite_existing=False,
            )
        ).extract(
            ObserverContext(
                source_path=source_video,
                metadata={"input_type": source.kind.value},
            )
        )
        stage_reporter = _OneShotStageReporter(emit_stage)
        clips_directory = run_directory / "clips"
        clips_directory.mkdir(parents=True, exist_ok=True)
        orchestrator = ProductionIncrementalOrchestrator(
            _IncrementalRendererWithStage(
                ClipRenderer(
                    ClipRendererConfig(
                        output_dir=clips_directory,
                        overwrite_existing=False,
                        maximum_clips=None,
                        burn_subtitles=False,
                    )
                ),
                stage_reporter,
            ),
            session_id=f"operator-{run_directory.name}",
            audio_observer=_ObserverWithStage(
                IncrementalWavAudioObserver(),
                stage_reporter,
            ),
            whisper_observer=_ObserverWithStage(
                IncrementalWavWhisperObserver(),
                stage_reporter,
            ),
            artifact_validator=artifact_validator,
            candidate_generator=_IncrementalGeneratorWithStage(
                CandidateGenerator(),
                stage_reporter,
            ),
            candidate_scorer=CandidateScorer(),
            candidate_selector=_SelectorWithStage(
                CandidateSelector(),
                stage_reporter,
            ),
        )
        report = orchestrator.run(source_video, extracted_audio.path)
        report_path = JsonProductionIncrementalReportWriter(
            run_directory / "reports" / "production-incremental-report.json"
        ).write(report)
        if report.status != "completed":
            raise RuntimeError(_production_failure_message(report))
        rendered_clips = _rendered_clip_outputs(report)
        return PipelineExecution(
            output_directory=clips_directory,
            rendered_clip_count=len(rendered_clips),
            report_path=report_path,
            rendered_clips=rendered_clips,
        )


def _stage_local_source(
    source: Path,
    destination_directory: Path,
    artifact_validator: Any,
) -> Path:
    """Place a local source and authoritative duration inside its run."""

    source = source.resolve(strict=True)
    destination_directory.mkdir(parents=True, exist_ok=True)
    destination = destination_directory / source.name
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    probe = artifact_validator.probe_source(destination)
    duration = _positive_duration(probe.duration_seconds)
    if duration is None:
        stream_durations = [
            value
            for stream in probe.streams
            if (value := _positive_duration(stream.duration_seconds)) is not None
        ]
        if not stream_durations:
            raise RuntimeError("Local source has no authoritative positive duration.")
        duration = max(stream_durations)
    metadata_path = destination.with_name(f"{destination.name}.metadata.json")
    metadata_path.write_text(
        json.dumps({"duration": duration}, sort_keys=True),
        encoding="utf-8",
    )
    return destination


def _positive_duration(value: object) -> float | None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        return None
    return float(value)


class _OneShotStageReporter:
    def __init__(self, emit_stage: StageCallback) -> None:
        self._emit_stage = emit_stage
        self._reported: set[PipelineStage] = set()

    def emit(self, stage: PipelineStage) -> None:
        if stage in self._reported:
            return
        self._reported.add(stage)
        self._emit_stage(stage)


class _StageProxy:
    def __init__(self, delegate: Any, reporter: _OneShotStageReporter) -> None:
        self._delegate = delegate
        self._reporter = reporter

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


class _ObserverWithStage(_StageProxy):
    def session(self, source: Path) -> Any:
        self._reporter.emit(PipelineStage.OBSERVING)
        return self._delegate.session(source)


class _IncrementalGeneratorWithStage(_StageProxy):
    def advance_incremental(self, *args: Any, **kwargs: Any) -> Any:
        self._reporter.emit(PipelineStage.GENERATING_CANDIDATES)
        return self._delegate.advance_incremental(*args, **kwargs)

    def finalize_incremental(self, *args: Any, **kwargs: Any) -> Any:
        self._reporter.emit(PipelineStage.GENERATING_CANDIDATES)
        return self._delegate.finalize_incremental(*args, **kwargs)


class _SelectorWithStage(_StageProxy):
    def select(self, scores: Any) -> Any:
        self._reporter.emit(PipelineStage.SELECTING_CANDIDATES)
        return self._delegate.select(scores)


class _IncrementalRendererWithStage(_StageProxy):
    def render_one(self, score: Any, identity: int) -> Any:
        self._reporter.emit(PipelineStage.RENDERING_CLIPS)
        return self._delegate.render_one(score, identity)

    def recover_render(self, score: Any, identity: int) -> Any:
        self._reporter.emit(PipelineStage.RENDERING_CLIPS)
        return self._delegate.recover_render(score, identity)


def _production_failure_message(report: Any) -> str:
    failures = [
        *report.observer_failures,
        *report.render_failures,
        *report.artifact_validation_failures,
    ]
    if failures:
        failure = failures[0]
        return (
            f"Production pipeline ended with status {report.status}: "
            f"{failure.phase}/{failure.error_type}: {failure.message}"
        )
    return f"Production pipeline ended with status {report.status}."


def _rendered_clip_outputs(report: Any) -> tuple[RenderedClipOutput, ...]:
    session_id = getattr(report, "session_id", None)
    if not isinstance(session_id, str) or not session_id.strip():
        raise RuntimeError("Production report has no render session identity.")
    outputs: list[RenderedClipOutput] = []
    identities: set[int] = set()
    for job in report.render_jobs:
        render_identity = job.metadata.get("incremental_render_identity")
        if (
            isinstance(render_identity, bool)
            or not isinstance(render_identity, int)
            or render_identity <= 0
            or render_identity in identities
        ):
            raise RuntimeError(
                "Production report has a missing or duplicate render identity."
            )
        identities.add(render_identity)
        candidate_title = job.candidate.title
        title = (
            candidate_title.strip()
            if isinstance(candidate_title, str) and candidate_title.strip()
            else f"{job.candidate.source_video_path.stem} — clip {render_identity}"
        )
        outputs.append(
            RenderedClipOutput(
                path=Path(job.output_path),
                identity=(
                    f"render:{session_id.strip()}:identity-{render_identity}"
                ),
                title=title[:100].rstrip(),
                description=job.candidate.reason,
            )
        )
    return tuple(outputs)


class OperatorPipelineController:
    """Own one background pipeline run and its UI-safe result callbacks."""

    def __init__(
        self,
        runner: PipelineRunner | None = None,
        *,
        run_directory_factory: Callable[[], Path] = default_run_directory,
    ) -> None:
        self._runner = runner or ProductionPipelineRunner()
        self._run_directory_factory = run_directory_factory
        self._lock = threading.Lock()
        self._active_thread: threading.Thread | None = None

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._active_thread is not None

    def start(
        self,
        source: str | Path | ValidatedSource,
        *,
        on_stage: StageCallback,
        on_success: SuccessCallback,
        on_failure: FailureCallback,
    ) -> threading.Thread:
        """Validate and start exactly one non-daemon background run."""

        validated = (
            source if isinstance(source, ValidatedSource) else validate_source(source)
        )
        with self._lock:
            if self._active_thread is not None:
                raise RunInProgressError("A pipeline run is already active.")
            thread = threading.Thread(
                target=self._execute,
                args=(validated, on_stage, on_success, on_failure),
                name="aitoclip-operator-pipeline",
                daemon=False,
            )
            self._active_thread = thread
            try:
                thread.start()
            except BaseException:
                self._active_thread = None
                raise
            return thread

    def _execute(
        self,
        source: ValidatedSource,
        on_stage: StageCallback,
        on_success: SuccessCallback,
        on_failure: FailureCallback,
    ) -> None:
        run_directory: Path | None = None
        log_path: Path | None = None
        try:
            run_directory = Path(self._run_directory_factory()).resolve(strict=False)
            run_directory.mkdir(parents=True, exist_ok=False)
            log_path = run_directory / "run.log"

            def emit(stage: PipelineStage) -> None:
                _append_log(log_path, f"stage={stage.value}")
                on_stage(stage)

            emit(PipelineStage.RESOLVING_SOURCE)
            emit(PipelineStage.READING_MEDIA)
            execution = self._runner.run(source, run_directory, emit)
            emit(PipelineStage.COMPLETED)
            _append_log(
                log_path,
                f"rendered_clip_count={execution.rendered_clip_count}",
            )
            on_success(
                PipelineRunSuccess(
                    run_directory=run_directory,
                    log_path=log_path,
                    output_directory=execution.output_directory,
                    rendered_clip_count=execution.rendered_clip_count,
                    report_path=execution.report_path,
                    rendered_clips=execution.rendered_clips,
                )
            )
        except BaseException as exc:
            details = _redact_sensitive(traceback.format_exc())
            if log_path is not None:
                _append_log(log_path, details)
            try:
                on_stage(PipelineStage.FAILED)
            finally:
                on_failure(
                    PipelineRunFailure(
                        message=_concise_failure(exc),
                        run_directory=run_directory,
                        log_path=log_path,
                    )
                )
        finally:
            with self._lock:
                if self._active_thread is threading.current_thread():
                    self._active_thread = None


def _append_log(path: Path, message: str) -> None:
    try:
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(f"{message.rstrip()}\n")
    except OSError:
        # Logging failure must not replace the real pipeline outcome.
        return


def _concise_failure(error: BaseException) -> str:
    message = _redact_sensitive(str(error).strip())
    if not message:
        message = type(error).__name__
    if len(message) > 240:
        message = f"{message[:237]}..."
    return f"{type(error).__name__}: {message}"


def _redact_sensitive(value: str) -> str:
    return _SENSITIVE_VALUE.sub(r"\1\2[redacted]", value)
