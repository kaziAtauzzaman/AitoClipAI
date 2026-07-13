"""Production composition of prerecorded incremental Audio and Whisper."""

from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Protocol

from audio_observer import IncrementalAudioBatch, IncrementalWavAudioObserver
from candidate_generation import CandidateGenerator
from candidate_scoring import CandidateScorer
from candidate_selection import CandidateSelector, SuppressedCandidate
from core import (
    AggregatedTimeline,
    ClipScore,
    FeatureTimeline,
    Observation,
    ObserverResult,
    RenderJob,
    TimelineGroup,
)
from pipeline.contracts import RenderedArtifactValidation
from pipeline.incremental import (
    IncrementalEOF,
    IncrementalPipelineConfig,
    IncrementalPrerecordedCoordinator,
    ObserverWatermarks,
)
from pipeline.validation import ArtifactValidator
from whisper_observer import IncrementalWavWhisperObserver, IncrementalWhisperBatch


class IncrementalAudioSession(Protocol):
    def read_batch(self) -> IncrementalAudioBatch | None: ...
    def close(self) -> None: ...


class IncrementalWhisperSession(Protocol):
    def read_batch(self) -> IncrementalWhisperBatch | None: ...
    def close(self) -> None: ...


class AudioSessionFactory(Protocol):
    def session(self, source: Path) -> IncrementalAudioSession: ...


class WhisperSessionFactory(Protocol):
    def session(self, source: Path) -> IncrementalWhisperSession: ...


class IncrementalRenderer(Protocol):
    def render_one(self, score: ClipScore, identity: int) -> RenderJob: ...


class IncrementalArtifactValidator(Protocol):
    def validate_jobs(self, jobs: list[RenderJob]) -> list[RenderedArtifactValidation]: ...


class ProductionIncrementalLifecycle(str, Enum):
    NEW = "new"
    ACTIVE = "active"
    FINISHED = "finished"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class TimedObserverBatch:
    operation_identity: str
    observer: str
    sequence: int
    elapsed_seconds: float
    watermark_seconds: float
    eof: bool
    succeeded: bool


@dataclass(frozen=True, slots=True)
class TimedOperation:
    operation: str
    operation_identity: str
    elapsed_seconds: float
    succeeded: bool


@dataclass(frozen=True, slots=True)
class IncrementalFailure:
    phase: str
    error_type: str
    message: str
    render_identity: int | None = None
    output_path: Path | None = None


@dataclass(slots=True)
class ProductionIncrementalReport:
    status: str
    source_video: Path
    audio_source: Path
    source_id: str | None = None
    observations: list[Observation] = field(default_factory=list)
    selected_scores: list[ClipScore] = field(default_factory=list)
    suppressed: list[SuppressedCandidate] = field(default_factory=list)
    render_jobs: list[RenderJob] = field(default_factory=list)
    render_failures: list[IncrementalFailure] = field(default_factory=list)
    artifact_validations: list[RenderedArtifactValidation] = field(default_factory=list)
    artifact_validation_failures: list[IncrementalFailure] = field(default_factory=list)
    observer_failures: list[IncrementalFailure] = field(default_factory=list)
    observer_timings: list[TimedObserverBatch] = field(default_factory=list)
    coordinator_timings: list[TimedOperation] = field(default_factory=list)
    render_timings: list[TimedOperation] = field(default_factory=list)
    validation_timings: list[TimedOperation] = field(default_factory=list)
    watermarks: dict[str, float] = field(default_factory=dict)
    total_wall_seconds: float = 0.0

    def to_dict(self) -> dict[str, object]:
        """Return strict canonical JSON-compatible report data."""

        value = _canonical_report_value(self)
        assert isinstance(value, dict)
        return value


class JsonProductionIncrementalReportWriter:
    """Persist one canonical production-incremental report."""

    def __init__(self, report_path: Path) -> None:
        self._report_path = Path(report_path)

    def write(self, report: ProductionIncrementalReport) -> Path:
        payload = report.to_dict()
        self._report_path.parent.mkdir(parents=True, exist_ok=True)
        self._report_path.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        return self._report_path


class _InstrumentedRenderer:
    def __init__(
        self,
        renderer: IncrementalRenderer,
        clock: Callable[[], float],
        report: ProductionIncrementalReport,
    ) -> None:
        self._renderer = renderer
        self._clock = clock
        self._report = report

    def render_one(self, score: ClipScore, identity: int) -> RenderJob:
        started = self._clock()
        succeeded = False
        try:
            job = self._renderer.render_one(score, identity)
            succeeded = True
            return replace(
                job,
                metadata={**job.metadata, "incremental_render_identity": identity},
            )
        except BaseException as exc:
            self._report.render_failures.append(
                IncrementalFailure("render", type(exc).__name__, str(exc), identity)
            )
            raise
        finally:
            self._report.render_timings.append(
                TimedOperation(
                    "render",
                    f"render:{identity}",
                    max(0.0, self._clock() - started),
                    succeeded,
                )
            )


class ProductionIncrementalOrchestrator:
    """Single-use orchestration for two real incremental WAV observers."""

    def __init__(
        self,
        renderer: IncrementalRenderer,
        *,
        audio_observer: AudioSessionFactory | None = None,
        whisper_observer: WhisperSessionFactory | None = None,
        artifact_validator: IncrementalArtifactValidator | None = None,
        candidate_generator=None,
        candidate_scorer=None,
        candidate_selector=None,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self._renderer = renderer
        self._audio = audio_observer or IncrementalWavAudioObserver()
        self._whisper = whisper_observer or IncrementalWavWhisperObserver()
        self._validator = artifact_validator or ArtifactValidator()
        self._generator = candidate_generator or CandidateGenerator()
        self._scorer = candidate_scorer or CandidateScorer()
        self._selector = candidate_selector or CandidateSelector()
        self._clock = clock
        self._lifecycle = ProductionIncrementalLifecycle.NEW
        self._validation_attempts: set[int] = set()

    @property
    def lifecycle(self) -> ProductionIncrementalLifecycle:
        return self._lifecycle

    def run(self, source_video: Path, audio_source: Path) -> ProductionIncrementalReport:
        if self._lifecycle is not ProductionIncrementalLifecycle.NEW:
            raise RuntimeError("Production incremental orchestrator is single-use.")
        source_video = Path(source_video)
        audio_source = Path(audio_source)
        if not source_video.is_file() or not audio_source.is_file():
            raise ValueError("Source video and extracted WAV must both exist.")
        report = ProductionIncrementalReport("running", source_video, audio_source)
        started = self._clock()
        coordinator: IncrementalPrerecordedCoordinator | None = None
        observations: dict[str, list[Observation]] = {"audio": [], "whisper": []}
        metadata: dict[str, dict[str, object]] = {"audio": {}, "whisper": {}}
        watermarks = {"audio": 0.0, "whisper": 0.0}
        eof_durations: dict[str, float] = {}
        sequences = {"audio": 0, "whisper": 0}
        self._lifecycle = ProductionIncrementalLifecycle.ACTIVE
        try:
            source_id = _source_id(source_video)
            report.source_id = source_id
            renderer = _InstrumentedRenderer(self._renderer, self._clock, report)
            coordinator = IncrementalPrerecordedCoordinator(
                self._generator,
                self._scorer,
                self._selector,
                renderer,
                IncrementalPipelineConfig(required_observers=("audio", "whisper")),
            )
            with ExitStack() as stack:
                audio_session = stack.enter_context(_closing(self._audio.session(audio_source)))
                whisper_session = stack.enter_context(
                    _closing(self._whisper.session(audio_source))
                )
                sessions = {"audio": audio_session, "whisper": whisper_session}
                completed: set[str] = set()
                while len(completed) < 2:
                    for observer in ("audio", "whisper"):
                        if observer in completed:
                            continue
                        batch = self._read_batch(
                            observer, sessions[observer], sequences[observer], report
                        )
                        sequences[observer] += 1
                        if batch is None:
                            raise RuntimeError(
                                f"{observer} session ended without authoritative EOF."
                            )
                        if batch.observer != observer:
                            raise RuntimeError(
                                f"Expected {observer} batch, received {batch.observer}."
                            )
                        observations[observer].extend(batch.observations)
                        metadata[observer].update(batch.metadata)
                        previous = watermarks[observer]
                        if batch.watermark_seconds < previous:
                            raise RuntimeError(f"{observer} watermark moved backwards.")
                        watermarks[observer] = batch.watermark_seconds
                        report.watermarks = dict(watermarks)
                        if batch.eof:
                            completed.add(observer)
                            eof_durations[observer] = _authoritative_eof_duration(
                                observer, batch
                            )
                        timeline = _timeline(
                            source_video, audio_source, source_id, observations, metadata
                        )
                        self._advance(coordinator, timeline, watermarks, report)
                duration = eof_durations["audio"]
                if not math.isclose(
                    duration, eof_durations["whisper"], rel_tol=0.0, abs_tol=1e-9
                ):
                    raise RuntimeError("Audio and Whisper EOF durations do not match.")
                timeline = _timeline(
                    source_video, audio_source, source_id, observations, metadata
                )
                self._flush(coordinator, timeline, watermarks, duration, report)
            self._lifecycle = ProductionIncrementalLifecycle.FINISHED
        except BaseException as exc:
            if not report.render_failures or report.render_failures[-1].message != str(exc):
                report.observer_failures.append(
                    IncrementalFailure("orchestration", type(exc).__name__, str(exc))
                )
            self._lifecycle = ProductionIncrementalLifecycle.FAILED
        finally:
            result = coordinator.result if coordinator is not None else None
            report.observations = sorted(
                observations["audio"] + observations["whisper"],
                key=lambda item: (item.timestamp_seconds, item.observer, item.type),
            )
            if result is not None:
                report.selected_scores = result.selected_scores
                report.suppressed = result.suppressed
                report.render_jobs = result.render_jobs
            report.total_wall_seconds = max(0.0, self._clock() - started)
            if self._lifecycle is ProductionIncrementalLifecycle.FINISHED:
                report.status = (
                    "completed_with_failures"
                    if report.render_failures or report.artifact_validation_failures
                    else "completed"
                )
            else:
                report.status = "failed"
        return report

    def _read_batch(self, observer, session, sequence, report):
        started = self._clock()
        batch = None
        succeeded = False
        try:
            batch = session.read_batch()
            succeeded = True
            return batch
        finally:
            elapsed = max(0.0, self._clock() - started)
            watermark = 0.0 if batch is None else batch.watermark_seconds
            eof = False if batch is None else batch.eof
            report.observer_timings.append(
                TimedObserverBatch(
                    f"{observer}:{sequence}",
                    observer,
                    sequence,
                    elapsed,
                    watermark,
                    eof,
                    succeeded,
                )
            )

    def _advance(self, coordinator, timeline, watermarks, report) -> None:
        started = self._clock()
        sequence = len(report.coordinator_timings)
        render_failure_count = len(report.render_failures)
        succeeded = False
        try:
            coordinator.advance(timeline, ObserverWatermarks(dict(watermarks)))
            succeeded = True
        except BaseException:
            if len(report.render_failures) > render_failure_count:
                self._validate(self._unvalidated_jobs(coordinator, report), report)
                return
            raise
        finally:
            report.coordinator_timings.append(
                TimedOperation(
                    "coordinator_advance",
                    f"advance:{sequence}",
                    max(0.0, self._clock() - started),
                    succeeded,
                )
            )
        self._validate(self._unvalidated_jobs(coordinator, report), report)

    def _flush(self, coordinator, timeline, watermarks, duration, report) -> None:
        started = self._clock()
        sequence = len(report.coordinator_timings)
        succeeded = False
        try:
            coordinator.flush(
                timeline,
                IncrementalEOF(duration, ObserverWatermarks(dict(watermarks))),
            )
            succeeded = True
        finally:
            report.coordinator_timings.append(
                TimedOperation(
                    "coordinator_flush",
                    f"flush:{sequence}",
                    max(0.0, self._clock() - started),
                    succeeded,
                )
            )
            self._validate(self._unvalidated_jobs(coordinator, report), report)

    def _unvalidated_jobs(self, coordinator, report) -> list[RenderJob]:
        return [
            job
            for job in coordinator.result.render_jobs
            if _render_identity(job) not in self._validation_attempts
        ]

    def _validate(self, jobs: list[RenderJob], report: ProductionIncrementalReport) -> None:
        for job in jobs:
            identity = _render_identity(job)
            if identity in self._validation_attempts:
                continue
            self._validation_attempts.add(identity)
            started = self._clock()
            succeeded = False
            try:
                results = self._validator.validate_jobs([job])
                validation = _validated_artifact(job, results)
                report.artifact_validations.append(validation)
                succeeded = True
            except BaseException as exc:
                report.artifact_validation_failures.append(
                    IncrementalFailure(
                        "artifact_validation",
                        type(exc).__name__,
                        str(exc),
                        render_identity=identity,
                        output_path=job.output_path,
                    )
                )
            finally:
                report.validation_timings.append(
                    TimedOperation(
                        "artifact_validation",
                        f"validation:{identity}",
                        max(0.0, self._clock() - started),
                        succeeded,
                    )
                )


class _closing:
    def __init__(self, value) -> None:
        self._value = value

    def __enter__(self):
        return self._value

    def __exit__(self, *_: object) -> None:
        self._value.close()


def _authoritative_eof_duration(observer: str, batch) -> float:
    sample_rate = batch.metadata.get("sample_rate_hz")
    if (
        isinstance(sample_rate, bool)
        or not isinstance(sample_rate, (int, float))
        or not math.isfinite(float(sample_rate))
        or float(sample_rate) <= 0
    ):
        raise RuntimeError(f"{observer} EOF requires a positive sample_rate_hz.")
    duration = batch.frames_processed / float(sample_rate)
    if not math.isfinite(duration) or duration < 0:
        raise RuntimeError(f"{observer} EOF duration is invalid.")
    if not math.isclose(
        batch.watermark_seconds, duration, rel_tol=0.0, abs_tol=1e-9
    ):
        raise RuntimeError(
            f"{observer} EOF watermark does not match its frame-derived duration."
        )
    return duration


def _render_identity(job: RenderJob) -> int:
    identity = job.metadata.get("incremental_render_identity")
    if isinstance(identity, bool) or not isinstance(identity, int) or identity <= 0:
        raise RuntimeError("Incremental render job is missing its stable identity.")
    return identity


def _validated_artifact(job: RenderJob, results: object) -> RenderedArtifactValidation:
    if not isinstance(results, list) or len(results) != 1:
        raise RuntimeError("Artifact validator must return exactly one result per job.")
    result = results[0]
    if not isinstance(result, RenderedArtifactValidation):
        raise RuntimeError("Artifact validator returned a malformed result.")
    if _portable_path(result.path) != _portable_path(job.output_path):
        raise RuntimeError("Artifact validator result does not match the render job.")
    return result


def _timeline(media, audio, source_id, observations, metadata) -> FeatureTimeline:
    results = [
        ObserverResult(name, list(observations[name]), dict(metadata[name]))
        for name in ("audio", "whisper")
    ]
    combined = sorted(
        observations["audio"] + observations["whisper"],
        key=lambda item: (item.timestamp_seconds, item.observer, item.type),
    )
    groups: dict[float, list[Observation]] = {}
    for item in combined:
        groups.setdefault(item.timestamp_seconds, []).append(item)
    return FeatureTimeline(
        media_path=media,
        audio_path=audio,
        timeline_path=audio.with_suffix(".incremental.json"),
        timeline=AggregatedTimeline(
            [TimelineGroup(key, groups[key]) for key in sorted(groups)], results
        ),
        metadata={"source_id": source_id, "input_type": "local"},
    )


def _source_id(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"local:sha256:{digest.hexdigest()}"


def _portable_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def _canonical_report_value(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _canonical_report_value(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Enum):
        return _canonical_report_value(value.value)
    if isinstance(value, Path):
        return _portable_path(value)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Report mappings require string keys.")
        return {
            key: _canonical_report_value(value[key])
            for key in sorted(value)
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonical_report_value(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Report floats must be finite.")
        return 0.0 if value == 0 else value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"Unsupported report value type: {type(value).__name__}.")
