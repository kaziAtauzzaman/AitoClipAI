"""Strict end-to-end composition for prerecorded-video validation."""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Callable, Iterable, Protocol, TypeVar

from audio_observer import (
    AudioObserver,
    FFmpegAudioExtractor,
    FFmpegAudioExtractorConfig,
)
from candidate_generation import CandidateGenerator
from candidate_scoring import CandidateScorer
from candidate_selection import CandidateSelectionResult, CandidateSelector
from clip_rendering import ClipRenderer, ClipRendererConfig
from core import ClipCandidate, ClipScore, FeatureTimeline, RenderJob
from downloader import DownloaderConfig, VideoDownloader
from observers import ObserverEngine, ObserverRegistry
from pipeline.contracts import (
    MediaProbeResult,
    PrerecordedPipelineResult,
    RenderedArtifactValidation,
)
from pipeline.errors import (
    ArtifactValidationError,
    NoCandidatesError,
    NoPassingCandidatesError,
    RequiredObserverError,
)
from pipeline.orchestrator import PipelineConfig, PipelineOrchestrator
from pipeline.validation import (
    ArtifactValidator,
    JsonValidationReportWriter,
    ValidationReportWriter,
    build_validation_report,
)
from whisper_observer import WhisperObserver


class AnalysisPipeline(Protocol):
    """Resolve, analyze, aggregate, and persist one source."""

    def analyze(self, source: str | Path) -> FeatureTimeline:
        """Return the source feature timeline."""


class CandidateGenerationService(Protocol):
    """Generate candidate clip windows from a feature timeline."""

    def generate(self, timeline: FeatureTimeline) -> list[ClipCandidate]:
        """Return candidate windows."""


class CandidateScoringService(Protocol):
    """Score and rank generated candidates."""

    def score(self, candidates: Iterable[ClipCandidate]) -> list[ClipScore]:
        """Return ranked scores."""


class CandidateSelectionService(Protocol):
    """Suppress weaker substantially overlapping passing scores."""

    def select(self, scores: Iterable[ClipScore]) -> CandidateSelectionResult:
        """Return scores selected for rendering and suppression provenance."""


class ClipRenderingService(Protocol):
    """Render selected passing scores without caption artifacts."""

    def render(self, scores: Iterable[ClipScore]) -> list[RenderJob]:
        """Render selected scores and return jobs."""


class PipelineArtifactValidator(Protocol):
    """Probe source metadata and validate rendered jobs."""

    def probe_source(self, path: Path) -> MediaProbeResult:
        """Return normalized source metadata."""

    def validate_jobs(
        self,
        jobs: list[RenderJob],
    ) -> list[RenderedArtifactValidation]:
        """Return successful validations for all rendered jobs."""


@dataclass(frozen=True, slots=True)
class PrerecordedPipelineConfig:
    """Strict policies for Pipeline Validation 0.1."""

    required_observers: tuple[str, ...] = ("audio", "whisper")
    run_dir: Path = Path("data") / "validation" / "pipeline-0.1"
    overwrite_existing: bool = False
    maximum_clips: int | None = 1


T = TypeVar("T")


class PrerecordedVideoPipeline:
    """Compose existing stages and enforce strict prerecorded MVP validation."""

    def __init__(
        self,
        analysis_pipeline: AnalysisPipeline | None = None,
        candidate_generator: CandidateGenerationService | None = None,
        candidate_scorer: CandidateScoringService | None = None,
        clip_renderer: ClipRenderingService | None = None,
        artifact_validator: PipelineArtifactValidator | None = None,
        report_writer: ValidationReportWriter | None = None,
        config: PrerecordedPipelineConfig | None = None,
        logger: logging.Logger | None = None,
        candidate_selector: CandidateSelectionService | None = None,
    ) -> None:
        self._config = config or PrerecordedPipelineConfig()
        if not {"audio", "whisper"}.issubset(self._config.required_observers):
            raise RequiredObserverError(
                "Pipeline Validation 0.1 requires audio and whisper observers."
            )
        self._analysis_pipeline = analysis_pipeline or self._default_analysis()
        self._candidate_generator = candidate_generator or CandidateGenerator()
        self._candidate_scorer = candidate_scorer or CandidateScorer()
        self._candidate_selector = candidate_selector or CandidateSelector()
        self._clip_renderer = clip_renderer or ClipRenderer(
            ClipRendererConfig(
                output_dir=self._config.run_dir / "clips",
                overwrite_existing=self._config.overwrite_existing,
                maximum_clips=self._config.maximum_clips,
                burn_subtitles=False,
            )
        )
        self._artifact_validator = artifact_validator or ArtifactValidator()
        self._report_writer = report_writer or JsonValidationReportWriter(
            self._config.run_dir / "reports" / "validation-report.json"
        )
        self._logger = logger or logging.getLogger(__name__)

    def run(self, source: str | Path) -> PrerecordedPipelineResult:
        """Run every prerecorded pipeline stage and validate playable output."""

        timeline = self._stage(
            "analysis",
            lambda: self._analysis_pipeline.analyze(source),
        )
        self._stage("required_observers", lambda: self._require_observers(timeline))
        candidates = self._stage(
            "candidate_generation",
            lambda: self._generate_candidates(timeline),
        )
        self._logger.info("stage_result stage=candidate_generation count=%d", len(candidates))

        scores, selected = self._stage(
            "candidate_scoring",
            lambda: self._score_candidates(candidates),
        )
        self._logger.info(
            "stage_result stage=candidate_scoring scores=%d passing=%d",
            len(scores),
            len(selected),
        )

        selection = self._stage(
            "candidate_selection",
            lambda: self._candidate_selector.select(selected),
        )
        self._logger.info(
            "stage_result stage=candidate_selection passing=%d selected=%d "
            "suppressed=%d",
            len(selected),
            len(selection.selected),
            len(selection.suppressed),
        )

        render_jobs = self._stage(
            "clip_rendering",
            lambda: self._render_scores(selection.selected),
        )
        source_metadata = self._stage(
            "source_probe",
            lambda: self._artifact_validator.probe_source(timeline.media_path),
        )
        rendered_artifacts = self._stage(
            "artifact_validation",
            lambda: self._artifact_validator.validate_jobs(render_jobs),
        )
        report = build_validation_report(
            timeline=timeline,
            required_observers=list(self._config.required_observers),
            candidates_count=len(candidates),
            scores_count=len(scores),
            passing_scores_count=len(selected),
            source_metadata=source_metadata,
            rendered_artifacts=rendered_artifacts,
        )
        report_path = self._stage(
            "report_persistence",
            lambda: self._report_writer.write(report),
        )
        self._logger.info(
            "pipeline_validation_passed rendered=%d report=%s",
            len(render_jobs),
            report_path,
        )
        return PrerecordedPipelineResult(
            feature_timeline=timeline,
            candidates=candidates,
            scores=scores,
            selected_scores=selected,
            render_jobs=render_jobs,
            validation_report=report,
            report_path=report_path,
        )

    def _default_analysis(self) -> PipelineOrchestrator:
        run_dir = self._config.run_dir
        return PipelineOrchestrator(
            downloader=VideoDownloader(
                DownloaderConfig(
                    downloads_dir=run_dir / "downloads",
                    overwrite_existing=self._config.overwrite_existing,
                )
            ),
            audio_extractor=FFmpegAudioExtractor(
                FFmpegAudioExtractorConfig(
                    output_dir=run_dir / "audio",
                    overwrite_existing=self._config.overwrite_existing,
                )
            ),
            observer_engine=ObserverEngine(
                ObserverRegistry(observers=[AudioObserver(), WhisperObserver()])
            ),
            config=PipelineConfig(timeline_dir=run_dir / "timelines"),
        )

    def _generate_candidates(
        self,
        timeline: FeatureTimeline,
    ) -> list[ClipCandidate]:
        candidates = self._candidate_generator.generate(timeline)
        if not candidates:
            raise NoCandidatesError("Candidate generation produced no clip windows.")
        return candidates

    def _score_candidates(
        self,
        candidates: list[ClipCandidate],
    ) -> tuple[list[ClipScore], list[ClipScore]]:
        scores = self._candidate_scorer.score(candidates)
        selected = [score for score in scores if score.passed_threshold is True]
        if not selected:
            raise NoPassingCandidatesError(
                "No scored candidate passed the configured threshold."
            )
        return scores, selected

    def _render_scores(self, selected: list[ClipScore]) -> list[RenderJob]:
        jobs = self._clip_renderer.render(selected)
        if not jobs:
            raise ArtifactValidationError(
                "Clip rendering returned no jobs for passing candidates."
            )
        return jobs

    def _require_observers(self, timeline: FeatureTimeline) -> None:
        required = set(self._config.required_observers)
        required_failures = [
            failure for failure in timeline.failures if failure.observer in required
        ]
        if required_failures:
            detail = ", ".join(
                f"{failure.observer}:{failure.error_type}"
                for failure in required_failures
            )
            raise RequiredObserverError(f"Required observer failure: {detail}")
        observed = {result.observer for result in timeline.timeline.observer_results}
        missing = sorted(required - observed)
        if missing:
            raise RequiredObserverError(
                f"Required observers are missing: {', '.join(missing)}"
            )

    def _stage(self, name: str, operation: Callable[[], T]) -> T:
        self._logger.info("stage_started stage=%s", name)
        try:
            result = operation()
        except Exception as exc:
            self._logger.exception(
                "stage_failed stage=%s error_type=%s",
                name,
                type(exc).__name__,
            )
            raise
        self._logger.info("stage_completed stage=%s", name)
        return result
