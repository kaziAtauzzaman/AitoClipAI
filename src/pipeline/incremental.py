"""Explicit-watermark incremental coordination and completed-timeline replay."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from typing import Protocol

from candidate_generation import CandidateGenerator
from candidate_scoring import CandidateScorer
from candidate_selection import (
    CandidateSelectionResult,
    CandidateSelector,
    SuppressedCandidate,
)
from core import (
    AggregatedTimeline,
    ClipCandidate,
    ClipScore,
    FeatureTimeline,
    Observation,
    ObserverResult,
    RenderJob,
    TimelineGroup,
)


class IncrementalCandidateGenerator(Protocol):
    """Deterministic generator with a bounded historical revision horizon."""

    @property
    def maximum_backtrack_seconds(self) -> float: ...

    @property
    def incremental_deterministic(self) -> bool: ...

    def generate(self, timeline: FeatureTimeline) -> list[ClipCandidate]: ...


class IncrementalCandidateScorer(Protocol):
    """Scorer whose result depends only on one candidate and fixed configuration."""

    @property
    def candidate_local_deterministic(self) -> bool: ...

    def score(self, candidates: Iterable[ClipCandidate]) -> list[ClipScore]: ...


class IncrementalCandidateSelector(Protocol):
    def select(self, scores: Iterable[ClipScore]) -> CandidateSelectionResult: ...


class IncrementalClipRenderer(Protocol):
    def render_one(self, score: ClipScore, identity: int) -> RenderJob: ...


@dataclass(frozen=True, slots=True)
class ObserverWatermarks:
    """Observer-confirmed timestamps before which observations are immutable."""

    stable_through: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class IncrementalEOF:
    """Authoritative end-of-input confirmation from every required observer."""

    media_duration_seconds: float
    final_watermarks: ObserverWatermarks


@dataclass(frozen=True, slots=True)
class IncrementalPipelineConfig:
    required_observers: tuple[str, ...] = ("audio", "whisper")

    def __post_init__(self) -> None:
        if not self.required_observers or any(
            not item.strip() for item in self.required_observers
        ):
            raise ValueError("At least one non-empty required observer is required.")
        if len(set(self.required_observers)) != len(self.required_observers):
            raise ValueError("Required observers must be unique.")


@dataclass(frozen=True, slots=True)
class CompletedTimelineReplayConfig:
    observation_batch_seconds: float = 30.0

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.observation_batch_seconds)
            or self.observation_batch_seconds <= 0
        ):
            raise ValueError("Observation batch duration must be finite and positive.")


@dataclass(slots=True)
class IncrementalPipelineResult:
    scores: list[ClipScore] = field(default_factory=list)
    selected_scores: list[ClipScore] = field(default_factory=list)
    suppressed: list[SuppressedCandidate] = field(default_factory=list)
    render_jobs: list[RenderJob] = field(default_factory=list)


class RenderLifecycleState(str, Enum):
    RENDERING = "rendering"
    RENDERED = "rendered"
    FAILED = "failed"


class CoordinatorLifecycle(str, Enum):
    NEW = "new"
    ACTIVE = "active"
    FLUSHED = "flushed"


class IncrementalPrerecordedCoordinator:
    """Finalize stable candidate groups from explicit observer watermarks.

    One instance is single-use for one stable source identifier. Generators must
    bound historical revisions, and scorers must be deterministic and candidate-local.
    """

    def __init__(
        self,
        candidate_generator: IncrementalCandidateGenerator | None = None,
        candidate_scorer: IncrementalCandidateScorer | None = None,
        candidate_selector: IncrementalCandidateSelector | None = None,
        clip_renderer: IncrementalClipRenderer | None = None,
        config: IncrementalPipelineConfig | None = None,
    ) -> None:
        if clip_renderer is None:
            raise ValueError("An incremental clip renderer is required.")
        self._generator = candidate_generator or CandidateGenerator()
        self._scorer = candidate_scorer or CandidateScorer()
        self._selector = candidate_selector or CandidateSelector()
        self._renderer = clip_renderer
        self._config = config or IncrementalPipelineConfig()
        self._competition_horizon = self._validate_compatibility()
        self._lifecycle = CoordinatorLifecycle.NEW
        self._source_id: str | None = None
        self._watermark = 0.0
        self._render_identity = 0
        self._render_identities: dict[str, int] = {}
        self._finalized_fingerprints: set[str] = set()
        self._render_states: dict[str, RenderLifecycleState] = {}
        self._result = IncrementalPipelineResult()

    @property
    def lifecycle(self) -> CoordinatorLifecycle:
        return self._lifecycle

    @property
    def watermark_seconds(self) -> float:
        return self._watermark

    @property
    def required_observers(self) -> tuple[str, ...]:
        """Return the immutable observer names required for safe progress."""

        return self._config.required_observers

    @property
    def result(self) -> IncrementalPipelineResult:
        """Return a snapshot of finalized coordinator output."""

        return IncrementalPipelineResult(
            scores=list(self._result.scores),
            selected_scores=list(self._result.selected_scores),
            suppressed=list(self._result.suppressed),
            render_jobs=list(self._result.render_jobs),
        )

    def render_state(self, score: ClipScore) -> RenderLifecycleState | None:
        return self._render_states.get(self._score_fingerprint(score))

    def advance(
        self,
        timeline: FeatureTimeline,
        watermarks: ObserverWatermarks,
    ) -> list[RenderJob]:
        """Consume only observations confirmed stable by required observers."""

        self._activate(timeline)
        stable = self._validated_global_watermark(watermarks)
        if stable < self._watermark:
            raise ValueError("Stable watermark cannot move backwards.")
        self._watermark = stable
        scores = self._scores_for(self._stable_prefix(timeline, watermarks))
        passing = [
            score
            for score in scores
            if score.passed_threshold is True
            and self._score_fingerprint(score) not in self._finalized_fingerprints
        ]
        safe: list[ClipScore] = []
        for group in self._overlap_groups(passing):
            group_end = max(score.candidate.end_seconds for score in group)
            if group_end + self._competition_horizon <= stable:
                safe.extend(group)
        return self._finalize(safe)

    def flush(
        self,
        timeline: FeatureTimeline,
        eof: IncrementalEOF,
    ) -> IncrementalPipelineResult:
        """Finalize once, after authoritative EOF from every required observer."""

        self._activate(timeline)
        self._validate_eof(eof)
        scores = self._scores_for(timeline)
        remaining = [
            score
            for score in scores
            if score.passed_threshold is True
            and self._score_fingerprint(score) not in self._finalized_fingerprints
        ]
        self._watermark = eof.media_duration_seconds
        self._finalize(remaining)
        self._lifecycle = CoordinatorLifecycle.FLUSHED
        return self._result

    def _validate_compatibility(self) -> float:
        backtrack = getattr(self._generator, "maximum_backtrack_seconds", None)
        deterministic = getattr(self._generator, "incremental_deterministic", False)
        candidate_local = getattr(self._scorer, "candidate_local_deterministic", False)
        if isinstance(backtrack, bool) or not isinstance(backtrack, int | float):
            raise ValueError("Incremental generator must declare maximum_backtrack_seconds.")
        backtrack = float(backtrack)
        if not math.isfinite(backtrack) or backtrack < 0:
            raise ValueError(
                "Generator maximum_backtrack_seconds must be finite and non-negative."
            )
        if deterministic is not True:
            raise ValueError("Incremental generator must declare deterministic prefix output.")
        if candidate_local is not True:
            raise ValueError("Incremental scorer must be deterministic and candidate-local.")
        return backtrack

    def _activate(self, timeline: FeatureTimeline) -> None:
        if self._lifecycle is CoordinatorLifecycle.FLUSHED:
            raise RuntimeError("Incremental coordinator has already been flushed.")
        source_id = stable_source_id(timeline)
        if self._source_id is None:
            self._source_id = source_id
            self._lifecycle = CoordinatorLifecycle.ACTIVE
        elif source_id != self._source_id:
            raise RuntimeError("Incremental coordinator is single-use for one source.")
        self._recover_stale_rendering()

    def _recover_stale_rendering(self) -> None:
        for fingerprint, state in list(self._render_states.items()):
            if state is RenderLifecycleState.RENDERING:
                self._render_states[fingerprint] = RenderLifecycleState.FAILED

    def _validated_global_watermark(self, watermarks: ObserverWatermarks) -> float:
        missing = [
            item
            for item in self._config.required_observers
            if item not in watermarks.stable_through
        ]
        if missing:
            raise ValueError(f"Missing required observer watermarks: {', '.join(missing)}")
        values = [
            float(watermarks.stable_through[item])
            for item in self._config.required_observers
        ]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("Observer watermarks must be finite and non-negative.")
        return min(values)

    def _validate_eof(self, eof: IncrementalEOF) -> None:
        duration = eof.media_duration_seconds
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("EOF media duration must be finite and non-negative.")
        stable = self._validated_global_watermark(eof.final_watermarks)
        if stable < duration:
            raise ValueError("Every required observer must confirm the final media duration.")

    def _scores_for(self, timeline: FeatureTimeline) -> list[ClipScore]:
        scores = self._scorer.score(self._generator.generate(timeline))
        self._result.scores = scores
        return scores

    def _finalize(self, scores: list[ClipScore]) -> list[RenderJob]:
        jobs: list[RenderJob] = []
        for group in self._overlap_groups(scores):
            selection = self._selector.select(group)
            for suppressed in selection.suppressed:
                fingerprint = self._score_fingerprint(suppressed.score)
                if fingerprint not in self._finalized_fingerprints:
                    self._result.suppressed.append(suppressed)
                    self._finalized_fingerprints.add(fingerprint)
            for winner in sorted(selection.selected, key=self._chronological_key):
                fingerprint = self._score_fingerprint(winner)
                if self._render_states.get(fingerprint) is RenderLifecycleState.RENDERED:
                    continue
                identity = self._render_identities.get(fingerprint)
                if identity is None:
                    self._render_identity += 1
                    identity = self._render_identity
                    self._render_identities[fingerprint] = identity
                self._render_states[fingerprint] = RenderLifecycleState.RENDERING
                try:
                    job = self._renderer.render_one(winner, identity)
                finally:
                    if self._render_states.get(fingerprint) is RenderLifecycleState.RENDERING:
                        self._render_states[fingerprint] = RenderLifecycleState.FAILED
                self._render_states[fingerprint] = RenderLifecycleState.RENDERED
                self._finalized_fingerprints.add(fingerprint)
                self._result.selected_scores.append(winner)
                self._result.render_jobs.append(job)
                jobs.append(job)
        return jobs

    def _score_fingerprint(self, score: ClipScore) -> str:
        assert self._source_id is not None
        return score_fingerprint(score, self._source_id)

    def _stable_prefix(
        self,
        timeline: FeatureTimeline,
        watermarks: ObserverWatermarks,
    ) -> FeatureTimeline:
        results: list[ObserverResult] = []
        for result in timeline.timeline.observer_results:
            stable = watermarks.stable_through.get(result.observer, self._watermark)
            observations = [
                item
                for item in result.observations
                if _observation_end(item) <= stable
            ]
            results.append(ObserverResult(result.observer, observations, result.metadata))
        return _timeline_with_results(timeline, results)

    @staticmethod
    def _overlap_groups(scores: list[ClipScore]) -> list[list[ClipScore]]:
        ordered = sorted(scores, key=IncrementalPrerecordedCoordinator._chronological_key)
        groups: list[list[ClipScore]] = []
        group_end = -1.0
        for score in ordered:
            if not groups or score.candidate.start_seconds >= group_end:
                groups.append([score])
                group_end = score.candidate.end_seconds
            else:
                groups[-1].append(score)
                group_end = max(group_end, score.candidate.end_seconds)
        return groups

    @staticmethod
    def _chronological_key(score: ClipScore) -> tuple[float, float, float]:
        return (
            score.candidate.start_seconds,
            score.candidate.end_seconds,
            -score.overall_score,
        )


class CompletedTimelineReplayAdapter:
    """Simulate incremental progress from a completed timeline.

    This adapter intentionally inspects future observations. It exists only for
    simulation and tests and must never be used as a real streaming observer.
    """

    def __init__(self, config: CompletedTimelineReplayConfig | None = None) -> None:
        self._config = config or CompletedTimelineReplayConfig()

    def run(
        self,
        coordinator: IncrementalPrerecordedCoordinator,
        timeline: FeatureTimeline,
        media_duration_seconds: float,
    ) -> IncrementalPipelineResult:
        requested = self._config.observation_batch_seconds
        while requested < media_duration_seconds:
            watermarks = self.watermarks_at(timeline, requested)
            coordinator.advance(_prefix_at(timeline, requested), watermarks)
            requested += self._config.observation_batch_seconds
        final = ObserverWatermarks(
            {
                name: media_duration_seconds
                for name in coordinator.required_observers
            }
        )
        return coordinator.flush(
            timeline,
            IncrementalEOF(media_duration_seconds, final),
        )

    @staticmethod
    def watermarks_at(timeline: FeatureTimeline, requested: float) -> ObserverWatermarks:
        stable: dict[str, float] = {}
        for result in timeline.timeline.observer_results:
            withheld_starts = [
                item.timestamp_seconds
                for item in result.observations
                if _observation_end(item) > requested
            ]
            stable[result.observer] = min([requested, *withheld_starts])
        return ObserverWatermarks(stable)


def stable_source_id(timeline: FeatureTimeline) -> str:
    value = timeline.metadata.get("source_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    if timeline.download is not None:
        return f"{timeline.download.provider}:{timeline.download.media_id}"
    raise ValueError("Incremental timelines require metadata['source_id'].")


def candidate_fingerprint(candidate: ClipCandidate, source_id: str) -> str:
    return _fingerprint({"source_id": source_id, "candidate": _candidate_value(candidate)})


def score_fingerprint(score: ClipScore, source_id: str) -> str:
    return _fingerprint(
        {
            "source_id": source_id,
            "candidate": _candidate_value(score.candidate),
            "overall_score": score.overall_score,
            "score_components": score.score_components,
            "rationale": score.rationale,
            "passed_threshold": score.passed_threshold,
        }
    )


def _candidate_value(candidate: ClipCandidate) -> dict[str, object]:
    return {
        "start_seconds": candidate.start_seconds,
        "end_seconds": candidate.end_seconds,
        "reason": candidate.reason,
        "source_signals": candidate.source_signals,
        "title": candidate.title,
        "metadata": candidate.metadata,
    }


def _fingerprint(value: object) -> str:
    encoded = json.dumps(
        _canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_value(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _canonical_value(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Fingerprint metadata dictionaries require string keys.")
        return {key: _canonical_value(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Fingerprint values must be finite.")
        normalized = "0" if value == 0 else format(value, ".17g")
        return {"$float": normalized}
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, Path):
        raise TypeError("Paths are not supported in fingerprint metadata; use source_id.")
    raise TypeError(f"Unsupported fingerprint value type: {type(value).__name__}.")


def _observation_end(observation: Observation) -> float:
    return observation.timestamp_seconds + (observation.duration_seconds or 0.0)


def _prefix_at(timeline: FeatureTimeline, requested: float) -> FeatureTimeline:
    results = [
        ObserverResult(
            result.observer,
            [item for item in result.observations if _observation_end(item) <= requested],
            result.metadata,
        )
        for result in timeline.timeline.observer_results
    ]
    return _timeline_with_results(timeline, results)


def _timeline_with_results(
    timeline: FeatureTimeline,
    results: list[ObserverResult],
) -> FeatureTimeline:
    grouped: dict[float, list[Observation]] = {}
    for result in results:
        for observation in result.observations:
            grouped.setdefault(observation.timestamp_seconds, []).append(observation)
    return FeatureTimeline(
        media_path=timeline.media_path,
        audio_path=timeline.audio_path,
        timeline_path=timeline.timeline_path,
        timeline=AggregatedTimeline(
            groups=[TimelineGroup(timestamp, grouped[timestamp]) for timestamp in sorted(grouped)],
            observer_results=results,
            metadata=timeline.timeline.metadata,
        ),
        source_url=timeline.source_url,
        download=timeline.download,
        failures=timeline.failures,
        metadata=timeline.metadata,
    )
