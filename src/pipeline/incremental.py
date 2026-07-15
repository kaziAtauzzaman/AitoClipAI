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
from whisper_observer.contracts import finalized_speech_segment_identity


class IncrementalCandidateGenerator(Protocol):
    """Deterministic generator with a bounded historical revision horizon."""

    @property
    def maximum_backtrack_seconds(self) -> float: ...

    @property
    def incremental_deterministic(self) -> bool: ...

    @property
    def maximum_competition_seconds(self) -> float: ...

    def revision_start_seconds(self, candidate: ClipCandidate) -> float: ...

    def revision_stable_after_seconds(self, candidate: ClipCandidate) -> float: ...

    def revision_partition_seconds(self, candidate: ClipCandidate) -> float: ...

    def earliest_unresolved_cluster_start_seconds(
        self,
        timeline: FeatureTimeline,
        stable_watermark_seconds: float,
    ) -> float | None: ...

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
    """Observer-confirmed timestamps before which observations are immutable.

    Whisper stabilizes complete speech intervals, so a speech observation must
    end at or before its watermark. Incremental Audio stabilizes emitted event
    values: diagnostic ``speaking_intensity`` and closed ``silence`` windows may
    start at or before the watermark even when their descriptive duration
    extends beyond it.
    """

    stable_through: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class IncrementalEOF:
    """Authoritative end-of-input confirmation from every required observer."""

    media_duration_seconds: float
    final_watermarks: ObserverWatermarks


@dataclass(frozen=True, slots=True)
class ObserverDeltaIdentity:
    """Durable chronological identity for one observer-owned stable delta."""

    source_id: str
    session_id: str
    observer: str
    sequence: int
    eof: bool = False

    def __post_init__(self) -> None:
        if not self.source_id.strip() or not self.session_id.strip():
            raise ValueError("Delta source and session identities must be non-empty.")
        if not self.observer.strip():
            raise ValueError("Delta observer must be non-empty.")
        if isinstance(self.sequence, bool) or self.sequence < 0:
            raise ValueError("Delta sequence must be a non-negative integer.")


@dataclass(frozen=True, slots=True)
class IncrementalPipelineConfig:
    required_observers: tuple[str, ...] = ("audio", "whisper")
    session_id: str | None = None

    def __post_init__(self) -> None:
        if not self.required_observers or any(
            not item.strip() for item in self.required_observers
        ):
            raise ValueError("At least one non-empty required observer is required.")
        if len(set(self.required_observers)) != len(self.required_observers):
            raise ValueError("Required observers must be unique.")
        if self.session_id is not None and not self.session_id.strip():
            raise ValueError("Incremental session identity must be non-empty.")


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


@dataclass(frozen=True, slots=True)
class IncrementalStateMetrics:
    """Read-only work counters for bounded-state regression and scaling tests."""

    generation_passes: int
    scored_candidates: int
    candidate_fingerprints: int
    score_fingerprints: int
    active_observations: int
    peak_active_observations: int
    active_scores: int
    finalized_scores: int
    immutable_score_fingerprints: int
    completed_render_jobs: int
    peak_active_scores: int
    peak_unresolved_group_size: int


class RenderLifecycleState(str, Enum):
    RENDERING = "rendering"
    RENDERED = "rendered"
    FAILED = "failed"


class DeltaAcceptanceState(str, Enum):
    RECEIVED = "received"
    INGESTED = "ingested"
    GENERATION_SCORED = "generation_scored"
    DECISIONS_COMMITTED = "decisions_committed"
    RENDERING = "rendering"
    COMPLETED = "completed"
    FAILED_RETRYABLE = "failed_retryable"


@dataclass(slots=True)
class _DeltaAcceptance:
    """Process-local receipt; deliberately not restored after process termination."""
    identity: ObserverDeltaIdentity
    payload: str
    state: DeltaAcceptanceState = DeltaAcceptanceState.RECEIVED
    scores: list[ClipScore] = field(default_factory=list)
    unresolved_revision_start: float | None = None
    safe_scores: list[ClipScore] = field(default_factory=list)
    pending_renders: list[tuple[ClipScore, str, int]] = field(default_factory=list)
    completed_jobs: list[RenderJob] = field(default_factory=list)


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
        self._backtrack_horizon, self._competition_horizon = self._validate_compatibility()
        self._lifecycle = CoordinatorLifecycle.NEW
        self._source_id: str | None = None
        self._watermark = 0.0
        self._render_identity = 0
        self._render_identities: dict[str, int] = {}
        self._finalized_fingerprints: set[str] = set()
        self._render_states: dict[str, RenderLifecycleState] = {}
        self._currently_rendering: set[str] = set()
        self._result = IncrementalPipelineResult()
        self._input_mode: str | None = None
        self._snapshot_seen: set[str] = set()
        self._active_observations: dict[str, list[Observation]] = {
            name: [] for name in self._config.required_observers
        }
        self._observer_metadata: dict[str, dict[str, object]] = {
            name: {} for name in self._config.required_observers
        }
        self._observer_watermarks = {
            name: 0.0 for name in self._config.required_observers
        }
        self._observer_frames = {
            name: 0 for name in self._config.required_observers
        }
        self._accepted_delta_sequences = {
            name: -1 for name in self._config.required_observers
        }
        self._pending_delta: _DeltaAcceptance | None = None
        self._pending_eof: _DeltaAcceptance | None = None
        self._pending_snapshot_renders: list[tuple[ClipScore, str, int]] = []
        self._completed_delta_receipts: dict[tuple[str, str, str, int, bool], str] = {}
        self._score_cache: dict[str, ClipScore] = {}
        self._score_fingerprint_cache: dict[str, str] = {}
        self._immutable_score_fingerprints: dict[str, str] = {}
        self._score_candidate_ids: dict[int, str] = {}
        self._candidate_object_ids: dict[int, str] = {}
        self._finalized_scores: dict[str, ClipScore] = {}
        self._immutable_through = 0.0
        self._finalized_generation_through = 0.0
        self._generation_passes = 0
        self._scored_candidates = 0
        self._candidate_fingerprint_count = 0
        self._score_fingerprint_count = 0
        self._peak_active_observations = 0
        self._peak_active_scores = 0
        self._peak_unresolved_group_size = 0
        self._active_scores: list[ClipScore] = []
        self._scores_dirty = True

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

        self._materialize_scores()
        return IncrementalPipelineResult(
            scores=list(self._result.scores),
            selected_scores=list(self._result.selected_scores),
            suppressed=list(self._result.suppressed),
            render_jobs=list(self._result.render_jobs),
        )

    def render_jobs_since(self, index: int) -> list[RenderJob]:
        """Return only newly completed render jobs without copying score history."""

        if index < 0 or index > len(self._result.render_jobs):
            raise ValueError("Render-job cursor is outside the completed job range.")
        return list(self._result.render_jobs[index:])

    def render_job_at(self, index: int) -> RenderJob | None:
        """Return one completed job in O(1), or None at the current end."""

        if index < 0 or index > len(self._result.render_jobs):
            raise ValueError("Render-job cursor is outside the completed job range.")
        if index == len(self._result.render_jobs):
            return None
        return self._result.render_jobs[index]

    @property
    def state_metrics(self) -> IncrementalStateMetrics:
        active = sum(len(items) for items in self._active_observations.values())
        return IncrementalStateMetrics(
            self._generation_passes,
            self._scored_candidates,
            self._candidate_fingerprint_count,
            self._score_fingerprint_count,
            active,
            self._peak_active_observations,
            len(self._active_scores),
            len(self._finalized_scores),
            len(self._immutable_score_fingerprints),
            len(self._result.render_jobs),
            self._peak_active_scores,
            self._peak_unresolved_group_size,
        )

    def render_state(self, score: ClipScore) -> RenderLifecycleState | None:
        return self._render_states.get(self._score_fingerprint(score))

    def advance(
        self,
        timeline: FeatureTimeline,
        watermarks: ObserverWatermarks,
    ) -> list[RenderJob]:
        """Consume only observations confirmed stable by required observers."""

        if self._input_mode not in (None, "snapshot"):
            raise RuntimeError("Cannot mix snapshot and delta coordinator input modes.")
        self._input_mode = "snapshot"
        delta_results: list[ObserverResult] = []
        for result in timeline.timeline.observer_results:
            identities = [self._observation_id(item) for item in result.observations]
            delta_results.append(
                ObserverResult(
                    result.observer,
                    [
                        item
                        for identity, item in zip(identities, result.observations)
                        if identity not in self._snapshot_seen
                    ],
                    dict(result.metadata),
                )
            )
            self._snapshot_seen.update(identities)
        return self._advance_delta(timeline, delta_results, watermarks)

    def advance_delta(
        self,
        timeline: FeatureTimeline,
        watermarks: ObserverWatermarks,
        identity: ObserverDeltaIdentity | None = None,
    ) -> list[RenderJob]:
        """Consume an append-only batch containing only newly stable observations."""

        if self._input_mode not in (None, "delta"):
            raise RuntimeError("Cannot mix snapshot and delta coordinator input modes.")
        self._input_mode = "delta"
        self._activate(timeline)
        results = list(timeline.timeline.observer_results)
        identity = identity or self._implicit_delta_identity(results, eof=False)
        return self._advance_delta(timeline, results, watermarks, identity)

    def _advance_delta(
        self,
        timeline: FeatureTimeline,
        delta_results: list[ObserverResult],
        watermarks: ObserverWatermarks,
        identity: ObserverDeltaIdentity | None = None,
    ) -> list[RenderJob]:
        self._activate(timeline)
        stable = self._validated_global_watermark(watermarks)
        if stable < self._watermark:
            raise ValueError("Stable watermark cannot move backwards.")
        if identity is not None:
            _validate_strict_audio_delta_metadata(delta_results)
            _validate_strict_whisper_delta_metadata(delta_results)
            payload = self._delta_payload_fingerprint(delta_results, watermarks, identity)
            completed = self._completed_delta_receipts.get(self._identity_key(identity))
            if completed is not None:
                if completed != payload:
                    raise ValueError("Delta identity was reused with different content.")
                return []
            if self._pending_delta is None:
                self._validate_delta_identity(identity, delta_results, watermarks)
                self._pending_delta = _DeltaAcceptance(identity, payload)
            elif (
                self._pending_delta.identity != identity
                or self._pending_delta.payload != payload
            ):
                raise RuntimeError("A previously accepted delta must finish before new input.")
            self._ensure_receipt_ingested(
                self._pending_delta,
                delta_results,
                watermarks,
            )
            if self._accepted_delta_sequences[identity.observer] < identity.sequence:
                self._accepted_delta_sequences[identity.observer] = identity.sequence
                self._observer_watermarks[identity.observer] = float(
                    watermarks.stable_through[identity.observer]
                )
                self._observer_frames[identity.observer] = self._delta_frames(
                    delta_results[0]
                )
        else:
            self._ingest_delta(delta_results, watermarks, strict=False)
        self._watermark = stable
        receipt = self._pending_delta if identity is not None else None
        if receipt is None or receipt.state is DeltaAcceptanceState.INGESTED:
            generation_timeline = self._generation_timeline(timeline)
            unresolved_revision_start = self._earliest_unresolved_revision_start(
                generation_timeline, stable
            )
            scores = (
                self._scores_for_active(
                    generation_timeline,
                    allow_historical=self._input_mode == "snapshot",
                )
                if self._should_generate(generation_timeline)
                else list(self._active_scores)
            )
            if receipt is not None:
                receipt.scores = scores
                receipt.unresolved_revision_start = unresolved_revision_start
                receipt.state = DeltaAcceptanceState.GENERATION_SCORED
        else:
            scores = receipt.scores
            unresolved_revision_start = receipt.unresolved_revision_start
        passing = [
            score
            for score in scores
            if score.passed_threshold is True
            and self._score_fingerprint(score) not in self._finalized_fingerprints
        ]
        safe: list[ClipScore] = []
        for group in self._overlap_groups(passing):
            group_end = max(score.candidate.end_seconds for score in group)
            group_revision_stable_after = max(
                self._revision_stable_after(score.candidate) for score in group
            )
            earliest_end = min(score.candidate.end_seconds for score in group)
            self._peak_unresolved_group_size = max(
                self._peak_unresolved_group_size, len(group)
            )
            if group_end - earliest_end > self._competition_horizon:
                raise RuntimeError(
                    "Continuous overlap competition exceeded the declared finite bound."
                )
            if max(
                group_end + self._competition_horizon,
                group_revision_stable_after,
            ) <= stable:
                safe.extend(group)
        safe_ids = {self._score_candidate_id(score) for score in safe}
        for score in scores:
            candidate_id = self._score_candidate_id(score)
            revision_stable = self._revision_stable_after(score.candidate) <= stable
            competition_closed = (
                score.candidate.end_seconds + self._competition_horizon <= stable
            )
            if candidate_id in safe_ids or (
                score.passed_threshold is not True
                and revision_stable
                and competition_closed
            ):
                self._finalized_scores[candidate_id] = score
                fingerprint = self._score_fingerprint_cache.pop(candidate_id, None)
                if fingerprint is not None:
                    self._immutable_score_fingerprints[candidate_id] = fingerprint
                self._finalized_generation_through = max(
                    self._finalized_generation_through,
                    self._revision_partition(score.candidate),
                )
        if receipt is not None and receipt.state is DeltaAcceptanceState.GENERATION_SCORED:
            receipt.safe_scores = safe
            receipt.pending_renders = self._commit_decisions(safe)
            receipt.state = DeltaAcceptanceState.DECISIONS_COMMITTED
        try:
            if receipt is None:
                jobs = self._finalize(safe)
            else:
                receipt.state = DeltaAcceptanceState.RENDERING
                jobs = self._render_plan(receipt.pending_renders)
                receipt.completed_jobs.extend(
                    item for item in jobs if item not in receipt.completed_jobs
                )
        except BaseException:
            if receipt is not None:
                receipt.state = DeltaAcceptanceState.FAILED_RETRYABLE
            raise
        unsafe_revision_starts = [
            self._revision_start(score.candidate)
            for score in scores
            if self._score_candidate_id(score) not in self._finalized_scores
        ]
        if unresolved_revision_start is not None:
            unsafe_revision_starts.append(unresolved_revision_start)
        if unsafe_revision_starts:
            earliest_unsafe = min(unsafe_revision_starts)
            immutable_through = (
                earliest_unsafe
                if earliest_unsafe <= stable
                else max(
                    0.0,
                    stable - self._competition_horizon - self._backtrack_horizon,
                )
            )
        else:
            immutable_through = max(
                self._finalized_generation_through,
                stable - self._competition_horizon - self._backtrack_horizon,
            )
        self._immutable_through = max(self._immutable_through, immutable_through)
        self._evict_inactive_state(stable, scores)
        self._set_active_scores(scores)
        if identity is not None:
            self._record_completed_delta(identity, payload)
            receipt.state = DeltaAcceptanceState.COMPLETED
            self._pending_delta = None
        return jobs

    def flush(
        self,
        timeline: FeatureTimeline,
        eof: IncrementalEOF,
    ) -> IncrementalPipelineResult:
        """Finalize once, after authoritative EOF from every required observer."""

        if self._input_mode not in (None, "snapshot"):
            raise RuntimeError("Cannot mix snapshot and delta coordinator input modes.")
        self._input_mode = "snapshot"
        delta_results: list[ObserverResult] = []
        for result in timeline.timeline.observer_results:
            identified = [
                (self._observation_id(item), item) for item in result.observations
            ]
            delta_results.append(
                ObserverResult(
                    result.observer,
                    [
                        item
                        for identity, item in identified
                        if identity not in self._snapshot_seen
                    ],
                    dict(result.metadata),
                )
            )
            self._snapshot_seen.update(identity for identity, _ in identified)
        return self._flush_delta(timeline, delta_results, eof)

    def flush_delta(
        self,
        timeline: FeatureTimeline,
        eof: IncrementalEOF,
        identities: tuple[ObserverDeltaIdentity, ...] = (),
    ) -> IncrementalPipelineResult:
        """Flush a delta-driven session after authoritative observer EOF."""

        if self._input_mode not in (None, "delta"):
            raise RuntimeError("Cannot mix snapshot and delta coordinator input modes.")
        self._input_mode = "delta"
        self._activate(timeline)
        results = list(timeline.timeline.observer_results)
        if results and not identities:
            identities = tuple(
                self._implicit_eof_identity(result, eof)
                for result in results
            )
        return self._flush_delta(timeline, results, eof, identities)

    def _flush_delta(
        self,
        timeline: FeatureTimeline,
        delta_results: list[ObserverResult],
        eof: IncrementalEOF,
        identities: tuple[ObserverDeltaIdentity, ...] = (),
    ) -> IncrementalPipelineResult:
        self._activate(timeline)
        self._validate_eof(eof)
        if self._pending_delta is not None:
            raise RuntimeError("A pending observer delta must finish before EOF.")
        if delta_results and not identities:
            self._ingest_delta(
                delta_results,
                eof.final_watermarks,
                strict=False,
            )
        for identity in identities:
            matching = [item for item in delta_results if item.observer == identity.observer]
            _validate_strict_audio_delta_metadata(matching)
            _validate_strict_whisper_delta_metadata(matching)
            payload = self._delta_payload_fingerprint(
                matching, eof.final_watermarks, identity
            )
            completed = self._completed_delta_receipts.get(self._identity_key(identity))
            if completed is not None:
                if completed != payload:
                    raise ValueError("Delta identity was reused with different content.")
                continue
            self._validate_delta_identity(identity, matching, eof.final_watermarks)
            self._ingest_delta(matching, eof.final_watermarks, strict=True)
            self._record_completed_delta(identity, payload)
            self._accepted_delta_sequences[identity.observer] = identity.sequence
            self._observer_watermarks[identity.observer] = float(
                eof.final_watermarks.stable_through[identity.observer]
            )
            self._observer_frames[identity.observer] = self._delta_frames(matching[0])
        eof_identity = ObserverDeltaIdentity(
            self._source_id or "unactivated",
            self._session_id(),
            "__combined_eof__",
            0,
            True,
        )
        eof_payload = _fingerprint(
            {
                "identity": eof_identity,
                "duration": eof.media_duration_seconds,
                "watermarks": dict(eof.final_watermarks.stable_through),
            }
        )
        if self._pending_eof is None:
            self._pending_eof = _DeltaAcceptance(
                eof_identity,
                eof_payload,
                DeltaAcceptanceState.INGESTED,
            )
        elif self._pending_eof.payload != eof_payload:
            raise ValueError("Combined EOF receipt was reused with different content.")
        receipt = self._pending_eof
        if receipt.state is DeltaAcceptanceState.INGESTED:
            generation_timeline = self._generation_timeline(timeline)
            receipt.scores = (
                self._scores_for_active(
                    generation_timeline,
                    allow_historical=self._input_mode == "snapshot",
                )
                if self._should_generate(generation_timeline)
                else list(self._active_scores)
            )
            receipt.state = DeltaAcceptanceState.GENERATION_SCORED
        scores = receipt.scores
        remaining = [
            score
            for score in scores
            if score.passed_threshold is True
            and self._score_fingerprint(score) not in self._finalized_fingerprints
        ]
        self._watermark = eof.media_duration_seconds
        if receipt.state is DeltaAcceptanceState.GENERATION_SCORED:
            receipt.safe_scores = remaining
            receipt.pending_renders = self._commit_decisions(remaining)
            receipt.state = DeltaAcceptanceState.DECISIONS_COMMITTED
        try:
            receipt.state = DeltaAcceptanceState.RENDERING
            jobs = self._render_plan(receipt.pending_renders)
            receipt.completed_jobs.extend(
                item for item in jobs if item not in receipt.completed_jobs
            )
        except BaseException:
            receipt.state = DeltaAcceptanceState.FAILED_RETRYABLE
            raise
        for score in scores:
            candidate_id = self._score_candidate_id(score)
            self._finalized_scores[candidate_id] = score
            fingerprint = self._score_fingerprint_cache.pop(candidate_id, None)
            if fingerprint is not None:
                self._immutable_score_fingerprints[candidate_id] = fingerprint
        self._immutable_through = eof.media_duration_seconds
        self._active_observations = {
            name: [] for name in self._config.required_observers
        }
        self._score_cache.clear()
        self._set_active_scores([])
        self._materialize_scores()
        receipt.state = DeltaAcceptanceState.COMPLETED
        self._lifecycle = CoordinatorLifecycle.FLUSHED
        return self._result

    def _validate_compatibility(self) -> tuple[float, float]:
        backtrack = getattr(self._generator, "maximum_backtrack_seconds", None)
        competition = getattr(self._generator, "maximum_competition_seconds", None)
        deterministic = getattr(self._generator, "incremental_deterministic", False)
        candidate_local = getattr(self._scorer, "candidate_local_deterministic", False)
        revision_start = getattr(self._generator, "revision_start_seconds", None)
        revision_stable_after = getattr(
            self._generator, "revision_stable_after_seconds", None
        )
        revision_partition = getattr(
            self._generator, "revision_partition_seconds", None
        )
        unresolved_cluster_start = getattr(
            self._generator, "earliest_unresolved_cluster_start_seconds", None
        )
        if isinstance(backtrack, bool) or not isinstance(backtrack, int | float):
            raise ValueError("Incremental generator must declare maximum_backtrack_seconds.")
        backtrack = float(backtrack)
        if not math.isfinite(backtrack) or backtrack < 0:
            raise ValueError(
                "Generator maximum_backtrack_seconds must be finite and non-negative."
            )
        if isinstance(competition, bool) or not isinstance(competition, int | float):
            raise ValueError(
                "Incremental generator must declare maximum_competition_seconds."
            )
        competition = float(competition)
        if not math.isfinite(competition) or competition < 0:
            raise ValueError(
                "Generator maximum_competition_seconds must be finite and non-negative."
            )
        if deterministic is not True:
            raise ValueError("Incremental generator must declare deterministic prefix output.")
        if candidate_local is not True:
            raise ValueError("Incremental scorer must be deterministic and candidate-local.")
        if (
            not callable(revision_start)
            or not callable(revision_stable_after)
            or not callable(revision_partition)
            or not callable(unresolved_cluster_start)
        ):
            raise ValueError(
                "Incremental generator must declare candidate revision boundaries."
            )
        return backtrack, competition

    def _revision_start(self, candidate: ClipCandidate) -> float:
        return self._revision_contract(candidate)[0]

    def _revision_partition(self, candidate: ClipCandidate) -> float:
        return self._revision_contract(candidate)[1]

    def _revision_stable_after(self, candidate: ClipCandidate) -> float:
        return self._revision_contract(candidate)[2]

    def _revision_contract(
        self, candidate: ClipCandidate
    ) -> tuple[float, float, float]:
        raw = (
            self._generator.revision_start_seconds(candidate),
            self._generator.revision_partition_seconds(candidate),
            self._generator.revision_stable_after_seconds(candidate),
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int | float)
            for value in raw
        ):
            raise ValueError("Candidate revision contract values must be numeric.")
        start, partition, stable_after = (float(value) for value in raw)
        if any(
            not math.isfinite(value) or value < 0
            for value in (start, partition, stable_after)
        ):
            raise ValueError(
                "Candidate revision contract values must be finite and non-negative."
            )
        if not start <= partition <= stable_after:
            raise ValueError(
                "Candidate revision contract must satisfy start <= partition <= stable-after."
            )
        if start > candidate.end_seconds or stable_after < candidate.end_seconds:
            raise ValueError("Candidate revision contract does not cover its candidate.")
        if stable_after - candidate.end_seconds > self._backtrack_horizon:
            raise ValueError(
                "Candidate revision stability exceeds maximum_backtrack_seconds."
            )
        return start, partition, stable_after

    def _earliest_unresolved_revision_start(
        self,
        timeline: FeatureTimeline,
        stable: float,
    ) -> float | None:
        value = self._generator.earliest_unresolved_cluster_start_seconds(
            timeline, stable
        )
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("Unresolved cluster start must be numeric or None.")
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                "Unresolved cluster start must be finite and non-negative."
            )
        return value

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
        for fingerprint in tuple(self._currently_rendering):
            if self._render_states.get(fingerprint) is RenderLifecycleState.RENDERING:
                self._render_states[fingerprint] = RenderLifecycleState.FAILED
            self._currently_rendering.discard(fingerprint)

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

    def _scores_for_active(
        self,
        timeline: FeatureTimeline,
        *,
        allow_historical: bool = False,
    ) -> list[ClipScore]:
        self._generation_passes += 1
        self._candidate_object_ids = {}
        self._score_candidate_ids = {
            id(score): identity
            for identity, score in self._score_cache.items()
        }
        generated = self._generator.generate(timeline)
        if not allow_historical and any(
            candidate.end_seconds <= self._immutable_through
            for candidate in generated
        ):
            raise RuntimeError("Generation input emitted an immutable historical candidate.")
        candidates = [
            candidate
            for candidate in generated
            if candidate.end_seconds > self._immutable_through
        ]
        identified = [(self._candidate_id(candidate), candidate) for candidate in candidates]
        missing = [candidate for identity, candidate in identified if identity not in self._score_cache]
        if missing:
            scored = self._scorer.score(missing)
            self._scored_candidates += len(scored)
            for score in scored:
                identity = self._score_candidate_id(score)
                self._score_cache[identity] = score
                self._score_candidate_ids[id(score)] = identity
        scores = [self._score_cache[identity] for identity, _ in identified]
        return sorted(scores, key=self._score_ordering_key)

    def _finalize(self, scores: list[ClipScore]) -> list[RenderJob]:
        if not self._pending_snapshot_renders:
            self._pending_snapshot_renders = self._commit_decisions(scores)
        try:
            jobs = self._render_plan(self._pending_snapshot_renders)
        except BaseException:
            raise
        self._pending_snapshot_renders = []
        return jobs

    def _commit_decisions(
        self,
        scores: list[ClipScore],
    ) -> list[tuple[ClipScore, str, int]]:
        plan: list[tuple[ClipScore, str, int]] = []
        for group in self._overlap_groups(scores):
            selection = self._selector.select(group)
            for suppressed in selection.suppressed:
                fingerprint = self._score_fingerprint(suppressed.score)
                if fingerprint not in self._finalized_fingerprints:
                    self._result.suppressed.append(suppressed)
                    self._finalized_fingerprints.add(fingerprint)
            for winner in sorted(selection.selected, key=self._chronological_key):
                fingerprint = self._score_fingerprint(winner)
                if fingerprint in self._finalized_fingerprints:
                    continue
                identity = self._render_identities.get(fingerprint)
                if identity is None:
                    self._render_identity += 1
                    identity = self._render_identity
                    self._render_identities[fingerprint] = identity
                self._finalized_fingerprints.add(fingerprint)
                self._result.selected_scores.append(winner)
                plan.append((winner, fingerprint, identity))
        return plan

    def _render_plan(
        self,
        plan: list[tuple[ClipScore, str, int]],
    ) -> list[RenderJob]:
        jobs: list[RenderJob] = []
        for winner, fingerprint, identity in plan:
            if self._render_states.get(fingerprint) is RenderLifecycleState.RENDERED:
                continue
            self._render_states[fingerprint] = RenderLifecycleState.RENDERING
            self._currently_rendering.add(fingerprint)
            try:
                job = self._renderer.render_one(winner, identity)
            finally:
                if self._render_states.get(fingerprint) is RenderLifecycleState.RENDERING:
                    self._render_states[fingerprint] = RenderLifecycleState.FAILED
                self._currently_rendering.discard(fingerprint)
            self._render_states[fingerprint] = RenderLifecycleState.RENDERED
            if not any(
                item.metadata.get("incremental_render_identity") == identity
                for item in self._result.render_jobs
            ):
                self._result.render_jobs.append(job)
                jobs.append(job)
        return jobs

    def _score_fingerprint(self, score: ClipScore) -> str:
        assert self._source_id is not None
        candidate_id = self._score_candidate_id(score)
        fingerprint = self._score_fingerprint_cache.get(candidate_id)
        if fingerprint is None:
            fingerprint = self._immutable_score_fingerprints.get(candidate_id)
        if fingerprint is None:
            fingerprint = score_fingerprint(score, self._source_id)
            self._score_fingerprint_cache[candidate_id] = fingerprint
            self._score_fingerprint_count += 1
        return fingerprint

    def _score_candidate_id(self, score: ClipScore) -> str:
        candidate_id = self._score_candidate_ids.get(id(score))
        if candidate_id is None:
            candidate_id = self._candidate_id(score.candidate)
            self._score_candidate_ids[id(score)] = candidate_id
        return candidate_id

    def _candidate_id(self, candidate: ClipCandidate) -> str:
        assert self._source_id is not None
        existing = self._candidate_object_ids.get(id(candidate))
        if existing is not None:
            return existing
        self._candidate_fingerprint_count += 1
        identity = candidate_fingerprint(candidate, self._source_id)
        self._candidate_object_ids[id(candidate)] = identity
        return identity

    @staticmethod
    def _observation_id(observation: Observation) -> str:
        return _fingerprint({"observation": observation})

    def _ingest_delta(
        self,
        results: list[ObserverResult],
        watermarks: ObserverWatermarks,
        *,
        strict: bool,
    ) -> None:
        retention_start = max(
            0.0,
            self._watermark
            - self._competition_horizon
            - self._backtrack_horizon,
        )
        for result in results:
            if strict and result.observer not in self._config.required_observers:
                raise ValueError(f"Unexpected incremental observer: {result.observer}.")
            self._active_observations.setdefault(result.observer, [])
            self._observer_metadata.setdefault(result.observer, {})
            observer_watermark = watermarks.stable_through.get(
                result.observer,
                min(float(item) for item in watermarks.stable_through.values()),
            )
            requires_audio_frontier = strict and any(
                item.observer == "audio"
                and item.type in {"speaking_intensity", "silence", "peak"}
                for item in result.observations
            )
            processed_frontier = (
                _validated_audio_processed_frontier(result)
                if requires_audio_frontier
                else None
            )
            whisper_frontier = (
                _validated_whisper_processed_frontier(result)
                if strict
                and any(
                    item.observer == "whisper" and item.type == "speech"
                    for item in result.observations
                )
                else None
            )
            for observation in result.observations:
                end = _observation_end(observation)
                stability_position = _observation_stability_position(observation)
                finalized_audio_peak = (
                    strict
                    and observation.observer == "audio"
                    and observation.type == "peak"
                )
                finalized_whisper_speech = (
                    strict
                    and observation.observer == "whisper"
                    and observation.type == "speech"
                )
                if (
                    not finalized_audio_peak
                    and not finalized_whisper_speech
                    and stability_position > observer_watermark
                ):
                    raise ValueError("Observation delta extends beyond its stable watermark.")
                if (
                    processed_frontier is not None
                    and observation.observer == "audio"
                    and observation.type in {"speaking_intensity", "silence", "peak"}
                    and end > processed_frontier
                ):
                    raise ValueError(
                        "Audio diagnostic observation extends beyond processed frames."
                    )
                if (
                    whisper_frontier is not None
                    and observation.observer == "whisper"
                    and observation.type == "speech"
                    and end > whisper_frontier
                ):
                    raise ValueError(
                        "Finalized Whisper speech extends beyond processed frames."
                    )
                if end < retention_start:
                    raise ValueError(
                        "Observation delta is older than the active revision horizon."
                    )
            self._active_observations[result.observer].extend(result.observations)
            self._active_observations[result.observer].sort(
                key=lambda item: (item.timestamp_seconds, _observation_end(item), item.type)
            )
            self._observer_metadata[result.observer].update(result.metadata)
        active = sum(len(items) for items in self._active_observations.values())
        self._peak_active_observations = max(self._peak_active_observations, active)

    def _generation_timeline(self, timeline: FeatureTimeline) -> FeatureTimeline:
        if self._input_mode == "snapshot":
            return _timeline_with_results(
                timeline,
                [
                    ObserverResult(
                        name,
                        list(observations),
                        dict(self._observer_metadata[name]),
                    )
                    for name, observations in self._active_observations.items()
                ],
            )
        boundary_ids: set[str] = set()
        for score in self._active_scores:
            if score.candidate.end_seconds <= self._immutable_through:
                continue
            observations = score.candidate.metadata.get("contributing_observations", [])
            if isinstance(observations, list):
                boundary_ids.update(
                    self._observation_id(item)
                    for item in observations
                    if isinstance(item, Observation)
                )
        return _timeline_with_results(
            timeline,
            [
                ObserverResult(
                    name,
                    [
                        item
                        for item in self._active_observations[name]
                        if item.timestamp_seconds >= self._immutable_through
                        or self._observation_id(item) in boundary_ids
                    ],
                    dict(self._observer_metadata[name]),
                )
                for name in self._active_observations
            ],
        )

    @staticmethod
    def _has_generation_input(timeline: FeatureTimeline) -> bool:
        return any(
            result.observations
            for result in timeline.timeline.observer_results
        )

    def _should_generate(self, timeline: FeatureTimeline) -> bool:
        return (
            self._has_generation_input(timeline)
            or (self._immutable_through == 0.0 and not self._finalized_scores)
        )

    def _evict_inactive_state(self, stable: float, scores: list[ClipScore]) -> None:
        retention_start = max(
            0.0,
            stable - self._competition_horizon - self._backtrack_horizon,
        )
        if self._input_mode != "snapshot":
            pending_observation_ids: set[str] = set()
            for score in scores:
                if score.candidate.end_seconds <= self._immutable_through:
                    continue
                observations = score.candidate.metadata.get(
                    "contributing_observations", []
                )
                if isinstance(observations, list):
                    pending_observation_ids.update(
                        self._observation_id(item)
                        for item in observations
                        if isinstance(item, Observation)
                    )
            for name, observations in self._active_observations.items():
                self._active_observations[name] = [
                    item
                    for item in observations
                    if _observation_end(item) >= retention_start
                    or self._observation_id(item) in pending_observation_ids
                ]
        active_ids = {self._score_candidate_id(score) for score in scores}
        self._score_cache = {
            identity: score
            for identity, score in self._score_cache.items()
            if identity in active_ids
        }
        self._score_fingerprint_cache = {
            identity: fingerprint
            for identity, fingerprint in self._score_fingerprint_cache.items()
            if identity in active_ids
        }

    def _set_active_scores(self, active: list[ClipScore]) -> None:
        self._active_scores = [
            score
            for score in active
            if score.candidate.end_seconds > self._immutable_through
        ]
        self._peak_active_scores = max(
            self._peak_active_scores, len(self._active_scores)
        )
        self._scores_dirty = True

    def _implicit_delta_identity(
        self,
        results: list[ObserverResult],
        *,
        eof: bool,
    ) -> ObserverDeltaIdentity | None:
        if not results:
            return None
        if len(results) != 1:
            raise ValueError("A delta batch must belong to exactly one observer.")
        observer = results[0].observer
        if self._pending_delta is not None:
            pending_identity = self._pending_delta.identity
            if pending_identity.observer == observer:
                return pending_identity
        sequence = self._accepted_delta_sequences.get(observer, -1) + 1
        return ObserverDeltaIdentity(
            self._source_id or "unactivated",
            self._session_id(),
            observer,
            sequence,
            eof,
        )

    def _implicit_eof_identity(
        self,
        result: ObserverResult,
        eof: IncrementalEOF,
    ) -> ObserverDeltaIdentity:
        observer = result.observer
        last_sequence = self._accepted_delta_sequences.get(observer, -1)
        if last_sequence >= 0:
            previous = ObserverDeltaIdentity(
                self._source_id or "unactivated",
                self._session_id(),
                observer,
                last_sequence,
                True,
            )
            payload = self._delta_payload_fingerprint(
                [result], eof.final_watermarks, previous
            )
            if self._completed_delta_receipts.get(self._identity_key(previous)) == payload:
                return previous
        return ObserverDeltaIdentity(
            self._source_id or "unactivated",
            self._session_id(),
            observer,
            last_sequence + 1,
            True,
        )

    def _validate_delta_identity(
        self,
        identity: ObserverDeltaIdentity,
        results: list[ObserverResult],
        watermarks: ObserverWatermarks,
    ) -> None:
        if identity.source_id != self._source_id or identity.session_id != self._session_id():
            raise ValueError("Delta identity does not belong to this source/session.")
        if identity.observer not in self._config.required_observers:
            raise ValueError(f"Unexpected incremental observer: {identity.observer}.")
        if len(results) != 1 or results[0].observer != identity.observer:
            raise ValueError("Delta identity must match exactly one observer result.")
        expected = self._accepted_delta_sequences[identity.observer] + 1
        if identity.sequence != expected:
            raise ValueError(
                f"Delta sequence for {identity.observer} must be {expected}."
            )
        observation_ids = [
            self._observation_id(item) for item in results[0].observations
        ]
        if any(
            item.observer != identity.observer
            for item in results[0].observations
        ):
            raise ValueError("Delta observations must be owned by the identified observer.")
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("Observer delta contains duplicate observations.")
        active_ids = {
            self._observation_id(item)
            for item in self._active_observations[identity.observer]
        }
        if active_ids.intersection(observation_ids):
            raise ValueError("Observer delta repeats an already accepted observation.")
        current = float(watermarks.stable_through[identity.observer])
        previous = self._observer_watermarks[identity.observer]
        frames = self._delta_frames(results[0])
        previous_frames = self._observer_frames[identity.observer]
        if frames < previous_frames:
            raise ValueError("Observer delta frame position cannot regress.")
        if current < previous:
            raise ValueError("Observer delta watermark cannot regress.")
        if not identity.eof and current <= previous and frames <= previous_frames:
            raise ValueError(
                "Non-EOF observer delta must advance its watermark or frame position."
            )

    @staticmethod
    def _delta_payload_fingerprint(
        results: list[ObserverResult],
        watermarks: ObserverWatermarks,
        identity: ObserverDeltaIdentity,
    ) -> str:
        return _fingerprint(
            {
                "identity": identity,
                "results": results,
                "watermarks": dict(watermarks.stable_through),
            }
        )

    def _record_completed_delta(
        self,
        identity: ObserverDeltaIdentity,
        payload: str,
    ) -> None:
        for key in tuple(self._completed_delta_receipts):
            if key[0:3] == (identity.source_id, identity.session_id, identity.observer):
                del self._completed_delta_receipts[key]
        self._completed_delta_receipts[self._identity_key(identity)] = payload

    def _ensure_receipt_ingested(
        self,
        receipt: _DeltaAcceptance,
        results: list[ObserverResult],
        watermarks: ObserverWatermarks,
    ) -> None:
        if receipt.state is not DeltaAcceptanceState.RECEIVED:
            return
        incoming = {
            self._observation_id(item)
            for result in results
            for item in result.observations
        }
        active = {
            self._observation_id(item)
            for result in results
            for item in self._active_observations[result.observer]
        }
        present = incoming.intersection(active)
        if present and present != incoming:
            raise RuntimeError("Delta ingestion was only partially committed.")
        if not incoming or not present:
            self._ingest_delta(results, watermarks, strict=True)
        receipt.state = DeltaAcceptanceState.INGESTED

    @staticmethod
    def _identity_key(identity: ObserverDeltaIdentity) -> tuple[str, str, str, int, bool]:
        return (
            identity.source_id,
            identity.session_id,
            identity.observer,
            identity.sequence,
            identity.eof,
        )

    def _session_id(self) -> str:
        assert self._source_id is not None
        return self._config.session_id or f"incremental:{self._source_id}"

    @staticmethod
    def _delta_frames(result: ObserverResult) -> int:
        value = result.metadata.get("incremental_frames_processed", 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("Delta frame position must be a non-negative integer.")
        return value

    def _materialize_scores(self) -> None:
        if not self._scores_dirty:
            return
        combined = dict(self._finalized_scores)
        for score in self._active_scores:
            identity = self._score_candidate_id(score)
            if identity not in combined:
                combined[identity] = score
        self._result.scores = sorted(combined.values(), key=self._score_ordering_key)
        self._scores_dirty = False

    @staticmethod
    def _score_ordering_key(score: ClipScore) -> tuple[float, float, float, str, str]:
        candidate = score.candidate
        return (
            -score.overall_score,
            candidate.start_seconds,
            candidate.end_seconds,
            candidate.reason,
            str(candidate.source_video_path),
        )

    def _overlap_groups(self, scores: list[ClipScore]) -> list[list[ClipScore]]:
        ordered = sorted(scores, key=IncrementalPrerecordedCoordinator._chronological_key)
        groups: list[list[ClipScore]] = []
        group_end = -1.0
        for score in ordered:
            competes = getattr(self._selector, "competes", None)
            joins = bool(groups) and any(
                competes(score.candidate, item.candidate)
                if competes is not None
                else score.candidate.start_seconds < item.candidate.end_seconds
                for item in groups[-1]
            )
            if not joins:
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


def _observation_stability_position(observation: Observation) -> float:
    """Return the observer-contract frontier required to accept an observation.

    Audio analysis windows and closed silence spans are immutable once emitted,
    but their duration is diagnostic context rather than an unresolved future
    interval. Their start therefore governs delta acceptance. Their full end is
    still used by rolling retention and candidate-boundary context.
    """

    if observation.observer == "audio" and observation.type in {
        "speaking_intensity",
        "silence",
    }:
        return observation.timestamp_seconds
    return _observation_end(observation)


def _validated_audio_processed_frontier(result: ObserverResult) -> float:
    """Validate the authoritative processed frontier for strict Audio deltas."""

    frames = result.metadata.get("incremental_frames_processed")
    sample_rate = result.metadata.get("sample_rate_hz")
    if (
        not isinstance(frames, int)
        or isinstance(frames, bool)
        or frames <= 0
    ):
        raise ValueError(
            "Strict Audio diagnostic deltas require a positive integer "
            "incremental_frames_processed value."
        )
    if (
        not isinstance(sample_rate, (int, float))
        or isinstance(sample_rate, bool)
        or not math.isfinite(float(sample_rate))
        or float(sample_rate) <= 0
    ):
        raise ValueError(
            "Strict Audio diagnostic deltas require a finite positive numeric "
            "sample_rate_hz value."
        )
    return frames / float(sample_rate)


def _validate_strict_audio_delta_metadata(results: list[ObserverResult]) -> None:
    for result in results:
        strict_audio = [
            item
            for item in result.observations
            if item.observer == "audio"
            and item.type in {"speaking_intensity", "silence", "peak"}
        ]
        if strict_audio:
            _validated_audio_processed_frontier(result)
        peaks = [item for item in strict_audio if item.type == "peak"]
        if peaks:
            _validate_finalized_audio_peaks(result, peaks)


def _validate_finalized_audio_peaks(
    result: ObserverResult, peaks: list[Observation]
) -> None:
    declared = result.metadata.get("finalized_peak_timestamps_seconds")
    if not isinstance(declared, (list, tuple)):
        raise ValueError(
            "Strict Audio peak deltas require finalized peak timestamp metadata."
        )
    values: list[float] = []
    for value in declared:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise ValueError(
                "Finalized Audio peak timestamps must be finite non-negative numbers."
            )
        values.append(float(value))
    if len(values) != len(set(values)):
        raise ValueError("Finalized Audio peak timestamps must be unique.")
    observed = sorted(item.timestamp_seconds for item in peaks)
    if len(observed) != len(set(observed)):
        raise ValueError("Audio peak observations must have unique timestamps.")
    if sorted(values) != observed:
        raise ValueError(
            "Audio peak observations must exactly match finalized peak metadata."
        )


def _validated_whisper_processed_frontier(result: ObserverResult) -> float:
    frames = result.metadata.get("incremental_frames_processed")
    sample_rate = result.metadata.get("sample_rate_hz")
    if (
        not isinstance(frames, int)
        or isinstance(frames, bool)
        or frames <= 0
    ):
        raise ValueError(
            "Strict Whisper speech deltas require a positive integer "
            "incremental_frames_processed value."
        )
    if (
        not isinstance(sample_rate, (int, float))
        or isinstance(sample_rate, bool)
        or not math.isfinite(float(sample_rate))
        or float(sample_rate) <= 0
    ):
        raise ValueError(
            "Strict Whisper speech deltas require a finite positive numeric "
            "sample_rate_hz value."
        )
    return frames / float(sample_rate)


def _validate_strict_whisper_delta_metadata(
    results: list[ObserverResult],
) -> None:
    for result in results:
        speech = [
            item
            for item in result.observations
            if item.observer == "whisper" and item.type == "speech"
        ]
        if not speech:
            continue
        _validated_whisper_processed_frontier(result)
        declared = result.metadata.get("finalized_speech_segment_identities")
        if not isinstance(declared, (list, tuple)):
            raise ValueError(
                "Strict Whisper speech deltas require finalized segment provenance."
            )
        if any(
            not isinstance(item, str)
            or len(item) != 64
            or any(character not in "0123456789abcdef" for character in item)
            for item in declared
        ):
            raise ValueError(
                "Finalized Whisper segment identities must be lowercase SHA-256 values."
            )
        if len(declared) != len(set(declared)):
            raise ValueError("Finalized Whisper segment identities must be unique.")
        observed = [finalized_speech_segment_identity(item) for item in speech]
        if len(observed) != len(set(observed)):
            raise ValueError("Finalized Whisper speech observations must be unique.")
        if sorted(declared) != sorted(observed):
            raise ValueError(
                "Whisper speech observations must exactly match finalized provenance."
            )


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
