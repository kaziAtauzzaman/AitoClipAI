"""Explicit-watermark incremental coordination and completed-timeline replay."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

from candidate_generation import (
    CandidateFamilyId,
    CandidateGenerationAdvance,
    CandidateGenerationCheckpoint,
    CandidateGenerator,
    ClosedCandidateFamily,
    IncrementalCandidateGenerator,
)
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
    SelectionPriorityContract,
    TimelineGroup,
)
from whisper_observer.contracts import finalized_speech_segment_identity


class IncrementalCandidateScorer(Protocol):
    """Scorer whose result depends only on one candidate and fixed configuration."""

    @property
    def candidate_local_deterministic(self) -> bool: ...

    @property
    def selection_priority_contract(self) -> SelectionPriorityContract: ...

    def score(self, candidates: Iterable[ClipCandidate]) -> list[ClipScore]: ...


class IncrementalCandidateSelector(Protocol):
    @property
    def selection_priority_contract(self) -> SelectionPriorityContract: ...

    def select(self, scores: Iterable[ClipScore]) -> CandidateSelectionResult: ...


class IncrementalClipRenderer(Protocol):
    def render_one(self, score: ClipScore, identity: int) -> RenderJob: ...

    def recover_render(
        self,
        score: ClipScore,
        identity: int,
    ) -> RenderJob | None: ...


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
    DECISION_PREPARED = "decision_prepared"
    DECISIONS_COMMITTED = "decisions_committed"
    RENDERING = "rendering"
    COMPLETED = "completed"
    FAILED_RETRYABLE = "failed_retryable"


@dataclass(frozen=True, slots=True)
class _ObserverAcceptanceState:
    """Immutable observer input accepted by the next decision publication."""

    watermarks: Mapping[str, float]
    acceptance_frontiers: Mapping[str, float]
    frames: Mapping[str, int]
    sequences: Mapping[str, int]
    recent_observation_ids: Mapping[str, Mapping[str, float]]


@dataclass(slots=True)
class _DeltaAcceptance:
    """Process-local receipt; deliberately not restored after process termination."""
    identity: ObserverDeltaIdentity
    payload: str
    state: DeltaAcceptanceState = DeltaAcceptanceState.RECEIVED
    observations: tuple[Observation, ...] = ()
    observer_state: _ObserverAcceptanceState | None = None
    generation: CandidateGenerationAdvance | None = None
    scores: list["_FamilyScore"] = field(default_factory=list)
    finalized_scores: list["_FamilyScore"] = field(default_factory=list)
    next_active_scores: list["_FamilyScore"] = field(default_factory=list)
    next_pending_suppressions: list["_PendingSuppression"] = field(
        default_factory=list
    )
    safe_scores: list["_FamilyScore"] = field(default_factory=list)
    prepared_decisions: "_PreparedDecisions | None" = None
    pending_renders: tuple[tuple["_FamilyScore", str, int], ...] = ()
    completed_jobs: list[RenderJob] = field(default_factory=list)
    completed_inputs: list[tuple[ObserverDeltaIdentity, str]] = field(
        default_factory=list
    )
    generation_passes: int = 0
    peak_unresolved_group_size: int = 0

    def release_payloads(self) -> None:
        """Release retry-only payloads after the receipt is durably complete."""

        self.observations = ()
        self.observer_state = None
        self.generation = None
        self.scores.clear()
        self.finalized_scores.clear()
        self.next_active_scores.clear()
        self.next_pending_suppressions.clear()
        self.safe_scores.clear()
        self.prepared_decisions = None
        self.pending_renders = ()
        self.completed_jobs.clear()
        self.completed_inputs.clear()


@dataclass(frozen=True, slots=True)
class _FamilyScore:
    family_id: CandidateFamilyId
    score: ClipScore
    candidate_fingerprint: str
    score_fingerprint: str


@dataclass(frozen=True, slots=True)
class _PendingSuppression:
    loser: _FamilyScore
    selected_neighbors: tuple[_FamilyScore, ...]


@dataclass(frozen=True, slots=True)
class _PreparedDecisions:
    suppressions: tuple[tuple[CandidateFamilyId, SuppressedCandidate], ...]
    selections: tuple[tuple[_FamilyScore, str, int], ...]
    render_identities: tuple[tuple[CandidateFamilyId, int], ...]
    finalized_family_ids: tuple[CandidateFamilyId, ...]
    next_render_identity: int


@dataclass(frozen=True, slots=True)
class _CoordinatorDecisionState:
    generation_checkpoint: CandidateGenerationCheckpoint | None
    selection_priority_identity: tuple[object, ...] = ()
    watermark_seconds: float = 0.0
    render_identity: int = 0
    render_identities: Mapping[CandidateFamilyId, int] = field(
        default_factory=lambda: MappingProxyType({})
    )
    render_family_ids_by_fingerprint: Mapping[
        str, frozenset[CandidateFamilyId]
    ] = field(default_factory=lambda: MappingProxyType({}))
    finalized_family_ids: frozenset[CandidateFamilyId] = frozenset()
    selected_scores: tuple[ClipScore, ...] = ()
    suppressions: tuple[SuppressedCandidate, ...] = ()
    suppression_family_ids: tuple[CandidateFamilyId, ...] = ()
    finalized_scores: Mapping[CandidateFamilyId, ClipScore] = field(
        default_factory=lambda: MappingProxyType({})
    )
    active_scores: tuple[_FamilyScore, ...] = ()
    pending_suppressions: tuple[_PendingSuppression, ...] = ()
    pending_render_plan: tuple[tuple[_FamilyScore, str, int], ...] = ()
    immutable_score_fingerprints: Mapping[CandidateFamilyId, str] = field(
        default_factory=lambda: MappingProxyType({})
    )
    score_family_ids: Mapping[str, frozenset[CandidateFamilyId]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    generation_passes: int = 0
    scored_candidates: int = 0
    candidate_fingerprints: int = 0
    score_fingerprints: int = 0
    peak_active_observations: int = 0
    peak_active_scores: int = 0
    peak_unresolved_group_size: int = 0
    observer_watermarks: Mapping[str, float] = field(
        default_factory=lambda: MappingProxyType({})
    )
    observer_acceptance_frontiers: Mapping[str, float] = field(
        default_factory=lambda: MappingProxyType({})
    )
    observer_frames: Mapping[str, int] = field(
        default_factory=lambda: MappingProxyType({})
    )
    accepted_delta_sequences: Mapping[str, int] = field(
        default_factory=lambda: MappingProxyType({})
    )
    recent_observation_ids: Mapping[str, Mapping[str, float]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    snapshot_observation_ids: frozenset[str] = frozenset()
    committed_snapshot_payload: str | None = None
    committed_snapshot_render_start_index: int = 0
    committed_snapshot_render_end_index: int = 0
    committed_receipt_key: tuple[str, str, str, int, bool] | None = None
    committed_receipt_payload: str | None = None


class CoordinatorLifecycle(str, Enum):
    NEW = "new"
    ACTIVE = "active"
    FLUSHED = "flushed"


@dataclass(frozen=True, slots=True)
class _RenderCompletion:
    render_key: str
    identity: int
    job: RenderJob


@dataclass(frozen=True, slots=True)
class _CompletedDeltaReceipt:
    payload: str
    render_start_index: int = 0
    render_end_index: int = 0


@dataclass(frozen=True, slots=True)
class _CoordinatorAuthoritativeState:
    decision_state: _CoordinatorDecisionState
    pending_delta: _DeltaAcceptance | None = None
    pending_eof: _DeltaAcceptance | None = None
    completed_delta_receipts: Mapping[
        tuple[str, str, str, int, bool], _CompletedDeltaReceipt
    ] = field(default_factory=lambda: MappingProxyType({}))
    completed_eof_fingerprint: str | None = None
    lifecycle: CoordinatorLifecycle = CoordinatorLifecycle.NEW
    render_completions: Mapping[str, _RenderCompletion] = field(
        default_factory=lambda: MappingProxyType({})
    )
    render_jobs: tuple[RenderJob, ...] = ()


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
        (
            self._backtrack_horizon,
            self._selection_priority,
            self._direct_competition_span_seconds,
        ) = self._validate_compatibility()
        self._maximum_selection_ownership_span_seconds = (
            self._direct_competition_span_seconds
            * self._selection_priority.maximum_strictly_improving_chain_length
        )
        self._source_id: str | None = None
        observer_names = self._config.required_observers
        decision_state = _CoordinatorDecisionState(
            None,
            selection_priority_identity=self._selection_priority.identity,
            observer_watermarks=MappingProxyType(
                {name: 0.0 for name in observer_names}
            ),
            observer_acceptance_frontiers=MappingProxyType(
                {name: 0.0 for name in observer_names}
            ),
            observer_frames=MappingProxyType({name: 0 for name in observer_names}),
            accepted_delta_sequences=MappingProxyType(
                {name: -1 for name in observer_names}
            ),
            recent_observation_ids=MappingProxyType(
                {name: MappingProxyType({}) for name in observer_names}
            ),
        )
        self._authoritative_state = _CoordinatorAuthoritativeState(decision_state)
        self._render_attempt_states: dict[str, RenderLifecycleState] = {}
        self._currently_rendering: set[str] = set()
        self._input_mode: str | None = None

    @property
    def _decision_state(self) -> _CoordinatorDecisionState:
        return self._authoritative_state.decision_state

    @_decision_state.setter
    def _decision_state(self, value: _CoordinatorDecisionState) -> None:
        self._authoritative_state = replace(
            self._authoritative_state,
            decision_state=value,
        )

    @property
    def _pending_delta(self) -> _DeltaAcceptance | None:
        return self._authoritative_state.pending_delta

    @_pending_delta.setter
    def _pending_delta(self, value: _DeltaAcceptance | None) -> None:
        self._authoritative_state = replace(
            self._authoritative_state,
            pending_delta=value,
        )

    @property
    def _pending_eof(self) -> _DeltaAcceptance | None:
        return self._authoritative_state.pending_eof

    @_pending_eof.setter
    def _pending_eof(self, value: _DeltaAcceptance | None) -> None:
        self._authoritative_state = replace(
            self._authoritative_state,
            pending_eof=value,
        )

    @property
    def _completed_delta_receipts(
        self,
    ) -> Mapping[tuple[str, str, str, int, bool], _CompletedDeltaReceipt]:
        return self._authoritative_state.completed_delta_receipts

    @property
    def _completed_eof_fingerprint(self) -> str | None:
        return self._authoritative_state.completed_eof_fingerprint

    @property
    def _lifecycle(self) -> CoordinatorLifecycle:
        return self._authoritative_state.lifecycle

    @_lifecycle.setter
    def _lifecycle(self, value: CoordinatorLifecycle) -> None:
        self._authoritative_state = replace(
            self._authoritative_state,
            lifecycle=value,
        )

    @property
    def _render_states(self) -> Mapping[str, RenderLifecycleState]:
        return MappingProxyType(
            {
                **self._render_attempt_states,
                **{
                    key: RenderLifecycleState.RENDERED
                    for key in self._authoritative_state.render_completions
                },
            }
        )

    @property
    def lifecycle(self) -> CoordinatorLifecycle:
        return self._lifecycle

    @property
    def _generation_checkpoint(self) -> CandidateGenerationCheckpoint | None:
        return self._decision_state.generation_checkpoint

    @property
    def _render_identities(self) -> Mapping[CandidateFamilyId, int]:
        return self._decision_state.render_identities

    @property
    def _finalized_scores(self) -> Mapping[CandidateFamilyId, ClipScore]:
        return self._decision_state.finalized_scores

    @property
    def _observer_watermarks(self) -> Mapping[str, float]:
        return self._decision_state.observer_watermarks

    @property
    def _observer_acceptance_frontiers(self) -> Mapping[str, float]:
        return self._decision_state.observer_acceptance_frontiers

    @property
    def _observer_frames(self) -> Mapping[str, int]:
        return self._decision_state.observer_frames

    @property
    def _accepted_delta_sequences(self) -> Mapping[str, int]:
        return self._decision_state.accepted_delta_sequences

    @property
    def _recent_observation_ids(self) -> Mapping[str, Mapping[str, float]]:
        return self._decision_state.recent_observation_ids

    @property
    def watermark_seconds(self) -> float:
        return self._decision_state.watermark_seconds

    @property
    def required_observers(self) -> tuple[str, ...]:
        """Return the immutable observer names required for safe progress."""

        return self._config.required_observers

    @property
    def result(self) -> IncrementalPipelineResult:
        """Return a snapshot of finalized coordinator output."""

        state = self._decision_state
        family_scores = [
            *(state.finalized_scores.items()),
            *((item.family_id, item.score) for item in state.active_scores),
            *(
                (item.loser.family_id, item.loser.score)
                for item in state.pending_suppressions
            ),
        ]
        scores = sorted(
            family_scores,
            key=lambda item: self._family_score_ordering_key(item[0], item[1]),
        )
        selected = [
            (family_id, state.finalized_scores[family_id])
            for family_id in state.render_identities
        ]
        if len(selected) != len(state.selected_scores):
            raise RuntimeError("Selected family ownership is inconsistent.")
        if len(state.suppression_family_ids) != len(state.suppressions):
            raise RuntimeError("Suppressed family ownership is inconsistent.")
        suppressed = list(zip(state.suppression_family_ids, state.suppressions))
        return IncrementalPipelineResult(
            scores=[score for _, score in scores],
            selected_scores=[
                score
                for _, score in sorted(
                    selected,
                    key=lambda item: self._family_score_ordering_key(
                        item[0], item[1]
                    ),
                )
            ],
            suppressed=[
                suppression
                for _, suppression in sorted(
                    suppressed,
                    key=lambda item: self._family_score_ordering_key(
                        item[0], item[1].score
                    ),
                )
            ],
            render_jobs=list(self._authoritative_state.render_jobs),
        )

    def render_jobs_since(self, index: int) -> list[RenderJob]:
        """Return only newly completed render jobs without copying score history."""

        jobs = self._authoritative_state.render_jobs
        if index < 0 or index > len(jobs):
            raise ValueError("Render-job cursor is outside the completed job range.")
        return list(jobs[index:])

    def render_job_at(self, index: int) -> RenderJob | None:
        """Return one completed job in O(1), or None at the current end."""

        jobs = self._authoritative_state.render_jobs
        if index < 0 or index > len(jobs):
            raise ValueError("Render-job cursor is outside the completed job range.")
        if index == len(jobs):
            return None
        return jobs[index]

    @property
    def state_metrics(self) -> IncrementalStateMetrics:
        state = self._decision_state
        active = self._retained_observation_count()
        active_scores = self._retained_active_score_count()
        return IncrementalStateMetrics(
            state.generation_passes,
            state.scored_candidates,
            state.candidate_fingerprints,
            state.score_fingerprints,
            active,
            max(state.peak_active_observations, active),
            active_scores,
            len(state.finalized_scores),
            len(state.immutable_score_fingerprints),
            len(self._authoritative_state.render_jobs),
            max(state.peak_active_scores, active_scores),
            state.peak_unresolved_group_size,
        )

    def _retained_observation_count(self) -> int:
        count = 0
        checkpoint = self._decision_state.generation_checkpoint
        if checkpoint is not None:
            count += checkpoint.retained_observation_count
        for receipt in (self._pending_delta, self._pending_eof):
            if receipt is None:
                continue
            count += self._unique_observation_count(receipt.observations)
            if receipt.generation is not None and receipt.generation.checkpoint is not None:
                count += receipt.generation.checkpoint.retained_observation_count
        return count

    def _retained_active_score_count(self) -> int:
        identities = {
            item.family_id for item in self._decision_state.active_scores
        }
        identities.update(
            item.loser.family_id
            for item in self._decision_state.pending_suppressions
        )
        identities.update(
            neighbor.family_id
            for item in self._decision_state.pending_suppressions
            for neighbor in item.selected_neighbors
        )
        for receipt in (self._pending_delta, self._pending_eof):
            if receipt is None:
                continue
            for collection in (
                receipt.scores,
                receipt.finalized_scores,
                receipt.next_active_scores,
                receipt.safe_scores,
            ):
                identities.update(item.family_id for item in collection)
            identities.update(
                item.loser.family_id
                for item in receipt.next_pending_suppressions
            )
            identities.update(
                neighbor.family_id
                for item in receipt.next_pending_suppressions
                for neighbor in item.selected_neighbors
            )
            identities.update(
                item.family_id for item, _, _ in receipt.pending_renders
            )
        return len(identities)

    @staticmethod
    def _unique_observation_count(observations: Sequence[Observation]) -> int:
        return len({id(item) for item in observations})

    def render_state(self, score: ClipScore) -> RenderLifecycleState | None:
        state = self._decision_state
        fingerprint = score_fingerprint(score, self._source_id or "unactivated")
        family_ids = state.score_family_ids.get(fingerprint, ())
        states = {
            self._render_states.get(self._render_key(item, fingerprint))
            for item in family_ids
        }
        states.discard(None)
        if RenderLifecycleState.FAILED in states:
            return RenderLifecycleState.FAILED
        if RenderLifecycleState.RENDERING in states:
            return RenderLifecycleState.RENDERING
        if RenderLifecycleState.RENDERED in states:
            return RenderLifecycleState.RENDERED
        return None

    def advance(
        self,
        timeline: FeatureTimeline,
        watermarks: ObserverWatermarks,
    ) -> list[RenderJob]:
        """Consume only observations confirmed stable by required observers."""

        if self._input_mode not in (None, "snapshot"):
            raise RuntimeError("Cannot mix snapshot and delta coordinator input modes.")
        self._input_mode = "snapshot"
        self._activate(timeline)
        snapshot_payload = self._snapshot_payload_fingerprint(timeline, watermarks)
        state = self._decision_state
        if state.committed_snapshot_payload == snapshot_payload:
            plan = state.pending_render_plan
            if plan:
                self._render_plan(plan)
                self._clear_committed_render_plan(plan)
            return list(
                self._authoritative_state.render_jobs[
                    state.committed_snapshot_render_start_index :
                    state.committed_snapshot_render_end_index
                ]
            )
        delta_results: list[ObserverResult] = []
        seen = self._decision_state.snapshot_observation_ids
        for result in timeline.timeline.observer_results:
            identities = [self._observation_id(item) for item in result.observations]
            delta_results.append(
                ObserverResult(
                    result.observer,
                    [
                        item
                        for identity, item in zip(identities, result.observations)
                        if identity not in seen
                    ],
                    dict(result.metadata),
                )
            )
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
        identity = identity or self._implicit_delta_identity(
            results,
            watermarks,
            eof=False,
        )
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
        if identity is None and stable < self._decision_state.watermark_seconds:
            raise ValueError("Stable watermark cannot move backwards.")
        observations = tuple(
            observation
            for result in delta_results
            for observation in result.observations
        )
        recovered_jobs: list[RenderJob] = []
        if identity is None and self._decision_state.pending_render_plan:
            plan = self._decision_state.pending_render_plan
            recovered_jobs = self._render_plan(plan)
            self._clear_committed_render_plan(plan)
            if not observations:
                return recovered_jobs
        if identity is not None:
            _validate_strict_audio_delta_metadata(delta_results)
            _validate_strict_whisper_delta_metadata(delta_results)
            payload = self._delta_payload_fingerprint(
                delta_results, watermarks, identity
            )
            completed = self._completed_delta_receipts.get(self._identity_key(identity))
            if completed is not None:
                if completed.payload != payload:
                    raise ValueError(
                        "Delta identity was reused with different content."
                    )
                return [
                    *self._authoritative_state.render_jobs[
                        completed.render_start_index : completed.render_end_index
                    ]
                ]
            if stable < self._decision_state.watermark_seconds:
                raise ValueError("Stable watermark cannot move backwards.")
            if self._pending_delta is None:
                receipt = _DeltaAcceptance(
                    identity,
                    payload,
                    observations=observations,
                )
                self._ensure_receipt_ingested(
                    receipt,
                    delta_results,
                    watermarks,
                )
                self._pending_delta = receipt
            elif (
                self._pending_delta.identity != identity
                or self._pending_delta.payload != payload
            ):
                raise RuntimeError(
                    "A previously accepted delta must finish before new input."
                )
            else:
                receipt = self._pending_delta
                self._ensure_receipt_ingested(
                    receipt,
                    delta_results,
                    watermarks,
                )
            if self._receipt_decision_published(receipt):
                receipt.pending_renders = self._decision_state.pending_render_plan
                receipt.state = DeltaAcceptanceState.DECISIONS_COMMITTED
                observer_state = self._current_observer_state()
            else:
                observer_state = receipt.observer_state
                assert observer_state is not None
        else:
            observer_state = self._propose_ingestion(
                delta_results,
                watermarks,
                strict=False,
            )
            observer_state = self._evicted_observer_state(observer_state)
            self._transaction_preparation_step("observer_proposal")
        receipt = self._pending_delta if identity is not None else None
        if receipt is None:
            generation = self._advance_generation(
                observations,
                stable,
                observer_state,
            )
            self._transaction_preparation_step("generator_proposal")
            scores = self._scores_for_closed_families(generation.closed_families)
            finalized, active, pending_suppressions, selected, suppressed, peak_group = (
                self._partition_family_scores(
                    scores,
                    self._earliest_future_candidate_start(generation),
                )
            )
            prepared = self._prepare_decisions(selected, suppressed)
            plan = self._publish_generation_decision(
                generation,
                scores,
                finalized,
                active,
                pending_suppressions,
                prepared,
                generation_passes=1,
                peak_unresolved_group_size=peak_group,
                receipt_observation_count=0,
                accepted_watermark=stable,
                observer_state=observer_state,
                snapshot_observation_ids=frozenset(
                    self._observation_id(observation)
                    for observation in observations
                ),
                committed_snapshot_payload=self._snapshot_payload_fingerprint(
                    timeline,
                    watermarks,
                ),
                committed_receipt=None,
            )
        else:
            if receipt.state is DeltaAcceptanceState.INGESTED:
                if receipt.generation is None:
                    assert receipt.observer_state is not None
                    receipt.generation = self._advance_generation(
                        receipt.observations,
                        stable,
                        receipt.observer_state,
                    )
                    receipt.generation_passes = 1
                    self._transaction_preparation_step("generator_proposal")
                receipt.scores = self._scores_for_closed_families(
                    receipt.generation.closed_families
                )
                receipt.state = DeltaAcceptanceState.GENERATION_SCORED
            if receipt.state is DeltaAcceptanceState.GENERATION_SCORED:
                assert receipt.generation is not None
                (
                    finalized,
                    active,
                    pending_suppressions,
                    selected,
                    suppressed,
                    peak_group,
                ) = self._partition_family_scores(
                    receipt.scores,
                    self._earliest_future_candidate_start(receipt.generation),
                )
                prepared = self._prepare_decisions(selected, suppressed)
                receipt.finalized_scores = finalized
                receipt.next_active_scores = active
                receipt.next_pending_suppressions = pending_suppressions
                receipt.safe_scores = [
                    *selected,
                    *(item.loser for item in suppressed),
                ]
                receipt.peak_unresolved_group_size = peak_group
                receipt.prepared_decisions = prepared
                receipt.state = DeltaAcceptanceState.DECISION_PREPARED
            if receipt.state is DeltaAcceptanceState.DECISION_PREPARED:
                assert receipt.generation is not None
                assert receipt.prepared_decisions is not None
                assert receipt.observer_state is not None
                receipt.pending_renders = self._publish_generation_decision(
                    receipt.generation,
                    receipt.scores,
                    receipt.finalized_scores,
                    receipt.next_active_scores,
                    receipt.next_pending_suppressions,
                    receipt.prepared_decisions,
                    generation_passes=receipt.generation_passes,
                    peak_unresolved_group_size=receipt.peak_unresolved_group_size,
                    receipt_observation_count=self._unique_observation_count(
                        receipt.observations
                    ),
                    accepted_watermark=stable,
                    observer_state=receipt.observer_state,
                    snapshot_observation_ids=frozenset(),
                    committed_snapshot_payload=None,
                    committed_receipt=(
                        self._identity_key(receipt.identity),
                        receipt.payload,
                    ),
                )
                receipt.state = DeltaAcceptanceState.DECISIONS_COMMITTED
        try:
            if receipt is None:
                plan = self._decision_state.pending_render_plan
                jobs = self._render_plan(plan)
                self._clear_committed_render_plan(plan)
            else:
                receipt.state = DeltaAcceptanceState.RENDERING
                jobs = self._render_plan(receipt.pending_renders)
                receipt.completed_jobs[:] = jobs
        except BaseException:
            if receipt is not None:
                receipt.state = DeltaAcceptanceState.FAILED_RETRYABLE
            raise
        if identity is not None:
            render_identities = [
                render_identity
                for _, _, render_identity in receipt.pending_renders
            ]
            if render_identities:
                render_start_index = min(render_identities) - 1
                render_end_index = max(render_identities)
                if (
                    sorted(render_identities)
                    != list(range(render_start_index + 1, render_end_index + 1))
                    or render_end_index
                    > len(self._authoritative_state.render_jobs)
                ):
                    raise RuntimeError(
                        "Committed receipt render identities are not contiguous."
                    )
            else:
                render_start_index = render_end_index = len(
                    self._authoritative_state.render_jobs
                )
            receipt.state = DeltaAcceptanceState.COMPLETED
            completed = self._completed_delta_receipts_after(
                self._completed_delta_receipts,
                identity,
                payload,
                render_start_index,
                render_end_index,
            )
            self._completion_preparation_step("delta_completion")
            self._authoritative_state = replace(
                self._authoritative_state,
                decision_state=replace(
                    self._decision_state,
                    pending_render_plan=(),
                ),
                pending_delta=None,
                completed_delta_receipts=completed,
            )
            receipt.release_payloads()
            self._completion_publication_step("after_delta_completion")
        return [*recovered_jobs, *jobs]

    def flush(
        self,
        timeline: FeatureTimeline,
        eof: IncrementalEOF,
    ) -> IncrementalPipelineResult:
        """Finalize once, after authoritative EOF from every required observer."""

        if self._input_mode not in (None, "snapshot"):
            raise RuntimeError("Cannot mix snapshot and delta coordinator input modes.")
        self._input_mode = "snapshot"
        if self._lifecycle is CoordinatorLifecycle.FLUSHED:
            self._validate_eof(eof)
            identities = {
                self._observation_id(item)
                for result in timeline.timeline.observer_results
                for item in result.observations
            }
            state = self._decision_state
            if (
                stable_source_id(timeline) == self._source_id
                and identities == state.snapshot_observation_ids
                and eof.media_duration_seconds == state.watermark_seconds
                and dict(eof.final_watermarks.stable_through)
                == dict(state.observer_watermarks)
            ):
                return self.result
            raise RuntimeError("Incremental coordinator has already been flushed.")
        delta_results: list[ObserverResult] = []
        seen = self._decision_state.snapshot_observation_ids
        pending_ids = (
            {
                self._observation_id(item)
                for item in self._pending_eof.observations
            }
            if self._pending_eof is not None
            else set()
        )
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
                        if identity not in seen or identity in pending_ids
                    ],
                    dict(result.metadata),
                )
            )
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
        if self._source_id is None:
            self._activate(timeline)
        results = list(timeline.timeline.observer_results)
        if results and not identities:
            if self._pending_eof is not None and self._pending_eof.completed_inputs:
                identities = tuple(
                    identity for identity, _ in self._pending_eof.completed_inputs
                )
            else:
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
        if self._lifecycle is CoordinatorLifecycle.FLUSHED:
            if stable_source_id(timeline) != self._source_id:
                raise RuntimeError("Incremental coordinator is single-use for one source.")
            self._validate_eof(eof)
            eof_payload = self._combined_eof_payload(
                delta_results,
                eof,
                identities,
            )
            if eof_payload != self._completed_eof_fingerprint:
                raise RuntimeError("Incremental coordinator has already been flushed.")
            return self.result
        self._activate(timeline)
        self._validate_eof(eof)
        if self._pending_delta is not None:
            raise RuntimeError("A pending observer delta must finish before EOF.")
        if self._decision_state.pending_render_plan:
            plan = self._decision_state.pending_render_plan
            self._render_plan(plan)
            self._clear_committed_render_plan(plan)
        eof_identity = ObserverDeltaIdentity(
            self._source_id or "unactivated",
            self._session_id(),
            "__combined_eof__",
            0,
            True,
        )
        eof_payload = self._combined_eof_payload(delta_results, eof, identities)
        if self._pending_eof is None:
            accepted_observations: list[Observation] = []
            completed_inputs: list[tuple[ObserverDeltaIdentity, str]] = []
            observer_state = self._current_observer_state()
            if delta_results and not identities:
                for result in delta_results:
                    observer_state = self._propose_ingestion(
                        [result],
                        eof.final_watermarks,
                        strict=False,
                        state=observer_state,
                    )
                    accepted_observations.extend(result.observations)
                    self._transaction_preparation_step(
                        f"eof_observer:{result.observer}"
                    )
            for identity in identities:
                matching = [
                    item for item in delta_results if item.observer == identity.observer
                ]
                _validate_strict_audio_delta_metadata(matching)
                _validate_strict_whisper_delta_metadata(matching)
                payload = self._delta_payload_fingerprint(
                    matching, eof.final_watermarks, identity
                )
                completed = self._completed_delta_receipts.get(
                    self._identity_key(identity)
                )
                if completed is not None:
                    if completed.payload != payload:
                        raise ValueError(
                            "Delta identity was reused with different content."
                        )
                    continue
                self._validate_delta_identity(
                    identity,
                    matching,
                    eof.final_watermarks,
                    observer_state,
                )
                observer_state = self._propose_ingestion(
                    matching,
                    eof.final_watermarks,
                    strict=True,
                    state=observer_state,
                )
                observer_state = self._propose_delta_identity(
                    observer_state,
                    identity,
                    matching[0],
                    eof.final_watermarks,
                )
                completed_inputs.append((identity, payload))
                accepted_observations.extend(matching[0].observations)
                self._transaction_preparation_step(
                    f"eof_observer:{identity.observer}"
                )
            observer_state = self._evicted_observer_state(
                observer_state,
                terminal=True,
            )
            self._transaction_preparation_step("observer_proposal")
            self._pending_eof = _DeltaAcceptance(
                eof_identity,
                eof_payload,
                DeltaAcceptanceState.INGESTED,
                observations=tuple(accepted_observations),
                observer_state=observer_state,
                completed_inputs=completed_inputs,
            )
        elif self._pending_eof.payload != eof_payload:
            raise ValueError("Combined EOF receipt was reused with different content.")
        receipt = self._pending_eof
        if self._receipt_decision_published(receipt):
            receipt.pending_renders = self._decision_state.pending_render_plan
            receipt.state = DeltaAcceptanceState.DECISIONS_COMMITTED
        if receipt.state is DeltaAcceptanceState.INGESTED:
            if receipt.generation is None:
                receipt.generation = self._finalize_generation(
                    receipt.observations,
                    eof.media_duration_seconds,
                )
                receipt.generation_passes = 1
                self._transaction_preparation_step("generator_proposal")
            receipt.scores = self._scores_for_closed_families(
                receipt.generation.closed_families
            )
            receipt.state = DeltaAcceptanceState.GENERATION_SCORED
        if receipt.state is DeltaAcceptanceState.GENERATION_SCORED:
            assert receipt.generation is not None
            (
                finalized,
                active,
                pending_suppressions,
                selected,
                suppressed,
                peak_group,
            ) = self._partition_family_scores(
                receipt.scores,
                float("inf"),
                eof=True,
            )
            prepared = self._prepare_decisions(selected, suppressed)
            receipt.finalized_scores = finalized
            receipt.next_active_scores = active
            receipt.next_pending_suppressions = pending_suppressions
            receipt.safe_scores = [
                *selected,
                *(item.loser for item in suppressed),
            ]
            receipt.peak_unresolved_group_size = peak_group
            receipt.prepared_decisions = prepared
            receipt.state = DeltaAcceptanceState.DECISION_PREPARED
        if receipt.state is DeltaAcceptanceState.DECISION_PREPARED:
            assert receipt.generation is not None
            assert receipt.prepared_decisions is not None
            assert receipt.observer_state is not None
            receipt.pending_renders = self._publish_generation_decision(
                receipt.generation,
                receipt.scores,
                receipt.finalized_scores,
                receipt.next_active_scores,
                receipt.next_pending_suppressions,
                receipt.prepared_decisions,
                generation_passes=receipt.generation_passes,
                peak_unresolved_group_size=receipt.peak_unresolved_group_size,
                receipt_observation_count=self._unique_observation_count(
                    receipt.observations
                ),
                accepted_watermark=eof.media_duration_seconds,
                observer_state=receipt.observer_state,
                snapshot_observation_ids=(
                    frozenset(
                        self._observation_id(observation)
                        for observation in receipt.observations
                    )
                    if self._input_mode == "snapshot"
                    else frozenset()
                ),
                committed_snapshot_payload=None,
                committed_receipt=(
                    self._identity_key(receipt.identity),
                    receipt.payload,
                ),
            )
            receipt.state = DeltaAcceptanceState.DECISIONS_COMMITTED
        try:
            receipt.state = DeltaAcceptanceState.RENDERING
            jobs = self._render_plan(receipt.pending_renders)
            receipt.completed_jobs[:] = jobs
        except BaseException:
            receipt.state = DeltaAcceptanceState.FAILED_RETRYABLE
            raise
        completed = self._completed_delta_receipts
        for input_identity, payload in receipt.completed_inputs:
            completed = self._completed_delta_receipts_after(
                completed,
                input_identity,
                payload,
                len(self._authoritative_state.render_jobs),
                len(self._authoritative_state.render_jobs),
            )
        receipt.state = DeltaAcceptanceState.COMPLETED
        output = self.result
        self._completion_preparation_step("eof_completion")
        self._authoritative_state = replace(
            self._authoritative_state,
            decision_state=replace(
                self._decision_state,
                pending_render_plan=(),
            ),
            pending_eof=None,
            completed_delta_receipts=completed,
            completed_eof_fingerprint=eof_payload,
            lifecycle=CoordinatorLifecycle.FLUSHED,
        )
        receipt.release_payloads()
        self._completion_publication_step("after_eof_completion")
        return output

    def _validate_compatibility(
        self,
    ) -> tuple[float, SelectionPriorityContract, float]:
        backtrack = getattr(self._generator, "maximum_backtrack_seconds", None)
        deterministic = getattr(self._generator, "incremental_deterministic", False)
        candidate_local = getattr(self._scorer, "candidate_local_deterministic", False)
        start_incremental = getattr(self._generator, "start_incremental", None)
        bind_incremental_publication = getattr(
            self._generator,
            "bind_incremental_publication",
            None,
        )
        advance_incremental = getattr(self._generator, "advance_incremental", None)
        finalize_incremental = getattr(self._generator, "finalize_incremental", None)
        future_candidate_start = getattr(
            self._generator,
            "earliest_future_candidate_start_seconds",
            None,
        )
        direct_competition_span = getattr(
            self._generator,
            "maximum_direct_competition_span_seconds",
            getattr(self._generator, "maximum_competition_seconds", None),
        )
        recover_render = getattr(self._renderer, "recover_render", None)
        if isinstance(backtrack, bool) or not isinstance(backtrack, int | float):
            raise ValueError(
                "Incremental generator must declare maximum_backtrack_seconds."
            )
        backtrack = float(backtrack)
        if not math.isfinite(backtrack) or backtrack < 0:
            raise ValueError(
                "Generator maximum_backtrack_seconds must be finite and non-negative."
            )
        if deterministic is not True:
            raise ValueError(
                "Incremental generator must declare deterministic prefix output."
            )
        if candidate_local is not True:
            raise ValueError(
                "Incremental scorer must be deterministic and candidate-local."
            )
        if not all(
            callable(item)
            for item in (
                start_incremental,
                bind_incremental_publication,
                advance_incremental,
                finalize_incremental,
            )
        ):
            raise ValueError(
                "Incremental generator must implement generator-owned continuation."
            )
        if not callable(future_candidate_start):
            raise ValueError(
                "Incremental generator must prove its earliest future candidate start."
            )
        if (
            isinstance(direct_competition_span, bool)
            or not isinstance(direct_competition_span, int | float)
            or not math.isfinite(float(direct_competition_span))
            or float(direct_competition_span) < 0
        ):
            raise ValueError(
                "Incremental generator must declare a finite non-negative direct "
                "competition span."
            )
        if not callable(recover_render):
            raise ValueError(
                "Incremental renderer must support durable render recovery by identity."
            )
        scorer_priority = getattr(
            self._scorer,
            "selection_priority_contract",
            None,
        )
        selector_priority = getattr(
            self._selector,
            "selection_priority_contract",
            None,
        )
        if not isinstance(scorer_priority, SelectionPriorityContract):
            raise ValueError(
                "Incremental scorer must expose a SelectionPriorityContract."
            )
        if not isinstance(selector_priority, SelectionPriorityContract):
            raise ValueError(
                "Incremental selector must expose a SelectionPriorityContract."
            )
        if scorer_priority.identity != selector_priority.identity:
            raise ValueError(
                "Incremental scorer and selector selection-priority contracts differ."
            )
        return backtrack, scorer_priority, float(direct_competition_span)

    def _validate_selection_priority_compatibility(self) -> None:
        """Reject configuration mutation or reuse under a different alphabet."""

        scorer_priority = getattr(
            self._scorer,
            "selection_priority_contract",
            None,
        )
        selector_priority = getattr(
            self._selector,
            "selection_priority_contract",
            None,
        )
        expected = self._decision_state.selection_priority_identity
        if (
            not isinstance(scorer_priority, SelectionPriorityContract)
            or not isinstance(selector_priority, SelectionPriorityContract)
            or scorer_priority.identity != expected
            or selector_priority.identity != expected
        ):
            raise RuntimeError(
                "Incremental selection-priority configuration changed during a session."
            )

    def _activate(self, timeline: FeatureTimeline) -> None:
        if self._lifecycle is CoordinatorLifecycle.FLUSHED:
            raise RuntimeError("Incremental coordinator has already been flushed.")
        self._validate_selection_priority_compatibility()
        source_id = stable_source_id(timeline)
        if self._source_id is None:
            checkpoint = self._generator.start_incremental(
                source_id=source_id,
                media_path=timeline.media_path,
                required_observers=self._config.required_observers,
            )
            if not isinstance(checkpoint, CandidateGenerationCheckpoint):
                raise TypeError(
                    "Incremental generator returned an unsupported initial checkpoint."
                )
            if (
                checkpoint.source_id != source_id
                or checkpoint.media_path != timeline.media_path
            ):
                raise RuntimeError(
                    "Incremental generator initial checkpoint changed source ownership."
                )
            if (
                checkpoint.stable_through_seconds != 0.0
                or checkpoint.next_family_ordinal != 0
                or checkpoint.retained_observation_count != 0
                or checkpoint._required_observers
                != self._config.required_observers
                or dict(checkpoint.observer_frontiers)
                != {
                    observer: 0.0
                    for observer in self._config.required_observers
                }
            ):
                raise RuntimeError(
                    "Incremental generator initial checkpoint was not empty."
                )
            self._generator.bind_incremental_publication(
                checkpoint,
                lambda: self._decision_state.generation_checkpoint,
            )
            self._source_id = source_id
            self._decision_state = replace(
                self._decision_state,
                generation_checkpoint=checkpoint,
            )
            self._lifecycle = CoordinatorLifecycle.ACTIVE
        elif source_id != self._source_id:
            raise RuntimeError("Incremental coordinator is single-use for one source.")
        elif (
            self._decision_state.generation_checkpoint is not None
            and self._decision_state.generation_checkpoint.media_path
            != timeline.media_path
        ):
            raise RuntimeError("Incremental coordinator source path changed.")
        self._recover_stale_rendering()

    def _recover_stale_rendering(self) -> None:
        for render_key in tuple(self._currently_rendering):
            if (
                render_key not in self._authoritative_state.render_completions
                and self._render_attempt_states.get(render_key)
                is RenderLifecycleState.RENDERING
            ):
                self._render_attempt_states[render_key] = RenderLifecycleState.FAILED
            self._currently_rendering.discard(render_key)

    def _validated_global_watermark(self, watermarks: ObserverWatermarks) -> float:
        missing = [
            item
            for item in self._config.required_observers
            if item not in watermarks.stable_through
        ]
        if missing:
            raise ValueError(
                f"Missing required observer watermarks: {', '.join(missing)}"
            )
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
            raise ValueError(
                "Every required observer must confirm the final media duration."
            )

    def _advance_generation(
        self,
        observations: tuple[Observation, ...],
        stable: float,
        observer_state: _ObserverAcceptanceState,
    ) -> CandidateGenerationAdvance:
        checkpoint = self._decision_state.generation_checkpoint
        if checkpoint is None:
            raise RuntimeError("Incremental generator has no active checkpoint.")
        output = self._generator.advance_incremental(
            checkpoint,
            observations,
            stable,
            {
                observer: observer_state.acceptance_frontiers[observer]
                for observer in self._config.required_observers
            },
        )
        self._validate_generation_output(
            output,
            checkpoint,
            terminal=False,
            expected_stable=stable,
            observer_state=observer_state,
        )
        return output

    def _finalize_generation(
        self,
        observations: tuple[Observation, ...],
        duration: float,
    ) -> CandidateGenerationAdvance:
        checkpoint = self._decision_state.generation_checkpoint
        if checkpoint is None:
            raise RuntimeError("Incremental generator has no active checkpoint.")
        output = self._generator.finalize_incremental(
            checkpoint,
            observations,
            duration,
        )
        self._validate_generation_output(output, checkpoint, terminal=True)
        return output

    def _validate_generation_output(
        self,
        output: CandidateGenerationAdvance,
        previous: CandidateGenerationCheckpoint,
        *,
        terminal: bool,
        expected_stable: float | None = None,
        observer_state: _ObserverAcceptanceState | None = None,
    ) -> None:
        if not isinstance(output, CandidateGenerationAdvance):
            raise TypeError("Incremental generator returned an unsupported result.")
        if terminal != (output.checkpoint is None):
            raise RuntimeError(
                "Incremental generator returned an invalid terminal checkpoint."
            )
        expected_ordinal = previous.next_family_ordinal
        for family in output.closed_families:
            if not isinstance(family, ClosedCandidateFamily):
                raise TypeError("Incremental generator returned a malformed family.")
            if family.family_id != CandidateFamilyId(
                previous.source_id, expected_ordinal
            ):
                raise RuntimeError(
                    "Incremental generator family lineage is not contiguous."
                )
            if (
                family.candidate is not None
                and Path(family.candidate.source_video_path) != previous.media_path
            ):
                raise RuntimeError(
                    "Incremental generator candidate belongs to another source."
                )
            expected_ordinal += 1
        if output.checkpoint is not None:
            checkpoint = output.checkpoint
            if (
                checkpoint.source_id != previous.source_id
                or checkpoint.media_path != previous.media_path
                or checkpoint._owner_token is not previous._owner_token
                or checkpoint._required_observers != previous._required_observers
            ):
                raise RuntimeError(
                    "Incremental generator checkpoint changed source ownership."
                )
            if checkpoint.next_family_ordinal != expected_ordinal:
                raise RuntimeError(
                    "Incremental generator checkpoint skipped family lineage."
                )
            if checkpoint.stable_through_seconds != expected_stable:
                raise RuntimeError(
                    "Incremental generator checkpoint changed the accepted frontier."
                )
            if dict(checkpoint.observer_frontiers) != dict(
                (
                    observer,
                    (
                        observer_state or self._current_observer_state()
                    ).acceptance_frontiers[observer],
                )
                for observer in self._config.required_observers
            ):
                raise RuntimeError(
                    "Incremental generator checkpoint changed observer frontiers."
                )

    def _scores_for_closed_families(
        self,
        families: tuple[ClosedCandidateFamily, ...],
    ) -> list[_FamilyScore]:
        candidates = [family for family in families if family.candidate is not None]
        if not candidates:
            return []
        generated_candidates = [item.candidate for item in candidates]
        scored = list(self._scorer.score(generated_candidates))
        if len(scored) != len(generated_candidates):
            raise RuntimeError(
                "Candidate scorer did not return one score per closed family."
            )
        assert self._source_id is not None
        unmatched = list(candidates)
        output: list[_FamilyScore] = []
        seen: set[CandidateFamilyId] = set()
        for score in scored:
            if not isinstance(score, ClipScore):
                raise RuntimeError("Candidate scorer returned a malformed score.")
            if (
                not isinstance(score.overall_score, float)
                or not math.isfinite(score.overall_score)
            ):
                raise RuntimeError(
                    "Incremental candidate scores must use finite float priorities."
                )
            try:
                self._selection_priority.normalize(score.overall_score)
            except ValueError as exc:
                raise RuntimeError(
                    "Incremental candidate score is outside the configured finite "
                    "selection-priority alphabet."
                ) from exc
            fingerprint = candidate_fingerprint(score.candidate, self._source_id)
            matching_index = next(
                (
                    index
                    for index, family in enumerate(unmatched)
                    if Path(family.candidate.source_video_path)
                    == Path(score.candidate.source_video_path)
                    and candidate_fingerprint(family.candidate, self._source_id)
                    == fingerprint
                ),
                None,
            )
            if matching_index is None:
                raise RuntimeError("Candidate scorer changed closed-family ownership.")
            family = unmatched.pop(matching_index)
            family_id = family.family_id
            if family_id in seen:
                raise RuntimeError("Candidate scorer repeated closed-family ownership.")
            seen.add(family_id)
            output.append(
                _FamilyScore(
                    family_id,
                    score,
                    fingerprint,
                    score_fingerprint(score, self._source_id),
                )
            )
        if unmatched or len(seen) != len(candidates):
            raise RuntimeError("Candidate scorer omitted a closed family.")
        return sorted(output, key=lambda item: self._score_ordering_key(item.score))

    def _partition_family_scores(
        self,
        new_scores: list[_FamilyScore],
        earliest_future_start: float,
        *,
        eof: bool = False,
    ) -> tuple[
        list[_FamilyScore],
        list[_FamilyScore],
        list[_PendingSuppression],
        list[_FamilyScore],
        list[_PendingSuppression],
        int,
    ]:
        """Resolve the greedy selector only where direct future competition is closed.

        Connected overlap components need not have a finite duration.  A score is
        nevertheless irrevocably selected once its own future-neighbor interval
        is closed and no higher-priority unresolved direct neighbor remains.
        Selected scores immediately eliminate their direct neighbors.  A loser's
        suppression provenance remains pending until all of its possible selected
        neighbors are known, preserving the completed-batch retained-score rule.
        Along a still-open forward dependency chain, selector priority must
        strictly improve. Equal normalized ranks retain generator emission order,
        so a later equal-rank candidate cannot improve the chain. The configured
        finite alphabet has ``rank_count`` ranks; therefore no forward improving
        dependency path can contain more than ``rank_count`` candidates. Direct
        competition locality bounds each retained rank to the generator's finite
        live candidate neighborhood, independently of stream duration, even when
        the connected overlap graph itself is arbitrarily long.
        """

        state = self._decision_state
        unresolved = [
            *(
                item
                for item in state.active_scores
                if item.family_id not in state.finalized_family_ids
            ),
            *new_scores,
        ]
        pending = list(state.pending_suppressions)
        family_ids = [
            *(item.family_id for item in unresolved),
            *(item.loser.family_id for item in pending),
        ]
        if len(family_ids) != len(set(family_ids)):
            raise RuntimeError(
                "Closed family was presented for scoring more than once."
            )
        failing = [
            item for item in unresolved if item.score.passed_threshold is not True
        ]
        unresolved = [
            item for item in unresolved if item.score.passed_threshold is True
        ]
        selected: list[_FamilyScore] = []
        ready_suppressions: list[_PendingSuppression] = []
        peak_group_size = len(unresolved) + len(pending)

        while True:
            changed = False
            ordered = sorted(
                unresolved,
                key=lambda item: self._score_ordering_key(item.score),
            )
            for item in ordered:
                candidate = item.score.candidate
                future_closed = eof or candidate.end_seconds <= earliest_future_start
                if not future_closed:
                    continue
                if any(
                    self._scores_compete(item.score, other.score)
                    and self._score_ordering_key(other.score)
                    < self._score_ordering_key(item.score)
                    for other in unresolved
                    if other.family_id != item.family_id
                ):
                    continue

                selected.append(item)
                unresolved = [
                    other
                    for other in unresolved
                    if other.family_id != item.family_id
                ]
                neighbors = [
                    other
                    for other in unresolved
                    if self._scores_compete(item.score, other.score)
                ]
                neighbor_ids = {other.family_id for other in neighbors}
                unresolved = [
                    other
                    for other in unresolved
                    if other.family_id not in neighbor_ids
                ]
                pending.extend(
                    _PendingSuppression(other, (item,)) for other in neighbors
                )
                pending = [
                    (
                        _PendingSuppression(
                            suppression.loser,
                            (*suppression.selected_neighbors, item),
                        )
                        if self._scores_compete(
                            item.score,
                            suppression.loser.score,
                        )
                        and all(
                            winner.family_id != item.family_id
                            for winner in suppression.selected_neighbors
                        )
                        else suppression
                    )
                    for suppression in pending
                ]
                changed = True
                break
            if not changed:
                break

        still_pending: list[_PendingSuppression] = []
        for suppression in pending:
            loser = suppression.loser
            future_closed = eof or (
                loser.score.candidate.end_seconds <= earliest_future_start
            )
            unresolved_neighbor = any(
                self._scores_compete(loser.score, item.score)
                for item in unresolved
            )
            if future_closed and not unresolved_neighbor:
                ready_suppressions.append(suppression)
            else:
                still_pending.append(suppression)

        finalized = [
            *failing,
            *selected,
            *(item.loser for item in ready_suppressions),
        ]
        peak_group_size = max(
            peak_group_size,
            len(unresolved) + len(still_pending),
        )
        if eof and (unresolved or still_pending):
            raise RuntimeError("EOF left unresolved selector ownership.")
        return (
            finalized,
            unresolved,
            still_pending,
            selected,
            ready_suppressions,
            peak_group_size,
        )

    def _prepare_decisions(
        self,
        selected_scores: list[_FamilyScore],
        suppressed_scores: list[_PendingSuppression],
    ) -> _PreparedDecisions:
        state = self._decision_state
        selections: list[tuple[_FamilyScore, str, int]] = []
        suppressions: list[tuple[CandidateFamilyId, SuppressedCandidate]] = []
        render_identities: list[tuple[CandidateFamilyId, int]] = []
        finalized: list[CandidateFamilyId] = []
        next_identity = state.render_identity
        for suppression in suppressed_scores:
            suppressed = self._suppression_result(suppression)
            suppressions.append((suppression.loser.family_id, suppressed))
            finalized.append(suppression.loser.family_id)
        for item in sorted(
            selected_scores,
            key=lambda value: self._chronological_key(value.score),
        ):
            selection = self._selector.select([item.score])
            if len(selection.selected) != 1 or selection.suppressed:
                raise RuntimeError("Selector changed resolved selection ownership.")
            winner = selection.selected[0]
            self._require_equivalent_score(winner, item.score)
            selected_item = _FamilyScore(
                item.family_id,
                winner,
                item.candidate_fingerprint,
                item.score_fingerprint,
            )
            fingerprint = selected_item.score_fingerprint
            identity = state.render_identities.get(item.family_id)
            if identity is None:
                next_identity += 1
                identity = next_identity
                render_identities.append((item.family_id, identity))
            selections.append((selected_item, fingerprint, identity))
            finalized.append(item.family_id)
        return _PreparedDecisions(
            tuple(suppressions),
            tuple(selections),
            tuple(render_identities),
            tuple(finalized),
            next_identity,
        )

    def _suppression_result(
        self,
        suppression: _PendingSuppression,
    ) -> SuppressedCandidate:
        ordered_winners = sorted(
            suppression.selected_neighbors,
            key=lambda item: self._score_ordering_key(item.score),
        )
        selection = self._selector.select(
            [
                *(item.score for item in ordered_winners),
                suppression.loser.score,
            ]
        )
        if len(selection.suppressed) != 1 or len(selection.selected) != len(
            ordered_winners
        ):
            raise RuntimeError("Selector changed resolved competition ownership.")
        for winner, expected in zip(selection.selected, ordered_winners):
            if not self._equivalent_score(winner, expected.score):
                raise RuntimeError("Selector changed resolved selection ownership.")
        suppressed = selection.suppressed[0]
        self._require_equivalent_score(suppressed.score, suppression.loser.score)
        retained = next(
            (
                item
                for item in ordered_winners
                if self._scores_compete(
                    item.score,
                    suppression.loser.score,
                )
            ),
            None,
        )
        if retained is None or not self._equivalent_score(
            suppressed.retained_score,
            retained.score,
        ):
            raise RuntimeError("Selector changed suppression provenance ownership.")
        return suppressed

    def _require_equivalent_score(
        self,
        actual: ClipScore,
        expected: ClipScore,
    ) -> None:
        if not self._equivalent_score(actual, expected):
            raise RuntimeError("Selector changed resolved score ownership.")

    def _equivalent_score(self, first: ClipScore, second: ClipScore) -> bool:
        if not isinstance(first, ClipScore) or not isinstance(second, ClipScore):
            return False
        assert self._source_id is not None
        return (
            Path(first.candidate.source_video_path)
            == Path(second.candidate.source_video_path)
            and score_fingerprint(first, self._source_id)
            == score_fingerprint(
                second,
                self._source_id,
            )
        )

    def _scores_compete(self, first: ClipScore, second: ClipScore) -> bool:
        competes = getattr(self._selector, "competes", None)
        if competes is not None:
            return bool(competes(first.candidate, second.candidate))
        selection = self._selector.select([first, second])
        if len(selection.selected) == 2 and not selection.suppressed:
            expected = sorted([first, second], key=self._score_ordering_key)
            self._require_equivalent_score(selection.selected[0], expected[0])
            self._require_equivalent_score(selection.selected[1], expected[1])
            return False
        if len(selection.selected) == 1 and len(selection.suppressed) == 1:
            retained = selection.selected[0]
            suppressed = selection.suppressed[0]
            expected = sorted([first, second], key=self._score_ordering_key)
            self._require_equivalent_score(retained, expected[0])
            self._require_equivalent_score(suppressed.score, expected[1])
            self._require_equivalent_score(suppressed.retained_score, expected[0])
            return True
        raise RuntimeError("Selector returned an invalid pairwise competition result.")

    def _earliest_future_candidate_start(
        self,
        generation: CandidateGenerationAdvance,
    ) -> float:
        checkpoint = generation.checkpoint
        if checkpoint is None:
            return float("inf")
        method = self._generator.earliest_future_candidate_start_seconds
        value = method(checkpoint)
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or math.isnan(float(value))
        ):
            raise RuntimeError("Generator returned an invalid future-candidate frontier.")
        return float(value)

    def _publish_generation_decision(
        self,
        generation: CandidateGenerationAdvance,
        newly_scored: list[_FamilyScore],
        finalized: list[_FamilyScore],
        active: list[_FamilyScore],
        pending_suppressions: list[_PendingSuppression],
        prepared: _PreparedDecisions,
        *,
        generation_passes: int,
        peak_unresolved_group_size: int,
        receipt_observation_count: int,
        accepted_watermark: float,
        observer_state: _ObserverAcceptanceState,
        snapshot_observation_ids: frozenset[str],
        committed_snapshot_payload: str | None,
        committed_receipt: tuple[
            tuple[str, str, str, int, bool], str
        ] | None,
    ) -> tuple[tuple[_FamilyScore, str, int], ...]:
        current = self._decision_state
        previous_checkpoint = current.generation_checkpoint
        if previous_checkpoint is None:
            raise RuntimeError("Incremental decision has no predecessor checkpoint.")

        self._decision_preparation_step("checkpoint")
        render_identities = dict(current.render_identities)
        render_identities.update(prepared.render_identities)
        render_families = {
            fingerprint: set(families)
            for fingerprint, families in current.render_family_ids_by_fingerprint.items()
        }
        for item, fingerprint, _ in prepared.selections:
            render_families.setdefault(fingerprint, set()).add(item.family_id)
        self._decision_preparation_step("render_identities")

        score_family_ids = {
            fingerprint: set(families)
            for fingerprint, families in current.score_family_ids.items()
        }
        for item in newly_scored:
            score_family_ids.setdefault(item.score_fingerprint, set()).add(
                item.family_id
            )
        existing_suppressions = set(current.finalized_family_ids)
        suppressions = list(current.suppressions)
        suppression_family_ids = list(current.suppression_family_ids)
        for family_id, suppressed in prepared.suppressions:
            if family_id not in existing_suppressions:
                suppressions.append(suppressed)
                suppression_family_ids.append(family_id)
                existing_suppressions.add(family_id)
        existing_selections = set(current.finalized_family_ids)
        selections = list(current.selected_scores)
        for item, fingerprint, _ in prepared.selections:
            if item.family_id not in existing_selections:
                selections.append(item.score)
                existing_selections.add(item.family_id)
        self._decision_preparation_step("selection_results")

        finalized_scores = dict(current.finalized_scores)
        immutable_fingerprints = dict(current.immutable_score_fingerprints)
        finalized_family_ids = set(current.finalized_family_ids)
        finalized_family_ids.update(prepared.finalized_family_ids)
        for item in finalized:
            finalized_scores[item.family_id] = item.score
            immutable_fingerprints[item.family_id] = item.score_fingerprint
            finalized_family_ids.add(item.family_id)
        self._decision_preparation_step("score_state")

        checkpoint_retained = (
            0
            if generation.checkpoint is None
            else generation.checkpoint.retained_observation_count
        )
        previous_checkpoint_retained = previous_checkpoint.retained_observation_count
        retained = (
            previous_checkpoint_retained
            + checkpoint_retained
            + receipt_observation_count
        )
        pending_score_count = len(
            {
                *(item.family_id for item in active),
                *(item.loser.family_id for item in pending_suppressions),
                *(
                    neighbor.family_id
                    for item in pending_suppressions
                    for neighbor in item.selected_neighbors
                ),
                *(item.family_id for item in newly_scored),
            }
        )
        proposed = _CoordinatorDecisionState(
            generation_checkpoint=generation.checkpoint,
            selection_priority_identity=current.selection_priority_identity,
            watermark_seconds=accepted_watermark,
            render_identity=prepared.next_render_identity,
            render_identities=MappingProxyType(render_identities),
            render_family_ids_by_fingerprint=MappingProxyType(
                {
                    fingerprint: frozenset(families)
                    for fingerprint, families in render_families.items()
                }
            ),
            finalized_family_ids=frozenset(finalized_family_ids),
            selected_scores=tuple(selections),
            suppressions=tuple(suppressions),
            suppression_family_ids=tuple(suppression_family_ids),
            finalized_scores=MappingProxyType(finalized_scores),
            active_scores=tuple(active),
            pending_suppressions=tuple(pending_suppressions),
            pending_render_plan=prepared.selections,
            immutable_score_fingerprints=MappingProxyType(immutable_fingerprints),
            score_family_ids=MappingProxyType(
                {
                    fingerprint: frozenset(families)
                    for fingerprint, families in score_family_ids.items()
                }
            ),
            generation_passes=current.generation_passes + generation_passes,
            scored_candidates=current.scored_candidates + len(newly_scored),
            candidate_fingerprints=current.candidate_fingerprints
            + len(newly_scored),
            score_fingerprints=current.score_fingerprints + len(newly_scored),
            peak_active_observations=max(
                current.peak_active_observations,
                retained,
            ),
            peak_active_scores=max(
                current.peak_active_scores,
                pending_score_count,
                len(
                    {
                        *(item.family_id for item in active),
                        *(item.loser.family_id for item in pending_suppressions),
                        *(
                            neighbor.family_id
                            for item in pending_suppressions
                            for neighbor in item.selected_neighbors
                        ),
                    }
                ),
            ),
            peak_unresolved_group_size=max(
                current.peak_unresolved_group_size,
                peak_unresolved_group_size,
            ),
            observer_watermarks=observer_state.watermarks,
            observer_acceptance_frontiers=observer_state.acceptance_frontiers,
            observer_frames=observer_state.frames,
            accepted_delta_sequences=observer_state.sequences,
            recent_observation_ids=observer_state.recent_observation_ids,
            snapshot_observation_ids=(
                current.snapshot_observation_ids | snapshot_observation_ids
            ),
            committed_snapshot_payload=committed_snapshot_payload,
            committed_snapshot_render_start_index=(
                min(identity for _, _, identity in prepared.selections) - 1
                if committed_snapshot_payload is not None and prepared.selections
                else current.render_identity
                if committed_snapshot_payload is not None
                else current.committed_snapshot_render_start_index
            ),
            committed_snapshot_render_end_index=(
                max(identity for _, _, identity in prepared.selections)
                if committed_snapshot_payload is not None and prepared.selections
                else current.render_identity
                if committed_snapshot_payload is not None
                else current.committed_snapshot_render_end_index
            ),
            committed_receipt_key=(
                None if committed_receipt is None else committed_receipt[0]
            ),
            committed_receipt_payload=(
                None if committed_receipt is None else committed_receipt[1]
            ),
        )
        self._decision_preparation_step("metrics")
        self._decision_preparation_step("render_plan")
        self._transaction_preparation_step("before_commit")
        self._transaction_preparation_step("former_generator_commit_gap")
        self._transaction_preparation_step("after_state_construction")
        self._decision_state = proposed
        self._transaction_publication_step("after_commit")
        return prepared.selections

    def _transaction_publication_step(self, step: str) -> None:
        """Failure-injection seam after an authoritative publication."""

    def _receipt_decision_published(self, receipt: _DeltaAcceptance) -> bool:
        state = self._decision_state
        return (
            state.committed_receipt_key == self._identity_key(receipt.identity)
            and state.committed_receipt_payload == receipt.payload
        )

    def _decision_preparation_step(self, step: str) -> None:
        """Failure-injection seam; preparation must not mutate coordinator state."""

    def _clear_committed_render_plan(
        self,
        plan: tuple[tuple[_FamilyScore, str, int], ...],
    ) -> None:
        if self._decision_state.pending_render_plan is plan:
            self._decision_state = replace(
                self._decision_state,
                pending_render_plan=(),
            )

    def _render_plan(
        self,
        plan: Sequence[tuple[_FamilyScore, str, int]],
    ) -> list[RenderJob]:
        jobs: list[RenderJob] = []
        for item, fingerprint, identity in plan:
            winner = item.score
            render_key = self._render_key(item.family_id, fingerprint)
            completion = self._authoritative_state.render_completions.get(render_key)
            if completion is not None:
                if completion.identity != identity:
                    raise RuntimeError(
                        "Completed render identity changed for an immutable plan."
                    )
                self._render_attempt_states.pop(render_key, None)
                jobs.append(completion.job)
                continue
            self._render_attempt_states[render_key] = RenderLifecycleState.RENDERING
            self._currently_rendering.add(render_key)
            job: RenderJob | None = None
            try:
                job = self._renderer.recover_render(winner, identity)
                if job is None:
                    job = self._renderer.render_one(winner, identity)
                self._render_publication_step("after_render_one")
                completion = _RenderCompletion(render_key, identity, job)
                self._render_publication_step("after_completion_construction")
                self._publish_render_completion(completion)
                self._render_publication_step("after_render_completion")
            except BaseException:
                # Failure injection after render_one returned must not discard a
                # successful external render. Durable renderer state closes even
                # the CALL/STORE_FAST interruption gap where no local job exists.
                if job is None:
                    try:
                        job = self._renderer.recover_render(winner, identity)
                    except Exception:
                        job = None
                if (
                    job is not None
                    and render_key
                    not in self._authoritative_state.render_completions
                ):
                    self._publish_render_completion(
                        _RenderCompletion(render_key, identity, job)
                    )
                if render_key not in self._authoritative_state.render_completions:
                    self._render_attempt_states[render_key] = (
                        RenderLifecycleState.FAILED
                    )
                else:
                    self._render_attempt_states.pop(render_key, None)
                raise
            finally:
                self._currently_rendering.discard(render_key)
            self._render_attempt_states.pop(render_key, None)
            committed = self._authoritative_state.render_completions.get(render_key)
            if committed is None:
                raise RuntimeError("Successful render completion was not published.")
            jobs.append(committed.job)
        return jobs

    def _publish_render_completion(self, completion: _RenderCompletion) -> None:
        current = self._authoritative_state
        existing = current.render_completions.get(completion.render_key)
        if existing is not None:
            if existing.identity != completion.identity or existing.job != completion.job:
                raise RuntimeError("Render completion was reused with different content.")
            return
        if any(
            item.identity == completion.identity
            for item in current.render_completions.values()
        ):
            raise RuntimeError("Render identity was reused by another completion.")
        completions = dict(current.render_completions)
        completions[completion.render_key] = completion
        self._authoritative_state = replace(
            current,
            render_completions=MappingProxyType(completions),
            render_jobs=(*current.render_jobs, completion.job),
        )

    def _render_publication_step(self, step: str) -> None:
        """Failure-injection seam around atomic render-result publication."""

    @staticmethod
    def _render_key(
        family_id: CandidateFamilyId | None,
        fingerprint: str,
    ) -> str:
        if family_id is None:
            return fingerprint
        return f"{family_id.source_id}:{family_id.ordinal}:{fingerprint}"

    @staticmethod
    def _observation_id(observation: Observation) -> str:
        return _fingerprint({"observation": observation})

    def _current_observer_state(self) -> _ObserverAcceptanceState:
        state = self._decision_state
        return _ObserverAcceptanceState(
            state.observer_watermarks,
            state.observer_acceptance_frontiers,
            state.observer_frames,
            state.accepted_delta_sequences,
            state.recent_observation_ids,
        )

    @staticmethod
    def _immutable_observer_state(
        watermarks: Mapping[str, float],
        frontiers: Mapping[str, float],
        frames: Mapping[str, int],
        sequences: Mapping[str, int],
        recent: Mapping[str, Mapping[str, float]],
    ) -> _ObserverAcceptanceState:
        return _ObserverAcceptanceState(
            MappingProxyType(dict(watermarks)),
            MappingProxyType(dict(frontiers)),
            MappingProxyType(dict(frames)),
            MappingProxyType(dict(sequences)),
            MappingProxyType(
                {
                    observer: MappingProxyType(dict(identities))
                    for observer, identities in recent.items()
                }
            ),
        )

    def _propose_ingestion(
        self,
        results: list[ObserverResult],
        watermarks: ObserverWatermarks,
        *,
        strict: bool,
        state: _ObserverAcceptanceState | None = None,
    ) -> _ObserverAcceptanceState:
        """Validate input and return an immutable observer-state proposal.

        Raw global time never rejects a new observation.  Observer sequence,
        processed position, finalized identity, duplicate ownership, and the
        observer-specific accepted frontier are the complete admission contract.
        """

        current = state or self._current_observer_state()
        proposed_watermarks = dict(current.watermarks)
        proposed_frontiers = dict(current.acceptance_frontiers)
        proposed_frames = dict(current.frames)
        proposed_sequences = dict(current.sequences)
        proposed_recent = {
            observer: dict(identities)
            for observer, identities in current.recent_observation_ids.items()
        }
        for result in results:
            if strict and result.observer not in self._config.required_observers:
                raise ValueError(f"Unexpected incremental observer: {result.observer}.")
            proposed_recent.setdefault(result.observer, {})
            observer_watermark = watermarks.stable_through.get(
                result.observer,
                min(float(item) for item in watermarks.stable_through.values()),
            )
            previous_acceptance = proposed_frontiers.get(
                result.observer,
                0.0,
            )
            acceptance_frontier = self._observer_acceptance_frontier(
                result,
                float(observer_watermark),
            )
            if acceptance_frontier < previous_acceptance:
                raise ValueError("Observer acceptance frontier cannot regress.")
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
                observation_identity = self._observation_id(observation)
                already_accepted = observation_identity in proposed_recent[
                    result.observer
                ]
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
                if (
                    strict
                    and not already_accepted
                    and proposed_sequences.get(result.observer, -1) >= 0
                    and end <= previous_acceptance
                ):
                    raise ValueError(
                        "Observation was already accepted or ends behind the "
                        "accepted observer frontier."
                    )
                if already_accepted:
                    continue
                proposed_recent[result.observer][observation_identity] = end
            proposed_frontiers[result.observer] = acceptance_frontier
        if not strict:
            for observer in self._config.required_observers:
                value = float(watermarks.stable_through[observer])
                if value < current.watermarks.get(observer, 0.0):
                    raise ValueError("Observer watermark cannot regress.")
                proposed_watermarks[observer] = value
                proposed_frontiers[observer] = value
        return self._immutable_observer_state(
            proposed_watermarks,
            proposed_frontiers,
            proposed_frames,
            proposed_sequences,
            proposed_recent,
        )

    @staticmethod
    def _observer_acceptance_frontier(
        result: ObserverResult,
        stable_watermark: float,
    ) -> float:
        frames = result.metadata.get("incremental_frames_processed")
        sample_rate = result.metadata.get("sample_rate_hz")
        if frames is None:
            return stable_watermark
        if (
            isinstance(frames, bool)
            or not isinstance(frames, int)
            or frames < 0
            or isinstance(sample_rate, bool)
            or not isinstance(sample_rate, int | float)
            or not math.isfinite(float(sample_rate))
            or float(sample_rate) <= 0
        ):
            raise ValueError(
                "Observer acceptance frontiers require valid frame and sample-rate metadata."
            )
        return min(stable_watermark, frames / float(sample_rate))

    def _evicted_observer_state(
        self,
        state: _ObserverAcceptanceState,
        *,
        terminal: bool = False,
    ) -> _ObserverAcceptanceState:
        """Drop duplicate evidence only after its observer frontier subsumes it."""

        recent = {
            observer: (
                {}
                if terminal
                else {
                    identity: end
                    for identity, end in identities.items()
                    if end > state.acceptance_frontiers.get(observer, 0.0)
                }
            )
            for observer, identities in state.recent_observation_ids.items()
        }
        return self._immutable_observer_state(
            state.watermarks,
            state.acceptance_frontiers,
            state.frames,
            state.sequences,
            recent,
        )

    def _implicit_delta_identity(
        self,
        results: list[ObserverResult],
        watermarks: ObserverWatermarks,
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
        last_sequence = self._accepted_delta_sequences.get(observer, -1)
        if last_sequence >= 0:
            previous = ObserverDeltaIdentity(
                self._source_id or "unactivated",
                self._session_id(),
                observer,
                last_sequence,
                eof,
            )
            completed = self._completed_delta_receipts.get(
                self._identity_key(previous)
            )
            if completed is not None and completed.payload == (
                self._delta_payload_fingerprint(results, watermarks, previous)
            ):
                return previous
        return ObserverDeltaIdentity(
            self._source_id or "unactivated",
            self._session_id(),
            observer,
            last_sequence + 1,
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
            completed = self._completed_delta_receipts.get(
                self._identity_key(previous)
            )
            if completed is not None and completed.payload == payload:
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
        state: _ObserverAcceptanceState | None = None,
    ) -> None:
        observer_state = state or self._current_observer_state()
        if identity.source_id != self._source_id or identity.session_id != self._session_id():
            raise ValueError("Delta identity does not belong to this source/session.")
        if identity.observer not in self._config.required_observers:
            raise ValueError(f"Unexpected incremental observer: {identity.observer}.")
        if len(results) != 1 or results[0].observer != identity.observer:
            raise ValueError("Delta identity must match exactly one observer result.")
        expected = observer_state.sequences[identity.observer] + 1
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
        active_ids = set(observer_state.recent_observation_ids[identity.observer])
        if active_ids.intersection(observation_ids):
            raise ValueError("Observer delta repeats an already accepted observation.")
        current = float(watermarks.stable_through[identity.observer])
        previous = observer_state.watermarks[identity.observer]
        frames = self._delta_frames(results[0])
        previous_frames = observer_state.frames[identity.observer]
        if frames < previous_frames:
            raise ValueError("Observer delta frame position cannot regress.")
        if current < previous:
            raise ValueError("Observer delta watermark cannot regress.")
        if not identity.eof and current <= previous and frames <= previous_frames:
            raise ValueError(
                "Non-EOF observer delta must advance its watermark or frame position."
            )

    def _propose_delta_identity(
        self,
        state: _ObserverAcceptanceState,
        identity: ObserverDeltaIdentity,
        result: ObserverResult,
        watermarks: ObserverWatermarks,
    ) -> _ObserverAcceptanceState:
        proposed_watermarks = dict(state.watermarks)
        proposed_frames = dict(state.frames)
        proposed_sequences = dict(state.sequences)
        proposed_sequences[identity.observer] = identity.sequence
        proposed_watermarks[identity.observer] = float(
            watermarks.stable_through[identity.observer]
        )
        proposed_frames[identity.observer] = self._delta_frames(result)
        return self._immutable_observer_state(
            proposed_watermarks,
            state.acceptance_frontiers,
            proposed_frames,
            proposed_sequences,
            state.recent_observation_ids,
        )

    def _delta_payload_fingerprint(
        self,
        results: list[ObserverResult],
        watermarks: ObserverWatermarks,
        identity: ObserverDeltaIdentity,
    ) -> str:
        return _fingerprint(
            {
                "identity": identity,
                "results": results,
                "watermarks": dict(watermarks.stable_through),
                "selection_priority": self._selection_priority.identity,
            }
        )

    def _snapshot_payload_fingerprint(
        self,
        timeline: FeatureTimeline,
        watermarks: ObserverWatermarks,
    ) -> str:
        return _fingerprint(
            {
                "results": timeline.timeline.observer_results,
                "watermarks": dict(watermarks.stable_through),
                "selection_priority": self._selection_priority.identity,
            }
        )

    def _combined_eof_payload(
        self,
        results: list[ObserverResult],
        eof: IncrementalEOF,
        identities: tuple[ObserverDeltaIdentity, ...],
    ) -> str:
        eof_identity = ObserverDeltaIdentity(
            self._source_id or "unactivated",
            self._session_id(),
            "__combined_eof__",
            0,
            True,
        )
        return _fingerprint(
            {
                "identity": eof_identity,
                "duration": eof.media_duration_seconds,
                "watermarks": dict(eof.final_watermarks.stable_through),
                "selection_priority": self._selection_priority.identity,
                "inputs": [
                    (
                        identity,
                        self._delta_payload_fingerprint(
                            [item for item in results if item.observer == identity.observer],
                            eof.final_watermarks,
                            identity,
                        ),
                    )
                    for identity in identities
                ],
                "unidentified_results": results if not identities else [],
            }
        )

    @staticmethod
    def _completed_delta_receipts_after(
        current: Mapping[
            tuple[str, str, str, int, bool], _CompletedDeltaReceipt
        ],
        identity: ObserverDeltaIdentity,
        payload: str,
        render_start_index: int,
        render_end_index: int,
    ) -> Mapping[
        tuple[str, str, str, int, bool], _CompletedDeltaReceipt
    ]:
        proposed = dict(current)
        for key in tuple(proposed):
            if key[0:3] == (identity.source_id, identity.session_id, identity.observer):
                del proposed[key]
        proposed[
            IncrementalPrerecordedCoordinator._identity_key(identity)
        ] = _CompletedDeltaReceipt(
            payload,
            render_start_index,
            render_end_index,
        )
        return MappingProxyType(proposed)

    def _completion_preparation_step(self, step: str) -> None:
        """Failure-injection seam before atomic receipt completion publication."""

    def _completion_publication_step(self, step: str) -> None:
        """Failure-injection seam after atomic receipt completion publication."""

    def _ensure_receipt_ingested(
        self,
        receipt: _DeltaAcceptance,
        results: list[ObserverResult],
        watermarks: ObserverWatermarks,
    ) -> None:
        if receipt.state is not DeltaAcceptanceState.RECEIVED:
            return
        if receipt.observer_state is None:
            current = self._current_observer_state()
            self._validate_delta_identity(
                receipt.identity,
                results,
                watermarks,
                current,
            )
            proposed = self._propose_ingestion(
                results,
                watermarks,
                strict=True,
                state=current,
            )
            proposed = self._propose_delta_identity(
                proposed,
                receipt.identity,
                results[0],
                watermarks,
            )
            receipt.observer_state = self._evicted_observer_state(proposed)
            self._transaction_preparation_step("observer_proposal")
        receipt.state = DeltaAcceptanceState.INGESTED

    def _transaction_preparation_step(self, step: str) -> None:
        """Failure-injection seam before the coordinator's single publication."""

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

    def _score_ordering_key(self, score: ClipScore) -> tuple[int]:
        return self._selection_priority.ordering_key(score.overall_score)

    def _family_score_ordering_key(
        self,
        family_id: CandidateFamilyId,
        score: ClipScore,
    ) -> tuple[int, int]:
        """Apply stable batch input order after the finite priority rank."""

        return (*self._score_ordering_key(score), family_id.ordinal)

    def _overlap_family_groups(
        self,
        scores: list[_FamilyScore],
    ) -> list[list[_FamilyScore]]:
        ordered = sorted(
            scores,
            key=lambda item: self._chronological_key(item.score),
        )
        groups: list[list[_FamilyScore]] = []
        for item in ordered:
            score = item.score
            competes = getattr(self._selector, "competes", None)
            joins = bool(groups) and any(
                competes(score.candidate, existing.score.candidate)
                if competes is not None
                else score.candidate.start_seconds
                < existing.score.candidate.end_seconds
                for existing in groups[-1]
            )
            if not joins:
                groups.append([item])
            else:
                groups[-1].append(item)
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
