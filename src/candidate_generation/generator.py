"""Deterministic candidate clip window generation."""

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Iterable, Sequence

from candidate_generation.config import CandidateGenerationConfig
from candidate_generation.contracts import (
    CandidateFamilyId,
    CandidateGenerationAdvance,
    CandidateGenerationCheckpoint,
    ClosedCandidateFamily,
)
from candidate_generation.errors import CandidateGenerationError
from candidate_generation.heuristics import (
    CandidateEvent,
    CandidateHeuristic,
    EventBoundaryRole,
    default_heuristics,
)
from core import ClipCandidate, FeatureTimeline, Observation


class _ImmutableList(tuple):
    """Tuple storage tagged so candidate output can restore list semantics."""

    __slots__ = ()

    def __new__(cls, items: Iterable[object]):
        return super().__new__(cls, tuple(items))


def _freeze_checkpoint_value(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                key: _freeze_checkpoint_value(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, list):
        return _ImmutableList(_freeze_checkpoint_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_checkpoint_value(item) for item in value)
    return value


def _plain_checkpoint_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            key: _plain_checkpoint_value(item)
            for key, item in value.items()
        }
    if isinstance(value, _ImmutableList):
        return [_plain_checkpoint_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_plain_checkpoint_value(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class _IncrementalSession:
    source_id: str
    media_path: Path
    configuration_identity: tuple[object, ...]
    committed_checkpoint: CandidateGenerationCheckpoint | None
    publication: Callable[[], CandidateGenerationCheckpoint | None] | None = None
    pending_key: str | None = None
    pending_output: CandidateGenerationAdvance | None = None
    pending_checkpoint: CandidateGenerationCheckpoint | None = None


class CandidateGenerator:
    """Transform a feature timeline into deterministic candidate clip windows."""

    def __init__(
        self,
        config: CandidateGenerationConfig | None = None,
        heuristics: Iterable[CandidateHeuristic] | None = None,
    ) -> None:
        self._config = config or CandidateGenerationConfig()
        self._validate_config()
        self._heuristics = list(
            default_heuristics(self._config) if heuristics is None else heuristics
        )
        self._incremental_sessions: dict[object, _IncrementalSession] = {}

    @property
    def maximum_backtrack_seconds(self) -> float:
        """Maximum distance a newly stable event can revise candidate history."""

        return self._config.maximum_clip_seconds

    @property
    def incremental_deterministic(self) -> bool:
        """Declare deterministic output for an identical stable timeline prefix."""

        return True

    @property
    def maximum_competition_seconds(self) -> float:
        """Maximum generator lookback used by legacy compatibility checks."""

        return self._config.maximum_clip_seconds

    @property
    def maximum_direct_competition_span_seconds(self) -> float:
        """Bound the start-time distance across one direct overlap edge.

        Every generated window is at most ``maximum_clip_seconds`` long. Two
        windows cannot overlap, and therefore cannot compete, when their starts
        are separated by that duration or more. This is a local edge bound, not
        a bound on an overlap component's total duration.
        """

        return self._config.maximum_clip_seconds

    def earliest_future_candidate_start_seconds(
        self,
        checkpoint: CandidateGenerationCheckpoint,
    ) -> float:
        """Return a lower bound for every candidate not emitted by ``checkpoint``.

        A family remains open until every required observer has advanced one
        maximum cluster span beyond its first event. Candidate minimum-duration
        expansion may move the rendered start before that first event, so that
        exact geometric backshift is included in the bound.
        """

        session = self._session_for_token(checkpoint._owner_token)
        self._validate_checkpoint_ownership(checkpoint, session)
        closure_frontier = (
            min(value for _, value in checkpoint._observer_frontiers)
            if checkpoint._observer_frontiers
            else checkpoint.stable_through_seconds
        )
        initial_window = self._config.pre_roll_seconds + self._config.post_roll_seconds
        minimum_expansion = max(
            0.0,
            (self._config.minimum_clip_seconds - initial_window) / 2.0,
        )
        candidate_backshift = self._config.pre_roll_seconds + minimum_expansion
        return (
            closure_frontier
            - self._config.maximum_clip_seconds
            - candidate_backshift
        )

    def start_incremental(
        self,
        *,
        source_id: str,
        media_path: Path,
        required_observers: Sequence[str] = (),
    ) -> CandidateGenerationCheckpoint:
        """Create an empty continuation bound to this generator and source."""

        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("Incremental source identity must be non-empty.")
        raw_observers = tuple(required_observers)
        if any(not isinstance(item, str) for item in raw_observers):
            raise ValueError("Incremental required observers must be strings.")
        observers = tuple(dict.fromkeys(item.strip() for item in raw_observers))
        if len(observers) != len(raw_observers) or any(not item for item in observers):
            raise ValueError("Incremental required observers must be unique and non-empty.")
        token = object()
        checkpoint = CandidateGenerationCheckpoint(
            token,
            source_id.strip(),
            Path(media_path),
            0.0,
            0,
            (),
            observers,
            tuple(sorted((observer, 0.0) for observer in observers)),
        )
        self._incremental_sessions[token] = _IncrementalSession(
            checkpoint.source_id,
            checkpoint.media_path,
            self._configuration_identity(),
            checkpoint,
        )
        return checkpoint

    def bind_incremental_publication(
        self,
        checkpoint: CandidateGenerationCheckpoint,
        committed_checkpoint: Callable[[], CandidateGenerationCheckpoint | None],
    ) -> None:
        """Delegate committed lineage to one coordinator-owned publication.

        Once bound, proposal generation is pure: advancing or finalizing cannot
        mutate generator session state.  The supplied callable exposes the
        checkpoint contained in the coordinator's single immutable state swap.
        """

        if not callable(committed_checkpoint):
            raise TypeError("Incremental publication must be callable.")
        session = self._validate_checkpoint(checkpoint)
        if session.publication is not None:
            raise ValueError("Incremental checkpoint publication is already bound.")
        self._incremental_sessions[checkpoint._owner_token] = _IncrementalSession(
            session.source_id,
            session.media_path,
            session.configuration_identity,
            session.committed_checkpoint,
            committed_checkpoint,
        )

    def advance_incremental(
        self,
        checkpoint: CandidateGenerationCheckpoint,
        observations: Sequence[Observation],
        stable_through_seconds: float,
        observer_frontiers: Mapping[str, float] | None = None,
    ) -> CandidateGenerationAdvance:
        """Advance exact unclosed event ownership with one observation delta."""

        session = self._validate_checkpoint(checkpoint)
        stable = self._validated_frontier(stable_through_seconds)
        if stable < checkpoint.stable_through_seconds:
            raise ValueError("Incremental stable frontier cannot move backwards.")
        frontiers = self._validated_observer_frontiers(
            checkpoint,
            observer_frontiers,
            stable,
        )
        transition_key = self._transition_key(
            "advance",
            observations,
            stable,
            frontiers,
        )
        replay = self._pending_replay(session, checkpoint, transition_key)
        if replay is not None:
            return replay
        events = self._merge_events(
            checkpoint._open_events,
            observations,
        )
        clusters = self._clusters(events)
        closed: list[ClosedCandidateFamily] = []
        closed_cluster_count = 0
        next_ordinal = checkpoint.next_family_ordinal
        closure_frontier = (
            min(value for _, value in frontiers) if frontiers else stable
        )
        for cluster in clusters:
            cluster_start = min(event.start_seconds for event in cluster)
            if closure_frontier < cluster_start + self._config.maximum_clip_seconds:
                break
            closed.append(
                ClosedCandidateFamily(
                    CandidateFamilyId(checkpoint.source_id, next_ordinal),
                    self._candidate(cluster, checkpoint.media_path, None),
                )
            )
            next_ordinal += 1
            closed_cluster_count += 1
        remaining = tuple(
            event
            for cluster in clusters[closed_cluster_count:]
            for event in cluster
        )
        next_checkpoint = CandidateGenerationCheckpoint(
            checkpoint._owner_token,
            checkpoint.source_id,
            checkpoint.media_path,
            stable,
            next_ordinal,
            remaining,
            checkpoint._required_observers,
            frontiers,
        )
        output = CandidateGenerationAdvance(next_checkpoint, tuple(closed))
        self._save_pending_transition(
            checkpoint._owner_token,
            session,
            checkpoint,
            transition_key,
            output,
        )
        return output

    def finalize_incremental(
        self,
        checkpoint: CandidateGenerationCheckpoint,
        observations: Sequence[Observation],
        media_duration_seconds: float,
    ) -> CandidateGenerationAdvance:
        """Close all remaining families after authoritative end of input."""

        session = self._validate_checkpoint(checkpoint)
        duration = self._validated_frontier(media_duration_seconds)
        transition_key = self._transition_key(
            "finalize",
            observations,
            duration,
            checkpoint._observer_frontiers,
        )
        replay = self._pending_replay(session, checkpoint, transition_key)
        if replay is not None:
            return replay
        events = self._merge_events(
            checkpoint._open_events,
            observations,
        )
        closed: list[ClosedCandidateFamily] = []
        next_ordinal = checkpoint.next_family_ordinal
        for cluster in self._clusters(events):
            closed.append(
                ClosedCandidateFamily(
                    CandidateFamilyId(checkpoint.source_id, next_ordinal),
                    self._candidate(cluster, checkpoint.media_path, duration),
                )
            )
            next_ordinal += 1
        output = CandidateGenerationAdvance(None, tuple(closed))
        self._save_pending_transition(
            checkpoint._owner_token,
            session,
            checkpoint,
            transition_key,
            output,
        )
        return output

    def commit_incremental(
        self,
        checkpoint: CandidateGenerationCheckpoint,
        advance: CandidateGenerationAdvance,
    ) -> None:
        """Accept one proposed transition and make its predecessor stale."""

        session = self._session_for_token(checkpoint._owner_token)
        self._validate_checkpoint_ownership(checkpoint, session)
        if session.publication is not None:
            raise ValueError(
                "Coordinator-owned incremental lineage is committed by publication."
            )
        if (
            session.committed_checkpoint is not checkpoint
            or session.pending_checkpoint is not checkpoint
            or session.pending_output is not advance
        ):
            raise ValueError(
                "Incremental transition is not the active uncommitted proposal."
            )
        if advance.checkpoint is not None:
            self._validate_checkpoint_ownership(advance.checkpoint, session)
        self._incremental_sessions[checkpoint._owner_token] = _IncrementalSession(
            session.source_id,
            session.media_path,
            session.configuration_identity,
            advance.checkpoint,
        )

    def revision_start_seconds(self, candidate: ClipCandidate) -> float:
        """Return the start of the generator cluster that owns ``candidate``."""

        return float(candidate.metadata["original_cluster_start"])

    def revision_stable_after_seconds(self, candidate: ClipCandidate) -> float:
        """Return the watermark after which this candidate cannot be revised."""

        return self.revision_start_seconds(candidate) + self._config.maximum_clip_seconds

    def revision_partition_seconds(self, candidate: ClipCandidate) -> float:
        """Return the closed cluster boundary safe to exclude from regeneration."""

        return float(candidate.metadata["original_cluster_end"])

    def earliest_unresolved_cluster_start_seconds(
        self,
        feature_timeline: FeatureTimeline,
        stable_watermark_seconds: float,
    ) -> float | None:
        """Return the earliest cluster that can still accept stable observations.

        This includes clusters that have not accumulated enough confidence to
        produce a candidate.  The hard padded-span limit closes every cluster
        no later than one ``maximum_clip_seconds`` span after its first event.
        """

        unresolved = [
            min(event.start_seconds for event in cluster)
            for cluster in self._clusters(self._events(feature_timeline))
            if stable_watermark_seconds
            < min(event.start_seconds for event in cluster)
            + self._config.maximum_clip_seconds
        ]
        return min(unresolved) if unresolved else None

    def generate(self, feature_timeline: FeatureTimeline) -> list[ClipCandidate]:
        """Generate candidate windows without modifying timeline observations."""

        events = self._events(feature_timeline)
        duration = self._media_duration(feature_timeline)
        candidates = [
            candidate
            for cluster in self._clusters(events)
            if (candidate := self._candidate(cluster, feature_timeline.media_path, duration))
            is not None
        ]
        return sorted(
            candidates,
            key=lambda candidate: (
                candidate.start_seconds,
                candidate.end_seconds,
                candidate.reason,
            ),
        )

    def _events(self, feature_timeline: FeatureTimeline) -> list[CandidateEvent]:
        observations = [
            observation
            for group in feature_timeline.timeline.groups
            for observation in group.observations
        ]
        return self._events_from_observations(observations)

    def _events_from_observations(
        self,
        observations: Iterable[Observation],
    ) -> list[CandidateEvent]:
        events = [
            event
            for observation in observations
            for heuristic in self._heuristics
            if (event := heuristic.detect(observation)) is not None
        ]
        return sorted(events, key=self._event_ordering_key)

    def _merge_events(
        self,
        existing: tuple[CandidateEvent, ...],
        observations: Sequence[Observation],
    ) -> list[CandidateEvent]:
        new_events: list[CandidateEvent] = []
        for observation in observations:
            frozen_observation = self._checkpoint_observation(observation)
            for heuristic in self._heuristics:
                event = heuristic.detect(observation)
                if event is None:
                    continue
                new_events.append(
                    CandidateEvent(
                        event.start_seconds,
                        event.end_seconds,
                        event.signal,
                        event.strength,
                        event.weight,
                        frozen_observation,
                        event.boundary_role,
                        event.sustained_strength,
                    )
                )
        return sorted([*existing, *new_events], key=self._event_ordering_key)

    @staticmethod
    def _checkpoint_observation(observation: Observation) -> Observation:
        if not isinstance(observation, Observation):
            raise TypeError(
                "Incremental observation deltas require Observation values."
            )
        return Observation(
            observation.timestamp_seconds,
            observation.observer,
            observation.type,
            _freeze_checkpoint_value(observation.value),
            observation.duration_seconds,
            observation.confidence,
            _freeze_checkpoint_value(observation.metadata),
        )

    @staticmethod
    def _event_ordering_key(event: CandidateEvent) -> tuple[object, ...]:
        return (
            event.start_seconds,
            event.end_seconds,
            event.signal,
            event.strength,
            event.weight,
        )

    def _validate_checkpoint(
        self,
        checkpoint: CandidateGenerationCheckpoint,
    ) -> _IncrementalSession:
        if not isinstance(checkpoint, CandidateGenerationCheckpoint):
            raise TypeError("Incremental checkpoint has an unsupported type.")
        session = self._session_for_token(checkpoint._owner_token)
        self._validate_checkpoint_ownership(checkpoint, session)
        committed = (
            session.publication()
            if session.publication is not None
            else session.committed_checkpoint
        )
        if committed is None:
            raise ValueError("Incremental checkpoint belongs to a finalized session.")
        if committed is not checkpoint:
            raise ValueError("Incremental checkpoint is stale or not yet committed.")
        return session

    def _session_for_token(self, token: object) -> _IncrementalSession:
        session = self._incremental_sessions.get(token)
        if session is None:
            raise ValueError(
                "Incremental checkpoint does not belong to the originating generator."
            )
        return session

    def _validate_checkpoint_ownership(
        self,
        checkpoint: CandidateGenerationCheckpoint,
        session: _IncrementalSession,
    ) -> None:
        if self._incremental_sessions.get(checkpoint._owner_token) is not session:
            raise ValueError(
                "Incremental checkpoint does not belong to the originating generator."
            )
        if (
            checkpoint.source_id != session.source_id
            or checkpoint.media_path != session.media_path
        ):
            raise ValueError("Incremental checkpoint belongs to another source.")
        if session.configuration_identity != self._configuration_identity():
            raise ValueError("Incremental checkpoint belongs to another configuration.")

    def _configuration_identity(self) -> tuple[object, ...]:
        return (
            self._config,
            tuple((type(item), id(item)) for item in self._heuristics),
        )

    def _validated_observer_frontiers(
        self,
        checkpoint: CandidateGenerationCheckpoint,
        values: Mapping[str, float] | None,
        stable: float,
    ) -> tuple[tuple[str, float], ...]:
        required = checkpoint._required_observers
        if not required:
            if values is None:
                return ()
            names = tuple(sorted(values))
        else:
            if values is None or set(values) != set(required):
                raise ValueError(
                    "Incremental observer frontiers must cover every required observer."
                )
            names = tuple(sorted(required))
        previous = dict(checkpoint._observer_frontiers)
        frontiers: list[tuple[str, float]] = []
        for name in names:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Incremental observer names must be non-empty.")
            value = self._validated_frontier(values[name])
            if value < previous.get(name, 0.0):
                raise ValueError("Incremental observer frontier cannot move backwards.")
            frontiers.append((name, value))
        if frontiers and min(value for _, value in frontiers) > stable:
            raise ValueError(
                "Incremental observer closure frontier cannot exceed the stable frontier."
            )
        return tuple(frontiers)

    def _transition_key(
        self,
        kind: str,
        observations: Sequence[Observation],
        position: float,
        frontiers: tuple[tuple[str, float], ...],
    ) -> str:
        return json.dumps(
            {
                "kind": kind,
                "position": position,
                "frontiers": frontiers,
                "observations": [
                    {
                        "timestamp_seconds": item.timestamp_seconds,
                        "observer": item.observer,
                        "type": item.type,
                        "value": _plain_checkpoint_value(item.value),
                        "duration_seconds": item.duration_seconds,
                        "confidence": item.confidence,
                        "metadata": _plain_checkpoint_value(item.metadata),
                    }
                    for item in observations
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    @staticmethod
    def _pending_replay(
        session: _IncrementalSession,
        checkpoint: CandidateGenerationCheckpoint,
        transition_key: str,
    ) -> CandidateGenerationAdvance | None:
        if session.publication is not None:
            return None
        if session.pending_output is None:
            return None
        if (
            session.pending_checkpoint is checkpoint
            and session.pending_key == transition_key
        ):
            return session.pending_output
        raise ValueError(
            "Incremental checkpoint already has a different uncommitted transition."
        )

    def _save_pending_transition(
        self,
        owner_token: object,
        session: _IncrementalSession,
        checkpoint: CandidateGenerationCheckpoint,
        transition_key: str,
        output: CandidateGenerationAdvance,
    ) -> None:
        if session.publication is not None:
            return
        self._incremental_sessions[owner_token] = _IncrementalSession(
            session.source_id,
            session.media_path,
            session.configuration_identity,
            session.committed_checkpoint,
            pending_key=transition_key,
            pending_output=output,
            pending_checkpoint=checkpoint,
        )

    @staticmethod
    def _validated_frontier(value: float) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise ValueError("Incremental frontier must be finite and non-negative.")
        return float(value)

    def _clusters(self, events: list[CandidateEvent]) -> list[list[CandidateEvent]]:
        clusters: list[list[CandidateEvent]] = []
        for event in events:
            if not clusters or not self._can_merge(clusters[-1], event):
                clusters.append([event])
            else:
                clusters[-1].append(event)
        return clusters

    def _can_merge(self, cluster: list[CandidateEvent], event: CandidateEvent) -> bool:
        cluster_end = max(item.end_seconds for item in cluster)
        if event.start_seconds - cluster_end > self._config.merge_gap_seconds:
            return False
        prospective_start = min(cluster[0].start_seconds, event.start_seconds)
        prospective_end = max(cluster_end, event.end_seconds)
        padded_span = (
            prospective_end
            - prospective_start
            + self._config.pre_roll_seconds
            + self._config.post_roll_seconds
        )
        return padded_span <= self._config.maximum_clip_seconds

    def _candidate(
        self,
        cluster: list[CandidateEvent],
        media_path: Path,
        media_duration: float | None,
    ) -> ClipCandidate | None:
        original_cluster_confidence = min(
            1.0, sum(event.contribution for event in cluster)
        )
        if original_cluster_confidence < self._config.minimum_candidate_confidence:
            return None

        cluster_start = min(event.start_seconds for event in cluster)
        cluster_end = max(event.end_seconds for event in cluster)
        core_start, core_end, refinement_reason = self._anchor_core(cluster)
        start = core_start - self._config.pre_roll_seconds
        end = core_end + self._config.post_roll_seconds
        start, end = self._bounded_window(start, end, media_duration)
        if end <= start:
            return None

        retained_events = [
            event
            for event in cluster
            if self._event_intersects(event, start, end)
        ]
        confidence = min(1.0, sum(event.contribution for event in retained_events))
        if confidence < self._config.minimum_candidate_confidence:
            return None
        signals = list(dict.fromkeys(event.signal for event in retained_events))
        explanation = self._explanation(start, end, signals)
        observations: list[Observation] = []
        observation_ids: set[int] = set()
        for event in retained_events:
            observation_id = id(event.observation)
            if observation_id not in observation_ids:
                observations.append(self._candidate_observation(event.observation))
                observation_ids.add(observation_id)
        rounded_confidence = round(confidence, 6)
        return ClipCandidate(
            source_video_path=media_path,
            start_seconds=start,
            end_seconds=end,
            reason=explanation,
            source_signals=signals,
            metadata={
                "start_time": start,
                "end_time": end,
                "confidence": rounded_confidence,
                "original_cluster_confidence": round(original_cluster_confidence, 6),
                "original_cluster_start": round(cluster_start, 6),
                "original_cluster_end": round(cluster_end, 6),
                "anchor_core_start": round(core_start, 6),
                "anchor_core_end": round(core_end, 6),
                "boundary_refinement": refinement_reason,
                "contributing_observations": observations,
                "signal_contributions": [
                    {
                        "signal": event.signal,
                        "strength": event.strength,
                        "weight": event.weight,
                        "contribution": event.contribution,
                        "boundary_role": event.boundary_role.value,
                        "sustained_strength": event.sustained_strength,
                    }
                    for event in retained_events
                ],
            },
        )

    @staticmethod
    def _candidate_observation(observation: Observation) -> Observation:
        if not isinstance(
            observation.value,
            (MappingProxyType, _ImmutableList),
        ) and not isinstance(
            observation.metadata,
            (MappingProxyType, _ImmutableList),
        ):
            return observation
        return Observation(
            observation.timestamp_seconds,
            observation.observer,
            observation.type,
            _plain_checkpoint_value(observation.value),
            observation.duration_seconds,
            observation.confidence,
            _plain_checkpoint_value(observation.metadata),
        )

    def _anchor_core(
        self,
        cluster: list[CandidateEvent],
    ) -> tuple[float, float, str]:
        cluster_start = min(event.start_seconds for event in cluster)
        cluster_end = max(event.end_seconds for event in cluster)
        anchor_target = self._anchor_target_seconds()
        if cluster_end - cluster_start <= anchor_target:
            return cluster_start, cluster_end, "cluster_within_anchor_target"

        boundary_events = [event for event in cluster if self._is_boundary_driving(event)]
        sustained = [
            event
            for event in boundary_events
            if event.sustained_strength >= self._config.sustained_event_contribution
            and event.end_seconds - event.start_seconds <= anchor_target
        ]
        sustained_core = self._strongest_sustained_core(sustained)
        if sustained_core is not None:
            return (*sustained_core, "sustained_high_signal_core")

        if not boundary_events:
            start, end = self._strongest_supporting_core(cluster)
            return start, end, "supporting_event_anchor"

        start, end = self._strongest_local_core(cluster, boundary_events)
        return start, end, "strongest_local_contribution_core"

    def _strongest_sustained_core(
        self,
        events: list[CandidateEvent],
    ) -> tuple[float, float] | None:
        if not events:
            return None
        chains: list[list[CandidateEvent]] = []
        chain_end = float("-inf")
        for event in events:
            if (
                not chains
                or event.start_seconds - chain_end > self._config.merge_gap_seconds
            ):
                chains.append([event])
                chain_end = event.end_seconds
            else:
                chains[-1].append(event)
                chain_end = max(chain_end, event.end_seconds)
        candidates = [
            chain
            for chain in chains
            if len({id(item.observation) for item in chain}) >= 2
            if max(item.end_seconds for item in chain)
            - min(item.start_seconds for item in chain)
            > self._anchor_target_seconds()
        ]
        if not candidates:
            return None
        chain = min(
            candidates,
            key=lambda items: (
                -sum(item.sustained_strength for item in items),
                -sum(item.contribution for item in items),
                min(item.start_seconds for item in items),
                max(item.end_seconds for item in items),
            ),
        )
        start = min(item.start_seconds for item in chain)
        end = max(item.end_seconds for item in chain)
        maximum_core = max(
            0.0,
            self._config.maximum_clip_seconds
            - self._config.pre_roll_seconds
            - self._config.post_roll_seconds,
        )
        if end - start > maximum_core:
            midpoint = (start + end) / 2.0
            start = midpoint - maximum_core / 2.0
            end = midpoint + maximum_core / 2.0
        return start, end

    def _strongest_supporting_core(
        self,
        cluster: list[CandidateEvent],
    ) -> tuple[float, float]:
        """Choose a fixed target window, preferring central ties over early ones."""

        cluster_start = min(event.start_seconds for event in cluster)
        cluster_end = max(event.end_seconds for event in cluster)
        target = self._anchor_target_seconds()
        latest_start = cluster_end - target
        cluster_midpoint = (cluster_start + cluster_end) / 2.0
        starts = {
            min(
                max((event.start_seconds + event.end_seconds - target) / 2.0, cluster_start),
                latest_start,
            )
            for event in cluster
        }

        def key(start: float) -> tuple[float, float, float]:
            end = start + target
            contribution = sum(
                event.contribution
                for event in cluster
                if event.end_seconds >= start and event.start_seconds <= end
            )
            return (
                -contribution,
                abs((start + end) / 2.0 - cluster_midpoint),
                start,
            )

        start = min(starts, key=key)
        return start, start + target

    def _strongest_local_core(
        self,
        cluster: list[CandidateEvent],
        boundary_events: list[CandidateEvent],
    ) -> tuple[float, float]:
        ordered_cluster = sorted(
            cluster,
            key=lambda event: (event.start_seconds, event.end_seconds),
        )
        best_core: tuple[float, float] | None = None
        best_key: tuple[float, float, float, float] | None = None
        for left_index, left in enumerate(boundary_events):
            start = left.start_seconds
            end = left.end_seconds
            pointer = 0
            total = 0.0
            boundary_total = 0.0
            for right in boundary_events[left_index:]:
                end = max(end, right.end_seconds)
                if end - start > self._anchor_target_seconds():
                    break
                while (
                    pointer < len(ordered_cluster)
                    and ordered_cluster[pointer].start_seconds <= end
                ):
                    event = ordered_cluster[pointer]
                    if event.end_seconds >= start:
                        total += event.contribution
                        if self._is_boundary_driving(event):
                            boundary_total += event.contribution
                    pointer += 1
                candidate_key = (-total, -boundary_total, end - start, start)
                if best_key is None or candidate_key < best_key:
                    best_key = candidate_key
                    best_core = (start, end)
        if best_core is None:
            anchor = min(
                boundary_events,
                key=lambda event: (
                    -event.contribution,
                    event.start_seconds,
                    event.end_seconds,
                    event.signal,
                ),
            )
            midpoint = (anchor.start_seconds + anchor.end_seconds) / 2.0
            half = self._anchor_target_seconds() / 2.0
            return midpoint - half, midpoint + half

        return best_core

    @staticmethod
    def _is_boundary_driving(event: CandidateEvent) -> bool:
        return event.boundary_role is EventBoundaryRole.DRIVING

    @staticmethod
    def _event_intersects(event: CandidateEvent, start: float, end: float) -> bool:
        return event.end_seconds >= start and event.start_seconds <= end

    def _anchor_target_seconds(self) -> float:
        return min(
            self._config.anchor_core_seconds,
            self._config.maximum_clip_seconds,
        )

    def _bounded_window(
        self,
        start: float,
        end: float,
        media_duration: float | None,
    ) -> tuple[float, float]:
        start = max(0.0, start)
        if media_duration is not None:
            end = min(end, media_duration)

        if end - start > self._config.maximum_clip_seconds:
            midpoint = (start + end) / 2.0
            half = self._config.maximum_clip_seconds / 2.0
            start, end = midpoint - half, midpoint + half

        if end - start < self._config.minimum_clip_seconds:
            midpoint = (start + end) / 2.0
            half = self._config.minimum_clip_seconds / 2.0
            start, end = midpoint - half, midpoint + half

        if start < 0:
            end -= start
            start = 0.0
        if media_duration is not None and end > media_duration:
            start = max(0.0, start - (end - media_duration))
            end = media_duration
        return round(start, 6), round(end, 6)

    def _media_duration(self, feature_timeline: FeatureTimeline) -> float | None:
        durations = [
            float(result.metadata["duration_seconds"])
            for result in feature_timeline.timeline.observer_results
            if isinstance(result.metadata.get("duration_seconds"), int | float)
            and not isinstance(result.metadata.get("duration_seconds"), bool)
        ]
        if feature_timeline.download and feature_timeline.download.duration_seconds:
            durations.append(feature_timeline.download.duration_seconds)
        return max(durations) if durations else None

    def _explanation(self, start: float, end: float, signals: list[str]) -> str:
        labels = {
            "whisper_speech": "Whisper speech",
            "audio_loudness": "audio loudness",
            "audio_peak": "audio peak",
            "silence_buildup": "silence buildup",
            "speaking_intensity": "speaking intensity",
        }
        readable = [labels.get(signal, signal.replace("_", " ")) for signal in signals]
        if len(readable) == 1:
            signal_text = readable[0]
        else:
            signal_text = ", ".join(readable[:-1]) + f", and {readable[-1]}"
        return f"Selected {start:.2f}s-{end:.2f}s from {signal_text}."

    def _validate_config(self) -> None:
        config = self._config
        if config.merge_gap_seconds < 0:
            raise CandidateGenerationError("Merge gap cannot be negative.")
        if config.pre_roll_seconds < 0 or config.post_roll_seconds < 0:
            raise CandidateGenerationError("Candidate roll durations cannot be negative.")
        if config.minimum_clip_seconds <= 0:
            raise CandidateGenerationError("Minimum clip duration must be positive.")
        if config.maximum_clip_seconds < config.minimum_clip_seconds:
            raise CandidateGenerationError(
                "Maximum clip duration cannot be shorter than the minimum."
            )
        if config.anchor_core_seconds <= 0:
            raise CandidateGenerationError(
                "Anchor core duration must be positive."
            )
        if not 0.0 <= config.sustained_event_contribution <= 1.0:
            raise CandidateGenerationError(
                "Sustained event contribution must be between zero and one."
            )
        if config.silence_reference_seconds <= 0:
            raise CandidateGenerationError("Silence reference must be positive.")
