"""Incremental candidate-generation continuation contracts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

from candidate_generation.heuristics import CandidateEvent
from core import ClipCandidate, FeatureTimeline, Observation


@dataclass(frozen=True, slots=True)
class CandidateFamilyId:
    """Stable identity assigned to one closed generator cluster."""

    source_id: str
    ordinal: int

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("Candidate family source identity must be non-empty.")
        if (
            isinstance(self.ordinal, bool)
            or not isinstance(self.ordinal, int)
            or self.ordinal < 0
        ):
            raise ValueError("Candidate family ordinal must be a non-negative integer.")


@dataclass(frozen=True, slots=True)
class ClosedCandidateFamily:
    """One immutable cluster closure and its optional final candidate."""

    family_id: CandidateFamilyId
    candidate: ClipCandidate | None

    def __post_init__(self) -> None:
        if not isinstance(self.family_id, CandidateFamilyId):
            raise TypeError("Closed families require a CandidateFamilyId.")
        if self.candidate is not None and not isinstance(
            self.candidate, ClipCandidate
        ):
            raise TypeError("Closed-family candidates must be ClipCandidate values.")


@dataclass(frozen=True, slots=True)
class CandidateGenerationCheckpoint:
    """Opaque process-local continuation owned by its originating generator.

    Coordinators may retain this value and read its bounded-state metric, but
    only the generator that created it may interpret or advance its events.
    """

    _owner_token: object
    source_id: str
    media_path: Path
    stable_through_seconds: float
    next_family_ordinal: int
    _open_events: tuple[CandidateEvent, ...]
    _required_observers: tuple[str, ...] = ()
    _observer_frontiers: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if self._owner_token is None:
            raise ValueError("Checkpoint owner identity must be non-empty.")
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("Checkpoint source identity must be non-empty.")
        if not isinstance(self.media_path, Path):
            raise TypeError("Checkpoint media ownership must use a Path.")
        stable = self.stable_through_seconds
        if (
            isinstance(stable, bool)
            or not isinstance(stable, int | float)
            or not math.isfinite(float(stable))
            or float(stable) < 0
        ):
            raise ValueError(
                "Checkpoint stable frontier must be finite and non-negative."
            )
        if (
            isinstance(self.next_family_ordinal, bool)
            or not isinstance(self.next_family_ordinal, int)
            or self.next_family_ordinal < 0
        ):
            raise ValueError("Checkpoint family ordinal must be non-negative.")
        if not isinstance(self._open_events, tuple):
            raise TypeError("Checkpoint events must be an immutable tuple.")
        if any(not isinstance(event, CandidateEvent) for event in self._open_events):
            raise TypeError(
                "Checkpoint continuation must contain CandidateEvent values."
            )
        if not isinstance(self._required_observers, tuple) or any(
            not isinstance(observer, str) or not observer.strip()
            for observer in self._required_observers
        ):
            raise TypeError("Checkpoint observer ownership must be an immutable tuple.")
        if len(self._required_observers) != len(set(self._required_observers)):
            raise ValueError("Checkpoint required observers must be unique.")
        if not isinstance(self._observer_frontiers, tuple):
            raise TypeError("Checkpoint observer frontiers must be an immutable tuple.")
        frontier_names: list[str] = []
        for item in self._observer_frontiers:
            if (
                not isinstance(item, tuple)
                or len(item) != 2
                or not isinstance(item[0], str)
                or not item[0].strip()
            ):
                raise TypeError("Checkpoint observer frontiers are malformed.")
            value = item[1]
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise ValueError(
                    "Checkpoint observer frontiers must be finite and non-negative."
                )
            frontier_names.append(item[0])
        if frontier_names != sorted(frontier_names) or len(frontier_names) != len(
            set(frontier_names)
        ):
            raise ValueError("Checkpoint observer frontiers must be unique and sorted.")
        if self._required_observers and set(frontier_names) != set(
            self._required_observers
        ):
            raise ValueError(
                "Checkpoint observer frontiers must cover every required observer."
            )

    @property
    def retained_observation_count(self) -> int:
        """Return unique observation references retained by unclosed events."""

        return len({id(event.observation) for event in self._open_events})

    @property
    def observer_frontiers(self) -> tuple[tuple[str, float], ...]:
        """Return immutable observer-specific closure frontiers."""

        return self._observer_frontiers


@dataclass(frozen=True, slots=True)
class CandidateGenerationAdvance:
    """Delta output from one deterministic checkpoint transition."""

    checkpoint: CandidateGenerationCheckpoint | None
    closed_families: tuple[ClosedCandidateFamily, ...]

    def __post_init__(self) -> None:
        if self.checkpoint is not None and not isinstance(
            self.checkpoint, CandidateGenerationCheckpoint
        ):
            raise TypeError("Generation advances require a checkpoint or None.")
        if not isinstance(self.closed_families, tuple):
            raise TypeError("Closed generation families must be an immutable tuple.")
        if any(
            not isinstance(family, ClosedCandidateFamily)
            for family in self.closed_families
        ):
            raise TypeError("Generation advances require ClosedCandidateFamily values.")


class IncrementalCandidateGenerator(Protocol):
    """Generator that owns exact bounded continuation between observation deltas."""

    @property
    def maximum_backtrack_seconds(self) -> float: ...

    @property
    def maximum_direct_competition_span_seconds(self) -> float: ...

    def earliest_future_candidate_start_seconds(
        self,
        checkpoint: CandidateGenerationCheckpoint,
    ) -> float: ...

    @property
    def incremental_deterministic(self) -> bool: ...

    def start_incremental(
        self,
        *,
        source_id: str,
        media_path: Path,
        required_observers: Sequence[str] = (),
    ) -> CandidateGenerationCheckpoint: ...

    def bind_incremental_publication(
        self,
        checkpoint: CandidateGenerationCheckpoint,
        committed_checkpoint: Callable[[], CandidateGenerationCheckpoint | None],
    ) -> None: ...

    def advance_incremental(
        self,
        checkpoint: CandidateGenerationCheckpoint,
        observations: Sequence[Observation],
        stable_through_seconds: float,
        observer_frontiers: Mapping[str, float] | None = None,
    ) -> CandidateGenerationAdvance: ...

    def finalize_incremental(
        self,
        checkpoint: CandidateGenerationCheckpoint,
        observations: Sequence[Observation],
        media_duration_seconds: float,
    ) -> CandidateGenerationAdvance: ...

    def commit_incremental(
        self,
        checkpoint: CandidateGenerationCheckpoint,
        advance: CandidateGenerationAdvance,
    ) -> None: ...

    def generate(self, feature_timeline: FeatureTimeline) -> list[ClipCandidate]: ...
