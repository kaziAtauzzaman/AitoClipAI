"""Test-only closed-family continuation for marker generators.

Production never adapts stateless generators. Marker fixtures use this mixin
because each marker observation already represents one complete, final family.
"""

from pathlib import Path

from aggregation import FeatureAggregator
from candidate_generation import (
    CandidateFamilyId,
    CandidateGenerationAdvance,
    CandidateGenerationCheckpoint,
    ClosedCandidateFamily,
)
from core import FeatureTimeline, Observation, ObserverResult


class DeltaClosedFamilyGeneratorMixin:
    def earliest_future_candidate_start_seconds(self, checkpoint):
        """Conservative future-start bound for chronological marker fixtures."""

        frontier = (
            min(value for _, value in checkpoint._observer_frontiers)
            if checkpoint._observer_frontiers
            else checkpoint.stable_through_seconds
        )
        return frontier - float(self.maximum_backtrack_seconds)

    def _checkpoint_owner(self) -> object:
        token = getattr(self, "_test_checkpoint_owner", None)
        if token is None:
            token = object()
            self._test_checkpoint_owner = token
        return token

    def start_incremental(
        self,
        *,
        source_id: str,
        media_path: Path,
        required_observers=(),
    ):
        observers = tuple(required_observers)
        checkpoint = CandidateGenerationCheckpoint(
            self._checkpoint_owner(),
            source_id,
            Path(media_path),
            0.0,
            0,
            (),
            observers,
            tuple(sorted((observer, 0.0) for observer in observers)),
        )
        self._test_committed_checkpoint = checkpoint
        self._test_pending_transition = None
        self._test_publication = None
        return checkpoint

    def bind_incremental_publication(self, checkpoint, committed_checkpoint):
        self._validate_test_checkpoint(checkpoint)
        if not callable(committed_checkpoint):
            raise TypeError("Test publication must be callable.")
        if getattr(self, "_test_publication", None) is not None:
            raise ValueError("Test publication is already bound.")
        self._test_publication = committed_checkpoint
        self._test_pending_transition = None

    def advance_incremental(
        self,
        checkpoint,
        observations,
        stable_through_seconds,
        observer_frontiers=None,
    ):
        self._validate_test_checkpoint(checkpoint)
        items = tuple(observations)
        frontiers = (
            tuple(sorted((name, float(value)) for name, value in observer_frontiers.items()))
            if observer_frontiers is not None
            else checkpoint._observer_frontiers
        )
        transition_key = (checkpoint, items, float(stable_through_seconds), frontiers)
        bound = getattr(self, "_test_publication", None) is not None
        pending = None if bound else getattr(self, "_test_pending_transition", None)
        if pending is not None:
            if pending[0] == transition_key:
                return pending[1]
            raise ValueError("Checkpoint already has another test transition.")
        candidates = self._candidates_for_delta(checkpoint, items)
        families = tuple(
            ClosedCandidateFamily(
                CandidateFamilyId(
                    checkpoint.source_id,
                    checkpoint.next_family_ordinal + index,
                ),
                candidate,
            )
            for index, candidate in enumerate(candidates)
        )
        output = CandidateGenerationAdvance(
            CandidateGenerationCheckpoint(
                self._checkpoint_owner(),
                checkpoint.source_id,
                checkpoint.media_path,
                float(stable_through_seconds),
                checkpoint.next_family_ordinal + len(families),
                (),
                checkpoint._required_observers,
                frontiers,
            ),
            families,
        )
        if not bound:
            self._test_pending_transition = (transition_key, output)
        return output

    def finalize_incremental(self, checkpoint, observations, media_duration_seconds):
        self._validate_test_checkpoint(checkpoint)
        items = tuple(observations)
        transition_key = (checkpoint, items, float(media_duration_seconds), "eof")
        bound = getattr(self, "_test_publication", None) is not None
        pending = None if bound else getattr(self, "_test_pending_transition", None)
        if pending is not None:
            if pending[0] == transition_key:
                return pending[1]
            raise ValueError("Checkpoint already has another test transition.")
        candidates = self._candidates_for_delta(
            checkpoint,
            items,
            include_empty=True,
        )
        families = tuple(
            ClosedCandidateFamily(
                CandidateFamilyId(
                    checkpoint.source_id,
                    checkpoint.next_family_ordinal + index,
                ),
                candidate,
            )
            for index, candidate in enumerate(candidates)
        )
        output = CandidateGenerationAdvance(None, families)
        if not bound:
            self._test_pending_transition = (transition_key, output)
        return output

    def commit_incremental(self, checkpoint, advance):
        if getattr(self, "_test_publication", None) is not None:
            raise ValueError("Coordinator-owned test lineage commits by publication.")
        pending = getattr(self, "_test_pending_transition", None)
        if (
            checkpoint is not getattr(self, "_test_committed_checkpoint", None)
            or pending is None
            or pending[1] is not advance
        ):
            raise ValueError("Test transition is not the active proposal.")
        self._test_committed_checkpoint = advance.checkpoint
        self._test_pending_transition = None

    def _candidates_for_delta(
        self,
        checkpoint: CandidateGenerationCheckpoint,
        observations,
        *,
        include_empty: bool = False,
    ):
        items = list(observations)
        if not items and not include_empty:
            return []
        grouped_results = []
        for observer in dict.fromkeys(item.observer for item in items):
            grouped_results.append(
                ObserverResult(
                    observer,
                    [item for item in items if item.observer == observer],
                )
            )
        timeline = FeatureTimeline(
            media_path=checkpoint.media_path,
            audio_path=checkpoint.media_path.with_suffix(".wav"),
            timeline_path=checkpoint.media_path.with_suffix(".json"),
            timeline=FeatureAggregator().aggregate(grouped_results),
            metadata={"source_id": checkpoint.source_id},
        )
        return list(self.generate(timeline))

    def _validate_test_checkpoint(self, checkpoint) -> None:
        if checkpoint._owner_token is not self._checkpoint_owner():
            raise ValueError("Checkpoint belongs to another test generator.")
        publication = getattr(self, "_test_publication", None)
        committed = (
            publication()
            if publication is not None
            else getattr(self, "_test_committed_checkpoint", None)
        )
        if committed is None:
            raise ValueError("Checkpoint belongs to a finalized test generator.")
        if checkpoint is not committed:
            raise ValueError("Checkpoint is stale or uncommitted.")

    def incremental_state_snapshot(self):
        publication = getattr(self, "_test_publication", None)
        return (
            getattr(self, "_test_committed_checkpoint", None),
            getattr(self, "_test_pending_transition", None),
            None if publication is None else publication(),
        )
