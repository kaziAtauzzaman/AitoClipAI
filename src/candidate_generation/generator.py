"""Deterministic candidate clip window generation."""

from pathlib import Path
from typing import Iterable

from candidate_generation.config import CandidateGenerationConfig
from candidate_generation.errors import CandidateGenerationError
from candidate_generation.heuristics import (
    CandidateEvent,
    CandidateHeuristic,
    EventBoundaryRole,
    default_heuristics,
)
from core import ClipCandidate, FeatureTimeline, Observation


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
        """Finite prefix span in which generated candidates may compete."""

        return self._config.maximum_clip_seconds

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
        events = [
            event
            for observation in observations
            for heuristic in self._heuristics
            if (event := heuristic.detect(observation)) is not None
        ]
        return sorted(
            events,
            key=lambda event: (
                event.start_seconds,
                event.end_seconds,
                event.signal,
                event.observation.observer,
                event.observation.type,
                event.boundary_role.value,
                event.sustained_strength,
                event.strength,
                event.weight,
            ),
        )

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
        for event in retained_events:
            if not any(event.observation is existing for existing in observations):
                observations.append(event.observation)
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
