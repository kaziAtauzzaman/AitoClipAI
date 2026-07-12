"""Deterministic candidate clip window generation."""

from pathlib import Path
from typing import Iterable

from candidate_generation.config import CandidateGenerationConfig
from candidate_generation.errors import CandidateGenerationError
from candidate_generation.heuristics import (
    CandidateEvent,
    CandidateHeuristic,
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
        confidence = min(1.0, sum(event.contribution for event in cluster))
        if confidence < self._config.minimum_candidate_confidence:
            return None

        start = min(event.start_seconds for event in cluster) - self._config.pre_roll_seconds
        end = max(event.end_seconds for event in cluster) + self._config.post_roll_seconds
        start, end = self._bounded_window(start, end, media_duration)
        if end <= start:
            return None

        signals = list(dict.fromkeys(event.signal for event in cluster))
        explanation = self._explanation(start, end, signals)
        observations: list[Observation] = []
        for event in cluster:
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
                "contributing_observations": observations,
                "signal_contributions": [
                    {
                        "signal": event.signal,
                        "strength": event.strength,
                        "weight": event.weight,
                        "contribution": event.contribution,
                    }
                    for event in cluster
                ],
            },
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
        if config.silence_reference_seconds <= 0:
            raise CandidateGenerationError("Silence reference must be positive.")
