"""Chronological aggregation for observer results."""

from collections import defaultdict
from typing import Iterable, Protocol

from core import AggregatedTimeline, Observation, ObserverResult, TimelineGroup


class TimelineGroupingStrategy(Protocol):
    """Protocol for grouping observations into a timeline."""

    def group(self, observations: Iterable[Observation]) -> list[TimelineGroup]:
        """Return grouped observations in chronological order."""


class ChronologicalObservationGrouper:
    """Group observations by exact timestamp in ascending chronological order."""

    def group(self, observations: Iterable[Observation]) -> list[TimelineGroup]:
        grouped: dict[float, list[Observation]] = defaultdict(list)

        for observation in observations:
            grouped[observation.timestamp_seconds].append(observation)

        return [
            TimelineGroup(
                timestamp_seconds=timestamp,
                observations=grouped[timestamp],
            )
            for timestamp in sorted(grouped)
        ]


class FeatureAggregator:
    """Build an aggregated timeline from observer result contracts."""

    def __init__(
        self,
        grouping_strategy: TimelineGroupingStrategy | None = None,
    ) -> None:
        self._grouping_strategy = grouping_strategy or ChronologicalObservationGrouper()

    def aggregate(
        self,
        observer_results: Iterable[ObserverResult],
    ) -> AggregatedTimeline:
        """Aggregate observer results without changing observer observations."""

        results = list(observer_results)
        observations = [
            observation
            for result in results
            for observation in result.observations
        ]

        return AggregatedTimeline(
            groups=self._grouping_strategy.group(observations),
            observer_results=results,
        )
