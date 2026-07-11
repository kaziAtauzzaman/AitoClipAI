from dataclasses import is_dataclass
from typing import Iterable

from aggregation import FeatureAggregator
from core import AggregatedTimeline, Observation, ObserverResult, TimelineGroup


def make_observation(
    timestamp_seconds: float,
    observer: str,
    observation_type: str,
    value: object,
) -> Observation:
    return Observation(
        timestamp_seconds=timestamp_seconds,
        observer=observer,
        type=observation_type,
        value=value,
    )


def test_aggregation_contracts_are_dataclasses() -> None:
    assert is_dataclass(ObserverResult)
    assert is_dataclass(TimelineGroup)
    assert is_dataclass(AggregatedTimeline)


def test_aggregator_groups_observations_chronologically() -> None:
    speech_observation = make_observation(12.0, "speech", "keyword", "launch")
    audio_observation = make_observation(8.0, "audio", "volume_spike", True)
    vision_observation = make_observation(12.0, "vision", "scene_change", True)

    timeline = FeatureAggregator().aggregate(
        [
            ObserverResult(
                observer="speech",
                observations=[speech_observation],
            ),
            ObserverResult(
                observer="audio",
                observations=[audio_observation],
            ),
            ObserverResult(
                observer="vision",
                observations=[vision_observation],
            ),
        ]
    )

    assert [group.timestamp_seconds for group in timeline.groups] == [8.0, 12.0]
    assert timeline.groups[0].observations == [audio_observation]
    assert timeline.groups[1].observations == [
        speech_observation,
        vision_observation,
    ]


def test_aggregator_preserves_observation_instances_exactly() -> None:
    original = Observation(
        timestamp_seconds=2.5,
        duration_seconds=1.0,
        observer="ocr",
        type="text",
        value={"text": "AitoClipAI"},
        confidence=0.9,
        metadata={"region": [1, 2, 3, 4]},
    )

    timeline = FeatureAggregator().aggregate(
        [ObserverResult(observer="ocr", observations=[original])]
    )

    grouped_observation = timeline.groups[0].observations[0]
    assert grouped_observation is original
    assert grouped_observation == original


def test_aggregator_accepts_any_number_of_observer_results() -> None:
    timeline = FeatureAggregator().aggregate([])

    assert timeline.groups == []
    assert timeline.observer_results == []


def test_aggregator_uses_injected_grouping_strategy() -> None:
    class ReverseGroupingStrategy:
        def group(self, observations: Iterable[Observation]) -> list[TimelineGroup]:
            return [
                TimelineGroup(
                    timestamp_seconds=observation.timestamp_seconds,
                    observations=[observation],
                )
                for observation in sorted(
                    observations,
                    key=lambda item: item.timestamp_seconds,
                    reverse=True,
                )
            ]

    early = make_observation(1.0, "audio", "early", True)
    late = make_observation(3.0, "audio", "late", True)

    timeline = FeatureAggregator(ReverseGroupingStrategy()).aggregate(
        [ObserverResult(observer="audio", observations=[early, late])]
    )

    assert [group.timestamp_seconds for group in timeline.groups] == [3.0, 1.0]
