# Aggregation Framework

The `src/aggregation/` package builds a unified timeline from observer output.
It is intentionally structural: it groups timestamped observations and does not
score, filter, rank, summarize, or make AI decisions.

## Contracts

Aggregation uses neutral contracts from `src/core/`:

- `Observation` represents one timestamped signal from any observer.
- `ObserverResult` wraps the observations emitted by one observer run.
- `TimelineGroup` contains observations that share the same timestamp.
- `AggregatedTimeline` is the final chronological timeline.

These contracts keep aggregation independent from concrete audio, speech,
vision, OCR, or future observer implementations.

## Flow

1. Observer implementations emit `ObserverResult` objects.
2. `FeatureAggregator.aggregate()` accepts any iterable of those results.
3. The aggregator flattens the contained `Observation` objects.
4. A grouping strategy turns the observations into `TimelineGroup` objects.
5. The aggregator returns an `AggregatedTimeline` with grouped observations and
   the original observer result objects.

The default `ChronologicalObservationGrouper` groups observations by exact
`timestamp_seconds` and orders groups from earliest to latest.

## Preservation Rule

Aggregation preserves every `Observation` object exactly. The framework does
not mutate observations, clone them, remove them, infer priority, normalize
confidence, or interpret observer-specific metadata.

Downstream scoring, clipping, ranking, or AI-assisted selection must happen in
separate modules after aggregation.

## Dependency Injection

`FeatureAggregator` accepts an optional `TimelineGroupingStrategy`. This keeps
the default behavior simple while allowing tests or future workflows to inject
alternate grouping behavior without changing observer code or the aggregator
contract.
