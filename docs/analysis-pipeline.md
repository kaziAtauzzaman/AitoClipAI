# End-to-End Analysis Pipeline

The `src/pipeline/` package composes the existing input and analysis stages. It
does not implement downloading, audio decoding, observation, or aggregation
logic itself.

`PipelineOrchestrator.analyze()` accepts either a local media path or an HTTP(S)
URL. URLs are resolved through `VideoDownloader`; local paths proceed directly.
The resolved media is converted to PCM WAV by `FFmpegAudioExtractor`, passed to
the configured `ObserverEngine` through `ObserverContext`, and aggregated by
`FeatureAggregator`.

The returned `FeatureTimeline` contains the source media path, extracted audio
path, canonical download result when applicable, aggregated observer timeline,
isolated observer failures, input metadata, and JSON artifact path.
`JsonFeatureTimelineWriter` writes the artifact beside the analyzed media using
the `.feature-timeline.json` suffix.

All orchestration collaborators are constructor-injected. The default wiring
registers one `AudioObserver`; callers can inject alternate downloaders,
extractors, observer engines, aggregators, and timeline writers without changing
the orchestration service.
