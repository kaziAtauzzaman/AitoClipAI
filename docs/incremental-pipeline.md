# Incremental pipeline foundation

`IncrementalPrerecordedCoordinator` accepts accumulated stable observations,
explicit per-observer watermarks, and an authoritative EOF contract. It does
not inspect observations that an observer has not emitted.

`CompletedTimelineReplayAdapter` is simulation and testing infrastructure only.
It replays an already completed `FeatureTimeline` and deliberately inspects
future observations to calculate conservative prototype watermarks. It must not
be used as, wrapped as, or substituted for a real streaming observer.

Future incremental Audio and Whisper implementations must emit their own
observer-confirmed watermarks and pass accumulated stable observations directly
to the coordinator.
