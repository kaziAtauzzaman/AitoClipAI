# Candidate Generation

The `src/candidate_generation/` package transforms a completed
`FeatureTimeline` into deterministic `ClipCandidate` windows. It does not score,
rank, render, caption, or upload candidates.

`CandidateGenerator` flattens the aggregated observations, applies injected
`CandidateHeuristic` strategies, merges nearby normalized events, bounds each
window to the known media duration, and filters windows below the configured
confidence threshold.

The default heuristic set detects Whisper speech, audio loudness and peaks,
silence buildup, and speaking-intensity windows. `CandidateGenerationConfig`
controls signal thresholds and weights, merge distance, pre/post roll, minimum
and maximum clip duration, and minimum candidate confidence.

The existing `ClipCandidate` contract remains unchanged. Candidate
`start_seconds` and `end_seconds` are the generated start and end times;
`reason` contains the human-readable explanation; `source_signals` identifies
the contributing signal families; and `metadata` preserves confidence, time
aliases, contribution details, and the original `Observation` objects.
