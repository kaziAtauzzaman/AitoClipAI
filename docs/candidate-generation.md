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

Long clusters receive anchor-aware boundary refinement after clustering. The
default core target is at most 30 seconds and is chosen from the strongest
deterministic local concentration of existing contributions. Whisper speech,
loudness, peaks, and injected signals drive boundaries. Speaking-intensity and
silence-end events remain supporting evidence but cannot extend boundaries
indefinitely. Contiguous events with normalized strength of at least `0.80` may retain a
longer core when they represent sustained high signal. Pre/post roll, minimum
duration, media clamping, and the 60-second hard maximum are applied afterward.
Sustained eligibility uses normalized event strength rather than weighted
contribution and requires at least two independently observed boundary-driving
events whose individual spans fit the anchor target. Custom heuristics default
to supporting evidence unless they explicitly declare a driving boundary role.

Candidate metadata records the original cluster and refined anchor bounds.
Only observations intersecting the final window remain contributing
observations for scoring.

The existing `ClipCandidate` contract remains unchanged. Candidate
`start_seconds` and `end_seconds` are the generated start and end times;
`reason` contains the human-readable explanation; `source_signals` identifies
the contributing signal families; and `metadata` preserves confidence, time
aliases, contribution details, and the original `Observation` objects.
