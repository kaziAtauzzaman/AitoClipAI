# Candidate Scoring

The `src/candidate_scoring/` package ranks existing `ClipCandidate` objects with
deterministic, explainable heuristics. It does not modify candidates and does
not render, caption, upload, or use machine learning.

`CandidateScorer` reads the contributing observations preserved by candidate
generation and calculates normalized components for speech excitement,
speaking intensity, loudness peaks, silence buildup, supporting-observation
count, and observation diversity.

`CandidateScoringConfig` injects component weights, the passing threshold, and
normalization settings. Custom `ScoringHeuristic` implementations can replace
or extend the default set when matching weights are supplied.

The service returns existing `ClipScore` contracts ordered from highest to
lowest `overall_score`. `score_components` contains each component's final
weighted contribution, so the components sum to the overall score. `rationale`
records the raw normalized value, configured weight, final contribution, and
the measurement used by each heuristic.
