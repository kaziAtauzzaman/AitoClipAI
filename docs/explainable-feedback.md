# Explainable Feedback

The `explainable_feedback` package creates a deterministic heuristic and
provenance report after a `PrerecordedVideoPipeline` run. It does not invoke a
language model, recalculate scores, or modify any pipeline artifact.

One aggregate UTF-8 JSON report contains one entry for every scored candidate.
Each entry traces the candidate window to overlapping timeline observations,
the candidate generator's reason and direct contributors, the scorer's exact
weighted `score_components` and rationale, its authoritative
`passed_threshold` result, and optional rendering and FFprobe validation data.

Candidate associations use the resolved source path plus start and end times
rounded to integer microseconds. Duplicate or unmatched score, render-job, or
artifact-validation associations fail with `FeedbackAssociationError` rather
than producing ambiguous provenance.

Selection statuses are:

- `rendered` when a matching validated render job exists
- `passed_not_rendered` when the score passed but was not rendered
- `below_threshold` when `passed_threshold` is false

Use `ExplainableFeedbackService` with an injected
`ExplainableFeedbackConfig(report_path=...)` after the pipeline returns. For a
validation run, the intended path is
`<run>/reports/explainable-feedback.json`.

The report deliberately contains no numeric passing threshold because the
existing score contract exposes only the authoritative boolean result. It also
does not parse the scorer rationale or attempt to recover raw heuristic values
and weights from that text.
