# Candidate Selection

The `candidate_selection` package is the deterministic post-scoring,
pre-render stage. It receives threshold-passing `ClipScore` objects without
changing their candidate windows, component scores, rationale, or threshold
results.

Scores are processed using the existing score ordering. A weaker candidate is
suppressed when its intersection with an already retained stronger candidate
is at least `1.0` second and covers at least `0.65` of the shorter candidate.
The ratio is:

```text
intersection duration / shorter candidate duration
```

Both values are configurable through `CandidateSelectionConfig`. Candidates
that do not substantially overlap are preserved. Suppression decisions include
the retained score, overlap duration, ratio, and a deterministic explanation.

`PrerecordedVideoPipeline` retains every original score and every
threshold-passing score in its existing result fields. Only selector outputs
are sent to rendering. Therefore explainable feedback continues to represent a
suppressed passing candidate as `passed_not_rendered`.
