# Production incremental orchestration

`ProductionIncrementalOrchestrator` is the prerecorded production composition
path for an existing source video and already extracted WAV. It is separate
from the batch pipeline and does not use completed-timeline replay.

The orchestrator alternates one Audio batch and one Whisper batch. Each batch's
stable observations are appended once, and its observer-confirmed watermark is
passed to `IncrementalPrerecordedCoordinator`. The coordinator takes the
minimum required-observer watermark, so faster Audio progress cannot finalize
work that slower Whisper has not confirmed. Rendering remains synchronous and
each successful render is validated immediately with the existing artifact
validator.

Both observers must emit authoritative EOF at the same frame-derived media
duration before the coordinator receives combined EOF. Each EOF must include
an explicit positive sample rate, and its watermark must equal its independently
frame-derived duration. Observer failures stop
further scheduling and close both sessions. Render attempts and artifact
validation have distinct failure records; a render failure remains retryable
under the coordinator's existing lifecycle rules.

The returned report preserves deterministic observation, selection,
suppression, render-job, failure, and watermark ordering. Timings record Audio
and Whisper batch work, coordinator advances, render attempts, artifact
validation, and total wall time. Every timing entry carries a stable operation
identity and success outcome; timing values are measurements and therefore are
not expected to be numerically identical across real runs.

Validation is attempted exactly once per persistent render identity. A
validator must return exactly one `RenderedArtifactValidation` whose output path
matches the requested job; empty, multiple, malformed, or mismatched results
are recorded as validation failures and are not silently retried.

`ProductionIncrementalReport.to_dict()` and
`JsonProductionIncrementalReportWriter` provide strict deterministic report
serialization. Paths use forward slashes, mapping keys are sorted, and
unsupported metadata, non-string keys, and non-finite floats are rejected. The
report includes the portable content-based source identity.
