# Incremental WAV audio observer

`IncrementalWavAudioObserver` analyzes an already extracted PCM WAV in
chronological frame chunks. It is a real forward-only observer: unlike
`CompletedTimelineReplayAdapter`, it never scans future observations or future
audio frames.

Each `IncrementalAudioBatch` contains newly immutable observations and an
observer-confirmed watermark. The watermark is the earliest of the processed
audio frontier, the next overlapping analysis-window start, an open silence
start, and a peak still inside its minimum-distance competition interval. This
can deliberately lag observations already emitted by the batch. The
coordinator may only treat the prefix through the watermark as globally stable.

The session retains only the samples required by the next overlapping analysis
window, plus bounded detector state. It carries open silence state and pending
peak competition across chunk boundaries. EOF processes the one remaining
full-or-partial analysis window, closes an open silence, emits a remaining peak,
and advances the audio watermark to the authoritative WAV duration exactly
once.

The existing whole-file observer emits one full-duration `loudness`
observation. That value cannot be known safely before EOF and could revise
already finalized candidates, so incremental mode does not emit it as an
observation. Its RMS loudness and peak amplitude are reported only in EOF batch
metadata, with `whole_file_loudness_is_candidate_signal` set to `false`. Local
`speaking_intensity` and `peak` observations remain candidate signals and retain
the existing whole-file analysis semantics.
