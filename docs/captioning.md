# Deterministic Caption Generation

The `src/captioning/` package generates one UTF-8 SRT artifact per scored clip
candidate. It uses every Whisper `speech` observation in the complete
`FeatureTimeline`, rather than the smaller observation subset that contributed
to candidate generation.

Speech segments are selected when they overlap a candidate window. Partial
segments are clipped to the window, and all cue timestamps are rebased to the
start of the rendered clip. One cue is produced per Whisper observation;
speaker labels and confidence remain available on `CaptionCue`.

`CaptionArtifact` is local to the captioning package and associates with a
candidate through resolved source path plus microsecond start and end times.
Duplicate identities are rejected. Core contracts remain unchanged.

`CaptionGeneratorConfig` controls output location, deterministic filename
template, overwrite behavior, UTF-8 encoding, empty-text handling, and speaker
labels. Formatting and persistence are dependency-injected through
`CaptionFormatter` and `CaptionWriter` protocols.

Expected failures use `CaptionGenerationError`, `InvalidCaptionSourceError`,
`InvalidCaptionTimingError`, and `CaptionPersistenceError`.
