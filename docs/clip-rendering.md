# Clip Rendering

The `src/clip_rendering/` package transforms ranked `ClipScore` objects into
rendered media artifacts and existing `RenderJob` contracts. It does not modify
scores or candidates and does not add captions, upload, or apply
platform-specific formatting.

Caption-free rendering remains the default. When `burn_subtitles` is enabled,
the renderer associates `CaptionArtifact` objects by resolved source path and
microsecond clip boundaries, then applies FFmpeg's subtitle filter after video
timestamps are reset. Successful jobs expose the SRT path through
`RenderJob.captions_path`.

`ClipRenderer` deterministically sorts scores, selects the configured number of
highest-ranked candidates, validates source media and time windows, and invokes
FFmpeg through an injected `RenderCommandRunner`.

Video and audio use the same start and end boundaries in one FFmpeg filter
graph. `trim` and `atrim` cut the streams, while `setpts` and `asetpts` reset
both timelines to zero before encoding. This preserves synchronization without
separate seek operations.

`ClipRendererConfig` controls the output directory, filename template,
overwrite behavior, container format, video codec, audio codec, maximum clip
count, and FFmpeg executable. Deterministic filename fields include source stem,
rank, millisecond boundaries, normalized score, and extension.

Subtitle filenames use dedicated FFmpeg filter escaping for Windows drive
letters, separators, quotes, commas, brackets, and semicolons. Subtitle FFmpeg
failures preserve diagnostics through `SubtitleRenderingError`.

Expected failures use `ClipRenderingError`, `RenderingFFmpegNotFoundError`, and
`InvalidRenderInputError`. Successful and reused artifacts are described by
`RenderJob` metadata containing ranking, score provenance, clip boundaries,
duration, encoding settings, and reuse status.
