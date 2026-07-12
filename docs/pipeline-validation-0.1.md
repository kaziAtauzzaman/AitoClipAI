# Pipeline Validation 0.1

Pipeline Validation 0.1 proves that the existing prerecorded-video stages can
produce playable clips from one source. It is strict: `AudioObserver` and
`WhisperObserver` must both succeed, at least one candidate must be generated,
at least one score must pass, and every rendered clip must pass FFprobe checks.

## Automated Validation

Automated tests generate local audio/video with FFmpeg, inject deterministic
source acquisition and transcription, and run acquisition, metadata, analysis,
candidate generation, scoring, caption-free rendering, and FFprobe validation
without network access or model downloads.

Rendered clips must exist, be nonempty, contain audio and video streams, have
positive duration, start within 0.05 seconds of zero, and keep audio/video
duration differences within 0.08 seconds.

## Manual Validation

Install development and transcription dependencies and confirm `ffmpeg` and
`ffprobe` are on `PATH`. Choose a short prerecorded video with clear speech.

```bash
pip install -e ".[dev,transcription]"
python scripts/validate_prerecorded_pipeline.py SOURCE \
  --whisper-model tiny \
  --run-dir data/validation/pipeline-0.1 \
  --overwrite
```

The run directory contains `downloads/`, `audio/`, `timelines/`, `clips/`,
`logs/`, and `reports/`. URL sources receive downloader metadata JSON. Local
sources do not fabricate `DownloadResult`; their FFprobe metadata appears only
in `reports/validation-report.json`.

Open every file in `clips/` and confirm that video decodes, audio plays, and
the selected boundaries are sensible. Captions are explicitly disabled for
this validation milestone.
