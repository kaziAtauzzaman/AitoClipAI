# AitoClipAI

AitoClipAI is a modular Python pipeline for turning prerecorded source video
into deterministic candidate clips. The current architecture supports source
download or local media, FFmpeg audio extraction, audio analysis, optional
Whisper transcription, timeline aggregation, candidate generation, explainable
scoring, FFmpeg clip rendering, optional burned-in SRT subtitles, and
deterministic post-pipeline heuristic/provenance feedback.

## Requirements

- Python 3.11 or newer
- FFmpeg and FFprobe available on `PATH`
- `yt-dlp` for URL acquisition
- Optional `openai-whisper` for real transcription
- Optional Google API dependencies for YouTube uploading

Install the project for development with transcription support:

```bash
pip install -e ".[dev,transcription]"
```

Install the optional YouTube client only when preparing a real upload:

```bash
pip install -e ".[youtube]"
```

## YouTube upload dry run

Plan an upload from an existing rendered clip without credentials or network:

```bash
python -m uploading --clip PATH_TO_CLIP --render-identity RENDER_ID \
  --title "Clip title" --description "Clip description" \
  --privacy-status private --dry-run
```

Run the complete automated suite:

```bash
pytest
```

## Pipeline Validation 0.1

The repository includes a narrow manual validation harness for one prerecorded
YouTube URL or local media file:

```bash
python scripts/validate_prerecorded_pipeline.py SOURCE \
  --whisper-model tiny \
  --run-dir data/validation/pipeline-0.1 \
  --overwrite
```

The harness requires successful audio and Whisper observers, passing candidate
scores, rendered clips with captions disabled, and FFprobe-confirmed playable
audio/video output. All generated files are contained under the selected run
directory. See `docs/pipeline-validation-0.1.md` for details.

## Packages

- `src/downloader/` — URL acquisition and metadata
- `src/audio_observer/` — FFmpeg extraction and deterministic audio signals
- `src/whisper_observer/` — dependency-injected timestamped transcription
- `src/observers/` — observer lifecycle and failure isolation
- `src/aggregation/` — feature timeline construction
- `src/candidate_generation/` — deterministic clip-window generation
- `src/candidate_scoring/` — explainable heuristic scoring
- `src/candidate_selection/` — deterministic overlap suppression before render
- `src/clip_rendering/` — synchronized FFmpeg rendering
- `src/captioning/` — optional SRT generation and subtitle burn-in support
- `src/pipeline/` — analysis and prerecorded-pipeline composition
- `src/explainable_feedback/` — deterministic scored-clip provenance reports
