# Core Data Contracts

The `src/core/` package defines shared dataclasses for information exchanged
between AitoClipAI pipeline modules. These contracts are intentionally free of
business logic. They only describe data shape.

No analyzers, clipping logic, rendering logic, AI logic, or upload integrations
are implemented here.

## Contracts

### `Observation`

Exists to represent one timestamped signal produced by any observer. It is
generic by design: audio, speech, vision, OCR, and future observers can use the
same contract while keeping observer-specific details in `value` and
`metadata`.

### `DownloadResult`

Exists to represent the downloader stage output in a platform-neutral way.
Downstream modules can locate the downloaded video and metadata without knowing
how the file was downloaded.

### `AudioFeatures`

Exists to carry audio-analysis results, such as duration, loudness, speech
ratio, and generic audio observations. Future scoring and clip selection can use
this without depending on a specific analyzer.

### `SpeechFeatures`

Exists to carry transcript and speech-recognition outputs plus generic speech
observations. Future captioning, search, scoring, and clip selection can
consume speech data through this shared shape.

### `VisionFeatures`

Exists to carry visual-analysis outputs as generic visual observations. Future
modules can use visual signals without depending on a specific model or
computer-vision library.

### `OCRFeatures`

Exists to carry text detected on screen as generic OCR observations. This lets
future scoring and search workflows use visual text independently of the OCR
engine.

### `AggregatedFeatures`

Exists to combine downloader, audio, speech, vision, and OCR outputs into one
handoff object for clip candidate generation and scoring.

### `ClipCandidate`

Exists to describe a proposed clip range before scoring or rendering. It stores
timing, source path, reason, signals, and optional metadata.

### `ClipScore`

Exists to represent ranking information for a `ClipCandidate`. It separates the
score data from the scoring implementation.

### `RenderJob`

Exists to describe the data needed to render a finished clip artifact. It keeps
render inputs separate from the selection logic that produced the candidate.

### `UploadJob`

Exists to describe a platform-neutral upload or export request for a rendered
clip. Destination-specific integrations can translate this into API calls later.
