"""Shared typed data contracts for AitoClipAI pipeline stages.

The classes in this module intentionally contain no business logic. They define
the data shape exchanged between downloader, analysis, clipping, rendering, and
upload modules so each stage can evolve behind a stable interface.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Observation:
    """Generic observation produced by any observer.

    This contract exists so audio, speech, vision, OCR, and future observers can
    emit timestamped signals through a single neutral data shape. It does not
    define how observations are detected, interpreted, ranked, or acted on.

    Attributes:
        timestamp_seconds: Start time of the observation in the source media.
        duration_seconds: Optional duration covered by the observation.
        observer: Observer family that produced the observation, such as audio,
            speech, vision, or ocr.
        type: Observer-defined observation type, such as laughter,
            volume_spike, or scene_change.
        value: Generic observed value. This may be a string, number, boolean,
            list, dictionary, or other JSON-compatible payload.
        confidence: Optional confidence score assigned by the observer.
        metadata: Additional observer-specific metadata.
    """

    timestamp_seconds: float
    observer: str
    type: str
    value: Any
    duration_seconds: float | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ObserverResult:
    """Observer output containing timestamped observations.

    This contract exists so aggregation can consume results from any observer
    without depending on observer implementations or feature-specific payloads.

    Attributes:
        observer: Observer family or implementation name that produced the
            result.
        observations: Timestamped observations emitted by the observer.
        metadata: Additional observer-specific run metadata.
    """

    observer: str
    observations: list[Observation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TimelineGroup:
    """Observations that share the same source-media timestamp.

    Attributes:
        timestamp_seconds: Shared timestamp for the grouped observations.
        observations: Original observations at this timestamp.
    """

    timestamp_seconds: float
    observations: list[Observation] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class AggregatedTimeline:
    """Chronological aggregation of observer observations.

    This contract preserves observations exactly as emitted by observers. It is
    a structural grouping only, with no scoring, filtering, ranking, or AI
    decision-making.

    Attributes:
        groups: Chronological groups of observations by timestamp.
        observer_results: Original observer result objects included in the
            aggregation.
        metadata: Optional aggregation metadata.
    """

    groups: list[TimelineGroup] = field(default_factory=list)
    observer_results: list[ObserverResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DownloadResult:
    """Output contract produced by the downloader stage.

    This contract exists so downstream modules can consume downloaded source
    media without knowing how the downloader talks to yt-dlp or any future
    platform-specific client.

    Attributes:
        source_url: Original URL requested by the user or upstream workflow.
        video_path: Local path to the downloaded source video.
        metadata_path: Local path to the JSON metadata saved beside the video.
        provider: Platform or extractor name, such as YouTube, Twitch, or Kick.
        media_id: Provider-specific media identifier.
        title: Human-readable source title when available.
        duration_seconds: Source duration in seconds when known.
        metadata: Normalized or raw metadata needed by downstream stages.
    """

    source_url: str
    video_path: Path
    metadata_path: Path
    provider: str
    media_id: str
    title: str | None = None
    duration_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AudioFeatures:
    """Audio-analysis contract for one source video.

    This contract exists so scoring and clipping modules can use audio signals
    without depending on a specific audio-analysis implementation.

    Attributes:
        source_video_path: Local path to the analyzed video.
        audio_path: Optional extracted audio file path.
        duration_seconds: Duration of the analyzed audio.
        sample_rate_hz: Audio sample rate used during analysis.
        channels: Number of audio channels.
        loudness_lufs: Integrated loudness when measured.
        peak_dbfs: Peak level when measured.
        speech_ratio: Portion of the audio estimated to contain speech.
        music_ratio: Portion of the audio estimated to contain music.
        observations: Generic timestamped observations emitted by audio
            observers.
    """

    source_video_path: Path
    audio_path: Path | None = None
    duration_seconds: float | None = None
    sample_rate_hz: int | None = None
    channels: int | None = None
    loudness_lufs: float | None = None
    peak_dbfs: float | None = None
    speech_ratio: float | None = None
    music_ratio: float | None = None
    observations: list[Observation] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SpeechFeatures:
    """Speech and transcript contract for one source video.

    This contract exists so future transcription, captioning, scoring, and clip
    selection stages can exchange speech data through one typed structure.

    Attributes:
        source_video_path: Local path to the source video.
        transcript_path: Optional path to a transcript or caption artifact.
        language: Detected or configured language code.
        full_text: Complete transcript text when available.
        keywords: Important terms or phrases found in the speech.
        confidence: Overall speech recognition confidence when available.
        observations: Generic timestamped observations emitted by speech
            observers.
    """

    source_video_path: Path
    transcript_path: Path | None = None
    language: str | None = None
    full_text: str | None = None
    keywords: list[str] = field(default_factory=list)
    confidence: float | None = None
    observations: list[Observation] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class VisionFeatures:
    """Visual-analysis contract for one source video.

    This contract exists so visual analyzers can provide scene and frame-level
    signals without coupling downstream modules to a specific computer-vision
    library or model.

    Attributes:
        source_video_path: Local path to the analyzed video.
        frame_sample_rate: Frame sampling rate used during analysis.
        observations: Generic timestamped observations emitted by visual
            observers.
    """

    source_video_path: Path
    frame_sample_rate: float | None = None
    observations: list[Observation] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class OCRFeatures:
    """Optical-character-recognition contract for one source video.

    This contract exists so text detected on screen can be used by scoring,
    search, and caption-aware workflows independently of the OCR engine.

    Attributes:
        source_video_path: Local path to the analyzed video.
        language: OCR language code or model hint.
        full_text: Combined detected text when available.
        confidence: Overall OCR confidence when available.
        observations: Generic timestamped observations emitted by OCR
            observers.
    """

    source_video_path: Path
    language: str | None = None
    full_text: str | None = None
    confidence: float | None = None
    observations: list[Observation] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class AggregatedFeatures:
    """Combined feature contract for clip selection and scoring.

    This contract exists as the handoff between independent analyzers and the
    clip-candidate generation stage.

    Attributes:
        download: Downloader output associated with these features.
        audio: Audio feature payload when audio analysis has run.
        speech: Speech feature payload when transcription has run.
        vision: Vision feature payload when visual analysis has run.
        ocr: OCR feature payload when OCR analysis has run.
        tags: Cross-modal labels or categories.
        summary: Short aggregate description produced by a future stage.
        extra: Extension point for non-core feature data.
    """

    download: DownloadResult
    audio: AudioFeatures | None = None
    speech: SpeechFeatures | None = None
    vision: VisionFeatures | None = None
    ocr: OCRFeatures | None = None
    tags: list[str] = field(default_factory=list)
    summary: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ClipCandidate:
    """Candidate clip contract before final scoring or rendering.

    This contract exists so clip generation can propose possible highlights
    while leaving scoring, ranking, and rendering to later stages.

    Attributes:
        source_video_path: Local path to the source video.
        start_seconds: Start time of the candidate clip.
        end_seconds: End time of the candidate clip.
        reason: Human-readable reason this range may be useful.
        source_signals: Feature signals that contributed to this candidate.
        title: Optional working title for the candidate.
        metadata: Additional candidate metadata for later stages.
    """

    source_video_path: Path
    start_seconds: float
    end_seconds: float
    reason: str
    source_signals: list[str] = field(default_factory=list)
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ClipScore:
    """Scoring contract for a candidate clip.

    This contract exists so ranking and selection can consume a score without
    knowing which scoring model, rules, or signals produced it.

    Attributes:
        candidate: Clip candidate being scored.
        overall_score: Final normalized score assigned to the candidate.
        score_components: Named component scores, such as speech or visual.
        rationale: Optional explanation of why the score was assigned.
        passed_threshold: Whether the candidate met the configured threshold.
    """

    candidate: ClipCandidate
    overall_score: float
    score_components: dict[str, float] = field(default_factory=dict)
    rationale: str | None = None
    passed_threshold: bool | None = None


@dataclass(frozen=True, slots=True)
class RenderJob:
    """Rendering contract for producing a finished clip artifact.

    This contract exists so rendering workers can receive all required render
    inputs without depending on clip selection internals.

    Attributes:
        candidate: Clip candidate to render.
        output_path: Target path for the rendered clip.
        aspect_ratio: Desired output aspect ratio, such as 9:16 or 16:9.
        resolution: Desired output resolution, such as 1080x1920.
        captions_path: Optional caption file to burn in or package.
        preset: Optional named render preset.
        metadata: Additional render settings or workflow metadata.
    """

    candidate: ClipCandidate
    output_path: Path
    aspect_ratio: str | None = None
    resolution: str | None = None
    captions_path: Path | None = None
    preset: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class UploadJob:
    """Upload contract for publishing or exporting a rendered clip.

    This contract exists so upload integrations can receive a platform-neutral
    publishing request without depending on renderer or scoring internals.

    Attributes:
        rendered_clip_path: Local path to the rendered clip.
        destination: Target platform or export destination.
        title: Upload title.
        description: Upload description.
        tags: Upload tags or hashtags.
        scheduled_time: Optional ISO-8601 timestamp for scheduled publishing.
        visibility: Platform visibility setting, such as private or public.
        metadata: Additional destination-specific upload settings.
    """

    rendered_clip_path: Path
    destination: str
    title: str
    description: str | None = None
    tags: list[str] = field(default_factory=list)
    scheduled_time: str | None = None
    visibility: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
