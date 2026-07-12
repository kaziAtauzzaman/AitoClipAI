"""Runtime configuration for audio extraction and observation."""

from dataclasses import dataclass
from pathlib import Path


DEFAULT_EXTRACTED_AUDIO_DIR = Path("data") / "audio"


@dataclass(frozen=True, slots=True)
class FFmpegAudioExtractorConfig:
    """Runtime configuration for deterministic FFmpeg audio extraction."""

    sample_rate_hz: int = 16_000
    channels: int = 1
    output_dir: Path = DEFAULT_EXTRACTED_AUDIO_DIR
    overwrite_existing: bool = False
    ffmpeg_binary: str = "ffmpeg"


@dataclass(frozen=True, slots=True)
class AudioObserverConfig:
    """Runtime configuration for the audio observer."""

    observer_name: str = "audio"
    order: int = 100
    window_seconds: float = 1.0
    hop_seconds: float = 0.5
    silence_threshold_dbfs: float = -45.0
    min_silence_seconds: float = 0.5
    peak_threshold: float = 0.85
    min_peak_distance_seconds: float = 0.25
    speaking_intensity_threshold_dbfs: float = -32.0
