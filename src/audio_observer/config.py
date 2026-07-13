"""Runtime configuration for audio extraction and observation."""

from dataclasses import dataclass
import math
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


@dataclass(frozen=True, slots=True)
class IncrementalAudioObserverConfig:
    """Chunk sizing and analysis policy for incremental WAV observation."""

    chunk_frames: int = 80_000
    analysis: AudioObserverConfig = AudioObserverConfig()

    def __post_init__(self) -> None:
        if self.chunk_frames <= 0:
            raise ValueError("Incremental audio chunk size must be positive.")
        numeric_values = {
            "window_seconds": self.analysis.window_seconds,
            "hop_seconds": self.analysis.hop_seconds,
            "min_silence_seconds": self.analysis.min_silence_seconds,
            "min_peak_distance_seconds": self.analysis.min_peak_distance_seconds,
        }
        if any(
            not math.isfinite(value) or value <= 0
            for value in numeric_values.values()
        ):
            raise ValueError(
                "Incremental audio durations must be finite and positive."
            )
