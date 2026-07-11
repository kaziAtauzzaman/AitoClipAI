"""Configurable audio observer thresholds."""

from dataclasses import dataclass


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
