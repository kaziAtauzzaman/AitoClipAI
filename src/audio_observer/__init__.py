"""Audio observer package for AitoClipAI."""

from audio_observer.analysis import (
    AudioAnalyzer,
    AudioAnalysisResult,
    LoudnessAnalyzer,
    PeakDetector,
    SilenceDetector,
    SpeakingIntensityAnalyzer,
)
from audio_observer.config import AudioObserverConfig
from audio_observer.contracts import AudioData, AudioSource
from audio_observer.errors import AudioObserverError
from audio_observer.extraction import AudioExtractor, ContextAudioExtractor
from audio_observer.loading import AudioLoader, WavAudioLoader
from audio_observer.observer import AudioObserver
from audio_observer.timestamping import TimestampGenerator

__all__ = [
    "AudioAnalyzer",
    "AudioAnalysisResult",
    "AudioData",
    "AudioExtractor",
    "AudioLoader",
    "AudioObserver",
    "AudioObserverConfig",
    "AudioObserverError",
    "AudioSource",
    "ContextAudioExtractor",
    "LoudnessAnalyzer",
    "PeakDetector",
    "SilenceDetector",
    "SpeakingIntensityAnalyzer",
    "TimestampGenerator",
    "WavAudioLoader",
]
