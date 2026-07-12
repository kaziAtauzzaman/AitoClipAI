"""Audio observer package for AitoClipAI."""

from audio_observer.analysis import (
    AudioAnalyzer,
    AudioAnalysisResult,
    LoudnessAnalyzer,
    PeakDetector,
    SilenceDetector,
    SpeakingIntensityAnalyzer,
)
from audio_observer.config import AudioObserverConfig, FFmpegAudioExtractorConfig
from audio_observer.contracts import AudioData, AudioSource
from audio_observer.errors import (
    AudioExtractionError,
    AudioObserverError,
    FFmpegNotFoundError,
)
from audio_observer.extraction import (
    AudioExtractor,
    CommandRunner,
    ContextAudioExtractor,
    FFmpegAudioExtractor,
    SubprocessCommandRunner,
)
from audio_observer.loading import AudioLoader, WavAudioLoader
from audio_observer.observer import AudioObserver
from audio_observer.timestamping import TimestampGenerator

__all__ = [
    "AudioAnalyzer",
    "AudioAnalysisResult",
    "AudioData",
    "AudioExtractionError",
    "AudioExtractor",
    "AudioLoader",
    "AudioObserver",
    "AudioObserverConfig",
    "AudioObserverError",
    "AudioSource",
    "CommandRunner",
    "ContextAudioExtractor",
    "FFmpegAudioExtractor",
    "FFmpegAudioExtractorConfig",
    "FFmpegNotFoundError",
    "LoudnessAnalyzer",
    "PeakDetector",
    "SilenceDetector",
    "SpeakingIntensityAnalyzer",
    "SubprocessCommandRunner",
    "TimestampGenerator",
    "WavAudioLoader",
]
