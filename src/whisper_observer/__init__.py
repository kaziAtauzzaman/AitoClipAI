"""Whisper speech observer package."""

from whisper_observer.backend import OpenAIWhisperBackend, OpenAIWhisperModelSession
from whisper_observer.config import (
    IncrementalWhisperObserverConfig,
    WhisperObserverConfig,
)
from whisper_observer.contracts import (
    IncrementalTranscriptionBackend,
    IncrementalTranscriptionSession,
    IncrementalWhisperAudioChunk,
    IncrementalWhisperBatch,
    IncrementalWhisperEOF,
    IncrementalWhisperLifecycle,
    finalized_speech_segment_identity,
    SegmentReconciliationPolicy,
    TranscriptionBackend,
    TranscriptionResult,
    TranscriptionSegment,
)
from whisper_observer.incremental import (
    IncrementalWhisperSessionCore,
    IncrementalWavWhisperObserver,
    IncrementalWavWhisperSession,
    TokenOverlapReconciliationPolicy,
)
from whisper_observer.live import LivePcmWhisperObserver, LivePcmWhisperSession
from whisper_observer.errors import (
    InvalidTranscriptionError,
    TranscriptionError,
    WhisperUnavailableError,
)
from whisper_observer.observer import WhisperObserver

__all__ = [
    "InvalidTranscriptionError",
    "IncrementalTranscriptionBackend",
    "IncrementalTranscriptionSession",
    "IncrementalWhisperAudioChunk",
    "IncrementalWhisperBatch",
    "IncrementalWhisperEOF",
    "IncrementalWhisperLifecycle",
    "finalized_speech_segment_identity",
    "IncrementalWhisperObserverConfig",
    "IncrementalWhisperSessionCore",
    "IncrementalWavWhisperObserver",
    "IncrementalWavWhisperSession",
    "LivePcmWhisperObserver",
    "LivePcmWhisperSession",
    "OpenAIWhisperBackend",
    "OpenAIWhisperModelSession",
    "SegmentReconciliationPolicy",
    "TokenOverlapReconciliationPolicy",
    "TranscriptionBackend",
    "TranscriptionError",
    "TranscriptionResult",
    "TranscriptionSegment",
    "WhisperObserver",
    "WhisperObserverConfig",
    "WhisperUnavailableError",
]
