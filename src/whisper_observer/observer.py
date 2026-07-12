"""Whisper observer implementation."""

from pathlib import Path

from core import Observation, ObserverResult
from observers import Observer, ObserverContext
from whisper_observer.backend import OpenAIWhisperBackend
from whisper_observer.config import WhisperObserverConfig
from whisper_observer.contracts import (
    TranscriptionBackend,
    TranscriptionResult,
    TranscriptionSegment,
)
from whisper_observer.errors import InvalidTranscriptionError, TranscriptionError


class WhisperObserver(Observer):
    """Transcribe extracted WAV audio into aggregation-compatible observations."""

    def __init__(
        self,
        config: WhisperObserverConfig | None = None,
        backend: TranscriptionBackend | None = None,
    ) -> None:
        self._config = config or WhisperObserverConfig()
        self._backend = backend or OpenAIWhisperBackend()

    @property
    def name(self) -> str:
        return self._config.observer_name

    @property
    def order(self) -> int:
        return self._config.order

    def observe(self, context: ObserverContext) -> ObserverResult:
        """Transcribe context audio and return timestamped speech observations."""

        audio_path = self._audio_path(context)
        try:
            transcription = self._backend.transcribe(audio_path, self._config)
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError(f"Transcription backend failed: {exc}") from exc

        self._validate(transcription)
        return ObserverResult(
            observer=self.name,
            observations=[self._observation(segment) for segment in transcription.segments],
            metadata={
                **transcription.metadata,
                "model_name": self._config.model_name,
                "language": transcription.language,
                "text": transcription.text,
                "segment_count": len(transcription.segments),
                "source_path": str(audio_path),
            },
        )

    def _audio_path(self, context: ObserverContext) -> Path:
        if context.source_path is None:
            raise TranscriptionError(
                "Whisper observer requires extracted audio in context.source_path."
            )
        audio_path = Path(context.source_path)
        if not audio_path.is_file():
            raise TranscriptionError(f"Transcription audio does not exist: {audio_path}")
        return audio_path

    def _validate(self, result: object) -> None:
        if not isinstance(result, TranscriptionResult):
            raise InvalidTranscriptionError(
                "Transcription backend must return TranscriptionResult."
            )
        for segment in result.segments:
            if not isinstance(segment, TranscriptionSegment):
                raise InvalidTranscriptionError(
                    "Transcription result contains a non-segment item."
                )
            if segment.start_seconds < 0 or segment.end_seconds < segment.start_seconds:
                raise InvalidTranscriptionError(
                    "Transcription segment timestamps are invalid."
                )

    def _observation(self, segment: TranscriptionSegment) -> Observation:
        metadata = {**segment.metadata}
        if segment.speaker is not None:
            metadata["speaker"] = segment.speaker
        return Observation(
            timestamp_seconds=segment.start_seconds,
            duration_seconds=segment.end_seconds - segment.start_seconds,
            observer=self.name,
            type="speech",
            value={"text": segment.text, "speaker": segment.speaker},
            confidence=segment.confidence,
            metadata=metadata,
        )
