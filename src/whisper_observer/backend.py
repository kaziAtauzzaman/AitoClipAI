"""Optional OpenAI Whisper transcription backend."""

import importlib
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from whisper_observer.config import WhisperObserverConfig
from whisper_observer.contracts import TranscriptionResult, TranscriptionSegment
from whisper_observer.errors import (
    InvalidTranscriptionError,
    TranscriptionError,
    WhisperUnavailableError,
)


class OpenAIWhisperBackend:
    """Lazy-loading adapter around the optional ``whisper`` Python package."""

    def __init__(
        self,
        module_loader: Callable[[str], ModuleType] | None = None,
    ) -> None:
        self._module_loader = module_loader or importlib.import_module
        self._models: dict[tuple[str, str | None], Any] = {}

    def transcribe(
        self,
        audio_path: Path,
        config: WhisperObserverConfig,
    ) -> TranscriptionResult:
        """Transcribe audio and normalize Whisper's result dictionary."""

        whisper = self._load_whisper()
        model = self._load_model(whisper, config)
        options = {**config.options, "task": config.task}
        if config.language is not None:
            options["language"] = config.language

        try:
            raw = model.transcribe(str(audio_path), **options)
        except Exception as exc:
            raise TranscriptionError(f"Whisper transcription failed: {exc}") from exc

        return self._normalize(raw)

    def _load_whisper(self) -> ModuleType:
        try:
            return self._module_loader("whisper")
        except (ImportError, ModuleNotFoundError) as exc:
            raise WhisperUnavailableError(
                "The optional 'whisper' package is not installed."
            ) from exc

    def _load_model(self, whisper: ModuleType, config: WhisperObserverConfig) -> Any:
        key = (config.model_name, config.device)
        if key not in self._models:
            kwargs = {"device": config.device} if config.device is not None else {}
            try:
                self._models[key] = whisper.load_model(config.model_name, **kwargs)
            except Exception as exc:
                raise TranscriptionError(
                    f"Failed to load Whisper model {config.model_name!r}: {exc}"
                ) from exc
        return self._models[key]

    def _normalize(self, raw: object) -> TranscriptionResult:
        if not isinstance(raw, dict):
            raise InvalidTranscriptionError(
                "Whisper returned a non-dictionary transcription result."
            )
        raw_segments = raw.get("segments", [])
        if not isinstance(raw_segments, list):
            raise InvalidTranscriptionError("Whisper segments must be a list.")

        segments = [self._normalize_segment(segment) for segment in raw_segments]
        return TranscriptionResult(
            segments=segments,
            text=str(raw.get("text", "")).strip(),
            language=_optional_string(raw.get("language")),
            metadata=dict(raw.get("metadata", {}))
            if isinstance(raw.get("metadata"), dict)
            else {},
        )

    def _normalize_segment(self, raw: object) -> TranscriptionSegment:
        if not isinstance(raw, dict):
            raise InvalidTranscriptionError("Whisper returned an invalid segment.")
        try:
            start = float(raw["start"])
            end = float(raw["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidTranscriptionError(
                "Whisper segment timestamps are missing or invalid."
            ) from exc

        preserved = {
            key: value
            for key, value in raw.items()
            if key not in {"start", "end", "text", "speaker", "confidence"}
        }
        return TranscriptionSegment(
            start_seconds=start,
            end_seconds=end,
            text=str(raw.get("text", "")).strip(),
            speaker=_optional_string(raw.get("speaker")),
            confidence=_optional_float(raw.get("confidence")),
            metadata=preserved,
        )


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
