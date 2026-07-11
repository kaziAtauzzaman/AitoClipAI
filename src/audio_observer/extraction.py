"""Audio extraction abstractions."""

from pathlib import Path
from typing import Protocol

from audio_observer.contracts import AudioSource
from audio_observer.errors import AudioObserverError
from observers import ObserverContext


class AudioExtractor(Protocol):
    """Resolve an audio source from an observer context."""

    def extract(self, context: ObserverContext) -> AudioSource:
        """Return an audio source ready for loading."""


class ContextAudioExtractor:
    """Use the context source path as the audio artifact."""

    def extract(self, context: ObserverContext) -> AudioSource:
        """Resolve the source path from the observer context."""

        if context.source_path is None:
            raise AudioObserverError("Audio observer requires context.source_path.")

        path = Path(context.source_path)
        if not path.exists():
            raise AudioObserverError(f"Audio source does not exist: {path}")

        return AudioSource(path=path, metadata={"source": "context.source_path"})
