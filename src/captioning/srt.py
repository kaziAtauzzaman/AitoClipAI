"""Deterministic SRT formatting and persistence."""

from pathlib import Path
from typing import Protocol

from captioning.config import CaptionGeneratorConfig
from captioning.contracts import CaptionCue
from captioning.errors import CaptionPersistenceError


class CaptionFormatter(Protocol):
    """Format normalized caption cues into a subtitle document."""

    def format(self, cues: list[CaptionCue]) -> str:
        """Return the complete subtitle document text."""


class CaptionWriter(Protocol):
    """Persist one formatted caption document."""

    def write(self, path: Path, content: str, *, overwrite: bool) -> Path:
        """Write or reuse a caption artifact and return its path."""


class SrtCaptionFormatter:
    """Format cues as deterministic millisecond-precision SRT."""

    def format(self, cues: list[CaptionCue]) -> str:
        blocks = [
            "\n".join(
                [
                    str(index),
                    f"{_timestamp(cue.start_seconds)} --> {_timestamp(cue.end_seconds)}",
                    _normalize_text(cue.text),
                ]
            )
            for index, cue in enumerate(cues, start=1)
        ]
        return "\n\n".join(blocks) + ("\n" if blocks else "")


class FileCaptionWriter:
    """Write caption documents using an injected caption configuration."""

    def __init__(self, config: CaptionGeneratorConfig) -> None:
        self._config = config

    def write(self, path: Path, content: str, *, overwrite: bool) -> Path:
        if path.exists() and not overwrite:
            return path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding=self._config.encoding, newline="\n")
        except (LookupError, OSError) as exc:
            raise CaptionPersistenceError(f"Failed to write SRT captions: {exc}") from exc
        return path


def _timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{millis:03d}"


def _normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()
