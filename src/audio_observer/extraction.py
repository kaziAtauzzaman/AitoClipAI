"""Audio extraction abstractions."""

import shutil
import subprocess
from pathlib import Path
from typing import Callable, Protocol, Sequence

from audio_observer.config import FFmpegAudioExtractorConfig
from audio_observer.contracts import AudioSource
from audio_observer.errors import (
    AudioExtractionError,
    AudioObserverError,
    FFmpegNotFoundError,
)
from observers import ObserverContext


class AudioExtractor(Protocol):
    """Resolve an audio source from an observer context."""

    def extract(self, context: ObserverContext) -> AudioSource:
        """Return an audio source ready for loading."""


class CommandRunner(Protocol):
    """Execute an external command and capture its result."""

    def run(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        """Run a command without raising for a non-zero exit status."""


class SubprocessCommandRunner:
    """Command runner backed by the standard-library subprocess module."""

    def run(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        """Run a command while capturing diagnostic output."""

        try:
            return subprocess.run(
                list(command),
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            raise AudioExtractionError(f"Failed to execute FFmpeg: {exc}") from exc


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


class FFmpegAudioExtractor:
    """Extract deterministic PCM WAV audio from context media using FFmpeg."""

    def __init__(
        self,
        config: FFmpegAudioExtractorConfig | None = None,
        runner: CommandRunner | None = None,
        executable_locator: Callable[[str], str | None] | None = None,
    ) -> None:
        self._config = config or FFmpegAudioExtractorConfig()
        self._runner = runner or SubprocessCommandRunner()
        self._executable_locator = executable_locator or shutil.which

    def extract(self, context: ObserverContext) -> AudioSource:
        """Extract configured PCM WAV audio from ``context.source_path``."""

        source_path = self._source_path(context)
        self._validate_config()
        ffmpeg_path = self._executable_locator(self._config.ffmpeg_binary)
        if ffmpeg_path is None:
            raise FFmpegNotFoundError(
                f"FFmpeg executable was not found: {self._config.ffmpeg_binary!r}."
            )

        output_path = self._output_path(source_path)
        if output_path.exists() and not self._config.overwrite_existing:
            return self._audio_source(source_path, output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = self._build_command(ffmpeg_path, source_path, output_path)
        result = self._runner.run(command)

        if result.returncode != 0:
            diagnostic = result.stderr.strip() or result.stdout.strip()
            detail = f": {diagnostic}" if diagnostic else ""
            raise AudioExtractionError(
                f"FFmpeg audio extraction failed with exit code "
                f"{result.returncode}{detail}"
            )
        if not output_path.is_file():
            raise AudioExtractionError(
                f"FFmpeg completed without creating audio output: {output_path}"
            )

        return self._audio_source(source_path, output_path)

    def _source_path(self, context: ObserverContext) -> Path:
        if context.source_path is None:
            raise AudioExtractionError(
                "FFmpeg audio extraction requires context.source_path."
            )

        source_path = Path(context.source_path)
        if not source_path.is_file():
            raise AudioExtractionError(f"Input media does not exist: {source_path}")
        return source_path

    def _validate_config(self) -> None:
        if self._config.sample_rate_hz <= 0:
            raise AudioExtractionError("FFmpeg sample rate must be positive.")
        if self._config.channels <= 0:
            raise AudioExtractionError("FFmpeg channel count must be positive.")

    def _output_path(self, source_path: Path) -> Path:
        filename = (
            f"{source_path.stem}.{self._config.sample_rate_hz}hz."
            f"{self._config.channels}ch.wav"
        )
        return self._config.output_dir / filename

    def _build_command(
        self,
        ffmpeg_path: str,
        source_path: Path,
        output_path: Path,
    ) -> list[str]:
        return [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y" if self._config.overwrite_existing else "-n",
            "-i",
            str(source_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(self._config.sample_rate_hz),
            "-ac",
            str(self._config.channels),
            str(output_path),
        ]

    def _audio_source(self, source_path: Path, output_path: Path) -> AudioSource:
        return AudioSource(
            path=output_path,
            metadata={
                "source": "ffmpeg",
                "input_path": str(source_path),
                "sample_rate_hz": self._config.sample_rate_hz,
                "channels": self._config.channels,
            },
        )
