"""Audio loading services."""

import wave
from pathlib import Path
from typing import Protocol

from audio_observer.contracts import AudioData, AudioSource
from audio_observer.errors import AudioObserverError


class AudioLoader(Protocol):
    """Load normalized samples from an audio source."""

    def load(self, source: AudioSource) -> AudioData:
        """Return loaded mono audio data."""


class WavAudioLoader:
    """Load PCM WAV audio using the Python standard library."""

    def load(self, source: AudioSource) -> AudioData:
        """Load mono-normalized samples from a WAV file."""

        try:
            with wave.open(str(source.path), "rb") as wav_file:
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                sample_rate_hz = wav_file.getframerate()
                frame_count = wav_file.getnframes()
                raw_frames = wav_file.readframes(frame_count)
        except (OSError, wave.Error) as exc:
            raise AudioObserverError(f"Failed to load WAV audio: {exc}") from exc

        if channels <= 0:
            raise AudioObserverError("WAV audio must contain at least one channel.")
        if sample_rate_hz <= 0:
            raise AudioObserverError("WAV audio must define a positive sample rate.")
        if sample_width not in {1, 2, 4}:
            raise AudioObserverError(
                f"Unsupported WAV sample width: {sample_width} bytes."
            )

        samples = self._decode_pcm(raw_frames, sample_width, channels)
        return AudioData(
            samples=samples,
            sample_rate_hz=sample_rate_hz,
            channels=channels,
            metadata={**source.metadata, "path": str(Path(source.path))},
        )

    def _decode_pcm(
        self,
        raw_frames: bytes,
        sample_width: int,
        channels: int,
    ) -> tuple[float, ...]:
        if not raw_frames:
            return ()

        frame_size = sample_width * channels
        frame_count = len(raw_frames) // frame_size
        samples: list[float] = []

        for frame_index in range(frame_count):
            frame_start = frame_index * frame_size
            channel_values = [
                self._decode_sample(
                    raw_frames[
                        frame_start
                        + channel_index * sample_width : frame_start
                        + (channel_index + 1) * sample_width
                    ],
                    sample_width,
                )
                for channel_index in range(channels)
            ]
            samples.append(sum(channel_values) / channels)

        return tuple(samples)

    def _decode_sample(self, sample_bytes: bytes, sample_width: int) -> float:
        if sample_width == 1:
            return (sample_bytes[0] - 128) / 128.0

        value = int.from_bytes(sample_bytes, byteorder="little", signed=True)
        max_amplitude = float(2 ** (8 * sample_width - 1))
        return max(-1.0, min(1.0, value / max_amplitude))
