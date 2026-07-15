"""Stateful incremental observation of an already extracted PCM WAV file."""

from collections.abc import Iterator
import math
from pathlib import Path
import wave

from audio_observer.analysis import MIN_DBFS, dbfs_to_intensity, rms_dbfs
from audio_observer.config import IncrementalAudioObserverConfig
from audio_observer.contracts import AudioSource, IncrementalAudioBatch
from audio_observer.errors import AudioObserverError
from core import Observation


class IncrementalWavAudioSession:
    """Read and analyze one WAV chronologically without loading it in full."""

    def __init__(
        self,
        source: AudioSource,
        config: IncrementalAudioObserverConfig | None = None,
    ) -> None:
        self._source = source
        self._config = config or IncrementalAudioObserverConfig()
        try:
            self._wav = wave.open(str(source.path), "rb")
        except (OSError, wave.Error) as exc:
            raise AudioObserverError(f"Failed to open incremental WAV audio: {exc}") from exc
        self._channels = self._wav.getnchannels()
        self._sample_width = self._wav.getsampwidth()
        self._sample_rate = self._wav.getframerate()
        if self._channels <= 0 or self._sample_rate <= 0:
            self.close()
            raise AudioObserverError("WAV audio must define channels and sample rate.")
        if self._sample_width not in {1, 2, 4}:
            self.close()
            raise AudioObserverError(
                f"Unsupported WAV sample width: {self._sample_width} bytes."
            )

        analysis = self._config.analysis
        self._window_size = max(1, round(analysis.window_seconds * self._sample_rate))
        self._hop_size = max(1, round(analysis.hop_seconds * self._sample_rate))
        self._min_peak_distance = max(
            1,
            round(analysis.min_peak_distance_seconds * self._sample_rate),
        )
        self._frames_processed = 0
        self._buffer_start = 0
        self._buffer: list[float] = []
        self._next_window_start = 0
        self._pending_peak: tuple[int, float] | None = None
        self._silence_start: float | None = None
        self._silence_end = 0.0
        self._silence_loudness_sum = 0.0
        self._silence_window_count = 0
        self._sum_squares = 0.0
        self._sample_count = 0
        self._peak_amplitude = 0.0
        self._eof_emitted = False
        self._closed = False

    @property
    def sample_rate_hz(self) -> int:
        return self._sample_rate

    def read_batch(self) -> IncrementalAudioBatch | None:
        """Read one chunk, or emit the single EOF flush batch."""

        if self._closed:
            return None
        try:
            raw = self._wav.readframes(self._config.chunk_frames)
        except (OSError, wave.Error) as exc:
            self.close()
            raise AudioObserverError(f"Failed to read incremental WAV audio: {exc}") from exc
        if not raw:
            return self.flush()

        samples = _decode_pcm(raw, self._sample_width, self._channels)
        chunk_start = self._frames_processed
        self._frames_processed += len(samples)
        self._buffer.extend(samples)
        self._accumulate_summary(samples)

        observations: list[Observation] = []
        observations.extend(self._process_peaks(samples, chunk_start, eof=False))
        observations.extend(self._process_complete_windows())
        return self._batch(observations, eof=False)

    def flush(self) -> IncrementalAudioBatch | None:
        """Emit final partial window, open silence, and pending peak exactly once."""

        if self._eof_emitted:
            return None
        if self._wav.tell() < self._wav.getnframes():
            raise AudioObserverError(
                "Cannot flush incremental WAV audio before the source is exhausted."
            )
        observations: list[Observation] = []
        if self._next_window_start < self._frames_processed:
            observations.extend(
                self._process_window(self._next_window_start, self._frames_processed)
            )
            self._next_window_start = self._frames_processed
            self._trim_buffer()
        observations.extend(self._close_silence())
        observations.extend(self._emit_pending_peak())
        self._eof_emitted = True
        duration = self._frames_processed / self._sample_rate
        batch = IncrementalAudioBatch(
            observer=self._config.analysis.observer_name,
            observations=tuple(sorted(observations, key=_observation_key)),
            watermark_seconds=duration,
            frames_processed=self._frames_processed,
            eof=True,
            metadata={
                **self._base_metadata(observations),
                "duration_seconds": duration,
                "overall_loudness_dbfs": self._overall_loudness(),
                "peak_amplitude": self._peak_amplitude,
                "whole_file_loudness_is_candidate_signal": False,
            },
        )
        self.close()
        return batch

    def close(self) -> None:
        if not self._closed:
            self._wav.close()
            self._closed = True

    def __enter__(self) -> "IncrementalWavAudioSession":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _process_complete_windows(self) -> list[Observation]:
        observations: list[Observation] = []
        while self._next_window_start + self._window_size < self._frames_processed:
            end = self._next_window_start + self._window_size
            observations.extend(self._process_window(self._next_window_start, end))
            self._next_window_start += self._hop_size
        self._trim_buffer()
        return observations

    def _process_window(self, start: int, end: int) -> list[Observation]:
        local_start = start - self._buffer_start
        local_end = end - self._buffer_start
        samples = tuple(self._buffer[local_start:local_end])
        loudness = rms_dbfs(samples)
        start_seconds = start / self._sample_rate
        duration_seconds = (end - start) / self._sample_rate
        observations: list[Observation] = []
        if loudness <= self._config.analysis.silence_threshold_dbfs:
            if self._silence_start is None:
                self._silence_start = start_seconds
                self._silence_loudness_sum = 0.0
                self._silence_window_count = 0
            self._silence_end = start_seconds + duration_seconds
            self._silence_loudness_sum += loudness
            self._silence_window_count += 1
        else:
            observations.extend(self._close_silence())
        if loudness >= self._config.analysis.speaking_intensity_threshold_dbfs:
            observations.append(
                Observation(
                    timestamp_seconds=start_seconds,
                    duration_seconds=duration_seconds,
                    observer=self._config.analysis.observer_name,
                    type="speaking_intensity",
                    value={
                        "intensity": dbfs_to_intensity(loudness),
                        "loudness_dbfs": loudness,
                    },
                )
            )
        return observations

    def _process_peaks(
        self,
        samples: tuple[float, ...],
        chunk_start: int,
        *,
        eof: bool,
    ) -> list[Observation]:
        observations: list[Observation] = []
        for offset, sample in enumerate(samples):
            amplitude = abs(sample)
            if amplitude < self._config.analysis.peak_threshold:
                continue
            index = chunk_start + offset
            if self._pending_peak is None:
                self._pending_peak = (index, amplitude)
                continue
            pending_index, pending_amplitude = self._pending_peak
            if index - pending_index < self._min_peak_distance:
                if amplitude > pending_amplitude:
                    self._pending_peak = (index, amplitude)
                continue
            observations.extend(self._emit_pending_peak())
            self._pending_peak = (index, amplitude)
        if eof or (
            self._pending_peak is not None
            and self._frames_processed - self._pending_peak[0]
            >= self._min_peak_distance
        ):
            observations.extend(self._emit_pending_peak())
        return observations

    def _emit_pending_peak(self) -> list[Observation]:
        if self._pending_peak is None:
            return []
        index, amplitude = self._pending_peak
        self._pending_peak = None
        return [
            Observation(
                timestamp_seconds=index / self._sample_rate,
                observer=self._config.analysis.observer_name,
                type="peak",
                value={"amplitude": amplitude},
            )
        ]

    def _close_silence(self) -> list[Observation]:
        if self._silence_start is None:
            return []
        start = self._silence_start
        duration = max(0.0, self._silence_end - start)
        loudness = (
            self._silence_loudness_sum / self._silence_window_count
            if self._silence_window_count
            else MIN_DBFS
        )
        self._silence_start = None
        self._silence_end = 0.0
        self._silence_loudness_sum = 0.0
        self._silence_window_count = 0
        if duration < self._config.analysis.min_silence_seconds:
            return []
        return [
            Observation(
                timestamp_seconds=start,
                duration_seconds=duration,
                observer=self._config.analysis.observer_name,
                type="silence",
                value={"loudness_dbfs": loudness},
            )
        ]

    def _batch(
        self,
        observations: list[Observation],
        *,
        eof: bool,
    ) -> IncrementalAudioBatch:
        return IncrementalAudioBatch(
            observer=self._config.analysis.observer_name,
            observations=tuple(sorted(observations, key=_observation_key)),
            watermark_seconds=self._stable_watermark(),
            frames_processed=self._frames_processed,
            eof=eof,
            metadata=self._base_metadata(observations),
        )

    def _stable_watermark(self) -> float:
        unstable = [
            self._frames_processed / self._sample_rate,
            self._next_window_start / self._sample_rate,
        ]
        if self._silence_start is not None:
            unstable.append(self._silence_start)
        if self._pending_peak is not None:
            unstable.append(self._pending_peak[0] / self._sample_rate)
        return min(unstable)

    def _trim_buffer(self) -> None:
        stable_frontier = min(self._next_window_start, self._frames_processed)
        drop = max(0, stable_frontier - self._buffer_start)
        if drop:
            del self._buffer[:drop]
            self._buffer_start += drop

    def _accumulate_summary(self, samples: tuple[float, ...]) -> None:
        self._sum_squares += sum(sample * sample for sample in samples)
        self._sample_count += len(samples)
        self._peak_amplitude = max(
            self._peak_amplitude,
            max((abs(sample) for sample in samples), default=0.0),
        )

    def _overall_loudness(self) -> float:
        if self._sample_count <= 0:
            return MIN_DBFS
        rms = math.sqrt(self._sum_squares / self._sample_count)
        if rms <= 0:
            return MIN_DBFS
        return max(MIN_DBFS, 20.0 * math.log10(rms))

    def _base_metadata(
        self, observations: list[Observation] | None = None
    ) -> dict[str, object]:
        metadata: dict[str, object] = {
            "source_path": str(self._source.path),
            "sample_rate_hz": self._sample_rate,
            "channels": self._channels,
        }
        finalized_peaks = tuple(
            item.timestamp_seconds
            for item in observations or []
            if item.type == "peak"
        )
        if finalized_peaks:
            metadata["finalized_peak_timestamps_seconds"] = finalized_peaks
        return metadata


class IncrementalWavAudioObserver:
    """Create chronological stable batches from an already extracted WAV.

    This observer reads only the next configured PCM frame chunk. It does not
    inspect future WAV frames and therefore its watermarks can be passed
    directly to ``IncrementalPrerecordedCoordinator``.
    """

    def __init__(self, config: IncrementalAudioObserverConfig | None = None) -> None:
        self._config = config or IncrementalAudioObserverConfig()

    def session(self, source: AudioSource | Path) -> IncrementalWavAudioSession:
        resolved = source if isinstance(source, AudioSource) else AudioSource(source)
        return IncrementalWavAudioSession(resolved, self._config)

    def batches(self, source: AudioSource | Path) -> Iterator[IncrementalAudioBatch]:
        with self.session(source) as session:
            while (batch := session.read_batch()) is not None:
                yield batch


def _decode_pcm(raw_frames: bytes, sample_width: int, channels: int) -> tuple[float, ...]:
    frame_size = sample_width * channels
    frame_count = len(raw_frames) // frame_size
    samples: list[float] = []
    for frame_index in range(frame_count):
        frame_start = frame_index * frame_size
        values = []
        for channel in range(channels):
            start = frame_start + channel * sample_width
            raw = raw_frames[start : start + sample_width]
            if sample_width == 1:
                values.append((raw[0] - 128) / 128.0)
            else:
                value = int.from_bytes(raw, "little", signed=True)
                values.append(value / float(2 ** (8 * sample_width - 1)))
        samples.append(max(-1.0, min(1.0, sum(values) / channels)))
    return tuple(samples)


def _observation_key(observation: Observation) -> tuple[float, str]:
    return observation.timestamp_seconds, observation.type
