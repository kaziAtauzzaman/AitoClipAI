"""Timestamp generation for audio analysis windows."""

from dataclasses import dataclass

from audio_observer.errors import AudioObserverError


@dataclass(frozen=True, slots=True)
class AudioWindow:
    """Sample range and timestamp for an analysis window."""

    start_index: int
    end_index: int
    start_seconds: float
    duration_seconds: float


class TimestampGenerator:
    """Generate deterministic analysis windows from sample counts."""

    def windows(
        self,
        sample_count: int,
        sample_rate_hz: int,
        window_seconds: float,
        hop_seconds: float,
    ) -> list[AudioWindow]:
        """Return analysis windows covering the available samples."""

        if sample_rate_hz <= 0:
            raise AudioObserverError("Sample rate must be positive.")
        if window_seconds <= 0:
            raise AudioObserverError("Window duration must be positive.")
        if hop_seconds <= 0:
            raise AudioObserverError("Hop duration must be positive.")
        if sample_count <= 0:
            return []

        window_size = max(1, int(round(window_seconds * sample_rate_hz)))
        hop_size = max(1, int(round(hop_seconds * sample_rate_hz)))
        windows: list[AudioWindow] = []
        start_index = 0

        while start_index < sample_count:
            end_index = min(sample_count, start_index + window_size)
            windows.append(
                AudioWindow(
                    start_index=start_index,
                    end_index=end_index,
                    start_seconds=start_index / sample_rate_hz,
                    duration_seconds=(end_index - start_index) / sample_rate_hz,
                )
            )
            if end_index == sample_count:
                break
            start_index += hop_size

        return windows
