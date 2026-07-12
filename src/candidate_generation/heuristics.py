"""Injectable observation heuristics for candidate generation."""

from dataclasses import dataclass
from typing import Protocol

from core import Observation
from candidate_generation.config import CandidateGenerationConfig


@dataclass(frozen=True, slots=True)
class CandidateEvent:
    """Normalized candidate signal produced from one observation."""

    start_seconds: float
    end_seconds: float
    signal: str
    strength: float
    weight: float
    observation: Observation

    @property
    def contribution(self) -> float:
        return self.strength * self.weight


class CandidateHeuristic(Protocol):
    """Convert a relevant observation into a normalized candidate event."""

    def detect(self, observation: Observation) -> CandidateEvent | None:
        """Return an event when the observation passes this heuristic."""


class WhisperSpeechHeuristic:
    """Detect timestamped Whisper speech with usable text and confidence."""

    def __init__(self, config: CandidateGenerationConfig) -> None:
        self._config = config

    def detect(self, observation: Observation) -> CandidateEvent | None:
        if observation.type != "speech" or observation.observer != "whisper":
            return None
        text = observation.value.get("text") if isinstance(observation.value, dict) else None
        if not isinstance(text, str) or not text.strip():
            return None
        confidence = observation.confidence if observation.confidence is not None else 0.75
        if confidence < self._config.minimum_speech_confidence:
            return None
        return _event(
            observation,
            signal="whisper_speech",
            strength=confidence,
            weight=self._config.speech_weight,
        )


class AudioLoudnessHeuristic:
    """Detect high overall loudness and discrete audio peaks."""

    def __init__(self, config: CandidateGenerationConfig) -> None:
        self._config = config

    def detect(self, observation: Observation) -> CandidateEvent | None:
        if observation.observer != "audio" or not isinstance(observation.value, dict):
            return None
        if observation.type == "loudness":
            loudness = _number(observation.value.get("loudness_dbfs"))
            if loudness is None or loudness < self._config.loudness_threshold_dbfs:
                return None
            denominator = abs(self._config.loudness_threshold_dbfs) or 1.0
            strength = _clamp(
                (loudness - self._config.loudness_threshold_dbfs) / denominator
            )
            return _point_event(
                observation,
                signal="audio_loudness",
                strength=strength,
                weight=self._config.loudness_weight,
            )
        if observation.type == "peak":
            amplitude = _number(observation.value.get("amplitude"))
            if amplitude is None or amplitude < self._config.peak_threshold:
                return None
            return _point_event(
                observation,
                signal="audio_peak",
                strength=_clamp(amplitude),
                weight=self._config.peak_weight,
            )
        return None


class SilenceBuildupHeuristic:
    """Treat the end of a sufficiently long silence as a candidate moment."""

    def __init__(self, config: CandidateGenerationConfig) -> None:
        self._config = config

    def detect(self, observation: Observation) -> CandidateEvent | None:
        if observation.observer != "audio" or observation.type != "silence":
            return None
        duration = observation.duration_seconds or 0.0
        if duration < self._config.minimum_silence_seconds:
            return None
        timestamp = observation.timestamp_seconds + duration
        return CandidateEvent(
            start_seconds=timestamp,
            end_seconds=timestamp,
            signal="silence_buildup",
            strength=_clamp(duration / self._config.silence_reference_seconds),
            weight=self._config.silence_weight,
            observation=observation,
        )


class SpeakingIntensityHeuristic:
    """Detect active speech windows with sufficient normalized intensity."""

    def __init__(self, config: CandidateGenerationConfig) -> None:
        self._config = config

    def detect(self, observation: Observation) -> CandidateEvent | None:
        if (
            observation.observer != "audio"
            or observation.type != "speaking_intensity"
            or not isinstance(observation.value, dict)
        ):
            return None
        intensity = _number(observation.value.get("intensity"))
        if intensity is None or intensity < self._config.speaking_intensity_threshold:
            return None
        return _event(
            observation,
            signal="speaking_intensity",
            strength=_clamp(intensity),
            weight=self._config.speaking_intensity_weight,
        )


def default_heuristics(
    config: CandidateGenerationConfig,
) -> list[CandidateHeuristic]:
    """Build the deterministic default heuristic set."""

    return [
        WhisperSpeechHeuristic(config),
        AudioLoudnessHeuristic(config),
        SilenceBuildupHeuristic(config),
        SpeakingIntensityHeuristic(config),
    ]


def _event(
    observation: Observation,
    *,
    signal: str,
    strength: float,
    weight: float,
) -> CandidateEvent:
    duration = max(0.0, observation.duration_seconds or 0.0)
    return CandidateEvent(
        start_seconds=observation.timestamp_seconds,
        end_seconds=observation.timestamp_seconds + duration,
        signal=signal,
        strength=_clamp(strength),
        weight=weight,
        observation=observation,
    )


def _point_event(
    observation: Observation,
    *,
    signal: str,
    strength: float,
    weight: float,
) -> CandidateEvent:
    return CandidateEvent(
        start_seconds=observation.timestamp_seconds,
        end_seconds=observation.timestamp_seconds,
        signal=signal,
        strength=_clamp(strength),
        weight=weight,
        observation=observation,
    )


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
