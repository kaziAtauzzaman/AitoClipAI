"""Explainable, injectable candidate scoring heuristics."""

from dataclasses import dataclass
from typing import Protocol

from candidate_scoring.config import CandidateScoringConfig
from core import ClipCandidate, Observation


@dataclass(frozen=True, slots=True)
class ComponentScore:
    """Normalized score and explanation returned by one heuristic."""

    value: float
    detail: str


class ScoringHeuristic(Protocol):
    """Score one explainable candidate signal on a zero-to-one scale."""

    @property
    def name(self) -> str:
        """Stable component name matching a configured weight."""

    def score(
        self,
        candidate: ClipCandidate,
        observations: list[Observation],
        config: CandidateScoringConfig,
    ) -> ComponentScore:
        """Return a normalized component score and human-readable detail."""


class SpeechExcitementHeuristic:
    name = "speech_excitement"

    def score(
        self,
        candidate: ClipCandidate,
        observations: list[Observation],
        config: CandidateScoringConfig,
    ) -> ComponentScore:
        speech = [item for item in observations if item.type == "speech"]
        values: list[float] = []
        for item in speech:
            text = _text(item)
            confidence = _clamp(item.confidence if item.confidence is not None else 0.5)
            punctuation = min(
                config.maximum_punctuation_boost,
                text.count("!") * config.exclamation_boost
                + text.count("?") * config.question_boost,
            )
            letters = [character for character in text if character.isalpha()]
            uppercase = (
                config.uppercase_boost
                if letters
                and sum(character.isupper() for character in letters) / len(letters) >= 0.6
                else 0.0
            )
            values.append(
                _clamp(
                    confidence * config.speech_confidence_factor
                    + punctuation
                    + uppercase
                )
            )
        value = max(values, default=0.0)
        return ComponentScore(
            value=value,
            detail=f"maximum excitement across {len(speech)} speech observations",
        )


class SpeakingIntensityScoreHeuristic:
    name = "speaking_intensity"

    def score(
        self,
        candidate: ClipCandidate,
        observations: list[Observation],
        config: CandidateScoringConfig,
    ) -> ComponentScore:
        intensities = [
            value
            for item in observations
            if item.type == "speaking_intensity"
            and isinstance(item.value, dict)
            if (value := _number(item.value.get("intensity"))) is not None
        ]
        value = _clamp(max(intensities, default=0.0))
        return ComponentScore(value, f"maximum of {len(intensities)} intensity values")


class LoudnessPeakScoreHeuristic:
    name = "loudness_peaks"

    def score(
        self,
        candidate: ClipCandidate,
        observations: list[Observation],
        config: CandidateScoringConfig,
    ) -> ComponentScore:
        peaks: list[float] = []
        for item in observations:
            if item.observer != "audio" or not isinstance(item.value, dict):
                continue
            if item.type == "peak":
                amplitude = _number(item.value.get("amplitude"))
                if amplitude is not None:
                    peaks.append(amplitude)
            elif item.type == "loudness":
                amplitude = _number(item.value.get("peak_amplitude"))
                if amplitude is not None:
                    peaks.append(amplitude)
        value = _clamp(max(peaks, default=0.0))
        return ComponentScore(value, f"maximum of {len(peaks)} peak measurements")


class SilenceBuildupScoreHeuristic:
    name = "silence_buildup"

    def score(
        self,
        candidate: ClipCandidate,
        observations: list[Observation],
        config: CandidateScoringConfig,
    ) -> ComponentScore:
        durations = [
            max(0.0, item.duration_seconds or 0.0)
            for item in observations
            if item.observer == "audio" and item.type == "silence"
        ]
        longest = max(durations, default=0.0)
        value = _clamp(longest / config.silence_reference_seconds)
        return ComponentScore(value, f"longest silence was {longest:.3f}s")


class SupportingObservationScoreHeuristic:
    name = "supporting_observations"

    def score(
        self,
        candidate: ClipCandidate,
        observations: list[Observation],
        config: CandidateScoringConfig,
    ) -> ComponentScore:
        count = len(observations)
        value = _clamp(count / config.supporting_observation_reference)
        return ComponentScore(value, f"{count} supporting observations")


class ObservationDiversityScoreHeuristic:
    name = "observation_diversity"

    def score(
        self,
        candidate: ClipCandidate,
        observations: list[Observation],
        config: CandidateScoringConfig,
    ) -> ComponentScore:
        families = {(item.observer, item.type) for item in observations}
        value = _clamp(len(families) / config.observation_diversity_reference)
        return ComponentScore(value, f"{len(families)} distinct observer/type families")


def default_heuristics() -> list[ScoringHeuristic]:
    """Return the default deterministic scoring components in stable order."""

    return [
        SpeechExcitementHeuristic(),
        SpeakingIntensityScoreHeuristic(),
        LoudnessPeakScoreHeuristic(),
        SilenceBuildupScoreHeuristic(),
        SupportingObservationScoreHeuristic(),
        ObservationDiversityScoreHeuristic(),
    ]


def candidate_observations(candidate: ClipCandidate) -> list[Observation]:
    """Read valid contributing observations without mutating candidate metadata."""

    raw = candidate.metadata.get("contributing_observations", [])
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, Observation)]


def _text(observation: Observation) -> str:
    if isinstance(observation.value, dict):
        value = observation.value.get("text", "")
        return str(value) if value is not None else ""
    return str(observation.value) if observation.value is not None else ""


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
