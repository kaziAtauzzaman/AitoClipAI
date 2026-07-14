"""Candidate-local Editorial Strength v1 shadow evaluator."""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import unicodedata

from candidate_scoring.heuristics import candidate_observations
from core import ClipCandidate, ClipScore, Observation
from decision_engine.config import EditorialStrengthConfig
from decision_engine.contracts import (
    EditorialNormalizedComponents,
    EditorialPenalties,
    EditorialRawEvidence,
    EditorialStrengthResult,
)
from decision_engine.errors import (
    EditorialStrengthError,
    InsufficientEditorialEvidenceError,
)


class EditorialStrengthEvaluator:
    """Compute v1 diagnostics without changing eligibility or selection.

    The evaluator is deterministic and candidate-local: its result depends only
    on the candidate, retained contributing observations, stable source id, and
    immutable configuration. It is therefore compatible with batch evaluation
    and with incremental evaluation after a candidate becomes stable.
    """

    candidate_local_deterministic = True
    incremental_compatible = True

    def __init__(self, config: EditorialStrengthConfig | None = None) -> None:
        self._config = config or EditorialStrengthConfig()
        self._validate_config()

    @property
    def formula_version(self) -> str:
        return self._config.formula_version

    def evaluate(
        self, scores: list[ClipScore], source_id: str
    ) -> list[EditorialStrengthResult]:
        """Evaluate every supplied score in deterministic candidate order."""

        if not isinstance(source_id, str) or not source_id.strip():
            raise EditorialStrengthError(
                "Editorial strength requires a stable source id."
            )
        ordered = sorted(
            scores,
            key=lambda item: (
                item.candidate.start_seconds,
                item.candidate.end_seconds,
                item.candidate.reason,
                candidate_identity(item.candidate, source_id.strip()),
            ),
        )
        return [self.evaluate_one(item, source_id.strip()) for item in ordered]

    def evaluate_one(self, score: ClipScore, source_id: str) -> EditorialStrengthResult:
        if not isinstance(source_id, str) or not source_id.strip():
            raise EditorialStrengthError(
                "Editorial strength requires a stable source id."
            )
        source_id = source_id.strip()
        candidate = score.candidate
        self._validate_candidate(candidate)
        observations = candidate_observations(candidate)
        evidence = self._raw_evidence(candidate, observations)
        self._require_sufficient_evidence(evidence)
        components = self._components(evidence)
        muted_base = _clamp(
            (evidence.silence_coverage - self._config.muted_silence_floor)
            / self._config.muted_silence_range
        )
        penalties = EditorialPenalties(
            muted_or_extremely_silent=_clamp(
                muted_base
                * (
                    1.0
                    - max(
                        components.meaningful_transition,
                        components.speech_information_completeness,
                    )
                )
            ),
            sustained_routine_intensity=_clamp(
                evidence.speaking_intensity_coverage
                * (1.0 - components.meaningful_transition)
            ),
            low_information_speech=_clamp(
                1.0 - components.speech_information_completeness
            ),
        )
        final = _clamp(
            self._config.rise_weight
            * components.silence_to_reaction_rise
            * evidence.audio_evidence_coverage
            + self._config.baseline_weight
            * components.local_baseline_contrast
            * evidence.audio_evidence_coverage
            + self._config.variability_weight
            * components.audio_variability
            * evidence.audio_evidence_coverage
            + self._config.transition_weight * components.meaningful_transition
            + self._config.speech_weight
            * components.speech_information_completeness
            - self._config.muted_penalty_weight
            * penalties.muted_or_extremely_silent
            - self._config.routine_penalty_weight
            * penalties.sustained_routine_intensity
            - self._config.low_information_penalty_weight
            * penalties.low_information_speech
        )
        return EditorialStrengthResult(
            formula_version=self._config.formula_version,
            candidate_identity=candidate_identity(candidate, source_id),
            editorial_score=_clean(final),
            raw_evidence=_clean_dataclass(evidence),
            normalized_components=_clean_dataclass(components),
            penalties=_clean_dataclass(penalties),
        )

    def _raw_evidence(
        self, candidate: ClipCandidate, observations: list[Observation]
    ) -> EditorialRawEvidence:
        duration = candidate.end_seconds - candidate.start_seconds
        silence: list[tuple[float, float, float]] = []
        intensity: list[tuple[float, float, float]] = []
        loudness: list[float] = []
        speech_texts: list[str] = []
        compression_ratios: list[float] = []

        for item in observations:
            self._validate_observation(item)
            observation_start = float(item.timestamp_seconds)
            observation_end = observation_start + float(item.duration_seconds or 0.0)
            start = max(candidate.start_seconds, observation_start)
            end = min(candidate.end_seconds, observation_end)
            if item.observer == "audio" and item.type in {
                "silence",
                "speaking_intensity",
            }:
                value = _required_mapping(item.value, item.type)
                measured = _required_finite_number(
                    value.get("loudness_dbfs"), "loudness_dbfs"
                )
                if end <= start:
                    continue
                row = (start, end, measured)
                (silence if item.type == "silence" else intensity).append(row)
                loudness.append(measured)
            elif item.observer == "whisper" and item.type == "speech":
                value = _required_mapping(item.value, "speech")
                text = value.get("text", "")
                if not isinstance(text, str):
                    raise EditorialStrengthError(
                        "Whisper speech text must be a string."
                    )
                speech_texts.append(text)
                if "compression_ratio" in item.metadata:
                    compression_ratios.append(
                        _required_finite_number(
                            item.metadata["compression_ratio"], "compression_ratio"
                        )
                    )

        rises = [
            reaction_db - silence_db
            for _, silence_end, silence_db in silence
            for reaction_start, _, reaction_db in intensity
            if 0.0
            <= reaction_start - silence_end
            <= self._config.reaction_max_gap_seconds
        ]
        tokens = _tokens(" ".join(speech_texts))
        unique_ratio = len(set(tokens)) / len(tokens) if tokens else 0.0
        longest_run_ratio = _longest_run(tokens) / len(tokens) if tokens else 1.0
        baseline = statistics.median(loudness) if loudness else None
        upper = _nearest_rank_percentile(loudness, 0.80) if loudness else None
        silence_coverage = _union_duration(silence) / duration
        intensity_coverage = _union_duration(intensity) / duration
        total_audio_coverage = _union_duration(silence + intensity) / duration
        coverage = _clamp(
            total_audio_coverage / self._config.audio_coverage_reference
        )
        return EditorialRawEvidence(
            candidate_duration_seconds=duration,
            audio_observation_count=len(silence) + len(intensity),
            speech_observation_count=len(speech_texts),
            silence_coverage=_clamp(silence_coverage),
            speaking_intensity_coverage=_clamp(intensity_coverage),
            total_audio_coverage=_clamp(total_audio_coverage),
            audio_evidence_coverage=coverage,
            maximum_silence_to_reaction_rise_db=max(rises, default=0.0),
            local_loudness_baseline_dbfs=baseline,
            local_loudness_p80_dbfs=upper,
            loudness_population_stddev_db=(
                statistics.pstdev(loudness) if len(loudness) > 1 else 0.0
            ),
            speech_token_count=len(tokens),
            unique_token_ratio=unique_ratio,
            longest_identical_token_run_ratio=longest_run_ratio,
            minimum_compression_ratio=(
                min(compression_ratios) if compression_ratios else None
            ),
            terminal_speech_present=any(
                text.strip().endswith((".", "!", "?")) for text in speech_texts
            ),
        )

    def _require_sufficient_evidence(
        self, evidence: EditorialRawEvidence
    ) -> None:
        if (
            evidence.audio_evidence_coverage
            < self._config.minimum_audio_evidence_coverage
            and evidence.speech_token_count < self._config.minimum_speech_tokens
        ):
            raise InsufficientEditorialEvidenceError(
                "Candidate lacks the minimum retained Audio or speech evidence."
            )

    def _components(
        self, evidence: EditorialRawEvidence
    ) -> EditorialNormalizedComponents:
        rise = _clamp(
            (
                evidence.maximum_silence_to_reaction_rise_db
                - self._config.reaction_rise_floor_db
            )
            / self._config.reaction_rise_range_db
        )
        contrast = 0.0
        if (
            evidence.local_loudness_baseline_dbfs is not None
            and evidence.local_loudness_p80_dbfs is not None
        ):
            contrast = _clamp(
                (
                    evidence.local_loudness_p80_dbfs
                    - evidence.local_loudness_baseline_dbfs
                )
                / self._config.baseline_contrast_range_db
            )
        variability = _clamp(
            (evidence.loudness_population_stddev_db - self._config.variability_floor_db)
            / self._config.variability_range_db
        )
        transition = _clamp(
            rise
            * evidence.audio_evidence_coverage
            * (
                1.0
                - self._config.sustained_intensity_discount
                * evidence.speaking_intensity_coverage
            )
        )
        unique = _clamp(
            (evidence.unique_token_ratio - self._config.unique_token_floor)
            / self._config.unique_token_range
        )
        compression = 0.0
        if evidence.minimum_compression_ratio is not None:
            compression = 1.0 - _clamp(
                (
                    evidence.minimum_compression_ratio
                    - self._config.compression_ratio_floor
                )
                / self._config.compression_ratio_range
            )
        repetition = 1.0 - _clamp(
            (
                evidence.longest_identical_token_run_ratio
                - self._config.repeated_run_floor
            )
            / self._config.repeated_run_range
        )
        speech = _clamp(
            0.40 * unique
            + 0.25 * compression
            + 0.25 * repetition
            + 0.10 * float(evidence.terminal_speech_present)
        )
        return EditorialNormalizedComponents(
            rise, contrast, variability, transition, speech
        )

    def _validate_candidate(self, candidate: ClipCandidate) -> None:
        start = _required_finite_number(candidate.start_seconds, "candidate start")
        end = _required_finite_number(candidate.end_seconds, "candidate end")
        if start < 0 or end <= start:
            raise EditorialStrengthError("Editorial candidate bounds are invalid.")
        raw = candidate.metadata.get("contributing_observations", [])
        if not isinstance(raw, list) or any(
            not isinstance(item, Observation) for item in raw
        ):
            raise EditorialStrengthError(
                "contributing_observations must be a list of Observation values."
            )

    @staticmethod
    def _validate_observation(item: Observation) -> None:
        timestamp = _required_finite_number(
            item.timestamp_seconds, "observation timestamp"
        )
        if timestamp < 0:
            raise EditorialStrengthError("Observation timestamp cannot be negative.")
        if item.duration_seconds is not None:
            duration = _required_finite_number(
                item.duration_seconds, "observation duration"
            )
            if duration < 0:
                raise EditorialStrengthError("Observation duration cannot be negative.")

    def _validate_config(self) -> None:
        config = self._config
        for item in fields(config):
            value = getattr(config, item.name)
            if item.name == "formula_version":
                if not isinstance(value, str) or not value:
                    raise EditorialStrengthError("Formula version must be non-empty.")
                continue
            number = _required_finite_number(value, item.name)
            if number < 0:
                raise EditorialStrengthError(f"{item.name} cannot be negative.")
        positive = (
            "audio_coverage_reference",
            "reaction_rise_range_db",
            "baseline_contrast_range_db",
            "variability_range_db",
            "unique_token_range",
            "compression_ratio_range",
            "repeated_run_range",
            "muted_silence_range",
        )
        if any(float(getattr(config, name)) <= 0 for name in positive):
            raise EditorialStrengthError(
                "Editorial normalization ranges must be positive."
            )
        if not 0.0 <= config.minimum_audio_evidence_coverage <= 1.0:
            raise EditorialStrengthError(
                "Minimum Audio evidence coverage must be between zero and one."
            )
        if (
            isinstance(config.minimum_speech_tokens, bool)
            or not isinstance(config.minimum_speech_tokens, int)
            or config.minimum_speech_tokens <= 0
        ):
            raise EditorialStrengthError(
                "Minimum speech token count must be a positive integer."
            )


def candidate_identity(candidate: ClipCandidate, source_id: str) -> str:
    """Return a portable semantic identity without including machine paths."""

    payload = {
        "source_id": source_id,
        "candidate": {
            "start_seconds": candidate.start_seconds,
            "end_seconds": candidate.end_seconds,
            "reason": candidate.reason,
            "source_signals": candidate.source_signals,
            "title": candidate.title,
            "metadata": candidate.metadata,
        },
    }
    encoded = json.dumps(
        _canonical_identity_value(payload), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def diagnostic_candidate_identity(candidate: ClipCandidate, source_id: str) -> str:
    """Return an identity that remains available when evidence metadata is invalid."""

    payload = {
        "source_id": source_id,
        "candidate_reference": {
            "start_seconds": candidate.start_seconds,
            "end_seconds": candidate.end_seconds,
            "reason": candidate.reason,
            "source_signals": candidate.source_signals,
            "title": candidate.title,
        },
    }
    encoded = json.dumps(
        _canonical_identity_value(payload), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_identity_value(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _canonical_identity_value(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise EditorialStrengthError("Identity metadata requires string keys.")
        return {
            key: _canonical_identity_value(value[key]) for key in sorted(value)
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_identity_value(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise EditorialStrengthError("Identity values must be finite.")
        return {"$float": "0" if value == 0 else format(value, ".17g")}
    if isinstance(value, Path):
        raise EditorialStrengthError("Identity metadata cannot contain paths.")
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise EditorialStrengthError(
        f"Unsupported identity metadata type: {type(value).__name__}."
    )


def _required_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise EditorialStrengthError(f"{label} value must be a string-keyed mapping.")
    return value


def _required_finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EditorialStrengthError(f"{label} must be numeric.")
    number = float(value)
    if not math.isfinite(number):
        raise EditorialStrengthError(f"{label} must be finite.")
    return number


def _tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.findall(r"[a-z0-9]+", normalized)


def _longest_run(tokens: list[str]) -> int:
    longest = 0
    current = 0
    previous: str | None = None
    for token in tokens:
        current = current + 1 if token == previous else 1
        longest = max(longest, current)
        previous = token
    return longest


def _nearest_rank_percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _union_duration(values: list[tuple[float, float, float]]) -> float:
    intervals = sorted((start, end) for start, end, _ in values if end > start)
    if not intervals:
        return 0.0
    total = 0.0
    start, end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def _clean(value: float) -> float:
    return 0.0 if value == 0 else round(value, 12)


def _clean_dataclass(value):
    return type(value)(
        **{
            item.name: (
                _clean(current) if isinstance(current, float) else current
            )
            for item in fields(value)
            if (current := getattr(value, item.name)) is not None
        },
        **{
            item.name: None
            for item in fields(value)
            if getattr(value, item.name) is None
        },
    )


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
