"""Serializable contracts for Editorial Strength v1 shadow diagnostics."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EditorialRawEvidence:
    candidate_duration_seconds: float
    audio_observation_count: int
    speech_observation_count: int
    silence_coverage: float
    speaking_intensity_coverage: float
    total_audio_coverage: float
    audio_evidence_coverage: float
    maximum_silence_to_reaction_rise_db: float
    local_loudness_baseline_dbfs: float | None
    local_loudness_p80_dbfs: float | None
    loudness_population_stddev_db: float
    speech_token_count: int
    unique_token_ratio: float
    longest_identical_token_run_ratio: float
    minimum_compression_ratio: float | None
    terminal_speech_present: bool


@dataclass(frozen=True, slots=True)
class EditorialNormalizedComponents:
    silence_to_reaction_rise: float
    local_baseline_contrast: float
    audio_variability: float
    meaningful_transition: float
    speech_information_completeness: float


@dataclass(frozen=True, slots=True)
class EditorialPenalties:
    muted_or_extremely_silent: float
    sustained_routine_intensity: float
    low_information_speech: float


@dataclass(frozen=True, slots=True)
class EditorialStrengthResult:
    formula_version: str
    candidate_identity: str
    editorial_score: float
    raw_evidence: EditorialRawEvidence
    normalized_components: EditorialNormalizedComponents
    penalties: EditorialPenalties


@dataclass(frozen=True, slots=True)
class EditorialStrengthFailure:
    formula_version: str
    candidate_identity: str
    code: str
    error_type: str
    message: str
