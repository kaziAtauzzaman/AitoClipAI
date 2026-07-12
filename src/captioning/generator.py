"""Deterministic SRT generation from complete feature timelines."""

from pathlib import Path
from typing import Iterable

from captioning.config import CaptionGeneratorConfig
from captioning.contracts import (
    CaptionArtifact,
    CaptionCue,
    CandidateCaptionIdentity,
    candidate_caption_identity,
)
from captioning.errors import (
    InvalidCaptionSourceError,
    InvalidCaptionTimingError,
)
from captioning.srt import (
    CaptionFormatter,
    CaptionWriter,
    FileCaptionWriter,
    SrtCaptionFormatter,
)
from core import ClipCandidate, ClipScore, FeatureTimeline, Observation


class CaptionGenerator:
    """Generate one deterministic clip-relative SRT artifact per candidate."""

    def __init__(
        self,
        config: CaptionGeneratorConfig | None = None,
        formatter: CaptionFormatter | None = None,
        writer: CaptionWriter | None = None,
    ) -> None:
        self._config = config or CaptionGeneratorConfig()
        self._formatter = formatter or SrtCaptionFormatter()
        self._writer = writer or FileCaptionWriter(self._config)
        self._validate_config()

    def generate(
        self,
        feature_timeline: FeatureTimeline,
        scores: Iterable[ClipScore],
    ) -> list[CaptionArtifact]:
        """Generate captions from all timeline speech overlapping each candidate."""

        ranked = sorted(scores, key=self._score_key)
        self._ensure_unique_candidates(score.candidate for score in ranked)
        speech = self._speech_observations(feature_timeline)
        language = self._language(feature_timeline)
        return [
            self._artifact(
                score.candidate,
                speech,
                language,
                feature_timeline.timeline_path,
                feature_timeline.media_path,
            )
            for score in ranked
        ]

    def _artifact(
        self,
        candidate: ClipCandidate,
        speech: list[Observation],
        language: str | None,
        timeline_path: Path,
        timeline_media_path: Path,
    ) -> CaptionArtifact:
        self._validate_candidate(candidate, timeline_media_path)
        overlapping = [item for item in speech if self._overlaps(item, candidate)]
        cues = [
            cue
            for index, item in enumerate(overlapping, start=1)
            if (cue := self._cue(index, item, candidate)) is not None
        ]
        cues = [
            CaptionCue(
                index=index,
                start_seconds=cue.start_seconds,
                end_seconds=cue.end_seconds,
                text=cue.text,
                speaker=cue.speaker,
                confidence=cue.confidence,
                metadata=cue.metadata,
            )
            for index, cue in enumerate(cues, start=1)
        ]
        path = self._caption_path(candidate)
        self._writer.write(
            path,
            self._formatter.format(cues),
            overwrite=self._config.overwrite_existing,
        )
        return CaptionArtifact(
            candidate=candidate,
            path=path,
            cues=cues,
            language=language,
            metadata={
                "source_timeline_path": str(timeline_path),
                "cue_count": len(cues),
                "format": "srt",
                "encoding": self._config.encoding,
            },
        )

    def _speech_observations(
        self,
        feature_timeline: FeatureTimeline,
    ) -> list[Observation]:
        speech = [
            observation
            for group in feature_timeline.timeline.groups
            for observation in group.observations
            if observation.observer == "whisper" and observation.type == "speech"
        ]
        return sorted(
            speech,
            key=lambda item: (
                item.timestamp_seconds,
                item.timestamp_seconds + (item.duration_seconds or 0.0),
                _speech_text(item),
            ),
        )

    def _overlaps(self, observation: Observation, candidate: ClipCandidate) -> bool:
        duration = observation.duration_seconds
        if duration is None or duration <= 0:
            raise InvalidCaptionTimingError(
                "Whisper speech observations require a positive duration."
            )
        if observation.timestamp_seconds < 0:
            raise InvalidCaptionTimingError(
                "Whisper speech timestamps cannot be negative."
            )
        end = observation.timestamp_seconds + duration
        return end > candidate.start_seconds and observation.timestamp_seconds < candidate.end_seconds

    def _cue(
        self,
        index: int,
        observation: Observation,
        candidate: ClipCandidate,
    ) -> CaptionCue | None:
        text = _speech_text(observation).strip()
        if not text and self._config.skip_empty_text:
            return None
        speaker = _speaker(observation)
        if speaker and self._config.include_speaker_labels:
            try:
                text = self._config.speaker_template.format(
                    speaker=speaker,
                    text=text,
                )
            except (KeyError, ValueError) as exc:
                raise InvalidCaptionSourceError(
                    f"Invalid speaker template: {exc}"
                ) from exc
        source_end = observation.timestamp_seconds + (observation.duration_seconds or 0.0)
        start = max(observation.timestamp_seconds, candidate.start_seconds)
        end = min(source_end, candidate.end_seconds)
        return CaptionCue(
            index=index,
            start_seconds=round(start - candidate.start_seconds, 6),
            end_seconds=round(end - candidate.start_seconds, 6),
            text=text,
            speaker=speaker,
            confidence=observation.confidence,
            metadata={
                **observation.metadata,
                "source_start_seconds": observation.timestamp_seconds,
                "source_end_seconds": source_end,
            },
        )

    def _caption_path(self, candidate: ClipCandidate) -> Path:
        extension_values = {
            "stem": candidate.source_video_path.stem,
            "start_ms": round(candidate.start_seconds * 1000),
            "end_ms": round(candidate.end_seconds * 1000),
            "start_microseconds": round(candidate.start_seconds * 1_000_000),
            "end_microseconds": round(candidate.end_seconds * 1_000_000),
        }
        try:
            filename = self._config.filename_template.format(**extension_values)
        except (KeyError, ValueError) as exc:
            raise InvalidCaptionSourceError(
                f"Invalid caption filename template: {exc}"
            ) from exc
        if not filename or Path(filename).name != filename:
            raise InvalidCaptionSourceError(
                "Caption filename template must produce one non-empty filename."
            )
        return self._config.output_dir / filename

    def _ensure_unique_candidates(self, candidates: Iterable[ClipCandidate]) -> None:
        seen: set[CandidateCaptionIdentity] = set()
        for candidate in candidates:
            identity = candidate_caption_identity(candidate)
            if identity in seen:
                raise InvalidCaptionSourceError(
                    "Multiple candidates share the same source path and time window."
                )
            seen.add(identity)

    def _validate_candidate(
        self,
        candidate: ClipCandidate,
        timeline_media_path: Path,
    ) -> None:
        if candidate.source_video_path.resolve() != timeline_media_path.resolve():
            raise InvalidCaptionSourceError(
                "Candidate source media does not match the feature timeline."
            )
        if candidate.start_seconds < 0:
            raise InvalidCaptionTimingError("Candidate start time cannot be negative.")
        if candidate.end_seconds <= candidate.start_seconds:
            raise InvalidCaptionTimingError(
                "Candidate end time must be after its start time."
            )

    def _language(self, feature_timeline: FeatureTimeline) -> str | None:
        languages = [
            result.metadata.get("language")
            for result in feature_timeline.timeline.observer_results
            if result.observer == "whisper"
            and isinstance(result.metadata.get("language"), str)
        ]
        return languages[0] if languages else None

    def _score_key(self, score: ClipScore) -> tuple[float, float, float, str, str]:
        candidate = score.candidate
        return (
            -score.overall_score,
            candidate.start_seconds,
            candidate.end_seconds,
            candidate.reason,
            str(candidate.source_video_path),
        )

    def _validate_config(self) -> None:
        if not self._config.encoding.strip():
            raise InvalidCaptionSourceError("Caption encoding cannot be empty.")


def _speech_text(observation: Observation) -> str:
    if isinstance(observation.value, dict):
        value = observation.value.get("text", "")
        return str(value) if value is not None else ""
    return str(observation.value) if observation.value is not None else ""


def _speaker(observation: Observation) -> str | None:
    if isinstance(observation.value, dict) and observation.value.get("speaker") is not None:
        return str(observation.value["speaker"])
    if observation.metadata.get("speaker") is not None:
        return str(observation.metadata["speaker"])
    return None
