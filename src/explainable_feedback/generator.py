"""Deterministic feedback generation from completed prerecorded pipeline results."""

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
import logging
from pathlib import Path
from typing import Protocol

from core import ClipCandidate, Observation, RenderJob
from pipeline import PrerecordedPipelineResult, RenderedArtifactValidation

from explainable_feedback.contracts import (
    CandidateIdentity,
    ClipFeedback,
    ExplainableFeedbackReport,
    ObserverEvidence,
    RenderFeedback,
    ScoreContribution,
)
from explainable_feedback.errors import FeedbackAssociationError


class CandidateIdentityStrategy(Protocol):
    """Create a deterministic identity for a candidate."""

    def identify(self, candidate: ClipCandidate) -> CandidateIdentity:
        """Return the candidate identity."""


class ResolvedPathCandidateIdentity:
    """Identify candidates by resolved source path and microsecond boundaries."""

    def identify(self, candidate: ClipCandidate) -> CandidateIdentity:
        return CandidateIdentity(
            resolved_source_path=candidate.source_video_path.resolve(strict=False),
            start_microseconds=_microseconds(candidate.start_seconds),
            end_microseconds=_microseconds(candidate.end_seconds),
        )


@dataclass(frozen=True, slots=True)
class _RenderAssociation:
    job: RenderJob
    artifact: RenderedArtifactValidation


class ExplainableFeedbackGenerator:
    """Generate an immutable provenance view without changing pipeline outputs."""

    def __init__(
        self,
        identity_strategy: CandidateIdentityStrategy | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._identity = identity_strategy or ResolvedPathCandidateIdentity()
        self._logger = logger or logging.getLogger(__name__)

    def generate(
        self,
        result: PrerecordedPipelineResult,
        *,
        schema_version: str = "1.0",
    ) -> ExplainableFeedbackReport:
        """Return one feedback entry for every score in scorer order."""

        self._logger.info(
            "feedback_generation_started scores=%d renders=%d",
            len(result.scores),
            len(result.render_jobs),
        )
        score_ids = self._unique_score_identities(result)
        render_by_id = self._render_associations(result)
        unknown = set(render_by_id).difference(score_ids)
        if unknown:
            raise FeedbackAssociationError(
                "A render job does not match exactly one scored candidate."
            )

        observations = [
            observation
            for group in result.feature_timeline.timeline.groups
            for observation in group.observations
        ]
        clips: list[ClipFeedback] = []
        for score in result.scores:
            identity = self._identity.identify(score.candidate)
            association = render_by_id.get(identity)
            evidence = self._evidence(score.candidate, observations)
            clips.append(
                ClipFeedback(
                    identity=identity,
                    selection_status=_selection_status(
                        bool(score.passed_threshold), association is not None
                    ),
                    candidate_reason=score.candidate.reason,
                    candidate_confidence=score.candidate.metadata.get("confidence"),
                    source_signals=list(score.candidate.source_signals),
                    candidate_signal_contributions=list(
                        score.candidate.metadata.get("signal_contributions", [])
                    ),
                    overall_score=score.overall_score,
                    passed_threshold=score.passed_threshold,
                    score_contributions=[
                        ScoreContribution(name, value)
                        for name, value in score.score_components.items()
                    ],
                    scorer_rationale=score.rationale,
                    supporting_evidence=evidence,
                    render=(
                        self._render_feedback(association)
                        if association is not None
                        else None
                    ),
                )
            )
        report = ExplainableFeedbackReport(
            schema_version=schema_version,
            report_type="explainable_heuristic_provenance",
            source_path=result.feature_timeline.media_path.resolve(strict=False),
            timeline_path=result.feature_timeline.timeline_path,
            scored_candidate_count=len(clips),
            rendered_clip_count=len(render_by_id),
            clips=clips,
        )
        self._logger.info(
            "feedback_generation_completed entries=%d rendered=%d",
            len(clips),
            len(render_by_id),
        )
        return report

    def _unique_score_identities(
        self, result: PrerecordedPipelineResult
    ) -> set[CandidateIdentity]:
        identities = [
            self._identity.identify(score.candidate) for score in result.scores
        ]
        if len(set(identities)) != len(identities):
            raise FeedbackAssociationError("Duplicate scored candidate identity.")
        return set(identities)

    def _render_associations(
        self, result: PrerecordedPipelineResult
    ) -> dict[CandidateIdentity, _RenderAssociation]:
        artifacts_by_path: dict[Path, RenderedArtifactValidation] = {}
        for artifact in result.validation_report.rendered_artifacts:
            path = artifact.path.resolve(strict=False)
            if path in artifacts_by_path:
                raise FeedbackAssociationError(
                    "Duplicate artifact validation output path."
                )
            artifacts_by_path[path] = artifact

        associations: dict[CandidateIdentity, _RenderAssociation] = {}
        used_artifacts: set[Path] = set()
        for job in result.render_jobs:
            identity = self._identity.identify(job.candidate)
            if identity in associations:
                raise FeedbackAssociationError(
                    "Duplicate render-job candidate identity."
                )
            output_path = job.output_path.resolve(strict=False)
            artifact = artifacts_by_path.get(output_path)
            if artifact is None:
                raise FeedbackAssociationError(
                    f"Render job has no artifact validation: {job.output_path}"
                )
            associations[identity] = _RenderAssociation(job, artifact)
            used_artifacts.add(output_path)
        if used_artifacts != set(artifacts_by_path):
            raise FeedbackAssociationError(
                "Artifact validation does not match exactly one render job."
            )
        return associations

    def _evidence(
        self,
        candidate: ClipCandidate,
        observations: list[Observation],
    ) -> list[ObserverEvidence]:
        direct = candidate.metadata.get("contributing_observations", [])
        if not isinstance(direct, list) or any(
            not isinstance(item, Observation) for item in direct
        ):
            raise FeedbackAssociationError(
                "Candidate contributing observations must be Observation objects."
            )
        return [
            ObserverEvidence(
                timestamp_seconds=item.timestamp_seconds,
                duration_seconds=item.duration_seconds,
                observer=item.observer,
                type=item.type,
                value=item.value,
                confidence=item.confidence,
                metadata=dict(item.metadata),
                direct_candidate_contributor=item in direct,
            )
            for item in observations
            if _overlaps(item, candidate)
        ]

    @staticmethod
    def _render_feedback(association: _RenderAssociation) -> RenderFeedback:
        job = association.job
        artifact = association.artifact
        return RenderFeedback(
            output_path=artifact.path,
            rank=job.metadata.get("rank"),
            duration_seconds=artifact.duration_seconds,
            size_bytes=artifact.size_bytes,
            video_codec=artifact.video_stream.codec_name,
            audio_codec=artifact.audio_stream.codec_name,
            checks=dict(artifact.checks),
        )


def _microseconds(seconds: float) -> int:
    return int(
        (Decimal(str(seconds)) * Decimal("1000000")).quantize(
            Decimal("1"), rounding=ROUND_HALF_UP
        )
    )


def _overlaps(observation: Observation, candidate: ClipCandidate) -> bool:
    observation_end = observation.timestamp_seconds + (
        observation.duration_seconds or 0.0
    )
    return (
        observation.timestamp_seconds <= candidate.end_seconds
        and observation_end >= candidate.start_seconds
    )


def _selection_status(passed: bool, rendered: bool) -> str:
    if rendered:
        return "rendered"
    if passed:
        return "passed_not_rendered"
    return "below_threshold"
