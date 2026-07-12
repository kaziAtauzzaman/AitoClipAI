"""Deterministic explainable heuristic and provenance feedback."""

from explainable_feedback.config import ExplainableFeedbackConfig
from explainable_feedback.contracts import (
    CandidateIdentity,
    ClipFeedback,
    ExplainableFeedbackReport,
    ObserverEvidence,
    RenderFeedback,
    ScoreContribution,
)
from explainable_feedback.errors import (
    ExplainableFeedbackError,
    FeedbackAssociationError,
    FeedbackPersistenceError,
)
from explainable_feedback.generator import (
    CandidateIdentityStrategy,
    ExplainableFeedbackGenerator,
    ResolvedPathCandidateIdentity,
)
from explainable_feedback.persistence import (
    FeedbackReportWriter,
    JsonExplainableFeedbackWriter,
)
from explainable_feedback.service import ExplainableFeedbackService

__all__ = [
    "CandidateIdentity",
    "CandidateIdentityStrategy",
    "ClipFeedback",
    "ExplainableFeedbackConfig",
    "ExplainableFeedbackError",
    "ExplainableFeedbackGenerator",
    "ExplainableFeedbackReport",
    "ExplainableFeedbackService",
    "FeedbackAssociationError",
    "FeedbackPersistenceError",
    "FeedbackReportWriter",
    "JsonExplainableFeedbackWriter",
    "ObserverEvidence",
    "RenderFeedback",
    "ResolvedPathCandidateIdentity",
    "ScoreContribution",
]
