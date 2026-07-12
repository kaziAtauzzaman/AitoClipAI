"""Typed errors for deterministic explainable feedback generation."""


class ExplainableFeedbackError(Exception):
    """Base error for feedback generation and persistence."""


class FeedbackAssociationError(ExplainableFeedbackError):
    """Raised when pipeline artifacts cannot be associated unambiguously."""


class FeedbackPersistenceError(ExplainableFeedbackError):
    """Raised when a feedback report cannot be persisted."""
