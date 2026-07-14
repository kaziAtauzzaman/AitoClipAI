"""Editorial-strength validation errors."""


class EditorialStrengthError(ValueError):
    """Raised when candidate evidence violates the v1 deterministic contract."""


class InsufficientEditorialEvidenceError(EditorialStrengthError):
    """Raised when retained observations cannot support a reliable diagnostic."""
