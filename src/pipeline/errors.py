"""Analysis pipeline exceptions."""


class PipelineError(Exception):
    """Raised when pipeline input resolution or persistence cannot continue."""


class PipelineValidationError(PipelineError):
    """Base error for strict prerecorded-pipeline validation failures."""


class RequiredObserverError(PipelineValidationError):
    """Raised when a required observer is missing or reports a failure."""


class NoCandidatesError(PipelineValidationError):
    """Raised when feature analysis produces no candidate windows."""


class NoPassingCandidatesError(PipelineValidationError):
    """Raised when no scored candidate passes the configured threshold."""


class MediaProbeError(PipelineValidationError):
    """Raised when FFprobe is unavailable or cannot inspect an artifact."""


class ArtifactValidationError(PipelineValidationError):
    """Raised when rendered media fails the required playback checks."""
