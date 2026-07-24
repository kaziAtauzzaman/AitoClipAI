"""Typed uploader failures with explicit retry classification."""

from facebook_auth_contracts import (
    FacebookAuthenticationIssue,
    FacebookCredentialDiagnostic,
    FacebookCredentialState,
)
from pathlib import Path


class UploadError(Exception):
    """Base class for expected uploader failures."""

    retryable = False


class RetryableUploadError(UploadError):
    """A transient failure for which retrying the same identity is safe."""

    retryable = True


class FacebookAuthenticationRequired(
    RetryableUploadError,
    FacebookAuthenticationIssue,
):
    """Facebook credentials need operator action before retrying."""

    def __init__(
        self,
        state: FacebookCredentialState,
        message: str | None = None,
        *,
        diagnostic: FacebookCredentialDiagnostic | None = None,
        diagnostic_log_path: Path | None = None,
    ) -> None:
        self.state = state
        self.diagnostic = diagnostic
        self.diagnostic_log_path = diagnostic_log_path
        super().__init__(message or state.label)


class PermanentUploadError(UploadError):
    """A configuration or request failure that requires user correction."""


class UploadLedgerError(RetryableUploadError):
    """A local persistence or locking failure."""


class UploadLedgerCorruptionError(PermanentUploadError):
    """A malformed ledger that must be repaired before uploads resume."""


class UploadIdentityConflictError(PermanentUploadError):
    """One stable identity was reused with different request content."""
