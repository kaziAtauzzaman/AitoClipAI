"""Typed uploader failures with explicit retry classification."""


class UploadError(Exception):
    """Base class for expected uploader failures."""

    retryable = False


class RetryableUploadError(UploadError):
    """A transient failure for which retrying the same identity is safe."""

    retryable = True


class PermanentUploadError(UploadError):
    """A configuration or request failure that requires user correction."""


class UploadLedgerError(RetryableUploadError):
    """A local persistence or locking failure."""


class UploadLedgerCorruptionError(PermanentUploadError):
    """A malformed ledger that must be repaired before uploads resume."""


class UploadIdentityConflictError(PermanentUploadError):
    """One stable identity was reused with different request content."""
