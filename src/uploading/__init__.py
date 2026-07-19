"""Persistent, platform-neutral clip uploading."""

from uploading.config import YouTubeUploadConfig
from uploading.contracts import (
    UploadAdapter,
    UploadLedgerRecord,
    UploadLedgerState,
    UploadPlan,
)
from uploading.errors import (
    PermanentUploadError,
    RetryableUploadError,
    UploadError,
    UploadIdentityConflictError,
    UploadLedgerCorruptionError,
    UploadLedgerError,
)
from uploading.identity import stable_upload_identity, upload_request_fingerprint
from uploading.ledger import JsonUploadLedger
from uploading.service import UploadService
from uploading.youtube import (
    GoogleYouTubeClient,
    YouTubeClient,
    YouTubeClientError,
    YouTubeRemoteVideo,
    YouTubeUploadAdapter,
)

__all__ = [
    "GoogleYouTubeClient",
    "JsonUploadLedger",
    "PermanentUploadError",
    "RetryableUploadError",
    "UploadAdapter",
    "UploadError",
    "UploadIdentityConflictError",
    "UploadLedgerCorruptionError",
    "UploadLedgerError",
    "UploadLedgerRecord",
    "UploadLedgerState",
    "UploadPlan",
    "UploadService",
    "YouTubeClient",
    "YouTubeClientError",
    "YouTubeRemoteVideo",
    "YouTubeUploadAdapter",
    "YouTubeUploadConfig",
    "stable_upload_identity",
    "upload_request_fingerprint",
]
