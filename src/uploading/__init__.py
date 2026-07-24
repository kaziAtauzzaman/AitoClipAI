"""Persistent, platform-neutral clip uploading."""

from facebook_auth_contracts import (
    FacebookCredentialDiagnostic,
    FacebookCredentialResolver,
    FacebookCredentialState,
    FacebookCredentialStore,
)
from uploading.config import YouTubeUploadConfig
from uploading.contracts import (
    UploadAdapter,
    UploadLedgerRecord,
    UploadLedgerState,
    UploadPlan,
)
from uploading.errors import (
    FacebookAuthenticationRequired,
    PermanentUploadError,
    RetryableUploadError,
    UploadError,
    UploadIdentityConflictError,
    UploadLedgerCorruptionError,
    UploadLedgerError,
)
from uploading.facebook import (
    FacebookClient,
    FacebookClientError,
    FacebookGraphClient,
    FacebookRemoteVideo,
    FacebookUploadAdapter,
)
from uploading.facebook_config import FacebookUploadConfig, FacebookUploadSettings
from uploading.facebook_credentials import (
    FacebookCredentialStoreError,
    FacebookCredentialValidation,
    FacebookCredentialValidator,
    FacebookGraphCredentialValidator,
    ValidatingFacebookCredentialResolver,
    WindowsFacebookCredentialStore,
    create_facebook_credential_resolver,
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
    "FacebookClient",
    "FacebookClientError",
    "FacebookAuthenticationRequired",
    "FacebookCredentialStoreError",
    "FacebookCredentialDiagnostic",
    "FacebookCredentialResolver",
    "FacebookCredentialState",
    "FacebookCredentialStore",
    "FacebookCredentialValidation",
    "FacebookCredentialValidator",
    "FacebookGraphClient",
    "FacebookGraphCredentialValidator",
    "FacebookRemoteVideo",
    "FacebookUploadAdapter",
    "FacebookUploadConfig",
    "FacebookUploadSettings",
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
    "ValidatingFacebookCredentialResolver",
    "WindowsFacebookCredentialStore",
    "YouTubeClient",
    "YouTubeClientError",
    "YouTubeRemoteVideo",
    "YouTubeUploadAdapter",
    "YouTubeUploadConfig",
    "stable_upload_identity",
    "create_facebook_credential_resolver",
    "upload_request_fingerprint",
]
