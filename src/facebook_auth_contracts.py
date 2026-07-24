"""Small platform-neutral contracts for Facebook credential ownership."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol


class FacebookCredentialState(str, Enum):
    """Operator-visible state for one configured Facebook Page credential."""

    CONNECTED = "connected"
    CREDENTIAL_STORED = "credential_stored"
    NOT_CONFIGURED = "not_configured"
    REAUTHORIZATION_REQUIRED = "reauthorization_required"
    PERMISSION_ERROR = "permission_error"
    WRONG_PAGE = "wrong_page"
    UNAVAILABLE = "unavailable"

    @property
    def label(self) -> str:
        return {
            self.CONNECTED: "Facebook Connected",
            self.CREDENTIAL_STORED: "Facebook Credential Stored",
            self.NOT_CONFIGURED: "Facebook Not Configured",
            self.REAUTHORIZATION_REQUIRED: (
                "Facebook Reauthorization Required"
            ),
            self.PERMISSION_ERROR: "Facebook Permission Error",
            self.WRONG_PAGE: "Facebook Wrong Page",
            self.UNAVAILABLE: "Facebook Unavailable",
        }[self]


class FacebookAuthenticationIssue(Exception):
    """Marker shared by safe UI handling and retryable upload failures."""

    state: FacebookCredentialState
    diagnostic: "FacebookCredentialDiagnostic | None"
    diagnostic_log_path: Path | None


@dataclass(frozen=True, slots=True)
class FacebookCredentialDiagnostic:
    """Strictly sanitized evidence from one credential configuration attempt."""

    stage: str
    windows_error_code: int | None = None
    http_status: int | None = None
    graph_error_code: int | None = None
    graph_error_type: str | None = None
    graph_error_message: str | None = None
    validation_succeeded: bool = False
    cred_write_attempted: bool = False
    cred_write_succeeded: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "windows_error_code": self.windows_error_code,
            "http_status": self.http_status,
            "graph_error_code": self.graph_error_code,
            "graph_error_type": self.graph_error_type,
            "graph_error_message": self.graph_error_message,
            "validation_succeeded": self.validation_succeeded,
            "cred_write_attempted": self.cred_write_attempted,
            "cred_write_succeeded": self.cred_write_succeeded,
        }


class FacebookCredentialStore(Protocol):
    """Secret storage boundary for one Page access token."""

    def read(self, page_id: str) -> str | None:
        """Return the token for ``page_id`` without exposing storage details."""

    def replace(self, page_id: str, token: str) -> None:
        """Atomically replace the token for ``page_id``."""


class FacebookCredentialResolver(Protocol):
    """Obtain and validate a Page token without changing uploader contracts."""

    def local_state(self) -> FacebookCredentialState:
        """Return a network-free state based on local credential presence."""

    def resolve(self) -> str:
        """Return a currently valid token for the configured Page."""

    def replace(self, token: str) -> None:
        """Validate ``token`` and save it only after successful validation."""
