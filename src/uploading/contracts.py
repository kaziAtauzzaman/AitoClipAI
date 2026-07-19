"""Platform-neutral uploader interfaces and persistence contracts."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from core import UploadJob, UploadResult


class UploadLedgerState(str, Enum):
    """Durable lifecycle states relevant to duplicate prevention."""

    PENDING = "pending"
    COMPLETED = "completed"
    PERMANENT_FAILURE = "permanent_failure"


@dataclass(frozen=True, slots=True)
class UploadPlan:
    """Credential-free description of one platform upload request."""

    upload_identity: str
    destination: str
    rendered_clip_identity: str
    rendered_clip_path: Path
    title: str
    description: str
    privacy_status: str
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class UploadLedgerRecord:
    """One durable upload identity and its latest authoritative state."""

    upload_identity: str
    request_fingerprint: str
    destination: str
    rendered_clip_identity: str
    state: UploadLedgerState
    result: UploadResult | None = None
    failure_type: str | None = None
    failure_message: str | None = None
    retryable: bool | None = None


class UploadAdapter(Protocol):
    """Platform boundary used by the neutral upload service."""

    destination: str

    def plan(self, job: UploadJob, upload_identity: str) -> UploadPlan:
        """Return a credential-free, deterministic platform plan."""

    def recover(
        self,
        job: UploadJob,
        upload_identity: str,
    ) -> UploadResult | None:
        """Return a previously completed remote upload when one exists."""

    def upload(self, job: UploadJob, upload_identity: str) -> UploadResult:
        """Submit one upload and return its completed result."""
