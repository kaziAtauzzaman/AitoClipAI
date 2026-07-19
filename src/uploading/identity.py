"""Stable identities and request fingerprints for uploader transactions."""

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from core import UploadJob
from uploading.errors import PermanentUploadError


def normalize_destination(value: str) -> str:
    """Return the canonical platform name used by identities and adapters."""

    if not isinstance(value, str) or not value.strip():
        raise PermanentUploadError("Upload destination must be a non-empty string.")
    return value.strip().lower()


def stable_upload_identity(job: UploadJob) -> str:
    """Derive identity exclusively from rendered ownership and platform."""

    rendered_identity = _rendered_identity(job)
    destination = normalize_destination(job.destination)
    digest = _digest(
        {
            "contract": "upload-identity-v1",
            "destination": destination,
            "rendered_clip_identity": rendered_identity,
        }
    )
    return f"{destination}:sha256:{digest}"


def upload_request_fingerprint(job: UploadJob) -> str:
    """Fingerprint mutable request content to reject identity reuse conflicts."""

    return _digest(
        {
            "contract": "upload-request-v1",
            "destination": normalize_destination(job.destination),
            "rendered_clip_identity": _rendered_identity(job),
            "rendered_clip_path": str(job.rendered_clip_path.resolve(strict=False)),
            "title": job.title,
            "description": job.description,
            "tags": job.tags,
            "scheduled_time": job.scheduled_time,
            "visibility": job.visibility,
            "metadata": job.metadata,
        }
    )


def _rendered_identity(job: UploadJob) -> str:
    value = job.rendered_clip_identity
    if not isinstance(value, str) or not value.strip():
        raise PermanentUploadError(
            "Upload jobs require a stable rendered clip identity."
        )
    return value.strip()


def _digest(value: object) -> str:
    try:
        encoded = json.dumps(
            _canonical(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PermanentUploadError(
            "Upload identity metadata must be JSON-compatible."
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Upload metadata mappings require string keys.")
        return {key: _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Upload metadata floats must be finite.")
        return 0.0 if value == 0 else value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"Unsupported upload metadata type: {type(value).__name__}.")
