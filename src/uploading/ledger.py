"""Atomic local upload ledger with a process-safe transaction lock."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
from typing import Iterator

from core import UploadResult, UploadStatus
from uploading.contracts import UploadLedgerRecord, UploadLedgerState
from uploading.errors import UploadLedgerCorruptionError, UploadLedgerError


_SCHEMA_VERSION = 1


class JsonUploadLedger:
    """Persist upload ownership and completion under one local JSON file."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock_path = self.path.with_name(f"{self.path.name}.lock")

    @contextmanager
    def locked(self) -> Iterator[None]:
        """Hold the ledger lock across remote recovery and submission."""

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            stream = self._lock_path.open("a+b")
        except OSError as exc:
            raise UploadLedgerError(
                f"Could not open upload ledger lock: {exc}"
            ) from exc
        acquired = False
        try:
            _lock_stream(stream)
            acquired = True
            yield
        finally:
            try:
                if acquired:
                    _unlock_stream(stream)
            finally:
                stream.close()

    def get(self, upload_identity: str) -> UploadLedgerRecord | None:
        payload = self._read()
        value = payload["uploads"].get(upload_identity)
        if value is None:
            return None
        try:
            return _record_from_dict(upload_identity, value)
        except (KeyError, TypeError, ValueError) as exc:
            raise UploadLedgerCorruptionError(
                f"Upload ledger record {upload_identity!r} is malformed."
            ) from exc

    def put_pending(self, record: UploadLedgerRecord) -> None:
        if record.state is not UploadLedgerState.PENDING:
            raise ValueError("Pending ledger writes require pending state.")
        self._put(record)

    def put_completed(
        self,
        record: UploadLedgerRecord,
        result: UploadResult,
    ) -> None:
        self._put(
            UploadLedgerRecord(
                record.upload_identity,
                record.request_fingerprint,
                record.destination,
                record.rendered_clip_identity,
                UploadLedgerState.COMPLETED,
                result=result,
            )
        )

    def put_failure(
        self,
        record: UploadLedgerRecord,
        error: Exception,
        *,
        retryable: bool,
    ) -> None:
        self._put(
            UploadLedgerRecord(
                record.upload_identity,
                record.request_fingerprint,
                record.destination,
                record.rendered_clip_identity,
                (
                    UploadLedgerState.PENDING
                    if retryable
                    else UploadLedgerState.PERMANENT_FAILURE
                ),
                failure_type=type(error).__name__,
                failure_message=str(error),
                retryable=retryable,
            )
        )

    def _put(self, record: UploadLedgerRecord) -> None:
        payload = self._read()
        payload["uploads"][record.upload_identity] = _record_to_dict(record)
        self._write(payload)

    def _read(self) -> dict[str, object]:
        if not self.path.exists():
            return {"schema_version": _SCHEMA_VERSION, "uploads": {}}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise UploadLedgerCorruptionError(
                "Upload ledger is not valid JSON."
            ) from exc
        except OSError as exc:
            raise UploadLedgerError(f"Could not read upload ledger: {exc}") from exc
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != _SCHEMA_VERSION
            or not isinstance(payload.get("uploads"), dict)
        ):
            raise UploadLedgerCorruptionError(
                "Upload ledger has an unsupported or malformed schema."
            )
        return payload

    def _write(self, payload: dict[str, object]) -> None:
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        except (OSError, TypeError, ValueError) as exc:
            raise UploadLedgerError(f"Could not persist upload ledger: {exc}") from exc


def _record_to_dict(record: UploadLedgerRecord) -> dict[str, object]:
    return {
        "request_fingerprint": record.request_fingerprint,
        "destination": record.destination,
        "rendered_clip_identity": record.rendered_clip_identity,
        "state": record.state.value,
        "result": _result_to_dict(record.result) if record.result is not None else None,
        "failure_type": record.failure_type,
        "failure_message": record.failure_message,
        "retryable": record.retryable,
    }


def _record_from_dict(
    upload_identity: str,
    value: object,
) -> UploadLedgerRecord:
    if not isinstance(value, dict):
        raise TypeError("Ledger record must be a mapping.")
    raw_result = value.get("result")
    return UploadLedgerRecord(
        upload_identity=upload_identity,
        request_fingerprint=str(value["request_fingerprint"]),
        destination=str(value["destination"]),
        rendered_clip_identity=str(value["rendered_clip_identity"]),
        state=UploadLedgerState(str(value["state"])),
        result=None if raw_result is None else _result_from_dict(raw_result),
        failure_type=_optional_string(value.get("failure_type")),
        failure_message=_optional_string(value.get("failure_message")),
        retryable=_optional_bool(value.get("retryable")),
    )


def _result_to_dict(result: UploadResult) -> dict[str, object]:
    return {
        "upload_identity": result.upload_identity,
        "rendered_clip_identity": result.rendered_clip_identity,
        "rendered_clip_path": str(result.rendered_clip_path),
        "destination": result.destination,
        "status": result.status.value,
        "remote_id": result.remote_id,
        "remote_url": result.remote_url,
        "recovered": result.recovered,
        "metadata": result.metadata,
    }


def _result_from_dict(value: object) -> UploadResult:
    if not isinstance(value, dict):
        raise TypeError("Upload result must be a mapping.")
    return UploadResult(
        upload_identity=str(value["upload_identity"]),
        rendered_clip_identity=str(value["rendered_clip_identity"]),
        rendered_clip_path=Path(str(value["rendered_clip_path"])),
        destination=str(value["destination"]),
        status=UploadStatus(str(value["status"])),
        remote_id=_optional_string(value.get("remote_id")),
        remote_url=_optional_string(value.get("remote_url")),
        recovered=bool(value.get("recovered", False)),
        metadata=dict(value.get("metadata", {})),
    )


def _optional_string(value: object) -> str | None:
    return None if value is None else str(value)


def _optional_bool(value: object) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    raise TypeError("Ledger retryability must be boolean or null.")


def _lock_stream(stream) -> None:
    try:
        if os.name == "nt":
            import msvcrt

            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        raise UploadLedgerError("Another upload transaction owns the ledger.") from exc


def _unlock_stream(stream) -> None:
    try:
        if os.name == "nt":
            import msvcrt

            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    except OSError as exc:
        raise UploadLedgerError("Could not release the upload ledger lock.") from exc
