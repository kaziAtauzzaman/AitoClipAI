"""Platform-neutral upload orchestration and exactly-once local ownership."""

from pathlib import Path
from typing import Iterable

from core import UploadJob, UploadResult, UploadStatus
from uploading.contracts import (
    UploadAdapter,
    UploadLedgerRecord,
    UploadLedgerState,
    UploadPlan,
)
from uploading.errors import (
    PermanentUploadError,
    UploadError,
    UploadIdentityConflictError,
    UploadLedgerCorruptionError,
)
from uploading.identity import (
    normalize_destination,
    stable_upload_identity,
    upload_request_fingerprint,
)
from uploading.ledger import JsonUploadLedger


class UploadService:
    """Plan or execute uploads while the ledger owns duplicate prevention."""

    def __init__(
        self,
        ledger: JsonUploadLedger,
        adapters: Iterable[UploadAdapter],
    ) -> None:
        self._ledger = ledger
        self._adapters: dict[str, UploadAdapter] = {}
        for adapter in adapters:
            destination = normalize_destination(adapter.destination)
            if destination in self._adapters:
                raise ValueError(f"Duplicate upload adapter: {destination}.")
            self._adapters[destination] = adapter

    def execute(self, job: UploadJob, *, dry_run: bool = False) -> UploadResult:
        """Plan without side effects or execute one ledger-owned upload."""

        self._validate_job(job)
        destination = normalize_destination(job.destination)
        adapter = self._adapters.get(destination)
        if adapter is None:
            raise PermanentUploadError(
                f"No upload adapter is configured for {destination!r}."
            )
        upload_identity = stable_upload_identity(job)
        request_fingerprint = upload_request_fingerprint(job)
        plan = adapter.plan(job, upload_identity)
        self._validate_plan(plan, job, upload_identity)
        if dry_run:
            return UploadResult(
                upload_identity=upload_identity,
                rendered_clip_identity=job.rendered_clip_identity,
                rendered_clip_path=job.rendered_clip_path,
                destination=destination,
                status=UploadStatus.DRY_RUN,
                metadata={"plan": _plan_value(plan)},
            )

        with self._ledger.locked():
            record = self._ledger.get(upload_identity)
            if record is not None:
                self._validate_record(record, request_fingerprint)
                if record.state is UploadLedgerState.COMPLETED:
                    if record.result is None:
                        raise UploadLedgerCorruptionError(
                            "Completed upload ledger record has no result."
                        )
                    return record.result
                if record.state is UploadLedgerState.PERMANENT_FAILURE:
                    raise PermanentUploadError(
                        record.failure_message
                        or "A previous permanent upload failure must be corrected."
                    )
            else:
                record = UploadLedgerRecord(
                    upload_identity=upload_identity,
                    request_fingerprint=request_fingerprint,
                    destination=destination,
                    rendered_clip_identity=job.rendered_clip_identity,
                    state=UploadLedgerState.PENDING,
                )
                self._ledger.put_pending(record)

            try:
                result = adapter.recover(job, upload_identity)
                if result is None:
                    result = adapter.upload(job, upload_identity)
                self._validate_result(result, job, upload_identity)
            except UploadError as exc:
                self._ledger.put_failure(record, exc, retryable=exc.retryable)
                raise
            self._ledger.put_completed(record, result)
            return result

    @staticmethod
    def _validate_job(job: UploadJob) -> None:
        path = Path(job.rendered_clip_path)
        if not path.is_file():
            raise PermanentUploadError("Rendered video path must be an existing file.")
        try:
            if path.stat().st_size <= 0:
                raise PermanentUploadError("Rendered video must not be empty.")
        except OSError as exc:
            raise PermanentUploadError(
                f"Rendered video could not be inspected: {exc}"
            ) from exc
        if not isinstance(job.title, str) or not job.title.strip():
            raise PermanentUploadError("Upload title must be a non-empty string.")
        if job.scheduled_time is not None:
            raise PermanentUploadError(
                "Scheduled publishing is outside the first uploader milestone."
            )

    @staticmethod
    def _validate_plan(
        plan: UploadPlan,
        job: UploadJob,
        upload_identity: str,
    ) -> None:
        if (
            plan.upload_identity != upload_identity
            or normalize_destination(plan.destination)
            != normalize_destination(job.destination)
            or plan.rendered_clip_identity != job.rendered_clip_identity
            or Path(plan.rendered_clip_path) != Path(job.rendered_clip_path)
        ):
            raise PermanentUploadError("Upload adapter returned a mismatched plan.")

    @staticmethod
    def _validate_result(
        result: UploadResult,
        job: UploadJob,
        upload_identity: str,
    ) -> None:
        if (
            not isinstance(result, UploadResult)
            or result.upload_identity != upload_identity
            or result.rendered_clip_identity != job.rendered_clip_identity
            or Path(result.rendered_clip_path) != Path(job.rendered_clip_path)
            or normalize_destination(result.destination)
            != normalize_destination(job.destination)
            or result.status is not UploadStatus.COMPLETED
            or not result.remote_id
        ):
            raise PermanentUploadError("Upload adapter returned a mismatched result.")

    @staticmethod
    def _validate_record(
        record: UploadLedgerRecord,
        request_fingerprint: str,
    ) -> None:
        if record.request_fingerprint != request_fingerprint:
            raise UploadIdentityConflictError(
                "Upload identity was reused with different request content."
            )


def _plan_value(plan: UploadPlan) -> dict[str, object]:
    return {
        "upload_identity": plan.upload_identity,
        "destination": plan.destination,
        "rendered_clip_identity": plan.rendered_clip_identity,
        "rendered_clip_path": str(plan.rendered_clip_path),
        "title": plan.title,
        "description": plan.description,
        "privacy_status": plan.privacy_status,
        "tags": list(plan.tags),
        "metadata": plan.metadata,
    }
