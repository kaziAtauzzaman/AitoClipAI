"""Fail-closed post-upload cleanup for operator-run rendered clips."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import json
import os
from pathlib import Path
import threading
from typing import Callable, Mapping
from uuid import uuid4

from core import UploadStatus
from operator_pipeline import OPERATOR_RUNS_DIRECTORY, RenderedClipOutput
from operator_upload import (
    SUPPORTED_DESTINATIONS,
    UploadQueueSummary,
)

__all__ = (
    "CLEANUP_POLICY_OPTIONS",
    "CleanupInProgressError",
    "CleanupPolicy",
    "CleanupRequest",
    "CleanupResult",
    "CleanupStatus",
    "OperatorCleanupController",
    "PostUploadCleanupCoordinator",
    "SanitizedCleanupError",
    "cleanup_policy_from_label",
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OPERATOR_RUNS_ROOT = _PROJECT_ROOT / OPERATOR_RUNS_DIRECTORY
_RECOVERY_DIAGNOSTIC_NAME = "post-upload-cleanup-recovery.json"


class CleanupPolicy(str, Enum):
    """Operator-selected lifetime policy for rendered clips."""

    KEEP_RENDERED_CLIPS = "keep_rendered_clips"
    DELETE_AFTER_SUCCESSFUL_UPLOADS = "delete_after_successful_uploads"

    @property
    def label(self) -> str:
        if self is CleanupPolicy.KEEP_RENDERED_CLIPS:
            return "Keep rendered clips"
        return "Delete rendered clips after all selected uploads complete"


CLEANUP_POLICY_OPTIONS = tuple(policy.label for policy in CleanupPolicy)


class CleanupStatus(str, Enum):
    """Sanitized terminal state written into the production report."""

    RETAINED_NO_DESTINATIONS = "retained_no_destinations"
    RETAINED_BY_POLICY = "retained_by_policy"
    RETAINED_INCOMPLETE_UPLOADS = "retained_incomplete_uploads"
    REFUSED_UNSAFE_PATH = "refused_unsafe_path"
    REFUSED_REPORT_UNAVAILABLE = "refused_report_unavailable"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ALREADY_CLEAN = "already_clean"
    PARTIAL_FAILURE = "partial_failure"


@dataclass(frozen=True, slots=True)
class SanitizedCleanupError:
    """Non-sensitive reason why cleanup retained one or more files."""

    stage: str
    code: str
    error_type: str | None = None
    rendered_clip_identity: str | None = None
    destination: str | None = None

    def to_dict(self) -> dict[str, str]:
        payload = {
            "stage": self.stage,
            "code": self.code,
        }
        if self.error_type is not None:
            payload["error_type"] = self.error_type
        if self.rendered_clip_identity is not None:
            payload["rendered_clip_identity"] = self.rendered_clip_identity
        if self.destination is not None:
            payload["destination"] = self.destination
        return payload


@dataclass(frozen=True, slots=True)
class CleanupRequest:
    """Complete evidence required for one post-upload cleanup decision."""

    run_directory: Path
    clips_directory: Path
    report_path: Path | None
    expected_rendered_clip_count: int
    rendered_clips: tuple[RenderedClipOutput, ...]
    selected_destinations: tuple[str, ...]
    upload_summary: UploadQueueSummary | None
    policy: CleanupPolicy


@dataclass(frozen=True, slots=True)
class CleanupResult:
    """Immutable cleanup outcome safe for UI and report serialization."""

    cleanup_policy: str
    eligible_clip_count: int
    deleted_clip_count: int
    retained_clip_count: int
    bytes_deleted: int
    cleanup_status: str
    sanitized_cleanup_errors: tuple[SanitizedCleanupError, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "cleanup_policy": self.cleanup_policy,
            "eligible_clip_count": self.eligible_clip_count,
            "deleted_clip_count": self.deleted_clip_count,
            "retained_clip_count": self.retained_clip_count,
            "bytes_deleted": self.bytes_deleted,
            "cleanup_status": self.cleanup_status,
            "sanitized_cleanup_errors": [
                issue.to_dict() for issue in self.sanitized_cleanup_errors
            ],
        }


class CleanupInProgressError(RuntimeError):
    """Raised when a second post-upload cleanup is started concurrently."""


@dataclass(frozen=True, slots=True)
class _CleanupContext:
    operator_runs_root: Path
    run_directory: Path
    clips_directory: Path
    reports_directory: Path
    report_path: Path | None


class PostUploadCleanupCoordinator:
    """Verify a complete upload matrix before deleting explicit clip paths."""

    def __init__(
        self,
        *,
        operator_runs_root: Path | None = None,
        unlink_file: Callable[[Path], None] | None = None,
    ) -> None:
        self._operator_runs_root = Path(
            operator_runs_root or _DEFAULT_OPERATOR_RUNS_ROOT
        )
        self._unlink_file = unlink_file or _unlink_file

    def run(self, request: CleanupRequest) -> CleanupResult:
        """Evaluate, safely apply, and record one cleanup request."""

        destinations, destination_issue = _normalize_destinations(
            request.selected_destinations
        )
        retained_before = _existing_clip_count(request.rendered_clips)
        context, boundary_issues = _validate_cleanup_context(
            request,
            self._operator_runs_root,
        )
        if context is None:
            return _retained_result(
                request,
                CleanupStatus.REFUSED_UNSAFE_PATH,
                retained_before,
                boundary_issues,
            )
        report_payload, report_issue = _load_report_payload(context)

        if not destinations and destination_issue is None:
            return _record_retained_result(
                request,
                CleanupStatus.RETAINED_NO_DESTINATIONS,
                retained_before,
                (),
                context,
                report_payload,
                report_issue,
            )

        if request.policy is CleanupPolicy.KEEP_RENDERED_CLIPS:
            return _record_retained_result(
                request,
                CleanupStatus.RETAINED_BY_POLICY,
                retained_before,
                (destination_issue,) if destination_issue else (),
                context,
                report_payload,
                report_issue,
            )

        if destination_issue is not None:
            return _record_retained_result(
                request,
                CleanupStatus.RETAINED_INCOMPLETE_UPLOADS,
                retained_before,
                (destination_issue,),
                context,
                report_payload,
                report_issue,
            )

        if report_issue is not None:
            return _retained_result(
                request,
                CleanupStatus.REFUSED_REPORT_UNAVAILABLE,
                retained_before,
                (report_issue,),
            )

        assert report_payload is not None
        assert context.report_path is not None
        path_issues, resolved_paths = _validate_clip_paths(request, context)
        if path_issues:
            return _record_retained_result(
                request,
                CleanupStatus.REFUSED_UNSAFE_PATH,
                retained_before,
                path_issues,
                context,
                report_payload,
                None,
            )

        upload_issues = _validate_upload_matrix(
            request,
            destinations,
            resolved_paths,
        )
        if upload_issues:
            return _record_retained_result(
                request,
                CleanupStatus.RETAINED_INCOMPLETE_UPLOADS,
                retained_before,
                upload_issues,
                context,
                report_payload,
                None,
            )

        in_progress_record = _in_progress_record(
            request,
            retained_before,
        )
        try:
            report_payload = _write_cleanup_record(
                context.report_path,
                report_payload,
                in_progress_record,
            )
        except BaseException as exc:
            issue = _cleanup_error(
                "record_report",
                "cleanup_in_progress_write_failed",
                exc=exc,
            )
            if not isinstance(exc, Exception):
                raise
            return _retained_result(
                request,
                CleanupStatus.REFUSED_REPORT_UNAVAILABLE,
                retained_before,
                (issue,),
            )

        deleted = 0
        bytes_deleted = 0
        delete_issues: list[SanitizedCleanupError] = []
        for clip in request.rendered_clips:
            path = Path(clip.path)
            try:
                if not path.exists():
                    continue
                if path.is_symlink():
                    raise _UnsafeCleanupPathError
                if path.resolve(strict=True) != resolved_paths[clip.identity]:
                    raise _UnsafeCleanupPathError
                if not path.is_file():
                    raise _UnsafeCleanupPathError
                size = path.stat().st_size
                self._unlink_file(path)
            except (OSError, RuntimeError, _UnsafeCleanupPathError) as exc:
                delete_issues.append(
                    _cleanup_error(
                        "delete",
                        (
                            "rendered_path_changed_after_verification"
                            if isinstance(exc, _UnsafeCleanupPathError)
                            else "clip_delete_failed"
                        ),
                        exc=exc,
                        rendered_clip_identity=clip.identity,
                    )
                )
                break
            deleted += 1
            bytes_deleted += size

        retained_after = _existing_clip_count(request.rendered_clips)
        if delete_issues:
            status = CleanupStatus.PARTIAL_FAILURE
        elif deleted == 0:
            status = CleanupStatus.ALREADY_CLEAN
        else:
            status = CleanupStatus.COMPLETED
        result = CleanupResult(
            cleanup_policy=request.policy.value,
            eligible_clip_count=len(request.rendered_clips),
            deleted_clip_count=deleted,
            retained_clip_count=retained_after,
            bytes_deleted=bytes_deleted,
            cleanup_status=status.value,
            sanitized_cleanup_errors=tuple(delete_issues),
        )
        try:
            _write_cleanup_record(
                context.report_path,
                report_payload,
                result.to_dict(),
            )
        except BaseException as exc:
            report_failure = _cleanup_error(
                "record_report",
                "production_report_write_failed",
                exc=exc,
            )
            failed_result = replace(
                result,
                cleanup_status=CleanupStatus.PARTIAL_FAILURE.value,
                sanitized_cleanup_errors=(
                    *result.sanitized_cleanup_errors,
                    report_failure,
                ),
            )
            diagnostic_issue = _write_recovery_diagnostic_safely(
                context,
                request,
                failed_result,
            )
            if diagnostic_issue is not None:
                failed_result = replace(
                    failed_result,
                    sanitized_cleanup_errors=(
                        *failed_result.sanitized_cleanup_errors,
                        diagnostic_issue,
                    ),
                )
            if not isinstance(exc, Exception):
                raise
            return failed_result
        return result


class OperatorCleanupController:
    """Run cleanup and report persistence without blocking Tkinter."""

    def __init__(
        self,
        coordinator: PostUploadCleanupCoordinator | None = None,
    ) -> None:
        self._coordinator = coordinator or PostUploadCleanupCoordinator()
        self._lock = threading.Lock()
        self._active_thread: threading.Thread | None = None

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._active_thread is not None

    def start(
        self,
        request: CleanupRequest,
        *,
        on_complete: Callable[[CleanupResult], None],
        on_failure: Callable[[str], None],
    ) -> threading.Thread:
        with self._lock:
            if self._active_thread is not None:
                raise CleanupInProgressError("Post-upload cleanup is already active.")
            thread = threading.Thread(
                target=self._execute,
                args=(request, on_complete, on_failure),
                name="aitoclip-operator-cleanup",
                daemon=False,
            )
            self._active_thread = thread
            try:
                thread.start()
            except BaseException:
                self._active_thread = None
                raise
            return thread

    def _execute(
        self,
        request: CleanupRequest,
        on_complete: Callable[[CleanupResult], None],
        on_failure: Callable[[str], None],
    ) -> None:
        try:
            on_complete(self._coordinator.run(request))
        except BaseException as exc:
            on_failure(type(exc).__name__)
            if not isinstance(exc, Exception):
                raise
        finally:
            with self._lock:
                if self._active_thread is threading.current_thread():
                    self._active_thread = None


def cleanup_policy_from_label(label: str) -> CleanupPolicy:
    """Resolve one fixed UI label without accepting ambiguous values."""

    for policy in CleanupPolicy:
        if label == policy.label:
            return policy
    raise ValueError("Unsupported cleanup policy.")


class _UnsafeCleanupPathError(RuntimeError):
    """Internal signal that a verified deletion path changed before unlink."""


def _normalize_destinations(
    destinations: tuple[str, ...],
) -> tuple[tuple[str, ...], SanitizedCleanupError | None]:
    selected = tuple(
        dict.fromkeys(str(destination).strip().lower() for destination in destinations)
    )
    unsupported = tuple(
        destination
        for destination in selected
        if destination not in SUPPORTED_DESTINATIONS
    )
    if unsupported:
        return (), _cleanup_error(
            "verify_uploads",
            "unsupported_destination",
            destination=unsupported[0],
        )
    return tuple(
        destination
        for destination in SUPPORTED_DESTINATIONS
        if destination in selected
    ), None


def _validate_cleanup_context(
    request: CleanupRequest,
    configured_operator_runs_root: Path,
) -> tuple[_CleanupContext | None, tuple[SanitizedCleanupError, ...]]:
    try:
        operator_runs_root = Path(configured_operator_runs_root).resolve(
            strict=True
        )
        run_directory = Path(request.run_directory).resolve(strict=True)
        clips_directory = Path(request.clips_directory).resolve(strict=True)
        expected_clips_directory = (run_directory / "clips").resolve(strict=True)
        reports_directory = (run_directory / "reports").resolve(strict=False)
        report_path = (
            None
            if request.report_path is None
            else Path(request.report_path).resolve(strict=False)
        )
    except (OSError, RuntimeError) as exc:
        return None, (
            _cleanup_error(
                "verify_paths",
                "run_or_clips_directory_unavailable",
                exc=exc,
            ),
        )
    if not operator_runs_root.is_dir():
        return None, (
            _cleanup_error(
                "verify_paths",
                "operator_runs_root_not_directory",
            ),
        )
    if (
        not run_directory.is_dir()
        or run_directory == operator_runs_root
        or not _is_relative_to(run_directory, operator_runs_root)
    ):
        return None, (
            _cleanup_error(
                "verify_paths",
                "run_directory_outside_operator_root",
            ),
        )
    if (
        not clips_directory.is_dir()
        or clips_directory != expected_clips_directory
        or not _is_relative_to(clips_directory, run_directory)
    ):
        return None, (
            _cleanup_error(
                "verify_paths",
                "clips_directory_outside_run",
            ),
        )
    if not _is_relative_to(reports_directory, run_directory):
        return None, (
            _cleanup_error(
                "verify_paths",
                "reports_directory_outside_run",
            ),
        )
    if report_path is not None and not _is_relative_to(
        report_path,
        reports_directory,
    ):
        return None, (
            _cleanup_error(
                "verify_paths",
                "production_report_outside_run",
            ),
        )
    return (
        _CleanupContext(
            operator_runs_root=operator_runs_root,
            run_directory=run_directory,
            clips_directory=clips_directory,
            reports_directory=reports_directory,
            report_path=report_path,
        ),
        (),
    )


def _validate_clip_paths(
    request: CleanupRequest,
    context: _CleanupContext,
) -> tuple[
    tuple[SanitizedCleanupError, ...],
    Mapping[str, Path],
]:
    issues: list[SanitizedCleanupError] = []
    resolved_paths: dict[str, Path] = {}
    resolved_path_owners: dict[Path, str] = {}
    run_directory = context.run_directory
    clips_directory = context.clips_directory

    for clip in request.rendered_clips:
        path = Path(clip.path)
        if clip.identity in resolved_paths:
            issues.append(
                _cleanup_error(
                    "verify_paths",
                    "duplicate_rendered_clip_identity",
                    rendered_clip_identity=clip.identity,
                )
            )
            continue
        try:
            if path.is_symlink():
                issues.append(
                    _cleanup_error(
                        "verify_paths",
                        "symbolic_link_refused",
                        rendered_clip_identity=clip.identity,
                    )
                )
                continue
            resolved = path.resolve(strict=False)
            exists = path.exists()
            if exists and not path.is_file():
                issues.append(
                    _cleanup_error(
                        "verify_paths",
                        "rendered_path_not_file",
                        rendered_clip_identity=clip.identity,
                    )
                )
                continue
        except (OSError, RuntimeError) as exc:
            issues.append(
                _cleanup_error(
                    "verify_paths",
                    "rendered_path_unavailable",
                    exc=exc,
                    rendered_clip_identity=clip.identity,
                )
            )
            continue
        if (
            not _is_relative_to(resolved, run_directory)
            or not _is_relative_to(resolved, clips_directory)
        ):
            issues.append(
                _cleanup_error(
                    "verify_paths",
                    "rendered_path_outside_clips_directory",
                    rendered_clip_identity=clip.identity,
                )
            )
            continue
        previous_owner = resolved_path_owners.get(resolved)
        if previous_owner is not None:
            issues.append(
                _cleanup_error(
                    "verify_paths",
                    "duplicate_rendered_clip_path",
                    rendered_clip_identity=clip.identity,
                )
            )
            continue
        resolved_paths[clip.identity] = resolved
        resolved_path_owners[resolved] = clip.identity
    return tuple(issues), resolved_paths


def _validate_upload_matrix(
    request: CleanupRequest,
    destinations: tuple[str, ...],
    resolved_paths: Mapping[str, Path],
) -> tuple[SanitizedCleanupError, ...]:
    summary = request.upload_summary
    if summary is None:
        return (
            _cleanup_error(
                "verify_uploads",
                "upload_summary_unavailable",
            ),
        )
    if request.expected_rendered_clip_count != len(request.rendered_clips):
        return (
            _cleanup_error(
                "verify_uploads",
                "rendered_clip_count_mismatch",
            ),
        )
    expected = {
        (clip.identity, destination): clip
        for clip in request.rendered_clips
        for destination in destinations
    }
    issues: list[SanitizedCleanupError] = []
    seen: set[tuple[str, str]] = set()
    if len(summary.attempts) != len(expected):
        issues.append(
            _cleanup_error(
                "verify_uploads",
                "upload_job_count_mismatch",
            )
        )
    for attempt in summary.attempts:
        key = (attempt.rendered_clip_identity, attempt.destination)
        clip = expected.get(key)
        if clip is None:
            issues.append(
                _cleanup_error(
                    "verify_uploads",
                    "unexpected_upload_job",
                    rendered_clip_identity=attempt.rendered_clip_identity,
                    destination=attempt.destination,
                )
            )
            continue
        if key in seen:
            issues.append(
                _cleanup_error(
                    "verify_uploads",
                    "duplicate_upload_job",
                    rendered_clip_identity=attempt.rendered_clip_identity,
                    destination=attempt.destination,
                )
            )
            continue
        seen.add(key)
        result = attempt.result
        if not attempt.completed or result is None:
            issues.append(
                _cleanup_error(
                    "verify_uploads",
                    "upload_incomplete",
                    error_type=attempt.error_type,
                    rendered_clip_identity=attempt.rendered_clip_identity,
                    destination=attempt.destination,
                )
            )
            continue
        try:
            result_path = Path(result.rendered_clip_path).resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            issues.append(
                _cleanup_error(
                    "verify_uploads",
                    "upload_result_path_unavailable",
                    exc=exc,
                    rendered_clip_identity=attempt.rendered_clip_identity,
                    destination=attempt.destination,
                )
            )
            continue
        if (
            result.status is not UploadStatus.COMPLETED
            or result.destination != attempt.destination
            or result.rendered_clip_identity != attempt.rendered_clip_identity
            or result_path != resolved_paths.get(clip.identity)
        ):
            issues.append(
                _cleanup_error(
                    "verify_uploads",
                    "upload_result_mismatch",
                    rendered_clip_identity=attempt.rendered_clip_identity,
                    destination=attempt.destination,
                )
            )
    for identity, destination in expected:
        if (identity, destination) not in seen:
            issues.append(
                _cleanup_error(
                    "verify_uploads",
                    "required_upload_job_missing",
                    rendered_clip_identity=identity,
                    destination=destination,
                )
            )
    return tuple(issues)


def _load_report_payload(
    context: _CleanupContext,
) -> tuple[dict[str, object] | None, SanitizedCleanupError | None]:
    if context.report_path is None:
        return None, _cleanup_error(
            "record_report",
            "production_report_unavailable",
        )
    try:
        if (
            not context.reports_directory.is_dir()
            or not context.report_path.is_file()
        ):
            raise ValueError("Production report is unavailable.")
        payload = json.loads(context.report_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Production report is not a JSON object.")
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        return None, _cleanup_error(
            "record_report",
            "production_report_unavailable",
            exc=exc,
        )
    return payload, None


def _record_retained_result(
    request: CleanupRequest,
    status: CleanupStatus,
    retained_count: int,
    issues: tuple[SanitizedCleanupError, ...],
    context: _CleanupContext,
    report_payload: dict[str, object] | None,
    report_issue: SanitizedCleanupError | None,
) -> CleanupResult:
    result = _retained_result(request, status, retained_count, issues)
    if report_payload is None or context.report_path is None:
        if report_issue is None:
            report_issue = _cleanup_error(
                "record_report",
                "production_report_unavailable",
            )
        return replace(
            result,
            sanitized_cleanup_errors=(
                *result.sanitized_cleanup_errors,
                report_issue,
            ),
        )
    try:
        _write_cleanup_record(
            context.report_path,
            report_payload,
            result.to_dict(),
        )
    except BaseException as exc:
        if not isinstance(exc, Exception):
            raise
        return replace(
            result,
            sanitized_cleanup_errors=(
                *result.sanitized_cleanup_errors,
                _cleanup_error(
                    "record_report",
                    "production_report_write_failed",
                    exc=exc,
                ),
            ),
        )
    return result


def _in_progress_record(
    request: CleanupRequest,
    retained_count: int,
) -> dict[str, object]:
    return {
        "cleanup_policy": request.policy.value,
        "eligible_clip_count": len(request.rendered_clips),
        "intended_rendered_clip_identities": [
            clip.identity for clip in request.rendered_clips
        ],
        "deleted_clip_count": 0,
        "retained_clip_count": retained_count,
        "bytes_deleted": 0,
        "cleanup_status": CleanupStatus.IN_PROGRESS.value,
        "sanitized_cleanup_errors": [],
    }


def _write_cleanup_record(
    report_path: Path,
    report_payload: dict[str, object],
    cleanup_record: Mapping[str, object],
) -> dict[str, object]:
    updated_payload = dict(report_payload)
    updated_payload["post_upload_cleanup"] = dict(cleanup_record)
    _atomic_write_json(report_path, updated_payload)
    return updated_payload


def _write_recovery_diagnostic_safely(
    context: _CleanupContext,
    request: CleanupRequest,
    result: CleanupResult,
) -> SanitizedCleanupError | None:
    payload = {
        "status": "cleanup_recovery_required",
        "cleanup_policy": request.policy.value,
        "intended_rendered_clip_identities": [
            clip.identity for clip in request.rendered_clips
        ],
        "cleanup_result": result.to_dict(),
    }
    try:
        _atomic_write_json(
            context.reports_directory / _RECOVERY_DIAGNOSTIC_NAME,
            payload,
        )
    except Exception as exc:
        return _cleanup_error(
            "record_recovery",
            "cleanup_recovery_diagnostic_write_failed",
            exc=exc,
        )
    return None


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary_path.open("x", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                sort_keys=True,
                separators=(",", ":"),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _retained_result(
    request: CleanupRequest,
    status: CleanupStatus,
    retained_count: int,
    issues: tuple[SanitizedCleanupError, ...],
) -> CleanupResult:
    return CleanupResult(
        cleanup_policy=request.policy.value,
        eligible_clip_count=0,
        deleted_clip_count=0,
        retained_clip_count=retained_count,
        bytes_deleted=0,
        cleanup_status=status.value,
        sanitized_cleanup_errors=issues,
    )


def _cleanup_error(
    stage: str,
    code: str,
    *,
    exc: BaseException | None = None,
    error_type: str | None = None,
    rendered_clip_identity: str | None = None,
    destination: str | None = None,
) -> SanitizedCleanupError:
    return SanitizedCleanupError(
        stage=stage,
        code=code,
        error_type=type(exc).__name__ if exc is not None else error_type,
        rendered_clip_identity=rendered_clip_identity,
        destination=destination,
    )


def _existing_clip_count(clips: tuple[RenderedClipOutput, ...]) -> int:
    count = 0
    for clip in clips:
        try:
            if Path(clip.path).is_file():
                count += 1
        except (OSError, RuntimeError):
            count += 1
    return count


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _unlink_file(path: Path) -> None:
    path.unlink()
