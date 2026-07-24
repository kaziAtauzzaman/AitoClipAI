import json
import os
from pathlib import Path
import subprocess
import threading

import pytest

import operator_cleanup
from core import UploadResult, UploadStatus
from operator_cleanup import (
    CleanupPolicy,
    CleanupRequest,
    CleanupStatus,
    OperatorCleanupController,
    PostUploadCleanupCoordinator,
)
from operator_pipeline import RenderedClipOutput
from operator_upload import (
    FACEBOOK_DESTINATION,
    YOUTUBE_DESTINATION,
    UploadAttempt,
    UploadQueueSummary,
)


def _operator_run(
    tmp_path: Path,
    *,
    clip_count: int = 2,
) -> tuple[Path, Path, Path, tuple[RenderedClipOutput, ...]]:
    run_directory = tmp_path / "operator-run"
    clips_directory = run_directory / "clips"
    reports_directory = run_directory / "reports"
    source_directory = run_directory / "source"
    clips_directory.mkdir(parents=True)
    reports_directory.mkdir()
    source_directory.mkdir()
    (run_directory / "run.log").write_text("safe run log", encoding="utf-8")
    (source_directory / "downloaded.mp4").write_bytes(b"source-media")
    report_path = reports_directory / "production-incremental-report.json"
    report_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "source_video": str(source_directory / "downloaded.mp4"),
                "render_jobs": [],
            }
        ),
        encoding="utf-8",
    )
    clips = []
    for number in range(1, clip_count + 1):
        path = clips_directory / f"clip-{number}.mp4"
        path.write_bytes(f"rendered-clip-{number}".encode())
        clips.append(
            RenderedClipOutput(
                path=path,
                identity=f"render:run:clip-{number}",
                title=f"Clip {number}",
                description=f"Description {number}",
            )
        )
    return run_directory, clips_directory, report_path, tuple(clips)


def _summary(
    clips: tuple[RenderedClipOutput, ...],
    destinations: tuple[str, ...],
    *,
    incomplete: set[tuple[str, str]] | None = None,
    error_type: str = "RetryableUploadError",
    result_status: UploadStatus = UploadStatus.COMPLETED,
) -> UploadQueueSummary:
    incomplete = incomplete or set()
    attempts = []
    for destination in destinations:
        for clip in clips:
            key = (clip.identity, destination)
            if key in incomplete:
                attempts.append(
                    UploadAttempt(
                        destination=destination,
                        rendered_clip_identity=clip.identity,
                        completed=False,
                        error_type=error_type,
                    )
                )
                continue
            result = UploadResult(
                upload_identity=f"{destination}:sha256:{clip.identity}",
                rendered_clip_identity=clip.identity,
                rendered_clip_path=clip.path,
                destination=destination,
                status=result_status,
                remote_id=f"{destination}-{clip.identity}",
            )
            attempts.append(
                UploadAttempt(
                    destination=destination,
                    rendered_clip_identity=clip.identity,
                    completed=True,
                    result=result,
                )
            )
    return UploadQueueSummary(tuple(attempts))


def _request(
    run_directory: Path,
    clips_directory: Path,
    report_path: Path,
    clips: tuple[RenderedClipOutput, ...],
    destinations: tuple[str, ...],
    *,
    policy: CleanupPolicy = CleanupPolicy.DELETE_AFTER_SUCCESSFUL_UPLOADS,
    summary: UploadQueueSummary | None = None,
) -> CleanupRequest:
    return CleanupRequest(
        run_directory=run_directory,
        clips_directory=clips_directory,
        report_path=report_path,
        expected_rendered_clip_count=len(clips),
        rendered_clips=clips,
        selected_destinations=destinations,
        upload_summary=summary,
        policy=policy,
    )


def _cleanup_report(report_path: Path) -> dict[str, object]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    return payload["post_upload_cleanup"]


def _coordinator_for(
    request: CleanupRequest,
    **kwargs,
) -> PostUploadCleanupCoordinator:
    return PostUploadCleanupCoordinator(
        operator_runs_root=Path(request.run_directory).parent,
        **kwargs,
    )


def _run_cleanup(
    request: CleanupRequest,
    **kwargs,
):
    return _coordinator_for(request, **kwargs).run(request)


def test_cleanup_module_exports_only_intended_public_contract() -> None:
    assert operator_cleanup.__all__ == (
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
    assert "UnsafeCleanupPathError" not in operator_cleanup.__all__


def test_no_destinations_selected_always_retains_clips(tmp_path: Path) -> None:
    run, clips_dir, report, clips = _operator_run(tmp_path)

    result = _run_cleanup(
        _request(run, clips_dir, report, clips, ())
    )

    assert result.cleanup_status == CleanupStatus.RETAINED_NO_DESTINATIONS.value
    assert result.deleted_clip_count == 0
    assert result.retained_clip_count == 2
    assert all(clip.path.is_file() for clip in clips)
    assert _cleanup_report(report) == result.to_dict()


def test_cleanup_disabled_retains_fully_uploaded_clips(tmp_path: Path) -> None:
    run, clips_dir, report, clips = _operator_run(tmp_path)
    destinations = (YOUTUBE_DESTINATION,)

    result = _run_cleanup(
        _request(
            run,
            clips_dir,
            report,
            clips,
            destinations,
            policy=CleanupPolicy.KEEP_RENDERED_CLIPS,
            summary=_summary(clips, destinations),
        )
    )

    assert result.cleanup_status == CleanupStatus.RETAINED_BY_POLICY.value
    assert result.eligible_clip_count == 0
    assert result.deleted_clip_count == 0
    assert all(clip.path.is_file() for clip in clips)


@pytest.mark.parametrize(
    "destinations",
    [
        pytest.param((YOUTUBE_DESTINATION,), id="youtube-only"),
        pytest.param((FACEBOOK_DESTINATION,), id="facebook-only"),
        pytest.param(
            (YOUTUBE_DESTINATION, FACEBOOK_DESTINATION),
            id="youtube-and-facebook",
        ),
    ],
)
def test_all_selected_uploads_complete_deletes_every_clip(
    tmp_path: Path,
    destinations: tuple[str, ...],
) -> None:
    run, clips_dir, report, clips = _operator_run(tmp_path)

    result = _run_cleanup(
        _request(
            run,
            clips_dir,
            report,
            clips,
            destinations,
            summary=_summary(clips, destinations),
        )
    )

    assert result.cleanup_status == CleanupStatus.COMPLETED.value
    assert result.eligible_clip_count == 2
    assert result.deleted_clip_count == 2
    assert result.retained_clip_count == 0
    assert result.bytes_deleted == len(b"rendered-clip-1") + len(
        b"rendered-clip-2"
    )
    assert all(not clip.path.exists() for clip in clips)


def test_one_destination_incomplete_retains_every_clip(tmp_path: Path) -> None:
    run, clips_dir, report, clips = _operator_run(tmp_path)
    destinations = (YOUTUBE_DESTINATION, FACEBOOK_DESTINATION)
    incomplete = {(clips[1].identity, FACEBOOK_DESTINATION)}

    result = _run_cleanup(
        _request(
            run,
            clips_dir,
            report,
            clips,
            destinations,
            summary=_summary(clips, destinations, incomplete=incomplete),
        )
    )

    assert result.cleanup_status == CleanupStatus.RETAINED_INCOMPLETE_UPLOADS.value
    assert result.deleted_clip_count == 0
    assert all(clip.path.is_file() for clip in clips)
    assert {
        issue.code for issue in result.sanitized_cleanup_errors
    } >= {"upload_incomplete"}


def test_one_failed_clip_upload_retains_every_clip(tmp_path: Path) -> None:
    run, clips_dir, report, clips = _operator_run(tmp_path, clip_count=3)
    destinations = (YOUTUBE_DESTINATION,)
    incomplete = {(clips[1].identity, YOUTUBE_DESTINATION)}

    result = _run_cleanup(
        _request(
            run,
            clips_dir,
            report,
            clips,
            destinations,
            summary=_summary(clips, destinations, incomplete=incomplete),
        )
    )

    assert result.deleted_clip_count == 0
    assert result.retained_clip_count == 3
    assert all(clip.path.is_file() for clip in clips)


@pytest.mark.parametrize(
    ("error_type", "case_name"),
    [
        ("FacebookAuthenticationRequired", "authentication-blocked"),
        ("PendingUpload", "pending-ledger"),
    ],
)
def test_nonterminal_or_authentication_blocked_upload_retains_all_clips(
    tmp_path: Path,
    error_type: str,
    case_name: str,
) -> None:
    del case_name
    run, clips_dir, report, clips = _operator_run(tmp_path)
    destinations = (FACEBOOK_DESTINATION,)
    incomplete = {(clips[0].identity, FACEBOOK_DESTINATION)}

    result = _run_cleanup(
        _request(
            run,
            clips_dir,
            report,
            clips,
            destinations,
            summary=_summary(
                clips,
                destinations,
                incomplete=incomplete,
                error_type=error_type,
            ),
        )
    )

    assert result.cleanup_status == CleanupStatus.RETAINED_INCOMPLETE_UPLOADS.value
    assert result.deleted_clip_count == 0
    assert all(clip.path.is_file() for clip in clips)


def test_dry_run_or_missing_upload_result_is_not_completion(tmp_path: Path) -> None:
    run, clips_dir, report, clips = _operator_run(tmp_path)
    destinations = (YOUTUBE_DESTINATION,)

    dry_run_result = _run_cleanup(
        _request(
            run,
            clips_dir,
            report,
            clips,
            destinations,
            summary=_summary(
                clips,
                destinations,
                result_status=UploadStatus.DRY_RUN,
            ),
        )
    )

    assert dry_run_result.deleted_clip_count == 0
    assert all(clip.path.is_file() for clip in clips)


def test_path_outside_current_run_refuses_all_deletion(tmp_path: Path) -> None:
    run, clips_dir, report, clips = _operator_run(tmp_path)
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"outside")
    unsafe_clips = (
        clips[0],
        RenderedClipOutput(
            path=outside,
            identity=clips[1].identity,
            title=clips[1].title,
            description=clips[1].description,
        ),
    )
    destinations = (YOUTUBE_DESTINATION,)

    result = _run_cleanup(
        _request(
            run,
            clips_dir,
            report,
            unsafe_clips,
            destinations,
            summary=_summary(unsafe_clips, destinations),
        )
    )

    assert result.cleanup_status == CleanupStatus.REFUSED_UNSAFE_PATH.value
    assert result.deleted_clip_count == 0
    assert clips[0].path.is_file()
    assert outside.is_file()
    assert any(
        issue.code == "rendered_path_outside_clips_directory"
        for issue in result.sanitized_cleanup_errors
    )


def test_dot_dot_escape_path_is_rejected(tmp_path: Path) -> None:
    run, clips_dir, report, clips = _operator_run(tmp_path)
    escaped_path = clips_dir / ".." / "source" / "downloaded.mp4"
    escaped_clip = RenderedClipOutput(
        path=escaped_path,
        identity=clips[0].identity,
        title=clips[0].title,
        description=clips[0].description,
    )
    destinations = (YOUTUBE_DESTINATION,)

    result = _run_cleanup(
        _request(
            run,
            clips_dir,
            report,
            (escaped_clip,),
            destinations,
            summary=_summary((escaped_clip,), destinations),
        )
    )

    assert result.cleanup_status == CleanupStatus.REFUSED_UNSAFE_PATH.value
    assert result.deleted_clip_count == 0
    assert (run / "source" / "downloaded.mp4").is_file()


@pytest.mark.skipif(
    os.name != "nt",
    reason="Windows case-insensitive path behavior is Windows-specific.",
)
def test_windows_case_variant_paths_resolve_within_trusted_run(
    tmp_path: Path,
) -> None:
    run, clips_dir, report, clips = _operator_run(tmp_path, clip_count=1)
    case_variant_clip = RenderedClipOutput(
        path=Path(str(clips[0].path).swapcase()),
        identity=clips[0].identity,
        title=clips[0].title,
        description=clips[0].description,
    )
    destinations = (YOUTUBE_DESTINATION,)
    request = CleanupRequest(
        run_directory=Path(str(run).swapcase()),
        clips_directory=Path(str(clips_dir).swapcase()),
        report_path=Path(str(report).swapcase()),
        expected_rendered_clip_count=1,
        rendered_clips=(case_variant_clip,),
        selected_destinations=destinations,
        upload_summary=_summary((case_variant_clip,), destinations),
        policy=CleanupPolicy.DELETE_AFTER_SUCCESSFUL_UPLOADS,
    )

    result = PostUploadCleanupCoordinator(
        operator_runs_root=Path(str(run.parent).swapcase())
    ).run(request)

    assert result.cleanup_status == CleanupStatus.COMPLETED.value
    assert result.deleted_clip_count == 1
    assert not clips[0].path.exists()


def test_final_component_symlink_is_rejected_when_supported(
    tmp_path: Path,
) -> None:
    run, clips_dir, report, clips = _operator_run(tmp_path, clip_count=1)
    linked_path = clips_dir / "linked-clip.mp4"
    try:
        linked_path.symlink_to(clips[0].path)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"File symlink creation is unavailable: {type(exc).__name__}.")
    linked_clip = RenderedClipOutput(
        path=linked_path,
        identity=clips[0].identity,
        title=clips[0].title,
        description=clips[0].description,
    )
    destinations = (YOUTUBE_DESTINATION,)

    result = _run_cleanup(
        _request(
            run,
            clips_dir,
            report,
            (linked_clip,),
            destinations,
            summary=_summary((linked_clip,), destinations),
        )
    )

    assert result.cleanup_status == CleanupStatus.REFUSED_UNSAFE_PATH.value
    assert linked_path.is_symlink()
    assert clips[0].path.is_file()
    assert "symbolic_link_refused" in {
        issue.code for issue in result.sanitized_cleanup_errors
    }


def test_parent_symlink_or_junction_escape_is_rejected_when_supported(
    tmp_path: Path,
) -> None:
    run, clips_dir, report, clips = _operator_run(tmp_path, clip_count=1)
    outside_directory = tmp_path / "outside-parent"
    outside_directory.mkdir()
    outside_file = outside_directory / "escaped.mp4"
    outside_file.write_bytes(b"must-not-delete")
    linked_directory = clips_dir / "escaped-parent"
    if os.name == "nt":
        completed = subprocess.run(
            [
                "cmd",
                "/c",
                "mklink",
                "/J",
                str(linked_directory),
                str(outside_directory),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            pytest.skip(
                "Windows junction creation is unavailable: "
                f"exit code {completed.returncode}."
            )
    else:
        try:
            linked_directory.symlink_to(
                outside_directory,
                target_is_directory=True,
            )
        except (NotImplementedError, OSError) as exc:
            pytest.skip(
                f"Directory symlink creation is unavailable: {type(exc).__name__}."
            )
    escaped_clip = RenderedClipOutput(
        path=linked_directory / outside_file.name,
        identity=clips[0].identity,
        title=clips[0].title,
        description=clips[0].description,
    )
    destinations = (YOUTUBE_DESTINATION,)

    try:
        result = _run_cleanup(
            _request(
                run,
                clips_dir,
                report,
                (escaped_clip,),
                destinations,
                summary=_summary((escaped_clip,), destinations),
            )
        )
    finally:
        if os.name == "nt" and linked_directory.exists():
            linked_directory.rmdir()

    assert result.cleanup_status == CleanupStatus.REFUSED_UNSAFE_PATH.value
    assert result.deleted_clip_count == 0
    assert outside_file.read_bytes() == b"must-not-delete"


def test_forged_run_outside_trusted_operator_root_is_rejected(
    tmp_path: Path,
) -> None:
    trusted_root = tmp_path / "project" / "data" / "runs" / "operator"
    trusted_root.mkdir(parents=True)
    run, clips_dir, report, clips = _operator_run(tmp_path / "forged")
    report_before = report.read_bytes()
    destinations = (YOUTUBE_DESTINATION,)
    request = _request(
        run,
        clips_dir,
        report,
        clips,
        destinations,
        summary=_summary(clips, destinations),
    )

    result = PostUploadCleanupCoordinator(
        operator_runs_root=trusted_root
    ).run(request)

    assert result.cleanup_status == CleanupStatus.REFUSED_UNSAFE_PATH.value
    assert result.deleted_clip_count == 0
    assert all(clip.path.is_file() for clip in clips)
    assert report.read_bytes() == report_before
    assert "run_directory_outside_operator_root" in {
        issue.code for issue in result.sanitized_cleanup_errors
    }


def test_nested_run_beneath_trusted_operator_root_is_accepted(
    tmp_path: Path,
) -> None:
    trusted_root = tmp_path / "data" / "runs" / "operator"
    run, clips_dir, report, clips = _operator_run(trusted_root / "nested")
    destinations = (YOUTUBE_DESTINATION,)
    request = _request(
        run,
        clips_dir,
        report,
        clips,
        destinations,
        summary=_summary(clips, destinations),
    )

    result = PostUploadCleanupCoordinator(
        operator_runs_root=trusted_root
    ).run(request)

    assert result.cleanup_status == CleanupStatus.COMPLETED.value
    assert result.deleted_clip_count == len(clips)


def test_validation06_shaped_directory_outside_operator_root_is_rejected(
    tmp_path: Path,
) -> None:
    trusted_root = tmp_path / "data" / "runs" / "operator"
    trusted_root.mkdir(parents=True)
    validation_root = (
        tmp_path
        / "data"
        / "validation"
        / "youtube-UMjrTuMomlc-endurance-qsv-06-production-media"
    )
    run, clips_dir, report, clips = _operator_run(validation_root)
    destinations = (FACEBOOK_DESTINATION,)
    request = _request(
        run,
        clips_dir,
        report,
        clips,
        destinations,
        summary=_summary(clips, destinations),
    )

    result = PostUploadCleanupCoordinator(
        operator_runs_root=trusted_root
    ).run(request)

    assert result.cleanup_status == CleanupStatus.REFUSED_UNSAFE_PATH.value
    assert result.deleted_clip_count == 0
    assert all(clip.path.is_file() for clip in clips)


@pytest.mark.parametrize(
    "protected_relative_path",
    [
        Path("source") / "downloaded.mp4",
        Path("run.log"),
        Path("reports") / "production-incremental-report.json",
    ],
    ids=["downloaded-source", "run-log", "production-report"],
)
def test_protected_run_files_cannot_be_supplied_as_rendered_clips(
    tmp_path: Path,
    protected_relative_path: Path,
) -> None:
    run, clips_dir, report, clips = _operator_run(tmp_path)
    protected_path = run / protected_relative_path
    protected_before = protected_path.read_bytes()
    unsafe_clip = RenderedClipOutput(
        path=protected_path,
        identity=clips[0].identity,
        title=clips[0].title,
        description=clips[0].description,
    )
    destinations = (YOUTUBE_DESTINATION,)

    result = _run_cleanup(
        _request(
            run,
            clips_dir,
            report,
            (unsafe_clip,),
            destinations,
            summary=_summary((unsafe_clip,), destinations),
        )
    )

    assert result.cleanup_status == CleanupStatus.REFUSED_UNSAFE_PATH.value
    assert result.deleted_clip_count == 0
    if protected_path == report:
        payload = json.loads(report.read_text(encoding="utf-8"))
        assert payload["status"] == "completed"
    else:
        assert protected_path.read_bytes() == protected_before


def test_missing_required_destination_or_duplicate_attempt_retains_all(
    tmp_path: Path,
) -> None:
    run, clips_dir, report, clips = _operator_run(tmp_path)
    destinations = (YOUTUBE_DESTINATION, FACEBOOK_DESTINATION)
    youtube_only = _summary(clips, (YOUTUBE_DESTINATION,))
    duplicate_summary = UploadQueueSummary(
        (*youtube_only.attempts, youtube_only.attempts[0])
    )

    missing = _run_cleanup(
        _request(
            run,
            clips_dir,
            report,
            clips,
            destinations,
            summary=youtube_only,
        )
    )
    duplicate = _run_cleanup(
        _request(
            run,
            clips_dir,
            report,
            clips,
            (YOUTUBE_DESTINATION,),
            summary=duplicate_summary,
        )
    )

    assert missing.deleted_clip_count == 0
    assert duplicate.deleted_clip_count == 0
    assert all(clip.path.is_file() for clip in clips)
    assert "required_upload_job_missing" in {
        issue.code for issue in missing.sanitized_cleanup_errors
    }
    assert "duplicate_upload_job" in {
        issue.code for issue in duplicate.sanitized_cleanup_errors
    }


def test_pipeline_rendered_count_mismatch_retains_all_clips(tmp_path: Path) -> None:
    run, clips_dir, report, clips = _operator_run(tmp_path)
    destinations = (YOUTUBE_DESTINATION,)
    request = _request(
        run,
        clips_dir,
        report,
        clips,
        destinations,
        summary=_summary(clips, destinations),
    )
    request = CleanupRequest(
        run_directory=request.run_directory,
        clips_directory=request.clips_directory,
        report_path=request.report_path,
        expected_rendered_clip_count=len(clips) + 1,
        rendered_clips=request.rendered_clips,
        selected_destinations=request.selected_destinations,
        upload_summary=request.upload_summary,
        policy=request.policy,
    )

    result = _run_cleanup(request)

    assert result.deleted_clip_count == 0
    assert all(clip.path.is_file() for clip in clips)
    assert "rendered_clip_count_mismatch" in {
        issue.code for issue in result.sanitized_cleanup_errors
    }


def test_cleanup_touches_only_explicit_clip_paths_and_report(
    tmp_path: Path,
) -> None:
    run, clips_dir, report, clips = _operator_run(tmp_path)
    source = run / "source" / "downloaded.mp4"
    run_log = run / "run.log"
    ledger = tmp_path / "upload-ledger.json"
    config = tmp_path / "facebook-upload.json"
    ledger.write_text('{"state":"completed"}', encoding="utf-8")
    config.write_text('{"page_id":"safe"}', encoding="utf-8")
    before = {
        source: source.read_bytes(),
        run_log: run_log.read_bytes(),
        ledger: ledger.read_bytes(),
        config: config.read_bytes(),
    }
    destinations = (FACEBOOK_DESTINATION,)

    result = _run_cleanup(
        _request(
            run,
            clips_dir,
            report,
            clips,
            destinations,
            summary=_summary(clips, destinations),
        )
    )

    assert result.deleted_clip_count == 2
    assert report.is_file()
    assert json.loads(report.read_text(encoding="utf-8"))["status"] == "completed"
    assert {path: path.read_bytes() for path in before} == before


def test_repeated_cleanup_safely_classifies_already_missing_clips(
    tmp_path: Path,
) -> None:
    run, clips_dir, report, clips = _operator_run(tmp_path)
    destinations = (YOUTUBE_DESTINATION,)
    request = _request(
        run,
        clips_dir,
        report,
        clips,
        destinations,
        summary=_summary(clips, destinations),
    )
    coordinator = PostUploadCleanupCoordinator(operator_runs_root=run.parent)

    first = coordinator.run(request)
    second = coordinator.run(request)

    assert first.cleanup_status == CleanupStatus.COMPLETED.value
    assert second.cleanup_status == CleanupStatus.ALREADY_CLEAN.value
    assert second.eligible_clip_count == 2
    assert second.deleted_clip_count == 0
    assert second.retained_clip_count == 0
    assert second.bytes_deleted == 0


def test_durable_in_progress_record_exists_before_first_unlink(
    tmp_path: Path,
) -> None:
    run, clips_dir, report, clips = _operator_run(tmp_path)
    destinations = (YOUTUBE_DESTINATION,)
    observed_records = []

    def observe_then_delete(path: Path) -> None:
        observed_records.append(_cleanup_report(report))
        path.unlink()

    request = _request(
        run,
        clips_dir,
        report,
        clips,
        destinations,
        summary=_summary(clips, destinations),
    )

    result = _run_cleanup(request, unlink_file=observe_then_delete)

    assert result.cleanup_status == CleanupStatus.COMPLETED.value
    assert len(observed_records) == 2
    assert all(
        record["cleanup_status"] == CleanupStatus.IN_PROGRESS.value
        for record in observed_records
    )
    assert observed_records[0]["eligible_clip_count"] == 2
    assert observed_records[0]["deleted_clip_count"] == 0
    assert observed_records[0]["bytes_deleted"] == 0
    assert observed_records[0]["intended_rendered_clip_identities"] == [
        clip.identity for clip in clips
    ]


def test_in_progress_report_failure_deletes_nothing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    run, clips_dir, report, clips = _operator_run(tmp_path)
    report_before = report.read_bytes()
    destinations = (FACEBOOK_DESTINATION,)

    def fail_atomic_write(path, payload):
        raise OSError("synthetic report write failure")

    monkeypatch.setattr(
        operator_cleanup,
        "_atomic_write_json",
        fail_atomic_write,
    )

    result = _run_cleanup(
        _request(
            run,
            clips_dir,
            report,
            clips,
            destinations,
            summary=_summary(clips, destinations),
        )
    )

    assert result.cleanup_status == CleanupStatus.REFUSED_REPORT_UNAVAILABLE.value
    assert result.deleted_clip_count == 0
    assert all(clip.path.is_file() for clip in clips)
    assert report.read_bytes() == report_before
    assert "cleanup_in_progress_write_failed" in {
        issue.code for issue in result.sanitized_cleanup_errors
    }


def test_final_report_failure_preserves_in_progress_and_writes_recovery(
    monkeypatch,
    tmp_path: Path,
) -> None:
    run, clips_dir, report, clips = _operator_run(tmp_path)
    destinations = (YOUTUBE_DESTINATION,)
    real_atomic_write = operator_cleanup._atomic_write_json
    writes = []

    def fail_final_report(path, payload):
        writes.append(Path(path))
        if len(writes) == 2:
            raise OSError("synthetic final report failure")
        real_atomic_write(path, payload)

    monkeypatch.setattr(
        operator_cleanup,
        "_atomic_write_json",
        fail_final_report,
    )

    result = _run_cleanup(
        _request(
            run,
            clips_dir,
            report,
            clips,
            destinations,
            summary=_summary(clips, destinations),
        )
    )

    assert result.cleanup_status == CleanupStatus.PARTIAL_FAILURE.value
    assert result.deleted_clip_count == 2
    assert all(not clip.path.exists() for clip in clips)
    durable_record = _cleanup_report(report)
    assert durable_record["cleanup_status"] == CleanupStatus.IN_PROGRESS.value
    assert durable_record["deleted_clip_count"] == 0
    recovery_path = (
        run / "reports" / "post-upload-cleanup-recovery.json"
    )
    recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
    assert recovery["status"] == "cleanup_recovery_required"
    assert recovery["cleanup_result"]["deleted_clip_count"] == 2
    assert recovery["cleanup_result"]["bytes_deleted"] == result.bytes_deleted
    assert len(writes) == 3


def test_keyboard_interrupt_cleans_temporary_report_and_is_not_swallowed(
    monkeypatch,
    tmp_path: Path,
) -> None:
    run, clips_dir, report, clips = _operator_run(tmp_path)
    destinations = (YOUTUBE_DESTINATION,)

    def interrupt_dump(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(operator_cleanup.json, "dump", interrupt_dump)

    with pytest.raises(KeyboardInterrupt):
        _run_cleanup(
            _request(
                run,
                clips_dir,
                report,
                clips,
                destinations,
                summary=_summary(clips, destinations),
            )
        )

    assert all(clip.path.is_file() for clip in clips)
    assert list((run / "reports").glob("*.tmp")) == []
    assert "post_upload_cleanup" not in json.loads(
        report.read_text(encoding="utf-8")
    )


def test_cleanup_failure_preserves_upload_success_and_remaining_files(
    tmp_path: Path,
) -> None:
    run, clips_dir, report, clips = _operator_run(tmp_path)
    destinations = (YOUTUBE_DESTINATION,)
    summary = _summary(clips, destinations)

    def fail_delete(path: Path) -> None:
        raise PermissionError("synthetic path detail must stay sanitized")

    request = _request(
        run,
        clips_dir,
        report,
        clips,
        destinations,
        summary=summary,
    )
    result = _run_cleanup(
        request,
        unlink_file=fail_delete,
    )

    assert result.cleanup_status == CleanupStatus.PARTIAL_FAILURE.value
    assert result.deleted_clip_count == 0
    assert all(clip.path.is_file() for clip in clips)
    assert all(attempt.completed for attempt in summary.attempts)
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    serialized_cleanup = json.dumps(payload["post_upload_cleanup"])
    assert "synthetic path detail" not in serialized_cleanup
    assert "PermissionError" in serialized_cleanup


def test_cleanup_controller_runs_off_thread_and_rejects_overlap(
    tmp_path: Path,
) -> None:
    run, clips_dir, report, clips = _operator_run(tmp_path)
    entered = threading.Event()
    release = threading.Event()
    caller_thread = threading.get_ident()
    worker_threads = []

    class BlockingCoordinator:
        def run(self, request):
            worker_threads.append(threading.get_ident())
            entered.set()
            release.wait(timeout=5)
            return _run_cleanup(request)

    controller = OperatorCleanupController(BlockingCoordinator())
    request = _request(
        run,
        clips_dir,
        report,
        clips,
        (),
    )
    results = []
    failures = []

    thread = controller.start(
        request,
        on_complete=results.append,
        on_failure=failures.append,
    )
    assert entered.wait(timeout=5)
    with pytest.raises(RuntimeError, match="already active"):
        controller.start(
            request,
            on_complete=results.append,
            on_failure=failures.append,
        )
    release.set()
    thread.join(timeout=5)

    assert worker_threads == [worker_threads[0]]
    assert worker_threads[0] != caller_thread
    assert len(results) == 1
    assert failures == []
    assert controller.is_running is False
