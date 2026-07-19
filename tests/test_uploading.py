import json
from pathlib import Path

import pytest

from core import UploadJob, UploadStatus
from uploading import (
    JsonUploadLedger,
    PermanentUploadError,
    RetryableUploadError,
    UploadIdentityConflictError,
    UploadService,
    YouTubeClientError,
    YouTubeRemoteVideo,
    YouTubeUploadAdapter,
    YouTubeUploadConfig,
    stable_upload_identity,
    upload_request_fingerprint,
)
from uploading.cli import main


class FakeYouTubeClient:
    def __init__(self) -> None:
        self.videos = {}
        self.find_calls = 0
        self.upload_calls = 0
        self.find_error = None
        self.upload_error = None
        self.interrupt_after_completion = False

    def find_video_by_upload_marker(self, marker):
        self.find_calls += 1
        if self.find_error is not None:
            raise self.find_error
        return self.videos.get(marker)

    def upload_video(self, plan):
        self.upload_calls += 1
        if self.upload_error is not None:
            raise self.upload_error
        marker = plan.metadata["upload_marker"]
        remote = YouTubeRemoteVideo("video-123", plan.privacy_status)
        self.videos[marker] = remote
        if self.interrupt_after_completion:
            self.interrupt_after_completion = False
            raise KeyboardInterrupt
        return remote


def upload_job(tmp_path: Path, **changes) -> UploadJob:
    clip = tmp_path / "rendered.mp4"
    clip.write_bytes(b"rendered-video")
    values = {
        "rendered_clip_path": clip,
        "rendered_clip_identity": "render:session-1:identity-7",
        "destination": "youtube",
        "title": "Prototype clip",
        "description": "Deterministic description",
        "tags": ["aitoclip"],
        "visibility": "private",
    }
    values.update(changes)
    return UploadJob(**values)


def service(tmp_path: Path, client=None) -> UploadService:
    return UploadService(
        JsonUploadLedger(tmp_path / "uploads" / "ledger.json"),
        [YouTubeUploadAdapter(client)],
    )


def test_upload_identity_uses_only_render_identity_and_platform(tmp_path: Path) -> None:
    first = upload_job(tmp_path)
    renamed = upload_job(tmp_path, title="A different title")
    another_render = upload_job(
        tmp_path,
        rendered_clip_identity="render:session-1:identity-8",
    )
    another_platform = upload_job(tmp_path, destination="example")

    assert stable_upload_identity(first) == stable_upload_identity(renamed)
    assert upload_request_fingerprint(first) != upload_request_fingerprint(renamed)
    assert stable_upload_identity(first) != stable_upload_identity(another_render)
    assert stable_upload_identity(first) != stable_upload_identity(another_platform)
    assert stable_upload_identity(first).startswith("youtube:sha256:")


def test_youtube_dry_run_needs_no_client_credentials_or_ledger(tmp_path: Path) -> None:
    ledger_path = tmp_path / "uploads" / "ledger.json"
    result = UploadService(
        JsonUploadLedger(ledger_path),
        [YouTubeUploadAdapter()],
    ).execute(upload_job(tmp_path, visibility="unlisted"), dry_run=True)

    assert result.status is UploadStatus.DRY_RUN
    assert result.remote_id is None
    assert result.metadata["plan"]["privacy_status"] == "unlisted"
    marker = result.metadata["plan"]["metadata"]["upload_marker"]
    assert marker in result.metadata["plan"]["tags"]
    assert not ledger_path.exists()
    assert not ledger_path.with_name("ledger.json.lock").exists()


def test_completed_ledger_result_prevents_duplicate_submission_after_restart(
    tmp_path: Path,
) -> None:
    client = FakeYouTubeClient()
    job = upload_job(tmp_path)
    first = service(tmp_path, client).execute(job)
    replacement_client = FakeYouTubeClient()
    second = service(tmp_path, replacement_client).execute(job)

    assert first == second
    assert first.status is UploadStatus.COMPLETED
    assert client.upload_calls == 1
    assert replacement_client.find_calls == 0
    assert replacement_client.upload_calls == 0


def test_interruption_after_remote_completion_recovers_without_resubmission(
    tmp_path: Path,
) -> None:
    client = FakeYouTubeClient()
    client.interrupt_after_completion = True
    uploader = service(tmp_path, client)
    job = upload_job(tmp_path)

    with pytest.raises(KeyboardInterrupt):
        uploader.execute(job)

    result = uploader.execute(job)

    assert result.status is UploadStatus.COMPLETED
    assert result.recovered is True
    assert result.remote_id == "video-123"
    assert client.upload_calls == 1
    assert client.find_calls == 2


@pytest.mark.parametrize(
    ("retryable", "error_type"),
    [(True, RetryableUploadError), (False, PermanentUploadError)],
)
def test_youtube_client_failures_are_classified_and_persisted(
    tmp_path: Path,
    retryable: bool,
    error_type,
) -> None:
    client = FakeYouTubeClient()
    client.find_error = YouTubeClientError("remote unavailable", retryable=retryable)
    uploader = service(tmp_path, client)
    job = upload_job(tmp_path)

    with pytest.raises(error_type, match="remote unavailable"):
        uploader.execute(job)

    if not retryable:
        client.find_error = None
        with pytest.raises(PermanentUploadError, match="remote unavailable"):
            uploader.execute(job)
        assert client.find_calls == 1


def test_retryable_failure_can_resume_the_same_ledger_identity(tmp_path: Path) -> None:
    client = FakeYouTubeClient()
    client.find_error = YouTubeClientError("temporary outage", retryable=True)
    uploader = service(tmp_path, client)
    job = upload_job(tmp_path)

    with pytest.raises(RetryableUploadError, match="temporary outage"):
        uploader.execute(job)

    client.find_error = None
    result = uploader.execute(job)

    assert result.status is UploadStatus.COMPLETED
    assert client.find_calls == 2
    assert client.upload_calls == 1


def test_same_upload_identity_rejects_changed_request_content(tmp_path: Path) -> None:
    client = FakeYouTubeClient()
    uploader = service(tmp_path, client)
    uploader.execute(upload_job(tmp_path))

    with pytest.raises(UploadIdentityConflictError):
        uploader.execute(upload_job(tmp_path, title="Changed after completion"))

    assert client.upload_calls == 1


def test_youtube_config_uses_file_paths_with_environment_overrides(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "youtube.json"
    config_path.write_text(
        json.dumps(
            {
                "client_secrets_path": "client.json",
                "token_path": "token.json",
                "ledger_path": "../uploads/ledger.json",
            }
        ),
        encoding="utf-8",
    )
    override = tmp_path / "private" / "override-client.json"

    config = YouTubeUploadConfig.from_sources(
        config_path=config_path,
        environ={"AITOCLIP_YOUTUBE_CLIENT_SECRETS_PATH": str(override)},
    )

    assert config.client_secrets_path == override
    assert config.token_path == config_dir / "token.json"
    assert config.ledger_path == tmp_path / "uploads" / "ledger.json"
    assert len(config.scopes) == 2


def test_dry_run_cli_prints_plan_without_writing_ledger(
    tmp_path: Path,
    capsys,
) -> None:
    job = upload_job(tmp_path)
    ledger = tmp_path / "ledger.json"

    result = main(
        [
            "--clip",
            str(job.rendered_clip_path),
            "--render-identity",
            job.rendered_clip_identity,
            "--title",
            job.title,
            "--description",
            job.description or "",
            "--privacy-status",
            "private",
            "--ledger",
            str(ledger),
            "--dry-run",
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert result == 0
    assert output["status"] == "dry_run"
    assert output["metadata"]["plan"]["title"] == job.title
    assert not ledger.exists()


def test_invalid_youtube_privacy_is_permanent_and_never_calls_client(
    tmp_path: Path,
) -> None:
    client = FakeYouTubeClient()

    with pytest.raises(PermanentUploadError, match="privacy status"):
        service(tmp_path, client).execute(
            upload_job(tmp_path, visibility="followers-only")
        )

    assert client.find_calls == 0
    assert client.upload_calls == 0
