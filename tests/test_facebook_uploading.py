import json
from pathlib import Path

import pytest

from core import UploadJob, UploadStatus
from facebook_auth_contracts import FacebookCredentialState
from uploading import (
    FacebookAuthenticationRequired,
    FacebookClientError,
    FacebookGraphClient,
    FacebookRemoteVideo,
    FacebookUploadAdapter,
    FacebookUploadConfig,
    JsonUploadLedger,
    PermanentUploadError,
    RetryableUploadError,
    UploadService,
    stable_upload_identity,
)
from uploading.facebook_cli import main


PAGE_ID = "123456789012345"


class FakeFacebookClient:
    def __init__(self, page_id: str = PAGE_ID) -> None:
        self.page_id = page_id
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
        remote = FacebookRemoteVideo(
            "facebook-video-123",
            published=plan.metadata["published"],
            permalink_url="/page/videos/facebook-video-123/",
        )
        self.videos[marker] = remote
        if self.interrupt_after_completion:
            self.interrupt_after_completion = False
            raise KeyboardInterrupt
        return remote


class FakeResponse:
    def __init__(self, status_code: int, payload) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, *, post_responses=(), get_responses=()) -> None:
        self.post_responses = list(post_responses)
        self.get_responses = list(get_responses)
        self.post_calls = []
        self.get_calls = []

    def post(self, url, **kwargs):
        source = kwargs["files"]["source"]
        self.post_calls.append(
            {
                "url": url,
                "data": dict(kwargs["data"]),
                "filename": source[0],
                "content": source[1].read(),
                "content_type": source[2],
                "timeout": kwargs["timeout"],
            }
        )
        return self.post_responses.pop(0)

    def get(self, url, **kwargs):
        self.get_calls.append({"url": url, **kwargs})
        return self.get_responses.pop(0)


def facebook_job(tmp_path: Path, **changes) -> UploadJob:
    clip = tmp_path / "rendered.mp4"
    clip.write_bytes(b"facebook-rendered-video")
    values = {
        "rendered_clip_path": clip,
        "rendered_clip_identity": "render:session-2:identity-9",
        "destination": "facebook",
        "title": "Facebook prototype clip",
        "description": "Deterministic Page caption",
        "visibility": "unpublished",
        "metadata": {"facebook_page_id": PAGE_ID},
    }
    values.update(changes)
    return UploadJob(**values)


def facebook_service(tmp_path: Path, client=None) -> UploadService:
    return UploadService(
        JsonUploadLedger(tmp_path / "uploads" / "ledger.json"),
        [FacebookUploadAdapter(client)],
    )


def graph_config(tmp_path: Path) -> FacebookUploadConfig:
    return FacebookUploadConfig(
        page_id=PAGE_ID,
        page_access_token="page-access-token",
        graph_api_version="v25.0",
        ledger_path=tmp_path / "ledger.json",
    )


def test_facebook_dry_run_needs_no_client_token_network_or_ledger(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "uploads" / "ledger.json"
    job = facebook_job(tmp_path)

    result = UploadService(
        JsonUploadLedger(ledger),
        [FacebookUploadAdapter()],
    ).execute(job, dry_run=True)

    assert result.status is UploadStatus.DRY_RUN
    assert result.upload_identity.startswith("facebook:sha256:")
    assert result.upload_identity == stable_upload_identity(job)
    plan = result.metadata["plan"]
    assert plan["privacy_status"] == "unpublished"
    assert plan["metadata"]["facebook_page_id"] == PAGE_ID
    assert plan["metadata"]["published"] is False
    assert plan["metadata"]["upload_marker"] in plan["description"]
    assert not ledger.exists()
    assert not ledger.with_name("ledger.json.lock").exists()


@pytest.mark.parametrize(
    ("input_state", "canonical_state", "published"),
    [
        ("public", "published", True),
        ("published", "published", True),
        ("draft", "unpublished", False),
        ("unpublished", "unpublished", False),
    ],
)
def test_facebook_publication_states_are_canonical(
    tmp_path: Path,
    input_state: str,
    canonical_state: str,
    published: bool,
) -> None:
    job = facebook_job(tmp_path, visibility=input_state)
    plan = FacebookUploadAdapter().plan(job, stable_upload_identity(job))

    assert plan.privacy_status == canonical_state
    assert plan.metadata["published"] is published


def test_facebook_rejects_personal_privacy_and_non_page_targets(
    tmp_path: Path,
) -> None:
    with pytest.raises(PermanentUploadError, match="publishing state"):
        facebook_service(tmp_path).execute(
            facebook_job(tmp_path, visibility="private"),
            dry_run=True,
        )

    with pytest.raises(PermanentUploadError, match="facebook_page_id"):
        facebook_service(tmp_path).execute(
            facebook_job(tmp_path, metadata={}),
            dry_run=True,
        )


def test_facebook_completed_ledger_prevents_duplicate_after_restart(
    tmp_path: Path,
) -> None:
    client = FakeFacebookClient()
    job = facebook_job(tmp_path)
    first = facebook_service(tmp_path, client).execute(job)
    replacement = FakeFacebookClient()
    second = facebook_service(tmp_path, replacement).execute(job)

    assert first == second
    assert first.remote_id == "facebook-video-123"
    assert first.remote_url == (
        "https://www.facebook.com/page/videos/facebook-video-123/"
    )
    assert client.upload_calls == 1
    assert replacement.find_calls == 0
    assert replacement.upload_calls == 0


def test_facebook_interruption_recovers_remote_marker_without_resubmission(
    tmp_path: Path,
) -> None:
    client = FakeFacebookClient()
    client.interrupt_after_completion = True
    uploader = facebook_service(tmp_path, client)
    job = facebook_job(tmp_path)

    with pytest.raises(KeyboardInterrupt):
        uploader.execute(job)

    result = uploader.execute(job)

    assert result.status is UploadStatus.COMPLETED
    assert result.recovered is True
    assert result.remote_id == "facebook-video-123"
    assert result.metadata["publishing_state"] == "unpublished"
    assert client.find_calls == 2
    assert client.upload_calls == 1


@pytest.mark.parametrize(
    ("retryable", "error_type"),
    [(True, RetryableUploadError), (False, PermanentUploadError)],
)
def test_facebook_client_failures_are_classified_and_persisted(
    tmp_path: Path,
    retryable: bool,
    error_type,
) -> None:
    client = FakeFacebookClient()
    client.find_error = FacebookClientError(
        "Facebook unavailable",
        retryable=retryable,
    )
    uploader = facebook_service(tmp_path, client)
    job = facebook_job(tmp_path)

    with pytest.raises(error_type, match="Facebook unavailable"):
        uploader.execute(job)

    if not retryable:
        client.find_error = None
        with pytest.raises(PermanentUploadError, match="Facebook unavailable"):
            uploader.execute(job)
        assert client.find_calls == 1


def test_facebook_retryable_failure_resumes_same_identity(tmp_path: Path) -> None:
    client = FakeFacebookClient()
    client.find_error = FacebookClientError("rate limited", retryable=True)
    uploader = facebook_service(tmp_path, client)
    job = facebook_job(tmp_path)

    with pytest.raises(RetryableUploadError, match="rate limited"):
        uploader.execute(job)

    client.find_error = None
    result = uploader.execute(job)

    assert result.status is UploadStatus.COMPLETED
    assert client.find_calls == 2
    assert client.upload_calls == 1


def test_facebook_authorization_failure_keeps_identity_retryable(
    tmp_path: Path,
) -> None:
    secret = "sensitive-token-value"
    client = FakeFacebookClient()
    client.find_error = FacebookClientError(
        f"expired credential {secret}",
        retryable=False,
        graph_code=190,
    )
    uploader = facebook_service(tmp_path, client)
    job = facebook_job(tmp_path)
    identity = stable_upload_identity(job)

    with pytest.raises(FacebookAuthenticationRequired) as captured:
        uploader.execute(job)

    assert (
        captured.value.state
        is FacebookCredentialState.REAUTHORIZATION_REQUIRED
    )
    ledger_path = tmp_path / "uploads" / "ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert ledger["uploads"][identity]["state"] == "pending"
    assert ledger["uploads"][identity]["retryable"] is True
    assert secret not in ledger_path.read_text(encoding="utf-8")

    client.find_error = None
    result = uploader.execute(job)

    assert result.upload_identity == identity
    assert client.find_calls == 2
    assert client.upload_calls == 1


def test_facebook_client_page_must_match_job_page(tmp_path: Path) -> None:
    client = FakeFacebookClient(page_id="999999999")

    with pytest.raises(PermanentUploadError, match="does not match"):
        facebook_service(tmp_path, client).execute(facebook_job(tmp_path))

    assert client.find_calls == 0
    assert client.upload_calls == 0


def test_facebook_graph_client_uploads_page_video_payload(tmp_path: Path) -> None:
    session = FakeSession(post_responses=[FakeResponse(200, {"id": "video-789"})])
    client = FacebookGraphClient(session, graph_config(tmp_path))
    job = facebook_job(tmp_path, visibility="published")
    plan = FacebookUploadAdapter().plan(job, stable_upload_identity(job))

    remote = client.upload_video(plan)

    assert remote == FacebookRemoteVideo("video-789", published=True)
    assert len(session.post_calls) == 1
    call = session.post_calls[0]
    assert call["url"] == (
        f"https://graph-video.facebook.com/v25.0/{PAGE_ID}/videos"
    )
    assert call["data"]["access_token"] == "page-access-token"
    assert call["data"]["title"] == job.title
    assert call["data"]["published"] == "true"
    assert plan.metadata["upload_marker"] in call["data"]["description"]
    assert call["filename"] == "rendered.mp4"
    assert call["content"] == b"facebook-rendered-video"


def test_facebook_graph_recovery_follows_page_pagination(tmp_path: Path) -> None:
    marker = "aitoclip-upload-marker"
    next_url = "https://graph.facebook.com/v25.0/next-page?after=cursor"
    session = FakeSession(
        get_responses=[
            FakeResponse(
                200,
                {
                    "data": [
                        {
                            "id": "other-video",
                            "description": "Another deterministic upload marker",
                        }
                    ],
                    "paging": {"next": next_url},
                },
            ),
            FakeResponse(
                200,
                {
                    "data": [
                        {
                            "id": "recovered-video",
                            "description": f"Caption\n\n[{marker}]",
                            "published": False,
                            "permalink_url": "/page/videos/recovered-video/",
                        }
                    ]
                },
            ),
        ]
    )
    client = FacebookGraphClient(session, graph_config(tmp_path))

    remote = client.find_video_by_upload_marker(marker)

    assert remote == FacebookRemoteVideo(
        "recovered-video",
        published=False,
        permalink_url="/page/videos/recovered-video/",
    )
    assert len(session.get_calls) == 2
    assert session.get_calls[0]["params"]["fields"] == (
        "id,description,permalink_url,published"
    )
    assert session.get_calls[0]["params"]["limit"] == 25
    assert session.get_calls[1]["url"] == next_url
    assert session.get_calls[1]["params"] is None


def test_facebook_graph_recovery_size_error_stays_retryable(tmp_path: Path) -> None:
    message = "Please reduce the amount of data you're asking for, then retry your request"
    session = FakeSession(
        get_responses=[
            FakeResponse(400, {"error": {"code": 1, "message": message}})
        ]
    )
    client = FacebookGraphClient(session, graph_config(tmp_path))

    with pytest.raises(FacebookClientError, match="Please reduce") as captured:
        client.find_video_by_upload_marker("aitoclip-upload-marker")

    assert captured.value.retryable is True
    assert len(session.get_calls) == 1
    assert session.get_calls[0]["params"]["limit"] == 25


@pytest.mark.parametrize(
    ("status", "error", "retryable"),
    [
        (429, {"code": 4, "message": "rate limited"}, True),
        (400, {"code": 190, "message": "invalid token"}, False),
    ],
)
def test_facebook_graph_errors_preserve_retry_classification(
    tmp_path: Path,
    status: int,
    error: dict,
    retryable: bool,
) -> None:
    session = FakeSession(post_responses=[FakeResponse(status, {"error": error})])
    client = FacebookGraphClient(session, graph_config(tmp_path))
    job = facebook_job(tmp_path)
    plan = FacebookUploadAdapter().plan(job, stable_upload_identity(job))

    with pytest.raises(FacebookClientError) as captured:
        client.upload_video(plan)

    assert captured.value.retryable is retryable


def test_facebook_config_binds_resolved_token_to_file_settings(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    path = config_dir / "facebook.json"
    path.write_text(
        json.dumps(
            {
                "page_id": PAGE_ID,
                "graph_api_version": "v25.0",
                "ledger_path": "../uploads/ledger.json",
            }
        ),
        encoding="utf-8",
    )

    config = FacebookUploadConfig.from_sources(
        config_path=path,
        environ={"AITOCLIP_FACEBOOK_PAGE_ACCESS_TOKEN": "page-token"},
        page_access_token="page-token",
    )

    assert config.page_id == PAGE_ID
    assert config.page_access_token == "page-token"
    assert config.ledger_path == tmp_path / "uploads" / "ledger.json"
    assert "page-token" not in repr(config)


def test_facebook_dry_run_cli_prints_plan_without_credentials_or_ledger(
    tmp_path: Path,
    capsys,
) -> None:
    job = facebook_job(tmp_path)
    ledger = tmp_path / "ledger.json"

    result = main(
        [
            "--clip",
            str(job.rendered_clip_path),
            "--render-identity",
            job.rendered_clip_identity,
            "--page-id",
            PAGE_ID,
            "--title",
            job.title,
            "--caption",
            job.description or "",
            "--publishing-state",
            "unpublished",
            "--ledger",
            str(ledger),
            "--dry-run",
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert result == 0
    assert output["status"] == "dry_run"
    assert output["destination"] == "facebook"
    assert output["metadata"]["plan"]["metadata"]["facebook_page_id"] == PAGE_ID
    assert not ledger.exists()
