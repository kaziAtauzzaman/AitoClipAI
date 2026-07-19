"""YouTube platform adapter and lazily loaded Google API client."""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Callable, Protocol

from core import UploadJob, UploadResult, UploadStatus
from uploading.config import YouTubeUploadConfig
from uploading.contracts import UploadPlan
from uploading.errors import PermanentUploadError, RetryableUploadError
from uploading.identity import normalize_destination


YOUTUBE_DESTINATION = "youtube"
YOUTUBE_VIDEO_URL = "https://www.youtube.com/watch?v={video_id}"
_RETRYABLE_HTTP_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
_PRIVACY_STATUSES = frozenset({"private", "unlisted", "public"})


@dataclass(frozen=True, slots=True)
class YouTubeRemoteVideo:
    """Minimum remote state needed for completion and recovery."""

    video_id: str
    privacy_status: str | None = None


class YouTubeClientError(Exception):
    """Client-level failure translated by the platform adapter."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


class YouTubeClient(Protocol):
    """Narrow client boundary used by fake and Google implementations."""

    def find_video_by_upload_marker(
        self,
        upload_marker: str,
    ) -> YouTubeRemoteVideo | None: ...

    def upload_video(self, plan: UploadPlan) -> YouTubeRemoteVideo: ...


class YouTubeUploadAdapter:
    """Translate neutral upload jobs into YouTube requests and recovery."""

    destination = YOUTUBE_DESTINATION

    def __init__(self, client: YouTubeClient | None = None) -> None:
        self._client = client

    def plan(self, job: UploadJob, upload_identity: str) -> UploadPlan:
        if normalize_destination(job.destination) != self.destination:
            raise PermanentUploadError("YouTube adapter received another destination.")
        title = job.title.strip()
        description = job.description or ""
        privacy = (job.visibility or "private").strip().lower()
        if len(title) > 100:
            raise PermanentUploadError("YouTube titles cannot exceed 100 characters.")
        if len(description) > 5_000:
            raise PermanentUploadError(
                "YouTube descriptions cannot exceed 5,000 characters."
            )
        if privacy not in _PRIVACY_STATUSES:
            raise PermanentUploadError(
                "YouTube privacy status must be private, unlisted, or public."
            )
        marker = _upload_marker(upload_identity)
        tags = tuple(dict.fromkeys([*job.tags, marker]))
        if any(not isinstance(item, str) or not item.strip() for item in tags):
            raise PermanentUploadError("YouTube tags must be non-empty strings.")
        if len(",".join(tags)) > 500:
            raise PermanentUploadError("YouTube tags exceed the 500-character limit.")
        return UploadPlan(
            upload_identity=upload_identity,
            destination=self.destination,
            rendered_clip_identity=job.rendered_clip_identity,
            rendered_clip_path=job.rendered_clip_path,
            title=title,
            description=description,
            privacy_status=privacy,
            tags=tags,
            metadata={"upload_marker": marker},
        )

    def recover(
        self,
        job: UploadJob,
        upload_identity: str,
    ) -> UploadResult | None:
        plan = self.plan(job, upload_identity)
        client = self._required_client()
        try:
            remote = client.find_video_by_upload_marker(
                str(plan.metadata["upload_marker"])
            )
        except YouTubeClientError as exc:
            raise _classified_error(exc) from exc
        if remote is None:
            return None
        return _result(job, upload_identity, remote, recovered=True)

    def upload(self, job: UploadJob, upload_identity: str) -> UploadResult:
        plan = self.plan(job, upload_identity)
        client = self._required_client()
        try:
            remote = client.upload_video(plan)
        except YouTubeClientError as exc:
            raise _classified_error(exc) from exc
        return _result(job, upload_identity, remote, recovered=False)

    def _required_client(self) -> YouTubeClient:
        if self._client is None:
            raise PermanentUploadError(
                "YouTube credentials and a client are required outside dry-run mode."
            )
        return self._client


class GoogleYouTubeClient:
    """Google API implementation with OAuth and marker-based recovery."""

    def __init__(
        self,
        service: Any,
        media_upload_factory: Callable[..., Any],
    ) -> None:
        self._service = service
        self._media_upload_factory = media_upload_factory

    @classmethod
    def from_oauth_config(
        cls,
        config: YouTubeUploadConfig,
    ) -> "GoogleYouTubeClient":
        """Build an authenticated client without importing Google in dry runs."""

        config.validate_for_oauth()
        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build
            from googleapiclient.http import MediaFileUpload
        except ImportError as exc:
            raise PermanentUploadError(
                'YouTube uploads require `pip install -e ".[youtube]"`.'
            ) from exc

        credentials = None
        try:
            if config.token_path.is_file():
                credentials = Credentials.from_authorized_user_file(
                    str(config.token_path),
                    list(config.scopes),
                )
            if (
                credentials is not None
                and credentials.expired
                and credentials.refresh_token
            ):
                credentials.refresh(Request())
            if credentials is None or not credentials.valid:
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(config.client_secrets_path),
                    list(config.scopes),
                )
                credentials = flow.run_local_server(port=0)
            _write_oauth_token(config.token_path, credentials.to_json())
            service = build(
                "youtube",
                "v3",
                credentials=credentials,
                cache_discovery=False,
            )
        except OSError as exc:
            raise PermanentUploadError(
                f"YouTube OAuth files could not be accessed: {exc}"
            ) from exc
        except Exception as exc:
            raise RetryableUploadError(f"YouTube OAuth failed: {exc}") from exc
        return cls(service, MediaFileUpload)

    def find_video_by_upload_marker(
        self,
        upload_marker: str,
    ) -> YouTubeRemoteVideo | None:
        try:
            channels = self._execute(
                self._service.channels().list(part="contentDetails", mine=True)
            ).get("items", [])
            if not channels:
                raise YouTubeClientError(
                    "Authenticated YouTube account has no channel.",
                    retryable=False,
                )
            uploads = channels[0]["contentDetails"]["relatedPlaylists"]["uploads"]
            page_token = None
            while True:
                page = self._execute(
                    self._service.playlistItems().list(
                        part="contentDetails",
                        playlistId=uploads,
                        maxResults=50,
                        pageToken=page_token,
                    )
                )
                video_ids = [
                    item["contentDetails"]["videoId"]
                    for item in page.get("items", [])
                ]
                if video_ids:
                    videos = self._execute(
                        self._service.videos().list(
                            part="snippet,status",
                            id=",".join(video_ids),
                        )
                    )
                    for item in videos.get("items", []):
                        if upload_marker in item.get("snippet", {}).get("tags", []):
                            return YouTubeRemoteVideo(
                                str(item["id"]),
                                item.get("status", {}).get("privacyStatus"),
                            )
                page_token = page.get("nextPageToken")
                if not page_token:
                    return None
        except YouTubeClientError:
            raise
        except Exception as exc:
            raise _client_error(exc) from exc

    def upload_video(self, plan: UploadPlan) -> YouTubeRemoteVideo:
        try:
            media = self._media_upload_factory(
                str(plan.rendered_clip_path),
                chunksize=-1,
                resumable=True,
            )
            request = self._service.videos().insert(
                part="snippet,status",
                body={
                    "snippet": {
                        "title": plan.title,
                        "description": plan.description,
                        "tags": list(plan.tags),
                    },
                    "status": {"privacyStatus": plan.privacy_status},
                },
                media_body=media,
            )
            response = None
            while response is None:
                _, response = request.next_chunk()
            video_id = response.get("id")
            if not isinstance(video_id, str) or not video_id:
                raise YouTubeClientError(
                    "YouTube upload returned no video ID.",
                    retryable=True,
                )
            return YouTubeRemoteVideo(video_id, plan.privacy_status)
        except YouTubeClientError:
            raise
        except Exception as exc:
            raise _client_error(exc) from exc

    @staticmethod
    def _execute(request: Any) -> dict[str, Any]:
        response = request.execute()
        if not isinstance(response, dict):
            raise YouTubeClientError(
                "YouTube API returned a malformed response.",
                retryable=True,
            )
        return response


def _result(
    job: UploadJob,
    upload_identity: str,
    remote: YouTubeRemoteVideo,
    *,
    recovered: bool,
) -> UploadResult:
    if not isinstance(remote.video_id, str) or not remote.video_id:
        raise PermanentUploadError("YouTube client returned an invalid video ID.")
    return UploadResult(
        upload_identity=upload_identity,
        rendered_clip_identity=job.rendered_clip_identity,
        rendered_clip_path=job.rendered_clip_path,
        destination=YOUTUBE_DESTINATION,
        status=UploadStatus.COMPLETED,
        remote_id=remote.video_id,
        remote_url=YOUTUBE_VIDEO_URL.format(video_id=remote.video_id),
        recovered=recovered,
        metadata={
            "privacy_status": remote.privacy_status or job.visibility or "private"
        },
    )


def _upload_marker(upload_identity: str) -> str:
    return f"aitoclip-upload-{upload_identity.rsplit(':', 1)[-1]}"


def _classified_error(error: YouTubeClientError):
    error_type = RetryableUploadError if error.retryable else PermanentUploadError
    return error_type(str(error))


def _client_error(error: Exception) -> YouTubeClientError:
    status = getattr(getattr(error, "resp", None), "status", None)
    if isinstance(status, int):
        return YouTubeClientError(
            f"YouTube API request failed with HTTP {status}: {error}",
            retryable=status in _RETRYABLE_HTTP_STATUSES,
        )
    return YouTubeClientError(
        f"YouTube API request failed: {error}",
        retryable=isinstance(error, (ConnectionError, TimeoutError, OSError)),
    )


def _write_oauth_token(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    try:
        os.chmod(temporary, 0o600)
    except OSError:
        pass
    os.replace(temporary, path)
