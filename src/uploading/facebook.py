"""Facebook Page video adapter and Graph API client."""

from dataclasses import dataclass
import mimetypes
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

from core import UploadJob, UploadResult, UploadStatus
from facebook_auth_contracts import FacebookCredentialState
from uploading.contracts import UploadPlan
from uploading.errors import (
    FacebookAuthenticationRequired,
    PermanentUploadError,
    RetryableUploadError,
)
from uploading.facebook_config import FacebookUploadConfig
from uploading.identity import normalize_destination


FACEBOOK_DESTINATION = "facebook"
FACEBOOK_VIDEO_URL = "https://www.facebook.com/{page_id}/videos/{video_id}/"
_RETRYABLE_HTTP_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
_RETRYABLE_GRAPH_CODES = frozenset({1, 2, 4, 17, 32, 613})
_AUTHENTICATION_GRAPH_CODES = frozenset({102, 190})
_PERMISSION_GRAPH_CODES = frozenset({10, 200, 299})
_RECOVERY_PAGE_SIZE = 25
_PUBLISHING_STATES = {
    "public": ("published", True),
    "published": ("published", True),
    "draft": ("unpublished", False),
    "unpublished": ("unpublished", False),
}


@dataclass(frozen=True, slots=True)
class FacebookRemoteVideo:
    """Minimum Facebook video state needed for completion and recovery."""

    video_id: str
    published: bool | None = None
    permalink_url: str | None = None


class FacebookClientError(Exception):
    """Client-level Graph API failure translated by the adapter."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool,
        graph_code: int | None = None,
        http_status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.graph_code = graph_code
        self.http_status = http_status


class FacebookClient(Protocol):
    """Narrow Page-video boundary used by fake and Graph API clients."""

    page_id: str

    def find_video_by_upload_marker(
        self,
        upload_marker: str,
    ) -> FacebookRemoteVideo | None: ...

    def upload_video(self, plan: UploadPlan) -> FacebookRemoteVideo: ...


class FacebookUploadAdapter:
    """Translate neutral upload jobs into Facebook Page video requests."""

    destination = FACEBOOK_DESTINATION

    def __init__(self, client: FacebookClient | None = None) -> None:
        self._client = client

    def plan(self, job: UploadJob, upload_identity: str) -> UploadPlan:
        if normalize_destination(job.destination) != self.destination:
            raise PermanentUploadError("Facebook adapter received another destination.")
        page_id = _job_page_id(job)
        publishing_state, published = _publishing_state(job.visibility)
        if job.tags:
            raise PermanentUploadError(
                "Facebook Page video tags are not supported in this milestone; "
                "include hashtags in the caption instead."
            )
        marker = _upload_marker(upload_identity)
        return UploadPlan(
            upload_identity=upload_identity,
            destination=self.destination,
            rendered_clip_identity=job.rendered_clip_identity,
            rendered_clip_path=job.rendered_clip_path,
            title=job.title.strip(),
            description=_caption_with_marker(job.description, marker),
            privacy_status=publishing_state,
            metadata={
                "facebook_page_id": page_id,
                "published": published,
                "upload_marker": marker,
            },
        )

    def recover(
        self,
        job: UploadJob,
        upload_identity: str,
    ) -> UploadResult | None:
        plan = self.plan(job, upload_identity)
        client = self._required_client(str(plan.metadata["facebook_page_id"]))
        try:
            remote = client.find_video_by_upload_marker(
                str(plan.metadata["upload_marker"])
            )
        except FacebookClientError as exc:
            raise _classified_error(exc) from exc
        if remote is None:
            return None
        return _result(job, plan, remote, recovered=True)

    def upload(self, job: UploadJob, upload_identity: str) -> UploadResult:
        plan = self.plan(job, upload_identity)
        client = self._required_client(str(plan.metadata["facebook_page_id"]))
        try:
            remote = client.upload_video(plan)
        except FacebookClientError as exc:
            raise _classified_error(exc) from exc
        return _result(job, plan, remote, recovered=False)

    def _required_client(self, page_id: str) -> FacebookClient:
        if self._client is None:
            raise PermanentUploadError(
                "A Facebook Page access token and client are required outside "
                "dry-run mode."
            )
        if self._client.page_id != page_id:
            raise PermanentUploadError(
                "Facebook client Page ID does not match the upload job."
            )
        return self._client


class FacebookGraphClient:
    """Requests-backed implementation of the Facebook Page video API."""

    def __init__(
        self,
        session: Any,
        config: FacebookUploadConfig,
        *,
        retryable_transport_errors: tuple[type[BaseException], ...] = (
            ConnectionError,
            TimeoutError,
            OSError,
        ),
        permanent_transport_errors: tuple[type[BaseException], ...] = (),
    ) -> None:
        config.validate_for_upload()
        self._session = session
        self._token = config.page_access_token
        self._version = config.graph_api_version
        self.page_id = config.page_id
        self._retryable_transport_errors = retryable_transport_errors
        self._permanent_transport_errors = permanent_transport_errors

    @classmethod
    def from_config(cls, config: FacebookUploadConfig) -> "FacebookGraphClient":
        """Construct the production HTTP client without affecting dry runs."""

        config.validate_for_upload()
        try:
            import requests
        except ImportError as exc:
            raise PermanentUploadError(
                'Facebook uploads require `pip install -e ".[facebook]"`.'
            ) from exc
        return cls(
            requests.Session(),
            config,
            retryable_transport_errors=(
                requests.Timeout,
                requests.ConnectionError,
            ),
            permanent_transport_errors=(requests.RequestException,),
        )

    def find_video_by_upload_marker(
        self,
        upload_marker: str,
    ) -> FacebookRemoteVideo | None:
        url = self._graph_url(f"{self.page_id}/videos")
        params: dict[str, object] | None = {
            "access_token": self._token,
            "fields": "id,description,permalink_url,published",
            "limit": _RECOVERY_PAGE_SIZE,
        }
        while url is not None:
            response = self._request("get", url, params=params, timeout=(10, 60))
            payload = _response_payload(response)
            data = payload.get("data", [])
            if not isinstance(data, list):
                raise FacebookClientError(
                    "Facebook video listing returned malformed data.",
                    retryable=True,
                )
            for item in data:
                if not isinstance(item, dict):
                    continue
                description = item.get("description", "")
                video_id = item.get("id")
                if (
                    isinstance(description, str)
                    and upload_marker in description
                    and isinstance(video_id, str)
                    and video_id
                ):
                    published = item.get("published")
                    return FacebookRemoteVideo(
                        video_id=video_id,
                        published=published if isinstance(published, bool) else None,
                        permalink_url=_optional_string(item.get("permalink_url")),
                    )
            paging = payload.get("paging", {})
            next_url = paging.get("next") if isinstance(paging, dict) else None
            url = _validated_next_url(next_url)
            params = None
        return None

    def upload_video(self, plan: UploadPlan) -> FacebookRemoteVideo:
        page_id = str(plan.metadata.get("facebook_page_id", ""))
        if page_id != self.page_id:
            raise FacebookClientError(
                "Facebook upload plan targets another Page.",
                retryable=False,
            )
        published = plan.metadata.get("published")
        if not isinstance(published, bool):
            raise FacebookClientError(
                "Facebook upload plan has no publication flag.",
                retryable=False,
            )
        path = Path(plan.rendered_clip_path)
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        try:
            with path.open("rb") as stream:
                response = self._request(
                    "post",
                    self._video_graph_url(f"{self.page_id}/videos"),
                    data={
                        "access_token": self._token,
                        "title": plan.title,
                        "description": plan.description,
                        "published": "true" if published else "false",
                    },
                    files={"source": (path.name, stream, content_type)},
                    timeout=(10, 3600),
                )
        except OSError as exc:
            raise FacebookClientError(
                f"Facebook upload source could not be read: {exc}",
                retryable=False,
            ) from exc
        payload = _response_payload(response)
        video_id = payload.get("id")
        if not isinstance(video_id, str) or not video_id:
            raise FacebookClientError(
                "Facebook upload returned no video ID.",
                retryable=True,
            )
        return FacebookRemoteVideo(video_id=video_id, published=published)

    def _request(self, method: str, url: str, **kwargs: object) -> Any:
        try:
            return getattr(self._session, method)(url, **kwargs)
        except self._retryable_transport_errors as exc:
            raise FacebookClientError(
                "Facebook API transport failed.",
                retryable=True,
            ) from exc
        except self._permanent_transport_errors as exc:
            raise FacebookClientError(
                "Facebook API request could not be constructed.",
                retryable=False,
            ) from exc

    def _graph_url(self, path: str) -> str:
        return f"https://graph.facebook.com/{self._version}/{path}"

    def _video_graph_url(self, path: str) -> str:
        return f"https://graph-video.facebook.com/{self._version}/{path}"


def _job_page_id(job: UploadJob) -> str:
    value = job.metadata.get("facebook_page_id")
    if not isinstance(value, str) or not value.strip().isdigit():
        raise PermanentUploadError(
            "Facebook uploads require a numeric facebook_page_id in job metadata."
        )
    return value.strip()


def _publishing_state(value: str | None) -> tuple[str, bool]:
    normalized = (value or "unpublished").strip().lower()
    state = _PUBLISHING_STATES.get(normalized)
    if state is None:
        raise PermanentUploadError(
            "Facebook Page publishing state must be published or unpublished."
        )
    return state


def _caption_with_marker(caption: str | None, marker: str) -> str:
    value = (caption or "").rstrip()
    suffix = f"[{marker}]"
    return f"{value}\n\n{suffix}" if value else suffix


def _upload_marker(upload_identity: str) -> str:
    return f"aitoclip-upload-{upload_identity.rsplit(':', 1)[-1]}"


def _result(
    job: UploadJob,
    plan: UploadPlan,
    remote: FacebookRemoteVideo,
    *,
    recovered: bool,
) -> UploadResult:
    if not isinstance(remote.video_id, str) or not remote.video_id:
        raise PermanentUploadError("Facebook client returned an invalid video ID.")
    page_id = str(plan.metadata["facebook_page_id"])
    expected_published = bool(plan.metadata["published"])
    published = expected_published if remote.published is None else remote.published
    return UploadResult(
        upload_identity=plan.upload_identity,
        rendered_clip_identity=job.rendered_clip_identity,
        rendered_clip_path=job.rendered_clip_path,
        destination=FACEBOOK_DESTINATION,
        status=UploadStatus.COMPLETED,
        remote_id=remote.video_id,
        remote_url=(
            _permalink(remote.permalink_url)
            or FACEBOOK_VIDEO_URL.format(page_id=page_id, video_id=remote.video_id)
        ),
        recovered=recovered,
        metadata={
            "facebook_page_id": page_id,
            "publishing_state": "published" if published else "unpublished",
        },
    )


def _response_payload(response: Any) -> dict[str, Any]:
    status = getattr(response, "status_code", None)
    try:
        payload = response.json()
    except (TypeError, ValueError) as exc:
        raise FacebookClientError(
            "Facebook API returned a non-JSON response.",
            retryable=isinstance(status, int) and status >= 500,
        ) from exc
    if not isinstance(payload, dict):
        raise FacebookClientError(
            "Facebook API returned a malformed response.",
            retryable=True,
        )
    error = payload.get("error")
    if isinstance(error, dict):
        code = error.get("code")
        retryable = (
            bool(error.get("is_transient"))
            or code in _RETRYABLE_GRAPH_CODES
            or status in _RETRYABLE_HTTP_STATUSES
        )
        message = error.get("message")
        raise FacebookClientError(
            str(message) if message else "Facebook Graph API rejected the request.",
            retryable=retryable,
            graph_code=code if isinstance(code, int) else None,
            http_status=status if isinstance(status, int) else None,
        )
    if isinstance(status, int) and status >= 400:
        raise FacebookClientError(
            f"Facebook API request failed with HTTP {status}.",
            retryable=status in _RETRYABLE_HTTP_STATUSES,
            http_status=status,
        )
    return payload


def _validated_next_url(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise FacebookClientError(
            "Facebook paging URL is malformed.",
            retryable=True,
        )
    parsed = urlparse(value)
    if parsed.scheme != "https" or parsed.hostname != "graph.facebook.com":
        raise FacebookClientError(
            "Facebook paging URL targets an unexpected host.",
            retryable=False,
        )
    return value


def _permalink(value: str | None) -> str | None:
    if value is None:
        return None
    if value.startswith("https://"):
        return value
    if value.startswith("/"):
        return f"https://www.facebook.com{value}"
    return None


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _classified_error(error: FacebookClientError):
    if (
        error.graph_code in _AUTHENTICATION_GRAPH_CODES
        or error.http_status == 401
    ):
        return FacebookAuthenticationRequired(
            FacebookCredentialState.REAUTHORIZATION_REQUIRED
        )
    if (
        error.graph_code in _PERMISSION_GRAPH_CODES
        or error.http_status == 403
    ):
        return FacebookAuthenticationRequired(
            FacebookCredentialState.PERMISSION_ERROR
        )
    error_type = RetryableUploadError if error.retryable else PermanentUploadError
    return error_type(str(error))
