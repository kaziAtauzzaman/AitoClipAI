"""Optional post-render upload queue for the operator interface."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import threading
from typing import TYPE_CHECKING, Callable, Protocol, Sequence

from core import UploadJob, UploadResult
from facebook_auth_contracts import (
    FacebookAuthenticationIssue,
    FacebookCredentialResolver,
    FacebookCredentialState,
)
from operator_pipeline import RenderedClipOutput

if TYPE_CHECKING:
    from uploading.facebook_config import FacebookUploadSettings


YOUTUBE_DESTINATION = "youtube"
FACEBOOK_DESTINATION = "facebook"
SUPPORTED_DESTINATIONS = (YOUTUBE_DESTINATION, FACEBOOK_DESTINATION)


class UploadQueueBusyError(RuntimeError):
    """Raised when another operator upload queue is still active."""


class UploadEventKind(str, Enum):
    """One safe UI-visible upload attempt transition."""

    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class UploadQueueEvent:
    """Progress for one clip/platform upload attempt."""

    kind: UploadEventKind
    destination: str
    clip_index: int
    clip_count: int
    error_type: str | None = None
    result: UploadResult | None = None
    authentication_state: FacebookCredentialState | None = None

    @property
    def platform_label(self) -> str:
        return "YouTube" if self.destination == YOUTUBE_DESTINATION else "Facebook"

    @property
    def message(self) -> str:
        if self.kind is UploadEventKind.STARTED:
            return (
                f"Uploading clip {self.clip_index} of {self.clip_count} "
                f"to {self.platform_label}..."
            )
        if self.kind is UploadEventKind.COMPLETED:
            return f"{self.platform_label} upload complete."
        if self.authentication_state is not None:
            return (
                f"Facebook uploads stopped: "
                f"{self.authentication_state.label}."
            )
        return (
            f"{self.platform_label} upload failed for clip {self.clip_index} "
            f"({self.error_type or 'UploadError'}). Continuing."
        )


@dataclass(frozen=True, slots=True)
class UploadAttempt:
    """Terminal outcome for one queued clip/platform pair."""

    destination: str
    rendered_clip_identity: str
    completed: bool
    result: UploadResult | None = None
    error_type: str | None = None


@dataclass(frozen=True, slots=True)
class UploadQueueSummary:
    """Aggregate terminal upload outcome displayed after the queue drains."""

    attempts: tuple[UploadAttempt, ...]

    @property
    def total(self) -> int:
        return len(self.attempts)

    @property
    def completed(self) -> int:
        return sum(attempt.completed for attempt in self.attempts)

    @property
    def failed(self) -> int:
        return self.total - self.completed


class UploadExecutor(Protocol):
    """Existing UploadService surface consumed by the operator queue."""

    def execute(self, job: UploadJob, *, dry_run: bool = False) -> UploadResult:
        """Execute one existing ledger-owned upload."""


@dataclass(frozen=True, slots=True)
class UploadRuntime:
    """One configured existing UploadService and platform-safe defaults."""

    service: UploadExecutor
    visibility: str
    facebook_page_id: str | None = None


class UploadRuntimeFactory(Protocol):
    """Lazily configure only a user-selected destination."""

    def __call__(self, destination: str) -> UploadRuntime:
        """Return an authenticated runtime for one destination."""


class FacebookCredentialResolverFactory(Protocol):
    """Construct the configured Facebook token provider on demand."""

    def __call__(
        self,
        settings: "FacebookUploadSettings",
    ) -> FacebookCredentialResolver:
        """Return a resolver for the configured Page."""


class ProductionUploadRuntimeFactory:
    """Compose existing adapters, credentials, and ledger without changing them."""

    def __init__(
        self,
        *,
        youtube_config_path: Path = Path("config") / "youtube-upload.json",
        facebook_config_path: Path = Path("config") / "facebook-upload.json",
        facebook_credential_resolver_factory: (
            FacebookCredentialResolverFactory | None
        ) = None,
    ) -> None:
        self._youtube_config_path = Path(youtube_config_path)
        self._facebook_config_path = Path(facebook_config_path)
        self._facebook_credential_resolver_factory = (
            facebook_credential_resolver_factory
            or _production_facebook_credential_resolver
        )

    def __call__(self, destination: str) -> UploadRuntime:
        if destination == YOUTUBE_DESTINATION:
            return self._youtube_runtime()
        if destination == FACEBOOK_DESTINATION:
            return self._facebook_runtime()
        raise ValueError(f"Unsupported upload destination: {destination!r}.")

    def _youtube_runtime(self) -> UploadRuntime:
        from uploading import (
            GoogleYouTubeClient,
            JsonUploadLedger,
            UploadService,
            YouTubeUploadAdapter,
            YouTubeUploadConfig,
        )

        config = YouTubeUploadConfig.from_sources(
            config_path=(
                self._youtube_config_path
                if self._youtube_config_path.is_file()
                else None
            )
        )
        client = GoogleYouTubeClient.from_oauth_config(config)
        return UploadRuntime(
            service=UploadService(
                JsonUploadLedger(config.ledger_path),
                [YouTubeUploadAdapter(client)],
            ),
            visibility="private",
        )

    def _facebook_runtime(self) -> UploadRuntime:
        from uploading import (
            FacebookGraphClient,
            FacebookUploadAdapter,
            FacebookUploadSettings,
            JsonUploadLedger,
            UploadService,
        )

        settings = FacebookUploadSettings.from_sources(
            config_path=(
                self._facebook_config_path
                if self._facebook_config_path.is_file()
                else None
            )
        )
        token = self._facebook_credential_resolver_factory(settings).resolve()
        config = settings.with_page_access_token(token)
        client = FacebookGraphClient.from_config(config)
        return UploadRuntime(
            service=UploadService(
                JsonUploadLedger(config.ledger_path),
                [FacebookUploadAdapter(client)],
            ),
            visibility="unpublished",
            facebook_page_id=config.page_id,
        )


class OperatorUploadQueue:
    """Convert rendered outputs into existing UploadJobs and drain every attempt."""

    def __init__(
        self,
        runtime_factory: UploadRuntimeFactory | None = None,
    ) -> None:
        self._runtime_factory = (
            runtime_factory or ProductionUploadRuntimeFactory()
        )

    def run(
        self,
        rendered_clips: Sequence[RenderedClipOutput],
        destinations: Sequence[str],
        *,
        on_event: Callable[[UploadQueueEvent], None],
    ) -> UploadQueueSummary:
        clips = tuple(rendered_clips)
        selected = _selected_destinations(destinations)
        attempts: list[UploadAttempt] = []
        if not clips:
            return UploadQueueSummary(())
        for destination in selected:
            try:
                runtime = self._runtime_factory(destination)
            except FacebookAuthenticationIssue as exc:
                attempts.extend(
                    UploadAttempt(
                        destination,
                        clip.identity,
                        False,
                        error_type=type(exc).__name__,
                    )
                    for clip in clips
                )
                on_event(
                    UploadQueueEvent(
                        UploadEventKind.FAILED,
                        destination,
                        1,
                        len(clips),
                        error_type=type(exc).__name__,
                        authentication_state=exc.state,
                    )
                )
                continue
            except Exception as exc:
                for index, clip in enumerate(clips, start=1):
                    on_event(
                        UploadQueueEvent(
                            UploadEventKind.STARTED,
                            destination,
                            index,
                            len(clips),
                        )
                    )
                    attempts.append(
                        UploadAttempt(
                            destination,
                            clip.identity,
                            False,
                            error_type=type(exc).__name__,
                        )
                    )
                    on_event(
                        UploadQueueEvent(
                            UploadEventKind.FAILED,
                            destination,
                            index,
                            len(clips),
                            error_type=type(exc).__name__,
                        )
                    )
                continue
            for index, clip in enumerate(clips, start=1):
                on_event(
                    UploadQueueEvent(
                        UploadEventKind.STARTED,
                        destination,
                        index,
                        len(clips),
                    )
                )
                try:
                    job = _upload_job(clip, destination, runtime)
                    result = runtime.service.execute(job, dry_run=False)
                except FacebookAuthenticationIssue as exc:
                    attempts.append(
                        UploadAttempt(
                            destination,
                            clip.identity,
                            False,
                            error_type=type(exc).__name__,
                        )
                    )
                    attempts.extend(
                        UploadAttempt(
                            destination,
                            remaining.identity,
                            False,
                            error_type=type(exc).__name__,
                        )
                        for remaining in clips[index:]
                    )
                    on_event(
                        UploadQueueEvent(
                            UploadEventKind.FAILED,
                            destination,
                            index,
                            len(clips),
                            error_type=type(exc).__name__,
                            authentication_state=exc.state,
                        )
                    )
                    break
                except Exception as exc:
                    attempts.append(
                        UploadAttempt(
                            destination,
                            clip.identity,
                            False,
                            error_type=type(exc).__name__,
                        )
                    )
                    on_event(
                        UploadQueueEvent(
                            UploadEventKind.FAILED,
                            destination,
                            index,
                            len(clips),
                            error_type=type(exc).__name__,
                        )
                    )
                    continue
                attempts.append(
                    UploadAttempt(
                        destination,
                        clip.identity,
                        True,
                        result=result,
                    )
                )
                on_event(
                    UploadQueueEvent(
                        UploadEventKind.COMPLETED,
                        destination,
                        index,
                        len(clips),
                        result=result,
                    )
                )
        return UploadQueueSummary(tuple(attempts))


class OperatorUploadController:
    """Run one optional upload queue without blocking Tkinter."""

    def __init__(self, queue: OperatorUploadQueue | None = None) -> None:
        self._queue = queue or OperatorUploadQueue()
        self._lock = threading.Lock()
        self._active_thread: threading.Thread | None = None

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._active_thread is not None

    def start(
        self,
        rendered_clips: Sequence[RenderedClipOutput],
        destinations: Sequence[str],
        *,
        on_event: Callable[[UploadQueueEvent], None],
        on_complete: Callable[[UploadQueueSummary], None],
        on_failure: Callable[[str], None],
    ) -> threading.Thread:
        clips = tuple(rendered_clips)
        selected = _selected_destinations(destinations)
        with self._lock:
            if self._active_thread is not None:
                raise UploadQueueBusyError("An upload queue is already active.")
            thread = threading.Thread(
                target=self._execute,
                args=(
                    clips,
                    selected,
                    on_event,
                    on_complete,
                    on_failure,
                ),
                name="aitoclip-operator-upload",
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
        rendered_clips: tuple[RenderedClipOutput, ...],
        destinations: tuple[str, ...],
        on_event: Callable[[UploadQueueEvent], None],
        on_complete: Callable[[UploadQueueSummary], None],
        on_failure: Callable[[str], None],
    ) -> None:
        try:
            summary = self._queue.run(
                rendered_clips,
                destinations,
                on_event=on_event,
            )
            on_complete(summary)
        except BaseException as exc:
            on_failure(type(exc).__name__)
        finally:
            with self._lock:
                if self._active_thread is threading.current_thread():
                    self._active_thread = None


def _selected_destinations(destinations: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(
        dict.fromkeys(str(value).strip().lower() for value in destinations)
    )
    unsupported = [value for value in selected if value not in SUPPORTED_DESTINATIONS]
    if unsupported:
        raise ValueError(f"Unsupported upload destination: {unsupported[0]!r}.")
    return tuple(value for value in SUPPORTED_DESTINATIONS if value in selected)


def _production_facebook_credential_resolver(
    settings: "FacebookUploadSettings",
) -> FacebookCredentialResolver:
    from uploading.facebook_credentials import (
        create_facebook_credential_resolver,
    )

    return create_facebook_credential_resolver(settings)


def _upload_job(
    clip: RenderedClipOutput,
    destination: str,
    runtime: UploadRuntime,
) -> UploadJob:
    metadata: dict[str, object] = {}
    if destination == FACEBOOK_DESTINATION:
        if not runtime.facebook_page_id:
            raise ValueError("Facebook upload runtime has no Page identity.")
        metadata["facebook_page_id"] = runtime.facebook_page_id
    return UploadJob(
        rendered_clip_path=clip.path,
        rendered_clip_identity=clip.identity,
        destination=destination,
        title=clip.title,
        description=clip.description,
        visibility=runtime.visibility,
        metadata=metadata,
    )
