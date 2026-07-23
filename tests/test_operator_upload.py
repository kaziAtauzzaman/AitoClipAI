from pathlib import Path
import socket
import threading

from core import UploadResult, UploadStatus
from operator_pipeline import RenderedClipOutput
from operator_upload import (
    FACEBOOK_DESTINATION,
    YOUTUBE_DESTINATION,
    OperatorUploadController,
    OperatorUploadQueue,
    UploadEventKind,
    UploadRuntime,
)


class _FakeUploadService:
    def __init__(self, failing_identities=()) -> None:
        self.failing_identities = set(failing_identities)
        self.calls = []
        self.thread_id = None

    def execute(self, job, *, dry_run=False):
        self.thread_id = threading.get_ident()
        self.calls.append((job, dry_run))
        if job.rendered_clip_identity in self.failing_identities:
            raise RuntimeError("synthetic upload failure")
        return UploadResult(
            upload_identity=f"{job.destination}:{job.rendered_clip_identity}",
            rendered_clip_identity=job.rendered_clip_identity,
            rendered_clip_path=job.rendered_clip_path,
            destination=job.destination,
            status=UploadStatus.COMPLETED,
            remote_id=f"remote-{len(self.calls)}",
        )


class _FakeRuntimeFactory:
    def __init__(self, services) -> None:
        self.services = services
        self.calls = []

    def __call__(self, destination):
        self.calls.append(destination)
        if destination == YOUTUBE_DESTINATION:
            return UploadRuntime(self.services[destination], "private")
        return UploadRuntime(
            self.services[destination],
            "unpublished",
            facebook_page_id="123456",
        )


def _rendered_clips(tmp_path: Path, count: int = 2):
    clips = []
    for number in range(1, count + 1):
        path = tmp_path / f"clip-{number}.mp4"
        path.write_bytes(f"rendered-{number}".encode())
        clips.append(
            RenderedClipOutput(
                path=path,
                identity=f"render:session:identity-{number}",
                title=f"Clip {number}",
                description=f"Description {number}",
            )
        )
    return tuple(clips)


def test_unchecked_destinations_never_construct_or_call_upload_service(
    tmp_path: Path,
) -> None:
    service = _FakeUploadService()
    factory = _FakeRuntimeFactory({YOUTUBE_DESTINATION: service})
    events = []

    summary = OperatorUploadQueue(factory).run(
        _rendered_clips(tmp_path),
        (),
        on_event=events.append,
    )

    assert summary.total == 0
    assert summary.completed == 0
    assert summary.failed == 0
    assert factory.calls == []
    assert service.calls == []
    assert events == []


def test_enabled_youtube_uploads_every_rendered_clip_without_network(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def reject_network(*args, **kwargs):
        raise AssertionError("operator upload test attempted network access")

    monkeypatch.setattr(socket, "socket", reject_network)
    service = _FakeUploadService()
    factory = _FakeRuntimeFactory({YOUTUBE_DESTINATION: service})
    clips = _rendered_clips(tmp_path)
    events = []

    summary = OperatorUploadQueue(factory).run(
        clips,
        (YOUTUBE_DESTINATION,),
        on_event=events.append,
    )

    assert summary.total == 2
    assert summary.completed == 2
    assert summary.failed == 0
    assert factory.calls == [YOUTUBE_DESTINATION]
    assert [call[0].rendered_clip_path for call in service.calls] == [
        clip.path for clip in clips
    ]
    assert [call[0].rendered_clip_identity for call in service.calls] == [
        clip.identity for clip in clips
    ]
    assert all(call[0].visibility == "private" for call in service.calls)
    assert all(call[1] is False for call in service.calls)
    assert [event.message for event in events] == [
        "Uploading clip 1 of 2 to YouTube...",
        "YouTube upload complete.",
        "Uploading clip 2 of 2 to YouTube...",
        "YouTube upload complete.",
    ]


def test_both_destinations_preserve_clip_multiplicity_and_platform_metadata(
    tmp_path: Path,
) -> None:
    youtube = _FakeUploadService()
    facebook = _FakeUploadService()
    factory = _FakeRuntimeFactory(
        {
            YOUTUBE_DESTINATION: youtube,
            FACEBOOK_DESTINATION: facebook,
        }
    )

    summary = OperatorUploadQueue(factory).run(
        _rendered_clips(tmp_path),
        (FACEBOOK_DESTINATION, YOUTUBE_DESTINATION),
        on_event=lambda event: None,
    )

    assert summary.total == 4
    assert summary.completed == 4
    assert factory.calls == [YOUTUBE_DESTINATION, FACEBOOK_DESTINATION]
    assert len(youtube.calls) == 2
    assert len(facebook.calls) == 2
    assert all(
        call[0].metadata == {"facebook_page_id": "123456"}
        and call[0].visibility == "unpublished"
        for call in facebook.calls
    )


def test_partial_upload_failure_continues_remaining_clips(
    tmp_path: Path,
) -> None:
    clips = _rendered_clips(tmp_path, count=3)
    service = _FakeUploadService(failing_identities={clips[1].identity})
    factory = _FakeRuntimeFactory({YOUTUBE_DESTINATION: service})
    events = []

    summary = OperatorUploadQueue(factory).run(
        clips,
        (YOUTUBE_DESTINATION,),
        on_event=events.append,
    )

    assert len(service.calls) == 3
    assert summary.total == 3
    assert summary.completed == 2
    assert summary.failed == 1
    assert [attempt.completed for attempt in summary.attempts] == [
        True,
        False,
        True,
    ]
    assert [event.kind for event in events].count(UploadEventKind.FAILED) == 1
    assert events[-1].kind is UploadEventKind.COMPLETED
    assert "Continuing." in events[3].message


def test_platform_setup_failure_does_not_block_other_destination(
    tmp_path: Path,
) -> None:
    facebook = _FakeUploadService()

    class PartiallyFailingFactory:
        def __call__(self, destination):
            if destination == YOUTUBE_DESTINATION:
                raise RuntimeError("synthetic configuration failure")
            return UploadRuntime(
                facebook,
                "unpublished",
                facebook_page_id="123456",
            )

    summary = OperatorUploadQueue(PartiallyFailingFactory()).run(
        _rendered_clips(tmp_path),
        (YOUTUBE_DESTINATION, FACEBOOK_DESTINATION),
        on_event=lambda event: None,
    )

    assert summary.total == 4
    assert summary.completed == 2
    assert summary.failed == 2
    assert len(facebook.calls) == 2


def test_upload_controller_runs_queue_off_the_calling_thread(
    tmp_path: Path,
) -> None:
    service = _FakeUploadService()
    factory = _FakeRuntimeFactory({YOUTUBE_DESTINATION: service})
    controller = OperatorUploadController(OperatorUploadQueue(factory))
    events = []
    summaries = []
    failures = []
    caller_thread = threading.get_ident()

    thread = controller.start(
        _rendered_clips(tmp_path, count=1),
        (YOUTUBE_DESTINATION,),
        on_event=events.append,
        on_complete=summaries.append,
        on_failure=failures.append,
    )
    thread.join(timeout=5)

    assert thread.is_alive() is False
    assert service.thread_id != caller_thread
    assert len(summaries) == 1
    assert summaries[0].completed == 1
    assert failures == []
    assert controller.is_running is False
