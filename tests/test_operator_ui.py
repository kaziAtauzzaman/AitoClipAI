import ast
from pathlib import Path
import socket
import subprocess
import sys
import threading

import pytest

from operator_cleanup import (
    CLEANUP_POLICY_OPTIONS,
    CleanupPolicy,
    CleanupResult,
    CleanupStatus,
)
from facebook_auth_contracts import (
    FacebookCredentialDiagnostic,
    FacebookCredentialState,
)
import operator_ui
from operator_facebook_auth import OperatorFacebookPreflightController
from operator_pipeline import (
    PipelineRunFailure,
    PipelineRunSuccess,
    RenderedClipOutput,
)
from operator_upload import (
    YOUTUBE_DESTINATION,
    UploadAttempt,
    UploadEventKind,
    UploadQueueEvent,
    UploadQueueSummary,
)
from operator_ui import (
    DEMO_STAGES,
    INITIAL_PROOF_ROWS,
    REPOSITORY_ROOT,
    START_BUTTON_LABEL,
    UPLOAD_OPTION_LABEL,
    DemoDataError,
    load_validation06,
)
from uploading import FacebookAuthenticationRequired


def test_demo_loads_validation06_without_network_or_writes(monkeypatch) -> None:
    def reject_network(*args, **kwargs):
        raise AssertionError("Demo Mode attempted to create a network socket.")

    monkeypatch.setattr(socket, "socket", reject_network)
    validation_directory = (
        REPOSITORY_ROOT / operator_ui.VALIDATION06_RELATIVE_PATH
    )
    summary = validation_directory / operator_ui.VALIDATION06_SUMMARY_NAME
    before = (summary.stat().st_size, summary.stat().st_mtime_ns)

    proof = load_validation06()

    assert proof.observations == 18_939
    assert proof.generated == 176
    assert proof.passing == 174
    assert proof.selected == 174
    assert proof.rendered == 174
    assert proof.rendered_before_eof == 172
    assert proof.youtube_upload_validated is True
    assert proof.facebook_upload_validated is True
    assert proof.clips_directory == (validation_directory / "clips").resolve()
    assert len(list(proof.clips_directory.glob("*.mp4"))) == 174
    assert (summary.stat().st_size, summary.stat().st_mtime_ns) == before


def test_demo_sequence_and_operator_integration_contract() -> None:
    assert DEMO_STAGES == (
        "Source resolved",
        "Observations loaded",
        "Candidates generated",
        "Candidates selected",
        "Clips rendered",
        "YouTube uploader validated",
        "Facebook uploader validated",
    )
    assert INITIAL_PROOF_ROWS == (
        ("observations", "18,939 observations"),
        ("generated", "176 generated"),
        ("passing", "174 passing"),
        ("selected", "174 selected"),
        ("rendered", "174 rendered"),
        ("before_eof", "172 rendered before EOF"),
        ("youtube", "✓ YouTube upload validated"),
        ("facebook", "✓ Facebook upload validated"),
    )
    assert START_BUTTON_LABEL == "Start Processing"
    assert UPLOAD_OPTION_LABEL == (
        "Upload after processing — optional and off by default."
    )

    source = Path(operator_ui.__file__).read_text(encoding="utf-8")
    imported_roots = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported_roots.update(
                alias.name.split(".", 1)[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.split(".", 1)[0])
    assert imported_roots.isdisjoint(
        {"http", "pipeline", "requests", "uploading", "urllib"}
    )
    assert "operator_pipeline" in imported_roots
    assert "operator_upload" in imported_roots
    assert "operator_cleanup" in imported_roots
    assert "self.youtube_enabled = tk.BooleanVar(value=False)" in source
    assert "self.facebook_enabled = tk.BooleanVar(value=False)" in source
    assert "value=CleanupPolicy.KEEP_RENDERED_CLIPS.label" in source
    assert "state=\"disabled\"" in source


def test_ui_import_does_not_load_pipeline() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import operator_ui; "
                "assert 'pipeline' not in sys.modules; "
                "assert 'uploading' not in sys.modules"
            ),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={**operator_ui.os.environ, "PYTHONPATH": str(REPOSITORY_ROOT / "src")},
    )

    assert completed.returncode == 0, completed.stderr


class _FakeVariable:
    def __init__(self, value=None) -> None:
        self.value = value

    def set(self, value) -> None:
        self.value = value

    def get(self):
        return self.value


class _FakeButton:
    def __init__(self) -> None:
        self.states: list[tuple[str, ...]] = []

    def state(self, values) -> None:
        self.states.append(tuple(values))


class _FakeUploadController:
    def __init__(self) -> None:
        self.is_running = False
        self.starts = []

    def start(self, clips, destinations, **callbacks):
        self.starts.append((tuple(clips), tuple(destinations), callbacks))


class _FakePipelineController:
    def __init__(self) -> None:
        self.is_running = False
        self.starts = []

    def start(self, source, **callbacks):
        self.starts.append((source, callbacks))


class _FakeFacebookPreflightController:
    def __init__(self) -> None:
        self.is_running = False
        self.starts = []

    def start(self, **callbacks):
        self.starts.append(callbacks)


class _FakeCleanupController:
    def __init__(self) -> None:
        self.is_running = False
        self.starts = []

    def start(self, request, **callbacks):
        self.starts.append((request, callbacks))


class _FakeFacebookCredentials:
    def __init__(
        self,
        state=FacebookCredentialState.NOT_CONFIGURED,
        failure=None,
    ) -> None:
        self.state = state
        self.failure = failure
        self.replacements = []
        self.validations = 0

    def current_state(self):
        return self.state

    def replace(self, token):
        self.replacements.append(token)
        if self.failure is not None:
            raise self.failure
        self.state = FacebookCredentialState.CONNECTED
        return self.state

    def validate(self):
        self.validations += 1
        if self.failure is not None:
            raise self.failure
        self.state = FacebookCredentialState.CONNECTED
        return self.state


def _terminal_state_app():
    app = operator_ui.AitoClipOperatorApp.__new__(
        operator_ui.AitoClipOperatorApp
    )
    app.current_stage = _FakeVariable()
    app.progress = _FakeVariable()
    app.open_button = _FakeButton()
    app.start_button = _FakeButton()
    app.demo_button = _FakeButton()
    app.youtube_checkbox = _FakeButton()
    app.facebook_checkbox = _FakeButton()
    app.facebook_credential_button = _FakeButton()
    app.cleanup_policy_control = _FakeButton()
    app.source = _FakeVariable(
        "https://www.youtube.com/watch?v=abcdefghijk"
    )
    app.youtube_enabled = _FakeVariable(False)
    app.facebook_enabled = _FakeVariable(False)
    app.cleanup_policy = _FakeVariable(CleanupPolicy.KEEP_RENDERED_CLIPS.label)
    app.facebook_credential_state = _FakeVariable()
    app.facebook_credential_action = _FakeVariable()
    app._facebook_credentials = _FakeFacebookCredentials()
    app._facebook_preflight_controller = _FakeFacebookPreflightController()
    app._controller = _FakePipelineController()
    app._upload_controller = _FakeUploadController()
    app._cleanup_controller = _FakeCleanupController()
    app._requested_destinations = ()
    app._requested_cleanup_policy = CleanupPolicy.KEEP_RENDERED_CLIPS
    app._completed_run = None
    app._upload_thread = None
    app._pending_source = None
    app._workflow_terminal_stage = "Completed"
    app._pipeline_events = operator_ui.queue.SimpleQueue()
    app._output_directory = None
    messages = []
    app._clear_log = messages.clear
    app._append_log = lambda message, tag: messages.append((message, tag))
    return app, messages


def test_workflow_disables_and_reenables_upload_checkboxes() -> None:
    app, _ = _terminal_state_app()

    app._set_workflow_controls(active=True)
    app._set_workflow_controls(active=False)

    assert app.youtube_checkbox.states == [
        ("disabled",),
        ("!disabled",),
    ]
    assert app.facebook_checkbox.states == [
        ("disabled",),
        ("!disabled",),
    ]
    assert app.facebook_credential_button.states == [
        ("disabled",),
        ("!disabled",),
    ]
    assert app.cleanup_policy_control.states == [
        ("disabled",),
        ("!disabled",),
    ]
    assert CLEANUP_POLICY_OPTIONS == (
        "Keep rendered clips",
        "Delete rendered clips after all selected uploads complete",
    )


def test_stored_valid_facebook_credential_starts_processing_after_preflight() -> None:
    app, messages = _terminal_state_app()
    app.facebook_enabled.set(True)

    app.start_processing()

    assert app.current_stage.value == operator_ui.FACEBOOK_PREFLIGHT_STAGE
    assert app._controller.starts == []
    assert len(app._facebook_preflight_controller.starts) == 1
    callbacks = app._facebook_preflight_controller.starts[0]
    callbacks["on_complete"](FacebookCredentialState.CONNECTED)
    event, state = app._pipeline_events.get_nowait()
    assert event == "facebook_preflight"

    app._show_facebook_preflight(state)

    assert len(app._controller.starts) == 1
    assert app._requested_destinations == ("facebook",)
    assert app.facebook_credential_state.value == "Facebook Connected"
    assert ("Facebook credential validated.", "success") in messages
    assert app._upload_controller.starts == []
    assert app._cleanup_controller.starts == []


@pytest.mark.parametrize(
    ("state", "expected_message"),
    [
        (
            FacebookCredentialState.REAUTHORIZATION_REQUIRED,
            "Facebook authorization expired or was revoked.",
        ),
        (
            FacebookCredentialState.NOT_CONFIGURED,
            "Facebook is not configured.",
        ),
        (
            FacebookCredentialState.WRONG_PAGE,
            "Facebook credential belongs to a different Page.",
        ),
        (
            FacebookCredentialState.PERMISSION_ERROR,
            "Facebook credential lacks the required Page publishing permission.",
        ),
        (
            FacebookCredentialState.UNAVAILABLE,
            "Facebook credential validation is unavailable.",
        ),
    ],
)
def test_facebook_preflight_failure_aborts_before_processing(
    state,
    expected_message,
) -> None:
    app, messages = _terminal_state_app()
    app.facebook_enabled.set(True)

    app.start_processing()
    callbacks = app._facebook_preflight_controller.starts[0]
    callbacks["on_failure"](state)
    event, queued_state = app._pipeline_events.get_nowait()
    assert event == "facebook_preflight"

    app._show_facebook_preflight(queued_state)

    assert app._controller.starts == []
    assert app._upload_controller.starts == []
    assert app._cleanup_controller.starts == []
    assert app._pending_source is None
    assert app.current_stage.value == "Failed"
    assert app.facebook_credential_state.value == state.label
    assert any(
        message.startswith(expected_message) and tag == "error"
        for message, tag in messages
    )


def test_facebook_unchecked_skips_facebook_preflight() -> None:
    app, _ = _terminal_state_app()

    app.start_processing()

    assert app._facebook_preflight_controller.starts == []
    assert len(app._controller.starts) == 1
    assert app._requested_destinations == ()


def test_youtube_only_processing_remains_unchanged() -> None:
    app, _ = _terminal_state_app()
    app.youtube_enabled.set(True)

    app.start_processing()

    assert app._facebook_preflight_controller.starts == []
    assert len(app._controller.starts) == 1
    assert app._requested_destinations == (YOUTUBE_DESTINATION,)


def test_facebook_preflight_keeps_ui_thread_responsive() -> None:
    app, messages = _terminal_state_app()
    entered = threading.Event()
    release = threading.Event()

    class BlockingCredentialManager:
        def validate(self):
            entered.set()
            assert release.wait(timeout=2)
            return FacebookCredentialState.CONNECTED

    controller = OperatorFacebookPreflightController(
        BlockingCredentialManager()
    )
    app._facebook_preflight_controller = controller
    app.facebook_enabled.set(True)

    app.start_processing()

    assert entered.wait(timeout=1)
    assert controller.is_running is True
    assert app._controller.starts == []
    app.start_processing()
    assert (
        "A processing or upload workflow is already active.",
        "error",
    ) in messages

    release.set()
    event, state = app._pipeline_events.get(timeout=1)
    assert event == "facebook_preflight"
    app._show_facebook_preflight(state)

    assert controller.is_running is False
    assert len(app._controller.starts) == 1


def test_facebook_preflight_does_not_expose_authentication_details() -> None:
    app, messages = _terminal_state_app()
    secret = "token-sentinel-that-must-not-leak"

    class ExpiredCredentialManager:
        def validate(self):
            raise FacebookAuthenticationRequired(
                FacebookCredentialState.REAUTHORIZATION_REQUIRED,
                message=secret,
                diagnostic=FacebookCredentialDiagnostic(
                    stage="graph_validation",
                    http_status=400,
                    graph_error_code=190,
                    graph_error_type="OAuthException",
                    graph_error_message=f"expired {secret}",
                ),
            )

    app._facebook_preflight_controller = OperatorFacebookPreflightController(
        ExpiredCredentialManager()
    )
    app.facebook_enabled.set(True)

    app.start_processing()
    event, state = app._pipeline_events.get(timeout=1)
    assert event == "facebook_preflight"
    assert state is FacebookCredentialState.REAUTHORIZATION_REQUIRED
    app._show_facebook_preflight(state)

    visible = "\n".join(message for message, _ in messages)
    assert secret not in visible
    assert secret not in repr(state)
    assert app._controller.starts == []


def test_demo_mode_never_starts_cleanup_or_deletes_cached_clips(
    monkeypatch,
    tmp_path: Path,
) -> None:
    app, _ = _terminal_state_app()
    clips_directory = tmp_path / "validation06" / "clips"
    clips_directory.mkdir(parents=True)
    cached_clip = clips_directory / "cached.mp4"
    cached_clip.write_bytes(b"cached-validation-artifact")

    class Root:
        def __init__(self) -> None:
            self.callbacks = []

        def after(self, *args) -> None:
            self.callbacks.append(args)

    proof = operator_ui.Validation06Proof(
        observations=18_939,
        generated=176,
        passing=174,
        selected=174,
        rendered=174,
        rendered_before_eof=172,
        youtube_upload_validated=True,
        facebook_upload_validated=True,
        source="cached source",
        clips_directory=clips_directory,
    )
    monkeypatch.setattr(operator_ui, "load_validation06", lambda: proof)
    monkeypatch.setattr(
        socket,
        "socket",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Demo Mode attempted network access")
        ),
    )
    app.root = Root()
    app.source = _FakeVariable()
    app._metric_values = {
        key: _FakeVariable() for key, _ in INITIAL_PROOF_ROWS
    }
    app._demo_run = 0
    app.last_demo_error = None
    app._clear_log = lambda: None

    app.run_demo()

    assert cached_clip.read_bytes() == b"cached-validation-artifact"
    assert app._cleanup_controller.starts == []
    assert app._upload_controller.starts == []
    assert app.current_stage.value == DEMO_STAGES[0]


def test_facebook_credential_dialog_is_masked_and_saves_after_validation(
    monkeypatch,
) -> None:
    app, messages = _terminal_state_app()
    app.root = object()
    prompts = []

    def fake_askstring(title, prompt, **kwargs):
        prompts.append((title, prompt, kwargs))
        return "secret-page-token"

    monkeypatch.setattr(operator_ui.simpledialog, "askstring", fake_askstring)

    app.replace_facebook_credential()

    assert app._facebook_credentials.replacements == ["secret-page-token"]
    assert prompts[0][2]["show"] == "*"
    assert prompts[0][2]["parent"] is app.root
    assert app.facebook_credential_state.value == "Facebook Connected"
    assert (
        app.facebook_credential_action.value
        == "Replace Facebook Credential"
    )
    assert all("secret-page-token" not in message for message, _ in messages)


def test_invalid_facebook_credential_updates_safe_state_without_leakage(
    monkeypatch,
) -> None:
    app, messages = _terminal_state_app()
    app.root = object()
    app._facebook_credentials = _FakeFacebookCredentials(
        failure=FacebookAuthenticationRequired(
            FacebookCredentialState.WRONG_PAGE
        )
    )
    monkeypatch.setattr(
        operator_ui.simpledialog,
        "askstring",
        lambda *args, **kwargs: "wrong-page-secret",
    )

    app.replace_facebook_credential()

    assert app.facebook_credential_state.value == "Facebook Wrong Page"
    assert all("wrong-page-secret" not in message for message, _ in messages)


def test_facebook_configuration_failure_shows_only_sanitized_stage(
    monkeypatch,
    tmp_path: Path,
) -> None:
    app, messages = _terminal_state_app()
    app.root = object()
    diagnostic_path = tmp_path / "facebook-diagnostic.json"
    app._facebook_credentials = _FakeFacebookCredentials(
        failure=FacebookAuthenticationRequired(
            FacebookCredentialState.UNAVAILABLE,
            diagnostic=FacebookCredentialDiagnostic(
                stage="graph_validation",
                http_status=400,
                graph_error_code=100,
                graph_error_type="OAuthException",
                graph_error_message="Detailed server message",
            ),
            diagnostic_log_path=diagnostic_path,
        )
    )
    monkeypatch.setattr(
        operator_ui.simpledialog,
        "askstring",
        lambda *args, **kwargs: "secret-page-token",
    )

    app.replace_facebook_credential()

    assert (
        "Facebook Unavailable during graph validation.",
        "error",
    ) in messages
    assert (
        f"Sanitized diagnostic: {diagnostic_path}",
        "muted",
    ) in messages
    rendered_messages = "\n".join(message for message, _ in messages)
    assert "secret-page-token" not in rendered_messages
    assert "Detailed server message" not in rendered_messages


def test_upload_auth_event_updates_facebook_reauthorization_state() -> None:
    app, messages = _terminal_state_app()
    event = UploadQueueEvent(
        UploadEventKind.FAILED,
        "facebook",
        1,
        2,
        error_type="FacebookAuthenticationRequired",
        authentication_state=FacebookCredentialState.REAUTHORIZATION_REQUIRED,
    )

    app._show_upload_event(event)

    assert (
        app.facebook_credential_state.value
        == "Facebook Reauthorization Required"
    )
    assert (
        app.facebook_credential_action.value
        == "Replace Facebook Credential"
    )
    assert ("Facebook uploads stopped: Facebook Reauthorization Required.", "error") in messages


def test_successful_pipeline_completion_updates_ui_state(tmp_path: Path) -> None:
    app, messages = _terminal_state_app()
    output = tmp_path / "clips"
    result = PipelineRunSuccess(
        run_directory=tmp_path,
        log_path=tmp_path / "run.log",
        output_directory=output,
        rendered_clip_count=7,
    )

    app._show_pipeline_success(result)

    assert app.current_stage.value == "Completed"
    assert app.progress.value == 100
    assert app._output_directory == output
    assert app.open_button.states[-1] == ("!disabled",)
    assert ("Completed with 7 rendered clips.", "success") in messages
    assert ("No upload was requested.", "muted") in messages
    assert app._upload_controller.starts == []
    assert len(app._cleanup_controller.starts) == 1
    cleanup_request, callbacks = app._cleanup_controller.starts[0]
    assert cleanup_request.selected_destinations == ()
    assert cleanup_request.policy is CleanupPolicy.KEEP_RENDERED_CLIPS

    callbacks["on_complete"](
        CleanupResult(
            cleanup_policy=CleanupPolicy.KEEP_RENDERED_CLIPS.value,
            eligible_clip_count=0,
            deleted_clip_count=0,
            retained_clip_count=0,
            bytes_deleted=0,
            cleanup_status=CleanupStatus.RETAINED_NO_DESTINATIONS.value,
        )
    )
    event_name, cleanup_result = app._pipeline_events.get_nowait()
    assert event_name == "cleanup_result"
    app._show_cleanup_result(cleanup_result)

    assert app.start_button.states[-1] == ("!disabled",)
    assert app.demo_button.states[-1] == ("!disabled",)
    assert app.youtube_checkbox.states[-1] == ("!disabled",)
    assert app.facebook_checkbox.states[-1] == ("!disabled",)


def test_checked_destination_queues_rendered_outputs_after_success(
    tmp_path: Path,
) -> None:
    app, messages = _terminal_state_app()
    app._requested_destinations = (YOUTUBE_DESTINATION,)
    clip = RenderedClipOutput(
        path=tmp_path / "clips" / "one.mp4",
        identity="render:session:identity-1",
        title="Clip 1",
        description="Description 1",
    )
    result = PipelineRunSuccess(
        run_directory=tmp_path,
        log_path=tmp_path / "run.log",
        output_directory=tmp_path / "clips",
        rendered_clip_count=1,
        rendered_clips=(clip,),
    )

    app._show_pipeline_success(result)

    assert app.current_stage.value == "Uploading"
    assert app.progress.value == 0
    assert len(app._upload_controller.starts) == 1
    clips, destinations, callbacks = app._upload_controller.starts[0]
    assert clips == (clip,)
    assert destinations == (YOUTUBE_DESTINATION,)
    assert app.start_button.states == []
    assert ("Queued 1 rendered clips for upload.", "muted") in messages

    callbacks["on_event"](
        UploadQueueEvent(
            UploadEventKind.STARTED,
            YOUTUBE_DESTINATION,
            1,
            1,
        )
    )
    callbacks["on_complete"](
        UploadQueueSummary(
            (
                UploadAttempt(
                    YOUTUBE_DESTINATION,
                    clip.identity,
                    True,
                ),
            )
        )
    )

    event_name, event = app._pipeline_events.get_nowait()
    summary_name, summary = app._pipeline_events.get_nowait()
    assert event_name == "upload_event"
    assert event.message == "Uploading clip 1 of 1 to YouTube..."
    assert summary_name == "upload_summary"
    assert summary.completed == 1


def test_upload_summary_restores_controls_and_reports_partial_failure() -> None:
    app, messages = _terminal_state_app()
    summary = UploadQueueSummary(
        (
            UploadAttempt(YOUTUBE_DESTINATION, "render:1", True),
            UploadAttempt(
                YOUTUBE_DESTINATION,
                "render:2",
                False,
                error_type="RetryableUploadError",
            ),
        )
    )

    app._show_upload_summary(summary)

    assert app.current_stage.value == "Completed with upload failures"
    assert app.progress.value == 100
    assert ("Upload summary: 1 completed, 1 failed.", "error") in messages
    assert app.start_button.states[-1] == ("!disabled",)
    assert app.youtube_checkbox.states[-1] == ("!disabled",)


def test_successful_upload_summary_starts_selected_cleanup_policy(
    tmp_path: Path,
) -> None:
    app, messages = _terminal_state_app()
    app._requested_destinations = (YOUTUBE_DESTINATION,)
    app._requested_cleanup_policy = CleanupPolicy.DELETE_AFTER_SUCCESSFUL_UPLOADS
    clip = RenderedClipOutput(
        path=tmp_path / "clips" / "one.mp4",
        identity="render:session:identity-1",
        title="Clip 1",
        description="Description 1",
    )
    app._completed_run = PipelineRunSuccess(
        run_directory=tmp_path,
        log_path=tmp_path / "run.log",
        output_directory=tmp_path / "clips",
        rendered_clip_count=1,
        report_path=tmp_path / "reports" / "report.json",
        rendered_clips=(clip,),
    )
    summary = UploadQueueSummary(
        (
            UploadAttempt(
                YOUTUBE_DESTINATION,
                clip.identity,
                True,
            ),
        )
    )

    app._show_upload_summary(summary)

    assert len(app._cleanup_controller.starts) == 1
    request, callbacks = app._cleanup_controller.starts[0]
    assert request.rendered_clips == (clip,)
    assert request.expected_rendered_clip_count == 1
    assert request.selected_destinations == (YOUTUBE_DESTINATION,)
    assert request.upload_summary is summary
    assert request.policy is CleanupPolicy.DELETE_AFTER_SUCCESSFUL_UPLOADS
    assert app.start_button.states == []

    callbacks["on_complete"](
        CleanupResult(
            cleanup_policy=CleanupPolicy.DELETE_AFTER_SUCCESSFUL_UPLOADS.value,
            eligible_clip_count=1,
            deleted_clip_count=1,
            retained_clip_count=0,
            bytes_deleted=2 * 1024 * 1024,
            cleanup_status=CleanupStatus.COMPLETED.value,
        )
    )
    event_name, cleanup_result = app._pipeline_events.get_nowait()
    assert event_name == "cleanup_result"
    app._show_cleanup_result(cleanup_result)

    assert app.current_stage.value == "Completed"
    assert ("Cleaning up uploaded clips...", "muted") in messages
    assert (
        "Deleted 1 rendered clips. Freed approximately 2.00 MB.",
        "success",
    ) in messages
    assert app.start_button.states[-1] == ("!disabled",)


def test_cleanup_waits_until_upload_worker_thread_has_exited(
    tmp_path: Path,
) -> None:
    app, _ = _terminal_state_app()
    app._requested_destinations = (YOUTUBE_DESTINATION,)
    clip = RenderedClipOutput(
        path=tmp_path / "clips" / "one.mp4",
        identity="render:session:identity-1",
        title="Clip 1",
        description="Description 1",
    )
    run = PipelineRunSuccess(
        run_directory=tmp_path,
        log_path=tmp_path / "run.log",
        output_directory=tmp_path / "clips",
        rendered_clip_count=1,
        report_path=tmp_path / "reports" / "report.json",
        rendered_clips=(clip,),
    )
    app._completed_run = run
    summary = UploadQueueSummary(
        (UploadAttempt(YOUTUBE_DESTINATION, clip.identity, True),)
    )

    class UploadThread:
        alive = True
        joins = 0

        def is_alive(self):
            return self.alive

        def join(self, timeout=None):
            assert timeout == 0
            self.joins += 1

    class Root:
        def __init__(self):
            self.callbacks = []

        def after(self, delay, callback, *args):
            self.callbacks.append((delay, callback, args))

    upload_thread = UploadThread()
    root = Root()
    app.root = root
    app._upload_thread = upload_thread

    app._show_upload_summary(summary)

    assert app._cleanup_controller.starts == []
    assert len(root.callbacks) == 1
    assert root.callbacks[0][0] == 25

    upload_thread.alive = False
    _, callback, args = root.callbacks.pop()
    callback(*args)

    assert upload_thread.joins == 1
    assert app._upload_thread is None
    assert len(app._cleanup_controller.starts) == 1
    cleanup_request, _ = app._cleanup_controller.starts[0]
    assert cleanup_request.upload_summary is summary


def test_failed_pipeline_completion_updates_ui_state(tmp_path: Path) -> None:
    app, messages = _terminal_state_app()
    failure = PipelineRunFailure(
        message="RuntimeError: observer failed",
        run_directory=tmp_path,
        log_path=tmp_path / "run.log",
    )

    app._show_pipeline_failure(failure)

    assert app.current_stage.value == "Failed"
    assert app.progress.value == 0
    assert app._output_directory == tmp_path
    assert app.open_button.states[-1] == ("!disabled",)
    assert app.start_button.states[-1] == ("!disabled",)
    assert app.demo_button.states[-1] == ("!disabled",)
    assert app.youtube_checkbox.states[-1] == ("!disabled",)
    assert app.facebook_checkbox.states[-1] == ("!disabled",)
    assert ("RuntimeError: observer failed", "error") in messages
    assert (f"Detailed log: {tmp_path / 'run.log'}", "muted") in messages


def test_output_folder_dispatches_without_opening_real_window(
    monkeypatch,
    tmp_path: Path,
) -> None:
    opened = []
    if operator_ui.os.name == "nt":
        monkeypatch.setattr(
            operator_ui.os,
            "startfile",
            lambda path: opened.append(path),
        )
    else:
        monkeypatch.setattr(
            operator_ui.subprocess,
            "Popen",
            lambda command: opened.append(command),
        )

    operator_ui.open_output_folder(tmp_path)

    assert str(tmp_path.resolve()) in str(opened[0])


def test_demo_rejects_missing_validation_artifacts(tmp_path: Path) -> None:
    with pytest.raises(DemoDataError, match="summary is unavailable"):
        load_validation06(tmp_path)
