import ast
from pathlib import Path
import socket
import subprocess
import sys

import pytest

import operator_ui
from operator_pipeline import PipelineRunFailure, PipelineRunSuccess
from operator_ui import (
    DEMO_STAGES,
    INITIAL_PROOF_ROWS,
    REPOSITORY_ROOT,
    START_BUTTON_LABEL,
    UPLOAD_DISABLED_LABEL,
    DemoDataError,
    load_validation06,
)


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
    assert UPLOAD_DISABLED_LABEL == (
        "Upload after processing — not enabled in this milestone."
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
    assert "state=\"disabled\"" in source


def test_ui_import_does_not_load_pipeline() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import operator_ui; "
                "assert 'pipeline' not in sys.modules"
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
    def __init__(self) -> None:
        self.value = None

    def set(self, value) -> None:
        self.value = value


class _FakeButton:
    def __init__(self) -> None:
        self.states: list[tuple[str, ...]] = []

    def state(self, values) -> None:
        self.states.append(tuple(values))


def _terminal_state_app():
    app = operator_ui.AitoClipOperatorApp.__new__(
        operator_ui.AitoClipOperatorApp
    )
    app.current_stage = _FakeVariable()
    app.progress = _FakeVariable()
    app.open_button = _FakeButton()
    app.start_button = _FakeButton()
    app.demo_button = _FakeButton()
    app._output_directory = None
    messages = []
    app._append_log = lambda message, tag: messages.append((message, tag))
    return app, messages


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
    assert app.start_button.states[-1] == ("!disabled",)
    assert app.demo_button.states[-1] == ("!disabled",)
    assert ("Completed with 7 rendered clips.", "success") in messages
    assert ("No network upload was performed.", "muted") in messages


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
