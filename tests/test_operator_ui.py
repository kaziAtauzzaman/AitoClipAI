import ast
from pathlib import Path
import socket

import pytest

import operator_ui
from operator_ui import (
    DEMO_STAGES,
    INITIAL_PROOF_ROWS,
    REPOSITORY_ROOT,
    START_BUTTON_LABEL,
    START_PROCESSING_MESSAGE,
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


def test_demo_sequence_and_disconnected_processing_contract() -> None:
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
    assert START_BUTTON_LABEL == "Prototype 1 Pipeline · Disabled for Demo"
    assert "not connected" in START_PROCESSING_MESSAGE.lower()

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
    assert "state=\"disabled\"" in source


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
