from pathlib import Path
import json
import socket
import threading
from types import SimpleNamespace

import pytest

from core import ClipCandidate, RenderJob
from operator_pipeline import (
    OperatorPipelineController,
    PipelineExecution,
    RenderedClipOutput,
    PipelineStage,
    RunInProgressError,
    SourceKind,
    SourceValidationError,
    _IncrementalGeneratorWithStage,
    _IncrementalRendererWithStage,
    _OneShotStageReporter,
    _SelectorWithStage,
    _rendered_clip_outputs,
    _stage_local_source,
    validate_source,
)


class _SuccessfulRunner:
    def __init__(self) -> None:
        self.calls = []
        self.thread_ident = None

    def run(self, source, run_directory, emit_stage):
        self.calls.append((source, run_directory))
        self.thread_ident = threading.get_ident()
        for stage in (
            PipelineStage.EXTRACTING_AUDIO,
            PipelineStage.OBSERVING,
            PipelineStage.GENERATING_CANDIDATES,
            PipelineStage.SELECTING_CANDIDATES,
            PipelineStage.RENDERING_CLIPS,
        ):
            emit_stage(stage)
        output = run_directory / "clips"
        output.mkdir()
        rendered_clips = tuple(
            RenderedClipOutput(
                path=output / f"clip-{identity}.mp4",
                identity=f"render:test:identity-{identity}",
                title=f"Clip {identity}",
                description=f"Rendered clip {identity}",
            )
            for identity in range(1, 4)
        )
        return PipelineExecution(
            output_directory=output,
            rendered_clip_count=3,
            report_path=run_directory / "reports" / "validation-report.json",
            rendered_clips=rendered_clips,
        )


class _BlockingRunner(_SuccessfulRunner):
    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def run(self, source, run_directory, emit_stage):
        self.entered.set()
        if not self.release.wait(timeout=5):
            raise TimeoutError("test did not release fake pipeline")
        return super().run(source, run_directory, emit_stage)


class _FailingRunner:
    def run(self, source, run_directory, emit_stage):
        emit_stage(PipelineStage.OBSERVING)
        raise RuntimeError("observer failed; access_token=example-sensitive-value")


class _InterruptedRunner:
    def run(self, source, run_directory, emit_stage):
        raise KeyboardInterrupt("pipeline interrupted")


def _local_video(tmp_path: Path) -> Path:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"not-real-media")
    return video


def test_source_validation_accepts_youtube_and_supported_local_file(
    tmp_path: Path,
) -> None:
    remote = validate_source(" https://www.youtube.com/watch?v=example ")
    local_path = _local_video(tmp_path)
    local = validate_source(local_path)

    assert remote.kind is SourceKind.YOUTUBE
    assert remote.value == "https://www.youtube.com/watch?v=example"
    assert local.kind is SourceKind.LOCAL
    assert local.value == local_path.resolve()


@pytest.mark.parametrize(
    "source, message",
    [
        ("", "Enter a YouTube URL"),
        ("https://twitch.tv/example", "Only YouTube URLs"),
        ("https://user:password@youtube.com/watch?v=x", "must not contain"),
        ("ftp://youtube.com/video", "Only HTTP or HTTPS"),
    ],
)
def test_source_validation_rejects_unsupported_remote_input(
    source: str,
    message: str,
) -> None:
    with pytest.raises(SourceValidationError, match=message):
        validate_source(source)


def test_source_validation_rejects_missing_and_unsupported_local_input(
    tmp_path: Path,
) -> None:
    with pytest.raises(SourceValidationError, match="does not exist"):
        validate_source(tmp_path / "missing.mp4")
    unsupported = tmp_path / "notes.txt"
    unsupported.write_text("not media", encoding="utf-8")
    with pytest.raises(SourceValidationError, match="Unsupported local media type"):
        validate_source(unsupported)


def test_local_source_staging_keeps_authoritative_duration_in_run(
    tmp_path: Path,
) -> None:
    source = _local_video(tmp_path)

    class FakeValidator:
        def probe_source(self, path):
            assert path.parent == tmp_path / "run" / "source"
            return SimpleNamespace(duration_seconds=12.75, streams=[])

    staged = _stage_local_source(
        source,
        tmp_path / "run" / "source",
        FakeValidator(),
    )

    assert staged.read_bytes() == source.read_bytes()
    metadata = staged.with_name(f"{staged.name}.metadata.json")
    assert json.loads(metadata.read_text(encoding="utf-8")) == {
        "duration": 12.75
    }
    assert not source.with_name(f"{source.name}.metadata.json").exists()


def test_stage_adapters_preserve_incremental_production_contract(
    tmp_path: Path,
) -> None:
    from candidate_generation import CandidateGenerator
    from candidate_scoring import CandidateScorer
    from candidate_selection import CandidateSelector
    from clip_rendering import ClipRenderer, ClipRendererConfig
    from pipeline import IncrementalPrerecordedCoordinator

    reported = []
    reporter = _OneShotStageReporter(reported.append)
    coordinator = IncrementalPrerecordedCoordinator(
        _IncrementalGeneratorWithStage(CandidateGenerator(), reporter),
        CandidateScorer(),
        _SelectorWithStage(CandidateSelector(), reporter),
        _IncrementalRendererWithStage(
            ClipRenderer(ClipRendererConfig(output_dir=tmp_path / "clips")),
            reporter,
        ),
    )

    assert coordinator.lifecycle.value == "new"
    assert reported == []


def test_production_render_jobs_become_stable_upload_outputs(
    tmp_path: Path,
) -> None:
    candidate = ClipCandidate(
        source_video_path=tmp_path / "source.mp4",
        start_seconds=1.0,
        end_seconds=9.0,
        reason="Strong speech and audio peak.",
    )
    report = SimpleNamespace(
        session_id="operator-session",
        render_jobs=[
            RenderJob(
                candidate=candidate,
                output_path=tmp_path / "clips" / "one.mp4",
                metadata={"incremental_render_identity": 1},
            ),
            RenderJob(
                candidate=candidate,
                output_path=tmp_path / "clips" / "two.mp4",
                metadata={"incremental_render_identity": 2},
            ),
        ],
    )

    outputs = _rendered_clip_outputs(report)

    assert [item.path.name for item in outputs] == ["one.mp4", "two.mp4"]
    assert [item.identity for item in outputs] == [
        "render:operator-session:identity-1",
        "render:operator-session:identity-2",
    ]
    assert outputs[0].title == "source — clip 1"
    assert outputs[0].description == "Strong speech and audio peak."


def test_controller_runs_fake_pipeline_once_in_background(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def reject_network(*args, **kwargs):
        raise AssertionError("operator controller attempted network access")

    monkeypatch.setattr(socket, "socket", reject_network)
    video = _local_video(tmp_path)
    run_directory = tmp_path / "operator-run"
    runner = _SuccessfulRunner()
    controller = OperatorPipelineController(
        runner,
        run_directory_factory=lambda: run_directory,
    )
    stages = []
    successes = []
    failures = []
    caller_thread = threading.get_ident()

    thread = controller.start(
        video,
        on_stage=stages.append,
        on_success=successes.append,
        on_failure=failures.append,
    )
    thread.join(timeout=5)

    assert thread.is_alive() is False
    assert runner.thread_ident != caller_thread
    assert len(runner.calls) == 1
    assert failures == []
    assert len(successes) == 1
    assert successes[0].run_directory == run_directory.resolve()
    assert successes[0].output_directory == run_directory.resolve() / "clips"
    assert successes[0].rendered_clip_count == 3
    assert len(successes[0].rendered_clips) == 3
    assert stages == [
        PipelineStage.RESOLVING_SOURCE,
        PipelineStage.READING_MEDIA,
        PipelineStage.EXTRACTING_AUDIO,
        PipelineStage.OBSERVING,
        PipelineStage.GENERATING_CANDIDATES,
        PipelineStage.SELECTING_CANDIDATES,
        PipelineStage.RENDERING_CLIPS,
        PipelineStage.COMPLETED,
    ]
    assert controller.is_running is False
    assert (run_directory / "run.log").read_text(encoding="utf-8").endswith(
        "rendered_clip_count=3\n"
    )


def test_controller_prevents_simultaneous_runs(tmp_path: Path) -> None:
    video = _local_video(tmp_path)
    runner = _BlockingRunner()
    controller = OperatorPipelineController(
        runner,
        run_directory_factory=lambda: tmp_path / "operator-run",
    )
    stages = []
    successes = []
    failures = []
    thread = controller.start(
        video,
        on_stage=stages.append,
        on_success=successes.append,
        on_failure=failures.append,
    )
    assert runner.entered.wait(timeout=5)

    with pytest.raises(RunInProgressError, match="already active"):
        controller.start(
            video,
            on_stage=stages.append,
            on_success=successes.append,
            on_failure=failures.append,
        )

    runner.release.set()
    thread.join(timeout=5)
    assert thread.is_alive() is False
    assert len(runner.calls) == 1
    assert len(successes) == 1
    assert failures == []


def test_controller_reports_failure_and_writes_redacted_traceback(
    tmp_path: Path,
) -> None:
    video = _local_video(tmp_path)
    run_directory = tmp_path / "failed-run"
    controller = OperatorPipelineController(
        _FailingRunner(),
        run_directory_factory=lambda: run_directory,
    )
    stages = []
    successes = []
    failures = []

    thread = controller.start(
        video,
        on_stage=stages.append,
        on_success=successes.append,
        on_failure=failures.append,
    )
    thread.join(timeout=5)

    assert thread.is_alive() is False
    assert successes == []
    assert len(failures) == 1
    assert failures[0].run_directory == run_directory.resolve()
    assert failures[0].log_path == run_directory.resolve() / "run.log"
    assert failures[0].message == "RuntimeError: observer failed; " \
        "access_token=[redacted]"
    assert stages[-1] is PipelineStage.FAILED
    log = (run_directory / "run.log").read_text(encoding="utf-8")
    assert "Traceback (most recent call last)" in log
    assert "access_token=[redacted]" in log
    assert "example-sensitive-value" not in log
    assert controller.is_running is False


def test_controller_contains_worker_interruption_at_ui_boundary(
    tmp_path: Path,
) -> None:
    video = _local_video(tmp_path)
    controller = OperatorPipelineController(
        _InterruptedRunner(),
        run_directory_factory=lambda: tmp_path / "interrupted-run",
    )
    failures = []
    stages = []
    successes = []

    thread = controller.start(
        video,
        on_stage=stages.append,
        on_success=successes.append,
        on_failure=failures.append,
    )
    thread.join(timeout=5)

    assert thread.is_alive() is False
    assert successes == []
    assert stages[-1] is PipelineStage.FAILED
    assert len(failures) == 1
    assert failures[0].message == "KeyboardInterrupt: pipeline interrupted"
    assert "KeyboardInterrupt: pipeline interrupted" in (
        tmp_path / "interrupted-run" / "run.log"
    ).read_text(encoding="utf-8")
    assert controller.is_running is False
