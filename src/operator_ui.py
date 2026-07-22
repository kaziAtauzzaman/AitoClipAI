"""Tkinter operator interface for AitoClipAI."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tkinter as tk
from tkinter import scrolledtext, ttk
from typing import Sequence

from operator_pipeline import (
    OperatorPipelineController,
    PipelineRunFailure,
    PipelineRunSuccess,
    PipelineStage,
    RunInProgressError,
    SourceValidationError,
    validate_source,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VALIDATION06_RELATIVE_PATH = (
    Path("data")
    / "validation"
    / "youtube-UMjrTuMomlc-endurance-qsv-06-production-media"
)
VALIDATION06_SUMMARY_NAME = "validation06-summary.json"
RENDERED_BEFORE_EOF = 172

DEMO_STAGES = (
    "Source resolved",
    "Observations loaded",
    "Candidates generated",
    "Candidates selected",
    "Clips rendered",
    "YouTube uploader validated",
    "Facebook uploader validated",
)
START_BUTTON_LABEL = "Start Processing"
UPLOAD_DISABLED_LABEL = "Upload after processing — not enabled in this milestone."
INITIAL_PROOF_ROWS = (
    ("observations", "18,939 observations"),
    ("generated", "176 generated"),
    ("passing", "174 passing"),
    ("selected", "174 selected"),
    ("rendered", "174 rendered"),
    ("before_eof", "172 rendered before EOF"),
    ("youtube", "✓ YouTube upload validated"),
    ("facebook", "✓ Facebook upload validated"),
)


class DemoDataError(RuntimeError):
    """Raised when cached Validation 06 proof cannot be loaded safely."""


@dataclass(frozen=True, slots=True)
class Validation06Proof:
    """Small immutable view of the completed Validation 06 artifacts."""

    observations: int
    generated: int
    passing: int
    selected: int
    rendered: int
    rendered_before_eof: int
    youtube_upload_validated: bool
    facebook_upload_validated: bool
    source: str
    clips_directory: Path


def load_validation06(
    repository_root: Path = REPOSITORY_ROOT,
) -> Validation06Proof:
    """Load Validation 06 proof without invoking engine or uploader code."""

    validation_directory = Path(repository_root) / VALIDATION06_RELATIVE_PATH
    summary_path = validation_directory / VALIDATION06_SUMMARY_NAME
    clips_directory = validation_directory / "clips"

    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DemoDataError(f"Validation 06 summary is unavailable: {exc}") from exc
    if not isinstance(payload, dict):
        raise DemoDataError("Validation 06 summary must be a JSON object.")
    if payload.get("production_status") != "completed":
        raise DemoDataError("Validation 06 production run is not completed.")

    counts = payload.get("counts")
    if not isinstance(counts, dict):
        raise DemoDataError("Validation 06 summary has no count proof.")
    observations = _required_count(payload, "observation_count")
    generated = _required_count(counts, "generated")
    passing = _required_count(counts, "passing")
    selected = _required_count(counts, "selected")
    rendered = _required_count(counts, "rendered")

    expected = {
        "observation_count": 18_939,
        "generated": 176,
        "passing": 174,
        "selected": 174,
        "rendered": 174,
    }
    actual = {
        "observation_count": observations,
        "generated": generated,
        "passing": passing,
        "selected": selected,
        "rendered": rendered,
    }
    if actual != expected:
        raise DemoDataError(
            f"Validation 06 count proof differs from the approved result: {actual}."
        )
    if not clips_directory.is_dir():
        raise DemoDataError("Validation 06 clips directory is unavailable.")
    clip_count = sum(
        item.is_file() and item.suffix.lower() == ".mp4"
        for item in clips_directory.iterdir()
    )
    if clip_count != rendered:
        raise DemoDataError(
            f"Validation 06 has {clip_count} clips; expected {rendered}."
        )

    paths = payload.get("paths")
    source = "Validation 06 cached production media"
    if isinstance(paths, dict) and isinstance(paths.get("video"), str):
        source = str(paths["video"])
    return Validation06Proof(
        observations=observations,
        generated=generated,
        passing=passing,
        selected=selected,
        rendered=rendered,
        rendered_before_eof=RENDERED_BEFORE_EOF,
        youtube_upload_validated=True,
        facebook_upload_validated=True,
        source=source,
        clips_directory=clips_directory.resolve(),
    )


def open_output_folder(path: Path) -> None:
    """Open one existing output directory with the host file manager."""

    folder = Path(path).resolve(strict=True)
    if not folder.is_dir():
        raise NotADirectoryError(str(folder))
    if os.name == "nt":
        os.startfile(str(folder))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(folder)])
    else:
        subprocess.Popen(["xdg-open", str(folder)])


def _required_count(values: dict[str, object], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise DemoDataError(f"Validation 06 count {key!r} is invalid.")
    return value


class AitoClipOperatorApp:
    """Dark Tkinter interface for production processing and cached proof."""

    _BACKGROUND = "#0d0f10"
    _PANEL = "#171a1d"
    _FIELD = "#111416"
    _TEXT = "#f3f5f7"
    _MUTED = "#9aa3aa"
    _GREEN = "#55d187"
    _RED = "#f16f6f"
    _BORDER = "#2a3035"

    def __init__(
        self,
        root: tk.Tk,
        controller: OperatorPipelineController | None = None,
    ) -> None:
        self.root = root
        self.root.title("AitoClipAI — Operator")
        self.root.geometry("1080x720")
        self.root.minsize(900, 620)
        self.root.configure(background=self._BACKGROUND)

        self._controller = controller or OperatorPipelineController()
        self._pipeline_events: queue.SimpleQueue[tuple[str, object]] = (
            queue.SimpleQueue()
        )
        self.source = tk.StringVar()
        self.youtube_enabled = tk.BooleanVar(value=False)
        self.facebook_enabled = tk.BooleanVar(value=False)
        self.current_stage = tk.StringVar(value="Ready")
        self.progress = tk.DoubleVar(value=0.0)
        self._metric_values: dict[str, tk.StringVar] = {}
        self._output_directory: Path | None = None
        self._demo_run = 0
        self.last_demo_error: str | None = None

        self._configure_style()
        self._build_layout()
        self.root.after(75, self._drain_pipeline_events)

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Root.TFrame", background=self._BACKGROUND)
        style.configure("Panel.TFrame", background=self._PANEL)
        style.configure(
            "Title.TLabel",
            background=self._BACKGROUND,
            foreground=self._TEXT,
            font=("Consolas", 24, "bold"),
        )
        style.configure(
            "Tagline.TLabel",
            background=self._BACKGROUND,
            foreground=self._GREEN,
            font=("Consolas", 11),
        )
        style.configure(
            "Section.TLabel",
            background=self._PANEL,
            foreground=self._TEXT,
            font=("Consolas", 11, "bold"),
        )
        style.configure(
            "Body.TLabel",
            background=self._PANEL,
            foreground=self._MUTED,
            font=("Consolas", 9),
        )
        style.configure(
            "Proof.TLabel",
            background=self._PANEL,
            foreground=self._TEXT,
            font=("Consolas", 10),
        )
        style.configure(
            "Dark.TCheckbutton",
            background=self._PANEL,
            foreground=self._TEXT,
            font=("Consolas", 10),
            indicatorbackground=self._FIELD,
            indicatorforeground=self._GREEN,
        )
        style.map(
            "Dark.TCheckbutton",
            background=[("active", self._PANEL)],
            foreground=[("active", self._TEXT)],
        )
        style.configure(
            "Primary.TButton",
            background=self._GREEN,
            foreground="#08110c",
            borderwidth=0,
            padding=(14, 9),
            font=("Consolas", 9, "bold"),
        )
        style.map("Primary.TButton", background=[("active", "#72dfa0")])
        style.configure(
            "Secondary.TButton",
            background="#252b30",
            foreground=self._TEXT,
            borderwidth=0,
            padding=(14, 9),
            font=("Consolas", 9),
        )
        style.map("Secondary.TButton", background=[("active", "#343c42")])
        style.configure(
            "Demo.Horizontal.TProgressbar",
            troughcolor=self._FIELD,
            background=self._GREEN,
            bordercolor=self._FIELD,
            lightcolor=self._GREEN,
            darkcolor=self._GREEN,
        )

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        header = ttk.Frame(self.root, style="Root.TFrame", padding=(24, 20, 24, 12))
        header.grid(row=0, column=0, sticky="ew")
        ttk.Label(header, text="AitoClipAI", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Capture the next frame.",
            style="Tagline.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        body = ttk.Frame(self.root, style="Root.TFrame", padding=(24, 0, 24, 24))
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        workspace = ttk.Frame(body, style="Panel.TFrame", padding=18)
        workspace.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        workspace.columnconfigure(0, weight=1)
        workspace.rowconfigure(7, weight=1)

        ttk.Label(workspace, text="SOURCE", style="Section.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            workspace,
            text="YouTube URL or supported local media path",
            style="Body.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(3, 7))
        source_entry = tk.Entry(
            workspace,
            textvariable=self.source,
            background=self._FIELD,
            foreground=self._TEXT,
            insertbackground=self._GREEN,
            relief="flat",
            highlightthickness=1,
            highlightbackground=self._BORDER,
            highlightcolor=self._GREEN,
            font=("Consolas", 10),
        )
        source_entry.grid(row=2, column=0, sticky="ew", ipady=9)

        destinations = ttk.Frame(workspace, style="Panel.TFrame")
        destinations.grid(row=3, column=0, sticky="ew", pady=(18, 12))
        ttk.Label(destinations, text="DESTINATIONS", style="Section.TLabel").pack(
            side="left", padx=(0, 20)
        )
        ttk.Checkbutton(
            destinations,
            text="YouTube",
            variable=self.youtube_enabled,
            style="Dark.TCheckbutton",
            state="disabled",
        ).pack(side="left", padx=(0, 12))
        ttk.Checkbutton(
            destinations,
            text="Facebook",
            variable=self.facebook_enabled,
            style="Dark.TCheckbutton",
            state="disabled",
        ).pack(side="left", padx=(0, 18))
        ttk.Label(
            destinations,
            text=UPLOAD_DISABLED_LABEL,
            style="Body.TLabel",
        ).pack(side="left")

        controls = ttk.Frame(workspace, style="Panel.TFrame")
        controls.grid(row=4, column=0, sticky="ew", pady=(0, 18))
        self.start_button = ttk.Button(
            controls,
            text=START_BUTTON_LABEL,
            command=self.start_processing,
            style="Primary.TButton",
        )
        self.start_button.pack(side="left", padx=(0, 8))
        self.demo_button = ttk.Button(
            controls,
            text="Demo Mode",
            command=self.run_demo,
            style="Secondary.TButton",
        )
        self.demo_button.pack(side="left", padx=(0, 8))
        self.open_button = ttk.Button(
            controls,
            text="Open Output Folder",
            command=self.open_output,
            state="disabled",
            style="Secondary.TButton",
        )
        self.open_button.pack(side="left")

        status_row = ttk.Frame(workspace, style="Panel.TFrame")
        status_row.grid(row=5, column=0, sticky="ew", pady=(0, 7))
        ttk.Label(status_row, text="CURRENT STAGE", style="Section.TLabel").pack(
            side="left"
        )
        ttk.Label(
            status_row,
            textvariable=self.current_stage,
            style="Body.TLabel",
        ).pack(side="right")
        ttk.Progressbar(
            workspace,
            variable=self.progress,
            maximum=100,
            style="Demo.Horizontal.TProgressbar",
        ).grid(row=6, column=0, sticky="ew", pady=(0, 10))

        self.log = scrolledtext.ScrolledText(
            workspace,
            background=self._FIELD,
            foreground=self._TEXT,
            insertbackground=self._GREEN,
            selectbackground="#315f45",
            relief="flat",
            borderwidth=0,
            wrap="word",
            font=("Consolas", 9),
            padx=12,
            pady=10,
            state="disabled",
        )
        self.log.grid(row=7, column=0, sticky="nsew")
        self.log.tag_configure("success", foreground=self._GREEN)
        self.log.tag_configure("error", foreground=self._RED)
        self.log.tag_configure("muted", foreground=self._MUTED)
        self._append_log(
            "Operator interface ready. Uploads are disabled; Demo Mode is local-only.",
            "muted",
        )

        proof = ttk.Frame(body, style="Panel.TFrame", padding=18)
        proof.grid(row=0, column=1, sticky="nsew")
        proof.columnconfigure(0, weight=1)
        ttk.Label(proof, text="VALIDATION 06 PROOF", style="Section.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 14)
        )
        for row, (key, label) in enumerate(INITIAL_PROOF_ROWS, start=1):
            value = tk.StringVar(value=label)
            self._metric_values[key] = value
            ttk.Label(proof, textvariable=value, style="Proof.TLabel").grid(
                row=row,
                column=0,
                sticky="w",
                pady=6,
            )
        ttk.Label(
            proof,
            text="Cached artifacts only\nNo network uploads",
            style="Body.TLabel",
            justify="left",
        ).grid(
            row=len(INITIAL_PROOF_ROWS) + 1,
            column=0,
            sticky="sw",
            pady=(24, 0),
        )

    def start_processing(self) -> None:
        """Validate input and dispatch one real pipeline run off the UI thread."""

        if self._controller.is_running:
            self._append_log("A pipeline run is already active.", "error")
            return
        try:
            source = validate_source(self.source.get())
        except SourceValidationError as exc:
            self.current_stage.set(PipelineStage.FAILED.value)
            self.progress.set(0)
            self._append_log(str(exc), "error")
            return

        self._clear_log()
        self.current_stage.set(PipelineStage.RESOLVING_SOURCE.value)
        self.progress.set(0)
        self._output_directory = None
        self.open_button.state(["disabled"])
        self.start_button.state(["disabled"])
        self.demo_button.state(["disabled"])
        self._append_log("Pipeline run accepted. Uploads remain disabled.", "muted")
        try:
            self._controller.start(
                source,
                on_stage=lambda stage: self._pipeline_events.put(("stage", stage)),
                on_success=lambda result: self._pipeline_events.put(
                    ("success", result)
                ),
                on_failure=lambda failure: self._pipeline_events.put(
                    ("failure", failure)
                ),
            )
        except (RunInProgressError, SourceValidationError) as exc:
            self._restore_idle_controls()
            self.current_stage.set(PipelineStage.FAILED.value)
            self._append_log(str(exc), "error")
        except Exception as exc:
            self._restore_idle_controls()
            self.current_stage.set(PipelineStage.FAILED.value)
            self._append_log(
                f"Could not start pipeline worker: {type(exc).__name__}.",
                "error",
            )

    def _drain_pipeline_events(self) -> None:
        """Apply worker callbacks on Tk's owning thread."""

        while True:
            try:
                event, payload = self._pipeline_events.get_nowait()
            except queue.Empty:
                break
            if event == "stage" and isinstance(payload, PipelineStage):
                self._show_pipeline_stage(payload)
            elif event == "success" and isinstance(payload, PipelineRunSuccess):
                self._show_pipeline_success(payload)
            elif event == "failure" and isinstance(payload, PipelineRunFailure):
                self._show_pipeline_failure(payload)
        self.root.after(75, self._drain_pipeline_events)

    def _show_pipeline_stage(self, stage: PipelineStage) -> None:
        self.current_stage.set(stage.value)
        if stage is PipelineStage.COMPLETED:
            self.progress.set(100)
            return
        if stage is PipelineStage.FAILED:
            self.progress.set(0)
            return
        self._append_log(stage.value, "success")

    def _show_pipeline_success(self, result: PipelineRunSuccess) -> None:
        self._output_directory = result.output_directory
        self.current_stage.set(PipelineStage.COMPLETED.value)
        self.progress.set(100)
        self._append_log(
            f"Completed with {result.rendered_clip_count} rendered clips.",
            "success",
        )
        self._append_log(f"Output directory: {result.output_directory}", "muted")
        self._append_log("No network upload was performed.", "muted")
        self.open_button.state(["!disabled"])
        self._restore_idle_controls()

    def _show_pipeline_failure(self, failure: PipelineRunFailure) -> None:
        self.current_stage.set(PipelineStage.FAILED.value)
        self.progress.set(0)
        self._append_log(failure.message, "error")
        if failure.log_path is not None:
            self._append_log(f"Detailed log: {failure.log_path}", "muted")
        if failure.run_directory is not None and failure.run_directory.is_dir():
            self._output_directory = failure.run_directory
            self.open_button.state(["!disabled"])
        self._restore_idle_controls()

    def _restore_idle_controls(self) -> None:
        self.start_button.state(["!disabled"])
        self.demo_button.state(["!disabled"])

    def run_demo(self) -> None:
        """Load cached proof and present a short local-only stage sequence."""

        if self._controller.is_running:
            self._append_log("Demo Mode is unavailable during a pipeline run.", "error")
            return
        self._demo_run += 1
        run = self._demo_run
        self.last_demo_error = None
        self.start_button.state(["disabled"])
        self.demo_button.state(["disabled"])
        self.current_stage.set("Loading cached Validation 06 proof")
        self.progress.set(0)
        self._clear_log()
        self._append_log(
            "Demo Mode started — no pipeline or upload will run.",
            "muted",
        )
        try:
            proof = load_validation06()
        except DemoDataError as exc:
            self.last_demo_error = str(exc)
            self.current_stage.set("Demo unavailable")
            self._append_log(str(exc), "error")
            self._restore_idle_controls()
            return

        self._output_directory = proof.clips_directory
        self.open_button.state(["!disabled"])
        self.source.set(proof.source)
        self._show_proof(proof)
        self._emit_demo_stage(run, 0)

    def _emit_demo_stage(self, run: int, index: int) -> None:
        if run != self._demo_run:
            return
        stage = DEMO_STAGES[index]
        self.current_stage.set(stage)
        self.progress.set(((index + 1) / len(DEMO_STAGES)) * 100)
        self._append_log(stage, "success")
        if index + 1 < len(DEMO_STAGES):
            self.root.after(140, self._emit_demo_stage, run, index + 1)
            return
        self.current_stage.set("Demo complete")
        self._append_log(
            f"Output ready: {self._output_directory}",
            "muted",
        )
        self._append_log("No network upload was performed.", "muted")
        self._restore_idle_controls()

    def _show_proof(self, proof: Validation06Proof) -> None:
        values = {
            "observations": f"{proof.observations:,} observations",
            "generated": f"{proof.generated:,} generated",
            "passing": f"{proof.passing:,} passing",
            "selected": f"{proof.selected:,} selected",
            "rendered": f"{proof.rendered:,} rendered",
            "before_eof": (
                f"{proof.rendered_before_eof:,} rendered before EOF"
            ),
            "youtube": "✓ YouTube upload validated",
            "facebook": "✓ Facebook upload validated",
        }
        for key, value in values.items():
            self._metric_values[key].set(value)

    def open_output(self) -> None:
        """Open the current run output or cached Demo Mode clips."""

        folder = self._output_directory
        if folder is None:
            folder = REPOSITORY_ROOT / VALIDATION06_RELATIVE_PATH / "clips"
        try:
            open_output_folder(folder)
        except (OSError, NotADirectoryError) as exc:
            self.current_stage.set("Output folder unavailable")
            self._append_log(f"Could not open output folder: {exc}", "error")
            return
        self.current_stage.set("Output folder opened")
        self._append_log(f"Opened output folder: {folder}", "muted")

    def _clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _append_log(self, message: str, tag: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", f"> {message}\n", tag)
        self.log.configure(state="disabled")
        self.log.see("end")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the AitoClipAI operator UI.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="load Demo Mode once and close automatically",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Launch the desktop operator interface."""

    args = _parser().parse_args(argv)
    root = tk.Tk()
    app = AitoClipOperatorApp(root)
    if args.smoke_test:
        root.after(50, app.run_demo)
        root.after(1_500, root.destroy)
    root.mainloop()
    return 1 if app.last_demo_error is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
