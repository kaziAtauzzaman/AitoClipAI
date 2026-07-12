"""Deterministic JSON persistence for feedback reports."""

from dataclasses import asdict
import json
from pathlib import Path
from typing import Protocol

from explainable_feedback.contracts import ExplainableFeedbackReport
from explainable_feedback.errors import FeedbackPersistenceError


class FeedbackReportWriter(Protocol):
    """Persist an aggregate feedback report."""

    def write(self, report: ExplainableFeedbackReport) -> Path:
        """Write and return the report path."""


class JsonExplainableFeedbackWriter:
    """Write stable, readable UTF-8 JSON under a configured reports directory."""

    def __init__(self, report_path: Path) -> None:
        self._report_path = report_path

    def write(self, report: ExplainableFeedbackReport) -> Path:
        try:
            self._report_path.parent.mkdir(parents=True, exist_ok=True)
            self._report_path.write_text(
                json.dumps(asdict(report), default=str, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except (OSError, TypeError, ValueError) as exc:
            raise FeedbackPersistenceError(
                f"Failed to write explainable feedback report: {exc}"
            ) from exc
        return self._report_path
