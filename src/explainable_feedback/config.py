"""Configuration for explainable feedback reports."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ExplainableFeedbackConfig:
    """Deterministic report location and schema settings."""

    report_path: Path
    schema_version: str = "1.0"
