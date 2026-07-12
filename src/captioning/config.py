"""Configuration for deterministic SRT caption generation."""

from dataclasses import dataclass
from pathlib import Path


DEFAULT_CAPTION_OUTPUT_DIR = Path("data") / "captions"


@dataclass(frozen=True, slots=True)
class CaptionGeneratorConfig:
    """Runtime settings for caption selection, naming, and text formatting."""

    output_dir: Path = DEFAULT_CAPTION_OUTPUT_DIR
    filename_template: str = "{stem}.captions-{start_ms}-{end_ms}.srt"
    overwrite_existing: bool = False
    include_speaker_labels: bool = True
    speaker_template: str = "[{speaker}] {text}"
    skip_empty_text: bool = True
    encoding: str = "utf-8"
