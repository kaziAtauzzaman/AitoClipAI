"""Configuration for deterministic FFmpeg clip rendering."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


DEFAULT_CLIP_OUTPUT_DIR = Path("data") / "clips"


class RendererBackend(str, Enum):
    """Explicit video-encoding backend; software remains the production default."""

    SOFTWARE = "software"
    INTEL_QSV = "intel_qsv"


@dataclass(frozen=True, slots=True)
class ClipRendererConfig:
    """Runtime settings for rendered clip selection, naming, and encoding."""

    output_dir: Path = DEFAULT_CLIP_OUTPUT_DIR
    filename_template: str = (
        "{stem}.clip-{rank:03d}-{start_ms}-{end_ms}-{score_millionths}.{ext}"
    )
    overwrite_existing: bool = False
    output_format: str = "mp4"
    video_codec: str = "libx264"
    renderer_backend: RendererBackend = RendererBackend.SOFTWARE
    audio_codec: str = "aac"
    maximum_clips: int | None = 1
    ffmpeg_binary: str = "ffmpeg"
    burn_subtitles: bool = False
    subtitle_character_encoding: str = "UTF-8"
