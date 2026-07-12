"""Feature timeline persistence services."""

from dataclasses import asdict
import json
from pathlib import Path
from typing import Protocol

from core import FeatureTimeline
from pipeline.errors import PipelineError


class TimelineWriter(Protocol):
    """Persist a complete feature timeline artifact."""

    def write(self, timeline: FeatureTimeline) -> Path:
        """Write a timeline and return its artifact path."""


class JsonFeatureTimelineWriter:
    """Write feature timelines as readable deterministic JSON."""

    def write(self, timeline: FeatureTimeline) -> Path:
        """Write the timeline JSON beside its analyzed media."""

        try:
            timeline.timeline_path.parent.mkdir(parents=True, exist_ok=True)
            timeline.timeline_path.write_text(
                json.dumps(
                    asdict(timeline),
                    default=str,
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        except (OSError, TypeError, ValueError) as exc:
            raise PipelineError(f"Failed to write feature timeline: {exc}") from exc

        return timeline.timeline_path
