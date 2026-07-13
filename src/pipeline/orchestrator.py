"""End-to-end media analysis pipeline orchestration."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse

from aggregation import FeatureAggregator
from audio_observer import AudioExtractor, AudioObserver, FFmpegAudioExtractor
from core import DownloadResult, FeatureTimeline, FeatureTimelineFailure
from downloader import VideoDownloader
from observers import ObserverContext, ObserverEngine, ObserverRegistry
from pipeline.errors import PipelineError
from pipeline.persistence import JsonFeatureTimelineWriter, TimelineWriter


class MediaDownloader(Protocol):
    """Download remote media into a local pipeline input."""

    def download(self, url: str) -> DownloadResult:
        """Download a URL and return its canonical core result."""


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Runtime configuration owned by the orchestration layer."""

    timeline_suffix: str = ".feature-timeline.json"
    timeline_dir: Path | None = None


class PipelineOrchestrator:
    """Compose existing download, extraction, observation, and aggregation stages."""

    def __init__(
        self,
        downloader: MediaDownloader | None = None,
        audio_extractor: AudioExtractor | None = None,
        observer_engine: ObserverEngine | None = None,
        aggregator: FeatureAggregator | None = None,
        timeline_writer: TimelineWriter | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        self._downloader = downloader or VideoDownloader()
        self._audio_extractor = audio_extractor or FFmpegAudioExtractor()
        self._observer_engine = observer_engine or ObserverEngine(
            ObserverRegistry(observers=[AudioObserver()])
        )
        self._aggregator = aggregator or FeatureAggregator()
        self._timeline_writer = timeline_writer or JsonFeatureTimelineWriter()
        self._config = config or PipelineConfig()

    def analyze(self, source: str | Path) -> FeatureTimeline:
        """Analyze a YouTube URL or local media path and persist its timeline."""

        media_path, source_url, input_type, download = self._resolve_input(source)
        media_context = ObserverContext(
            source_path=media_path,
            metadata={"input_type": input_type, "source_url": source_url},
        )
        audio_source = self._audio_extractor.extract(media_context)
        observer_context = ObserverContext(
            source_path=audio_source.path,
            source=media_path,
            metadata={
                "input_type": input_type,
                "media_path": str(media_path),
                "source_url": source_url,
                "audio": audio_source.metadata,
            },
        )
        engine_result = self._observer_engine.run(observer_context)
        aggregated = self._aggregator.aggregate(engine_result.results)
        timeline_metadata: dict[str, object] = {
            "input_type": input_type,
            "observer_count": engine_result.metadata.get("observer_count", 0),
        }
        if input_type == "local":
            timeline_metadata["source_id"] = self._local_source_id(media_path)
        timeline = FeatureTimeline(
            media_path=media_path,
            audio_path=audio_source.path,
            timeline_path=self._timeline_path(media_path),
            timeline=aggregated,
            source_url=source_url,
            download=download,
            failures=[
                FeatureTimelineFailure(
                    observer=failure.observer,
                    error_type=failure.error_type,
                    message=failure.message,
                    metadata=failure.metadata,
                )
                for failure in engine_result.failures
            ],
            metadata=timeline_metadata,
        )
        written_path = self._timeline_writer.write(timeline)
        if written_path != timeline.timeline_path:
            raise PipelineError(
                "Timeline writer returned a path different from the requested path."
            )
        return timeline

    def _resolve_input(
        self,
        source: str | Path,
    ) -> tuple[Path, str | None, str, DownloadResult | None]:
        if isinstance(source, Path):
            return self._local_media(source)

        source_value = source.strip()
        if not source_value:
            raise PipelineError("A non-empty media path or URL is required.")
        if self._is_remote_url(source_value):
            download = self._downloader.download(source_value)
            if not download.video_path.is_file():
                raise PipelineError(
                    f"Downloaded media does not exist: {download.video_path}"
                )
            return download.video_path, source_value, "download", download
        return self._local_media(Path(source_value))

    def _local_media(self, path: Path) -> tuple[Path, None, str, None]:
        if not path.is_file():
            raise PipelineError(f"Local media does not exist: {path}")
        return path, None, "local", None

    def _is_remote_url(self, source: str) -> bool:
        parsed = urlparse(source)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

    def _timeline_path(self, media_path: Path) -> Path:
        filename = f"{media_path.name}{self._config.timeline_suffix}"
        if self._config.timeline_dir is not None:
            return self._config.timeline_dir / filename
        return media_path.with_name(filename)

    @staticmethod
    def _local_source_id(media_path: Path) -> str:
        """Return a portable content identity independent of the local path."""

        digest = hashlib.sha256()
        try:
            with media_path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as exc:
            raise PipelineError(
                f"Failed to fingerprint local media: {media_path}: {exc}"
            ) from exc
        return f"local:sha256:{digest.hexdigest()}"
