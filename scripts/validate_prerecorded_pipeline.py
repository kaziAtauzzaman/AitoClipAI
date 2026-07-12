"""Narrow manual harness for Pipeline Validation 0.1."""

import argparse
import logging
from pathlib import Path

from audio_observer import AudioObserver, FFmpegAudioExtractor, FFmpegAudioExtractorConfig
from candidate_generation import CandidateGenerator
from candidate_scoring import CandidateScorer
from clip_rendering import ClipRenderer, ClipRendererConfig
from downloader import DownloaderConfig, VideoDownloader
from observers import ObserverEngine, ObserverRegistry
from pipeline import (
    ArtifactValidator,
    JsonValidationReportWriter,
    PipelineConfig,
    PipelineOrchestrator,
    PrerecordedVideoPipeline,
)
from whisper_observer import WhisperObserver, WhisperObserverConfig


def main() -> int:
    args = _arguments()
    run_dir = args.run_dir.resolve()
    directories = {
        name: run_dir / name
        for name in ("downloads", "audio", "timelines", "clips", "logs", "reports")
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    logger = _logger(directories["logs"] / "validation.log")
    observer_engine = ObserverEngine(
        ObserverRegistry(
            observers=[
                AudioObserver(),
                WhisperObserver(
                    WhisperObserverConfig(model_name=args.whisper_model)
                ),
            ]
        ),
        logger=logger,
    )
    analysis = PipelineOrchestrator(
        downloader=VideoDownloader(
            DownloaderConfig(
                downloads_dir=directories["downloads"],
                overwrite_existing=args.overwrite,
            )
        ),
        audio_extractor=FFmpegAudioExtractor(
            FFmpegAudioExtractorConfig(
                output_dir=directories["audio"],
                overwrite_existing=args.overwrite,
            )
        ),
        observer_engine=observer_engine,
        config=PipelineConfig(timeline_dir=directories["timelines"]),
    )
    pipeline = PrerecordedVideoPipeline(
        analysis_pipeline=analysis,
        candidate_generator=CandidateGenerator(),
        candidate_scorer=CandidateScorer(),
        clip_renderer=ClipRenderer(
            ClipRendererConfig(
                output_dir=directories["clips"],
                overwrite_existing=args.overwrite,
                maximum_clips=args.maximum_clips,
                burn_subtitles=False,
            )
        ),
        artifact_validator=ArtifactValidator(),
        report_writer=JsonValidationReportWriter(
            directories["reports"] / "validation-report.json"
        ),
        logger=logger,
    )
    result = pipeline.run(args.source)
    logger.info("validation_report path=%s", result.report_path)
    for job in result.render_jobs:
        logger.info("playable_output path=%s", job.output_path)
    return 0


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the narrow AitoClipAI Pipeline Validation 0.1 harness."
    )
    parser.add_argument("source", help="One prerecorded YouTube URL or local file")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("data") / "validation" / "pipeline-0.1",
    )
    parser.add_argument("--whisper-model", default="tiny")
    parser.add_argument("--maximum-clips", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("aitoclipai.pipeline_validation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(levelname)s %(message)s")
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


if __name__ == "__main__":
    raise SystemExit(main())
