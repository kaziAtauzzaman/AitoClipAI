import json
import shutil
import subprocess
import wave
from pathlib import Path

import pytest

from audio_observer import AudioObserver, FFmpegAudioExtractor, FFmpegAudioExtractorConfig
from candidate_generation import CandidateGenerator
from candidate_scoring import CandidateScorer, CandidateScoringConfig
from clip_rendering import ClipRenderer, ClipRendererConfig
from explainable_feedback import ExplainableFeedbackConfig, ExplainableFeedbackService
from observers import ObserverEngine, ObserverRegistry
from pipeline import (
    ArtifactValidator,
    JsonValidationReportWriter,
    PipelineConfig,
    PipelineOrchestrator,
    PrerecordedVideoPipeline,
)
from whisper_observer import (
    TranscriptionResult,
    TranscriptionSegment,
    WhisperObserver,
    WhisperObserverConfig,
)


class OfflineWhisperBackend:
    def transcribe(
        self, audio_path: Path, config: WhisperObserverConfig
    ) -> TranscriptionResult:
        with wave.open(str(audio_path), "rb") as audio:
            duration = audio.getnframes() / audio.getframerate()
        return TranscriptionResult(
            [
                TranscriptionSegment(
                    0.2,
                    min(duration, 2.8),
                    "THIS OFFLINE CLIP IS EXCITING!",
                    confidence=0.98,
                )
            ],
            "THIS OFFLINE CLIP IS EXCITING!",
            "en",
        )


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg and FFprobe are required",
)
def test_complete_offline_pipeline_produces_feedback_for_playable_clip(
    tmp_path: Path,
) -> None:
    source = tmp_path / "fixture.mp4"
    subprocess.run(
        [
            shutil.which("ffmpeg") or "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x180:rate=30:duration=3",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=3",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(source),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    run_dir = tmp_path / "run"
    analysis = PipelineOrchestrator(
        audio_extractor=FFmpegAudioExtractor(
            FFmpegAudioExtractorConfig(output_dir=run_dir / "audio")
        ),
        observer_engine=ObserverEngine(
            ObserverRegistry(
                observers=[
                    AudioObserver(),
                    WhisperObserver(backend=OfflineWhisperBackend()),
                ]
            )
        ),
        config=PipelineConfig(timeline_dir=run_dir / "timelines"),
    )
    pipeline = PrerecordedVideoPipeline(
        analysis_pipeline=analysis,
        candidate_generator=CandidateGenerator(),
        candidate_scorer=CandidateScorer(
            CandidateScoringConfig(passing_score=0.1)
        ),
        clip_renderer=ClipRenderer(
            ClipRendererConfig(
                output_dir=run_dir / "clips",
                overwrite_existing=True,
                burn_subtitles=False,
            )
        ),
        artifact_validator=ArtifactValidator(),
        report_writer=JsonValidationReportWriter(
            run_dir / "reports" / "validation-report.json"
        ),
    )

    pipeline_result = pipeline.run(source)
    report, report_path = ExplainableFeedbackService(
        ExplainableFeedbackConfig(
            run_dir / "reports" / "explainable-feedback.json"
        )
    ).create(pipeline_result)

    assert len(report.clips) == len(pipeline_result.scores)
    assert report.clips[0].scorer_rationale == pipeline_result.scores[0].rationale
    assert [
        (item.signal, item.contribution)
        for item in report.clips[0].score_contributions
    ] == list(pipeline_result.scores[0].score_components.items())
    rendered = [clip for clip in report.clips if clip.selection_status == "rendered"]
    assert rendered
    assert rendered[0].render is not None
    assert all(rendered[0].render.checks.values())
    assert rendered[0].render.output_path.is_file()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["scored_candidate_count"] == len(pipeline_result.scores)
    assert report_path.parent.name == "reports"
