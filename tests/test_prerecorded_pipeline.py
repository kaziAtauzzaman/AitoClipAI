import json
import logging
import shutil
import subprocess
import wave
from pathlib import Path

import pytest

from audio_observer import AudioObserver, FFmpegAudioExtractor, FFmpegAudioExtractorConfig
from candidate_generation import CandidateGenerator
from candidate_scoring import CandidateScorer, CandidateScoringConfig
from clip_rendering import ClipRenderer, ClipRendererConfig
from core import (
    AggregatedTimeline,
    ClipCandidate,
    ClipScore,
    DownloadResult,
    FeatureTimeline,
    FeatureTimelineFailure,
    ObserverResult,
    RenderJob,
)
from downloader import DownloaderConfig, VideoDownloader
from explainable_feedback import ExplainableFeedbackGenerator
from observers import ObserverEngine, ObserverRegistry
from pipeline import (
    ArtifactValidationError,
    ArtifactValidator,
    JsonValidationReportWriter,
    MediaProbeResult,
    MediaStreamProbe,
    NoCandidatesError,
    NoPassingCandidatesError,
    PipelineConfig,
    PipelineOrchestrator,
    PrerecordedPipelineConfig,
    PrerecordedVideoPipeline,
    RenderedArtifactValidation,
    RequiredObserverError,
)
from whisper_observer import (
    TranscriptionResult,
    TranscriptionSegment,
    WhisperObserver,
    WhisperObserverConfig,
)


def timeline(
    tmp_path: Path,
    *,
    observers: tuple[str, ...] = ("audio", "whisper"),
    failures: list[FeatureTimelineFailure] | None = None,
    input_type: str = "local",
) -> FeatureTimeline:
    return FeatureTimeline(
        media_path=tmp_path / "source.mp4",
        audio_path=tmp_path / "audio.wav",
        timeline_path=tmp_path / "timeline.json",
        timeline=AggregatedTimeline(
            observer_results=[ObserverResult(observer=name) for name in observers]
        ),
        failures=failures or [],
        metadata={"input_type": input_type},
    )


class FakeAnalysis:
    def __init__(self, result: FeatureTimeline) -> None:
        self.result = result
        self.sources: list[str | Path] = []

    def analyze(self, source: str | Path) -> FeatureTimeline:
        self.sources.append(source)
        return self.result


class FakeGenerator:
    def __init__(self, candidates: list[ClipCandidate]) -> None:
        self.candidates = candidates

    def generate(self, feature_timeline: FeatureTimeline) -> list[ClipCandidate]:
        return self.candidates


class FakeScorer:
    def __init__(self, scores: list[ClipScore]) -> None:
        self.scores = scores

    def score(self, candidates) -> list[ClipScore]:
        return self.scores


class FakeRenderer:
    def __init__(self, jobs: list[RenderJob]) -> None:
        self.jobs = jobs
        self.received: list[ClipScore] = []

    def render(self, scores) -> list[RenderJob]:
        self.received = list(scores)
        return self.jobs


class FakeValidator:
    def __init__(self, source_path: Path, artifact: RenderedArtifactValidation) -> None:
        stream = MediaStreamProbe("video", "h264", 0.0, 1.0)
        self.source = MediaProbeResult(source_path, "mp4", 1.0, [stream])
        self.artifact = artifact

    def probe_source(self, path: Path) -> MediaProbeResult:
        return self.source

    def validate_jobs(self, jobs: list[RenderJob]) -> list[RenderedArtifactValidation]:
        return [self.artifact for _ in jobs]


class FakeReportWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.reports = []

    def write(self, report) -> Path:
        self.reports.append(report)
        self.path.write_text("report", encoding="utf-8")
        return self.path


def policy_fixture(tmp_path: Path):
    clip = ClipCandidate(tmp_path / "source.mp4", 0.0, 1.0, "candidate")
    passing = ClipScore(clip, 0.8, passed_threshold=True)
    failing = ClipScore(clip, 0.2, passed_threshold=False)
    output = tmp_path / "clip.mp4"
    output.write_bytes(b"clip")
    job = RenderJob(clip, output)
    video = MediaStreamProbe("video", "h264", 0.0, 1.0)
    audio = MediaStreamProbe("audio", "aac", 0.0, 1.0)
    artifact = RenderedArtifactValidation(
        output,
        output.stat().st_size,
        video,
        audio,
        1.0,
        {"valid": True},
    )
    return clip, passing, failing, job, artifact


def test_prerecorded_pipeline_renders_only_passing_scores_and_reports(
    tmp_path: Path,
    caplog,
) -> None:
    clip, passing, failing, job, artifact = policy_fixture(tmp_path)
    renderer = FakeRenderer([job])
    writer = FakeReportWriter(tmp_path / "report.json")
    pipeline = PrerecordedVideoPipeline(
        analysis_pipeline=FakeAnalysis(timeline(tmp_path)),
        candidate_generator=FakeGenerator([clip]),
        candidate_scorer=FakeScorer([passing, failing]),
        clip_renderer=renderer,
        artifact_validator=FakeValidator(tmp_path / "source.mp4", artifact),
        report_writer=writer,
        logger=logging.getLogger("test.validation.pipeline"),
    )

    with caplog.at_level(logging.INFO, logger="test.validation.pipeline"):
        result = pipeline.run(tmp_path / "source.mp4")

    assert renderer.received == [passing]
    assert result.selected_scores == [passing]
    assert result.validation_report.status == "passed"
    assert result.validation_report.passing_score_count == 1
    assert result.feature_timeline.download is None
    assert result.validation_report.source_metadata is not None
    assert result.report_path == tmp_path / "report.json"
    assert "stage_started stage=analysis" in caplog.text
    assert "pipeline_validation_passed" in caplog.text


def test_pipeline_suppresses_overlap_before_rendering_and_preserves_scores(
    tmp_path: Path,
    caplog,
) -> None:
    source = tmp_path / "source.mp4"
    stronger_candidate = ClipCandidate(source, 0.0, 10.0, "stronger")
    weaker_candidate = ClipCandidate(source, 1.0, 9.0, "weaker")
    stronger = ClipScore(stronger_candidate, 0.9, passed_threshold=True)
    weaker = ClipScore(weaker_candidate, 0.8, passed_threshold=True)
    output = tmp_path / "clip.mp4"
    output.write_bytes(b"clip")
    job = RenderJob(stronger_candidate, output)
    video = MediaStreamProbe("video", "h264", 0.0, 10.0)
    audio = MediaStreamProbe("audio", "aac", 0.0, 10.0)
    artifact = RenderedArtifactValidation(
        output,
        output.stat().st_size,
        video,
        audio,
        10.0,
        {"valid": True},
    )
    renderer = FakeRenderer([job])
    logger = logging.getLogger("test.validation.selection")
    pipeline = PrerecordedVideoPipeline(
        analysis_pipeline=FakeAnalysis(timeline(tmp_path)),
        candidate_generator=FakeGenerator([stronger_candidate, weaker_candidate]),
        candidate_scorer=FakeScorer([weaker, stronger]),
        clip_renderer=renderer,
        artifact_validator=FakeValidator(source, artifact),
        report_writer=FakeReportWriter(tmp_path / "report.json"),
        logger=logger,
    )

    with caplog.at_level(logging.INFO, logger="test.validation.selection"):
        result = pipeline.run(source)

    assert result.scores == [weaker, stronger]
    assert result.selected_scores == [weaker, stronger]
    assert renderer.received == [stronger]
    assert result.validation_report.passing_score_count == 2
    assert "selected=1 suppressed=1" in caplog.text
    feedback = ExplainableFeedbackGenerator().generate(result)
    statuses = {
        clip.identity.start_microseconds: clip.selection_status
        for clip in feedback.clips
    }
    assert statuses == {1_000_000: "passed_not_rendered", 0: "rendered"}


@pytest.mark.parametrize("missing", ["audio", "whisper"])
def test_prerecorded_pipeline_requires_both_observers(
    tmp_path: Path,
    missing: str,
) -> None:
    observed = tuple(name for name in ("audio", "whisper") if name != missing)
    pipeline = PrerecordedVideoPipeline(
        analysis_pipeline=FakeAnalysis(timeline(tmp_path, observers=observed))
    )

    with pytest.raises(RequiredObserverError, match=missing):
        pipeline.run("source")


def test_prerecorded_pipeline_configuration_cannot_remove_required_observers(
    tmp_path: Path,
) -> None:
    with pytest.raises(RequiredObserverError, match="requires audio and whisper"):
        PrerecordedVideoPipeline(
            analysis_pipeline=FakeAnalysis(timeline(tmp_path)),
            config=PrerecordedPipelineConfig(required_observers=("audio",)),
        )


@pytest.mark.parametrize("observer", ["audio", "whisper"])
def test_prerecorded_pipeline_fails_required_observer_errors(
    tmp_path: Path,
    observer: str,
) -> None:
    failure = FeatureTimelineFailure(observer, "RuntimeError", "failed")
    pipeline = PrerecordedVideoPipeline(
        analysis_pipeline=FakeAnalysis(timeline(tmp_path, failures=[failure]))
    )

    with pytest.raises(RequiredObserverError, match=observer):
        pipeline.run("source")


def test_prerecorded_pipeline_fails_without_candidates(tmp_path: Path) -> None:
    pipeline = PrerecordedVideoPipeline(
        analysis_pipeline=FakeAnalysis(timeline(tmp_path)),
        candidate_generator=FakeGenerator([]),
    )

    with pytest.raises(NoCandidatesError):
        pipeline.run("source")


def test_prerecorded_pipeline_fails_without_passing_scores(tmp_path: Path) -> None:
    clip, _, failing, _, _ = policy_fixture(tmp_path)
    renderer = FakeRenderer([])
    pipeline = PrerecordedVideoPipeline(
        analysis_pipeline=FakeAnalysis(timeline(tmp_path)),
        candidate_generator=FakeGenerator([clip]),
        candidate_scorer=FakeScorer([failing]),
        clip_renderer=renderer,
    )

    with pytest.raises(NoPassingCandidatesError):
        pipeline.run("source")
    assert renderer.received == []


def test_prerecorded_pipeline_requires_render_jobs(tmp_path: Path) -> None:
    clip, passing, _, _, artifact = policy_fixture(tmp_path)
    pipeline = PrerecordedVideoPipeline(
        analysis_pipeline=FakeAnalysis(timeline(tmp_path)),
        candidate_generator=FakeGenerator([clip]),
        candidate_scorer=FakeScorer([passing]),
        clip_renderer=FakeRenderer([]),
        artifact_validator=FakeValidator(tmp_path / "source.mp4", artifact),
    )

    with pytest.raises(ArtifactValidationError, match="no jobs"):
        pipeline.run("source")


class FixtureDownloader:
    def __init__(self, fixture: Path, destination: Path) -> None:
        self.fixture = fixture
        self.destination = destination

    def download(self, url: str) -> DownloadResult:
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.fixture, self.destination)
        metadata_path = self.destination.with_name(
            f"{self.destination.name}.metadata.json"
        )
        metadata_path.write_text(
            json.dumps({"id": "offline-fixture", "source_url": url}),
            encoding="utf-8",
        )
        return DownloadResult(
            source_url=url,
            video_path=self.destination,
            metadata_path=metadata_path,
            provider="OfflineFixture",
            media_id="offline-fixture",
            title="Offline Fixture",
            duration_seconds=3.0,
        )


class OfflineWhisperBackend:
    def transcribe(
        self,
        audio_path: Path,
        config: WhisperObserverConfig,
    ) -> TranscriptionResult:
        with wave.open(str(audio_path), "rb") as audio:
            duration = audio.getnframes() / audio.getframerate()
        return TranscriptionResult(
            segments=[
                TranscriptionSegment(
                    0.25,
                    min(duration, 2.75),
                    "THIS OFFLINE PIPELINE IS EXCITING!",
                    confidence=0.98,
                )
            ],
            text="THIS OFFLINE PIPELINE IS EXCITING!",
            language="en",
        )


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg and FFprobe are required",
)
def test_prerecorded_pipeline_complete_offline_integration(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture.mp4"
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
            str(fixture),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    run_dir = tmp_path / "validation-run"
    directories = {
        name: run_dir / name
        for name in ("downloads", "audio", "timelines", "clips", "reports")
    }
    downloaded = directories["downloads"] / "fixture.mp4"
    engine = ObserverEngine(
        ObserverRegistry(
            observers=[
                AudioObserver(),
                WhisperObserver(backend=OfflineWhisperBackend()),
            ]
        )
    )
    analysis = PipelineOrchestrator(
        downloader=FixtureDownloader(fixture, downloaded),
        audio_extractor=FFmpegAudioExtractor(
            FFmpegAudioExtractorConfig(output_dir=directories["audio"])
        ),
        observer_engine=engine,
        config=PipelineConfig(timeline_dir=directories["timelines"]),
    )
    weights = {
        "speech_excitement": 0.30,
        "speaking_intensity": 0.20,
        "loudness_peaks": 0.18,
        "silence_buildup": 0.12,
        "supporting_observations": 0.10,
        "observation_diversity": 0.10,
    }
    pipeline = PrerecordedVideoPipeline(
        analysis_pipeline=analysis,
        candidate_generator=CandidateGenerator(),
        candidate_scorer=CandidateScorer(
            CandidateScoringConfig(weights=weights, passing_score=0.1)
        ),
        clip_renderer=ClipRenderer(
            ClipRendererConfig(
                output_dir=directories["clips"],
                overwrite_existing=True,
                burn_subtitles=False,
            )
        ),
        artifact_validator=ArtifactValidator(),
        report_writer=JsonValidationReportWriter(
            directories["reports"] / "validation-report.json"
        ),
    )

    result = pipeline.run("https://example.test/offline-fixture")

    assert result.feature_timeline.download is not None
    assert result.feature_timeline.download.metadata_path.is_file()
    assert result.feature_timeline.audio_path.is_file()
    assert result.feature_timeline.timeline_path.is_file()
    assert [
        item.observer for item in result.feature_timeline.timeline.observer_results
    ] == ["audio", "whisper"]
    assert result.candidates
    assert result.selected_scores
    assert result.render_jobs
    assert all(job.captions_path is None for job in result.render_jobs)
    assert all(job.output_path.is_file() for job in result.render_jobs)
    assert result.validation_report.status == "passed"
    assert result.validation_report.source_type == "download"
    assert result.validation_report.rendered_artifacts
    assert all(
        all(artifact.checks.values())
        for artifact in result.validation_report.rendered_artifacts
    )
    report_data = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report_data["status"] == "passed"
    assert report_data["passing_score_count"] >= 1
    assert set(path.parent.name for path in [
        result.feature_timeline.download.video_path,
        result.feature_timeline.audio_path,
        result.feature_timeline.timeline_path,
        result.render_jobs[0].output_path,
        result.report_path,
    ]) == {"downloads", "audio", "timelines", "clips", "reports"}
