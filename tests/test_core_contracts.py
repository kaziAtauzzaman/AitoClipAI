from dataclasses import is_dataclass
from pathlib import Path

from core import (
    AggregatedFeatures,
    AudioFeatures,
    ClipCandidate,
    ClipScore,
    DownloadResult,
    OCRFeatures,
    Observation,
    RenderJob,
    SpeechFeatures,
    UploadJob,
    VisionFeatures,
)


def test_core_contracts_are_dataclasses() -> None:
    contracts = [
        AggregatedFeatures,
        AudioFeatures,
        ClipCandidate,
        ClipScore,
        DownloadResult,
        OCRFeatures,
        Observation,
        RenderJob,
        SpeechFeatures,
        UploadJob,
        VisionFeatures,
    ]

    assert all(is_dataclass(contract) for contract in contracts)


def test_aggregated_features_can_reference_stage_contracts(tmp_path: Path) -> None:
    download = DownloadResult(
        source_url="https://example.test/video",
        video_path=tmp_path / "video.mp4",
        metadata_path=tmp_path / "video.mp4.metadata.json",
        provider="Example",
        media_id="abc123",
    )

    observation = Observation(
        timestamp_seconds=12.5,
        duration_seconds=1.0,
        observer="audio",
        type="volume_spike",
        value={"peak_dbfs": -2.0},
        confidence=0.91,
        metadata={"window_seconds": 1.0},
    )

    features = AggregatedFeatures(
        download=download,
        audio=AudioFeatures(
            source_video_path=download.video_path,
            observations=[observation],
        ),
        speech=SpeechFeatures(
            source_video_path=download.video_path,
            observations=[
                Observation(
                    timestamp_seconds=14.0,
                    observer="speech",
                    type="keyword",
                    value="example",
                )
            ],
        ),
        vision=VisionFeatures(
            source_video_path=download.video_path,
            observations=[
                Observation(
                    timestamp_seconds=20.0,
                    observer="vision",
                    type="scene_change",
                    value=True,
                )
            ],
        ),
        ocr=OCRFeatures(
            source_video_path=download.video_path,
            observations=[
                Observation(
                    timestamp_seconds=21.0,
                    observer="ocr",
                    type="text",
                    value="Example",
                    confidence=0.88,
                )
            ],
        ),
    )

    assert features.download.media_id == "abc123"
    assert features.audio is not None
    assert features.speech is not None
    assert features.vision is not None
    assert features.ocr is not None
    assert features.audio.observations[0].type == "volume_spike"


def test_clip_score_render_and_upload_jobs_share_candidate(tmp_path: Path) -> None:
    candidate = ClipCandidate(
        source_video_path=tmp_path / "source.mp4",
        start_seconds=10.0,
        end_seconds=40.0,
        reason="Example candidate",
    )
    score = ClipScore(candidate=candidate, overall_score=0.85)
    render_job = RenderJob(
        candidate=candidate,
        output_path=tmp_path / "clips" / "clip.mp4",
    )
    upload_job = UploadJob(
        rendered_clip_path=render_job.output_path,
        destination="example-platform",
        title="Example Clip",
    )

    assert score.candidate == candidate
    assert render_job.candidate == candidate
    assert upload_job.rendered_clip_path == render_job.output_path
