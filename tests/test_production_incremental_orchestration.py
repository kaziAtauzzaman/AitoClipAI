import json
import struct
import wave
from pathlib import Path

import pytest

from audio_observer import (
    AudioObserverConfig,
    IncrementalAudioBatch,
    IncrementalAudioObserverConfig,
    IncrementalWavAudioObserver,
)
from candidate_selection import CandidateSelector
from core import (
    ClipCandidate,
    ClipScore,
    Observation,
    RenderJob,
)
from pipeline import (
    ArtifactValidationError,
    ProductionIncrementalLifecycle,
    ProductionIncrementalOrchestrator,
    JsonProductionIncrementalReportWriter,
    RenderedArtifactValidation,
)
from pipeline.incremental import IncrementalPipelineResult
import pipeline.production_incremental as production_incremental
from pipeline.contracts import MediaStreamProbe
from whisper_observer import IncrementalWhisperBatch


class MarkerGenerator:
    maximum_backtrack_seconds = 0.0
    incremental_deterministic = True

    def generate(self, timeline):
        return [
            ClipCandidate(
                timeline.media_path,
                float(item.value["start"]),
                float(item.value["end"]),
                str(item.value["name"]),
                metadata={"score": float(item.value["score"]), "observation": item},
            )
            for result in timeline.timeline.observer_results
            for item in result.observations
            if item.type == "candidate"
        ]


class MarkerScorer:
    candidate_local_deterministic = True

    def score(self, candidates):
        return [
            ClipScore(item, item.metadata["score"], passed_threshold=True)
            for item in candidates
        ]


class EditorialEvidenceGenerator:
    maximum_backtrack_seconds = 0.0
    incremental_deterministic = True

    def generate(self, timeline):
        valid = [
            Observation(
                1.0,
                "audio",
                "silence",
                {"loudness_dbfs": -60.0},
                duration_seconds=0.5,
            ),
            Observation(
                1.5,
                "audio",
                "speaking_intensity",
                {"loudness_dbfs": -20.0, "intensity": 0.8},
                duration_seconds=0.5,
            ),
        ]
        malformed = [
            Observation(
                4.0,
                "audio",
                "speaking_intensity",
                {"loudness_dbfs": "invalid", "intensity": 0.8},
                duration_seconds=1.0,
            )
        ]
        return [
            ClipCandidate(
                timeline.media_path,
                1.0,
                2.0,
                "valid editorial evidence",
                metadata={
                    "score": 0.8,
                    "contributing_observations": valid,
                },
            ),
            ClipCandidate(
                timeline.media_path,
                4.0,
                5.0,
                "malformed editorial evidence",
                metadata={
                    "score": 0.7,
                    "contributing_observations": malformed,
                },
            ),
        ]


class Session:
    def __init__(self, batches, failure_at=None):
        self.batches = list(batches)
        self.failure_at = failure_at
        self.calls = 0
        self.closed = False

    def read_batch(self):
        if self.calls == self.failure_at:
            raise RuntimeError("observer unavailable")
        self.calls += 1
        return self.batches.pop(0) if self.batches else None

    def close(self):
        self.closed = True


class Factory:
    def __init__(self, session):
        self.value = session

    def session(self, source):
        return self.value


class Renderer:
    def __init__(self, output_dir, fail=False, fail_identities=()):
        self.output_dir = output_dir
        self.fail = fail
        self.fail_identities = set(fail_identities)
        self.calls = []

    def render_one(self, score, identity):
        self.calls.append((score, identity))
        if self.fail or identity in self.fail_identities:
            raise RuntimeError("render failed")
        return RenderJob(score.candidate, self.output_dir / f"clip-{identity}.mp4")


class Validator:
    def __init__(self, fail=False):
        self.fail = fail
        self.jobs = []

    def validate_jobs(self, jobs):
        self.jobs.extend(jobs)
        if self.fail:
            raise ArtifactValidationError("invalid artifact")
        stream_video = MediaStreamProbe("video", "h264", 0.0, 1.0)
        stream_audio = MediaStreamProbe("audio", "aac", 0.0, 1.0)
        return [
            RenderedArtifactValidation(
                item.output_path, 1, stream_video, stream_audio, 1.0
            )
            for item in jobs
        ]


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        self.value += 0.001
        return self.value


def marker(observer, at, start, end, name, score):
    return Observation(
        at,
        observer,
        "candidate",
        {"start": start, "end": end, "name": name, "score": score},
    )


def audio_batch(watermark, observations=(), *, eof=False, frames=None, sample_rate=10):
    return IncrementalAudioBatch(
        "audio",
        tuple(observations),
        watermark,
        round(watermark * 10) if frames is None else frames,
        eof,
        {} if sample_rate is None else {"sample_rate_hz": sample_rate},
    )


def whisper_batch(watermark, observations=(), *, eof=False, frames=None, sample_rate=10):
    return IncrementalWhisperBatch(
        "whisper",
        tuple(observations),
        watermark,
        round(watermark * 10) if frames is None else frames,
        eof,
        {} if sample_rate is None else {"sample_rate_hz": sample_rate},
    )


def sources(tmp_path):
    video = tmp_path / "source.mp4"
    wav = tmp_path / "audio.wav"
    video.write_bytes(b"video")
    wav.write_bytes(b"wav")
    return video, wav


def orchestrator(
    tmp_path, audio, whisper, renderer=None, validator=None, generator=None
):
    return ProductionIncrementalOrchestrator(
        renderer or Renderer(tmp_path),
        audio_observer=Factory(audio),
        whisper_observer=Factory(whisper),
        artifact_validator=validator or Validator(),
        candidate_generator=generator or MarkerGenerator(),
        candidate_scorer=MarkerScorer(),
        candidate_selector=CandidateSelector(),
        clock=Clock(),
    )


def test_slower_whisper_blocks_then_multiple_watermarks_render_chronologically(tmp_path):
    first = marker("audio", 1.0, 1.0, 2.0, "first", 0.8)
    second = marker("audio", 4.0, 4.0, 5.0, "second", 0.9)
    audio = Session([
        audio_batch(3.0, [first]),
        audio_batch(6.0, [second]),
        audio_batch(7.0, eof=True),
    ])
    whisper = Session([
        whisper_batch(0.5),
        whisper_batch(3.0),
        whisper_batch(7.0, eof=True),
    ])
    renderer = Renderer(tmp_path)
    report = orchestrator(tmp_path, audio, whisper, renderer).run(*sources(tmp_path))

    assert [item.candidate.reason for item in report.selected_scores] == ["first", "second"]
    assert [identity for _, identity in renderer.calls] == [1, 2]
    assert report.watermarks == {"audio": 7.0, "whisper": 7.0}
    assert report.status == "completed"


def test_combined_eof_flushes_overlap_winner_once(tmp_path):
    weak = marker("audio", 1.0, 1.0, 3.0, "weak", 0.5)
    strong = marker("whisper", 1.5, 1.5, 3.5, "strong", 0.9)
    audio = Session([audio_batch(4.0, [weak], eof=True)])
    whisper = Session([whisper_batch(4.0, [strong], eof=True)])
    renderer = Renderer(tmp_path)
    report = orchestrator(tmp_path, audio, whisper, renderer).run(*sources(tmp_path))

    assert [item.candidate.reason for item in report.selected_scores] == ["strong"]
    assert [item.score.candidate.reason for item in report.suppressed] == ["weak"]
    assert len(report.render_jobs) == 1
    assert report.editorial_strength_results == []
    assert [item.code for item in report.editorial_strength_failures] == [
        "insufficient_evidence",
        "insufficient_evidence",
    ]


def test_editorial_failure_isolated_per_candidate_without_changing_decisions(
    tmp_path,
):
    renderer = Renderer(tmp_path)
    report = orchestrator(
        tmp_path,
        Session([audio_batch(6.0, eof=True)]),
        Session([whisper_batch(6.0, eof=True)]),
        renderer=renderer,
        generator=EditorialEvidenceGenerator(),
    ).run(*sources(tmp_path))

    assert [item.candidate.reason for item in report.selected_scores] == [
        "valid editorial evidence",
        "malformed editorial evidence",
    ]
    assert len(renderer.calls) == 2
    assert len(report.artifact_validations) == 2
    assert len(report.editorial_strength_results) == 1
    assert len(report.editorial_strength_failures) == 1
    failure = report.editorial_strength_failures[0]
    assert failure.code == "invalid_evidence"
    assert failure.candidate_identity
    assert report.status == "completed"


def test_real_incremental_audio_and_deterministic_whisper_drive_orchestrator(tmp_path):
    video = tmp_path / "source.mp4"
    wav = tmp_path / "audio.wav"
    video.write_bytes(b"video")
    with wave.open(str(wav), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(10)
        output.writeframes(b"".join(struct.pack("<h", 16000) for _ in range(30)))
    audio = IncrementalWavAudioObserver(
        IncrementalAudioObserverConfig(
            chunk_frames=5,
            analysis=AudioObserverConfig(
                window_seconds=0.5,
                hop_seconds=0.5,
                speaking_intensity_threshold_dbfs=-40,
                peak_threshold=1.0,
            ),
        )
    )
    whisper = Session([
        whisper_batch(1.0), whisper_batch(2.0), whisper_batch(3.0, eof=True)
    ])
    report = ProductionIncrementalOrchestrator(
        Renderer(tmp_path),
        audio_observer=audio,
        whisper_observer=Factory(whisper),
        artifact_validator=Validator(),
        candidate_generator=MarkerGenerator(),
        candidate_scorer=MarkerScorer(),
        clock=Clock(),
    ).run(video, wav)

    assert report.status == "completed"
    assert any(item.observer == "audio" for item in report.observations)
    assert len([item for item in report.observer_timings if item.observer == "audio"]) > 1


def test_observer_failure_stops_progress_and_cleanup_is_deterministic(tmp_path):
    audio = Session([audio_batch(1.0)], failure_at=1)
    whisper = Session([whisper_batch(0.5), whisper_batch(1.0, eof=True)])
    runner = orchestrator(tmp_path, audio, whisper)
    video, wav = sources(tmp_path)
    report = runner.run(video, wav)

    assert report.status == "failed"
    assert report.observer_failures[0].error_type == "RuntimeError"
    assert audio.closed and whisper.closed
    assert runner.lifecycle is ProductionIncrementalLifecycle.FAILED
    with pytest.raises(RuntimeError, match="single-use"):
        runner.run(video, wav)


def test_render_and_artifact_failures_are_reported_separately(tmp_path):
    candidate = marker("audio", 1.0, 1.0, 2.0, "clip", 0.8)
    audio = Session([audio_batch(3.0, [candidate]), audio_batch(4.0, eof=True)])
    whisper = Session([whisper_batch(3.0), whisper_batch(4.0, eof=True)])
    renderer = Renderer(tmp_path, fail=True)
    report = orchestrator(tmp_path, audio, whisper, renderer).run(*sources(tmp_path))
    assert report.render_failures
    assert report.render_jobs == []
    assert report.artifact_validation_failures == []

    audio2 = Session([audio_batch(4.0, [candidate], eof=True)])
    whisper2 = Session([whisper_batch(4.0, eof=True)])
    validator = Validator(fail=True)
    report2 = orchestrator(
        tmp_path, audio2, whisper2, validator=validator
    ).run(*sources(tmp_path))
    assert report2.render_failures == []
    assert len(report2.artifact_validation_failures) == 1
    assert len(report2.render_jobs) == 1


def test_report_order_and_timing_fields_are_deterministic(tmp_path):
    item = marker("audio", 1.0, 1.0, 2.0, "clip", 0.8)
    report = orchestrator(
        tmp_path,
        Session([audio_batch(3.0, [item], eof=True)]),
        Session([whisper_batch(3.0, eof=True)]),
    ).run(*sources(tmp_path))

    assert [entry.observer for entry in report.observer_timings] == ["audio", "whisper"]
    assert report.coordinator_timings
    assert report.render_timings
    assert report.validation_timings
    assert report.total_wall_seconds > 0
    assert all(item.operation_identity for item in report.observer_timings)
    assert all(item.succeeded for item in report.observer_timings)


def test_post_activation_setup_failure_returns_failed_report(
    tmp_path, monkeypatch
):
    video, wav = sources(tmp_path)
    monkeypatch.setattr(
        production_incremental,
        "_source_id",
        lambda path: (_ for _ in ()).throw(OSError("fingerprint failed")),
    )
    runner = orchestrator(tmp_path, Session([]), Session([]))

    report = runner.run(video, wav)

    assert report.status == "failed"
    assert report.source_id is None
    assert report.observer_failures[0].message == "fingerprint failed"
    assert runner.lifecycle is ProductionIncrementalLifecycle.FAILED
    with pytest.raises(RuntimeError, match="single-use"):
        runner.run(video, wav)


def test_coordinator_advance_exception_closes_sessions(tmp_path, monkeypatch):
    class FailingCoordinator:
        def __init__(self, *args, **kwargs):
            self.result = IncrementalPipelineResult()

        def advance(self, timeline, watermarks):
            raise RuntimeError("advance failed")

    monkeypatch.setattr(
        production_incremental, "IncrementalPrerecordedCoordinator", FailingCoordinator
    )
    audio = Session([audio_batch(1.0)])
    whisper = Session([whisper_batch(1.0)])
    report = orchestrator(tmp_path, audio, whisper).run(*sources(tmp_path))

    assert report.status == "failed"
    assert report.observer_failures[0].message == "advance failed"
    assert audio.closed and whisper.closed


@pytest.mark.parametrize(
    ("audio_eof", "whisper_eof", "message"),
    [
        (audio_batch(1.0, eof=True, sample_rate=None), whisper_batch(1.0, eof=True), "sample_rate_hz"),
        (audio_batch(1.0, eof=True, sample_rate=0), whisper_batch(1.0, eof=True), "sample_rate_hz"),
        (audio_batch(1.0, eof=True, sample_rate=float("nan")), whisper_batch(1.0, eof=True), "sample_rate_hz"),
        (audio_batch(1.0, eof=True, frames=20), whisper_batch(1.0, eof=True), "watermark"),
        (audio_batch(1.0, eof=True), whisper_batch(2.0, eof=True), "durations do not match"),
    ],
)
def test_authoritative_eof_metadata_is_strict(
    tmp_path, audio_eof, whisper_eof, message
):
    audio = Session([audio_eof])
    whisper = Session([whisper_eof])
    report = orchestrator(tmp_path, audio, whisper).run(*sources(tmp_path))

    assert report.status == "failed"
    assert message in report.observer_failures[0].message
    assert audio.closed and whisper.closed


def test_flush_validates_success_before_later_render_failure(tmp_path):
    class FlushOnlyGenerator(MarkerGenerator):
        maximum_backtrack_seconds = 10.0

    first = marker("audio", 1.0, 1.0, 2.0, "first", 0.8)
    second = marker("audio", 3.0, 3.0, 4.0, "second", 0.9)
    audio = Session([audio_batch(5.0, [first, second], eof=True)])
    whisper = Session([whisper_batch(5.0, eof=True)])
    renderer = Renderer(tmp_path, fail_identities={2})
    validator = Validator()
    report = orchestrator(
        tmp_path,
        audio,
        whisper,
        renderer=renderer,
        validator=validator,
        generator=FlushOnlyGenerator(),
    ).run(*sources(tmp_path))

    assert report.status == "failed"
    assert len(report.render_jobs) == 1
    assert len(report.artifact_validations) == 1
    assert len(report.render_failures) == 1
    assert report.validation_timings[0].operation_identity == "validation:1"


@pytest.mark.parametrize("mode", ["empty", "multiple", "mismatch", "malformed"])
def test_validator_contract_violations_are_attempted_once(tmp_path, mode):
    class ContractValidator:
        def __init__(self):
            self.calls = 0

        def validate_jobs(self, jobs):
            self.calls += 1
            valid = Validator().validate_jobs(jobs)[0]
            if mode == "empty":
                return []
            if mode == "multiple":
                return [valid, valid]
            if mode == "mismatch":
                return [
                    RenderedArtifactValidation(
                        tmp_path / "other.mp4",
                        valid.size_bytes,
                        valid.video_stream,
                        valid.audio_stream,
                        valid.duration_seconds,
                    )
                ]
            return [object()]

    item = marker("audio", 1.0, 1.0, 2.0, "clip", 0.8)
    validator = ContractValidator()
    report = orchestrator(
        tmp_path,
        Session([audio_batch(3.0, [item]), audio_batch(4.0, eof=True)]),
        Session([whisper_batch(3.0), whisper_batch(4.0, eof=True)]),
        validator=validator,
    ).run(*sources(tmp_path))

    assert validator.calls == 1
    assert len(report.artifact_validation_failures) == 1
    assert report.artifact_validation_failures[0].render_identity == 1
    assert report.validation_timings[0].succeeded is False


def test_report_serialization_is_canonical_and_strict(tmp_path):
    item = marker("audio", 1.0, 1.0, 2.0, "clip", 0.8)
    report = orchestrator(
        tmp_path,
        Session([audio_batch(3.0, [item], eof=True)]),
        Session([whisper_batch(3.0, eof=True)]),
    ).run(*sources(tmp_path))
    first = report.to_dict()
    second = report.to_dict()
    json.dumps(first)
    assert first == second
    assert "\\" not in first["source_video"]
    output = JsonProductionIncrementalReportWriter(tmp_path / "report.json").write(report)
    assert json.loads(output.read_text(encoding="utf-8")) == first

    report.observations[0].metadata["unsupported"] = object()
    with pytest.raises(TypeError, match="Unsupported report value"):
        report.to_dict()


def test_report_serialization_rejects_nonfinite_and_nonstring_keys(tmp_path):
    report = orchestrator(
        tmp_path,
        Session([audio_batch(1.0, eof=True)]),
        Session([whisper_batch(1.0, eof=True)]),
    ).run(*sources(tmp_path))
    report.watermarks["audio"] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        report.to_dict()
    report.watermarks = {1: 1.0}
    with pytest.raises(TypeError, match="string keys"):
        report.to_dict()
