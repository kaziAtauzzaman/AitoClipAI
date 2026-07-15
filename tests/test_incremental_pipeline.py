from pathlib import Path
import struct
import subprocess
import wave

import pytest

from aggregation import FeatureAggregator
from audio_observer import (
    AudioObserverConfig,
    IncrementalAudioObserverConfig,
    IncrementalWavAudioObserver,
)
from candidate_generation import CandidateGenerator
from candidate_scoring import CandidateScorer, CandidateScoringConfig
from candidate_selection import CandidateSelector
from clip_rendering import ClipRenderer, ClipRendererConfig
from core import (
    AggregatedTimeline,
    ClipCandidate,
    ClipScore,
    FeatureTimeline,
    Observation,
    ObserverResult,
    RenderJob,
    TimelineGroup,
)
from decision_engine import EditorialStrengthEvaluator
from pipeline import (
    candidate_fingerprint,
    CompletedTimelineReplayAdapter,
    CompletedTimelineReplayConfig,
    CoordinatorLifecycle,
    IncrementalEOF,
    IncrementalPipelineConfig,
    IncrementalPrerecordedCoordinator,
    ObserverWatermarks,
    RenderLifecycleState,
)
from whisper_observer import (
    IncrementalWhisperObserverConfig,
    IncrementalWavWhisperObserver,
    TranscriptionResult,
    TranscriptionSegment,
)


class MarkerGenerator:
    maximum_backtrack_seconds = 5.0
    maximum_competition_seconds = 5.0
    incremental_deterministic = True

    def generate(self, timeline: FeatureTimeline) -> list[ClipCandidate]:
        return [
            ClipCandidate(
                timeline.media_path,
                float(item.value["start"]),
                float(item.value["end"]),
                str(item.value["name"]),
                metadata={
                    "score": float(item.value["score"]),
                    "contributing_observations": [item],
                },
            )
            for result in timeline.timeline.observer_results
            for item in result.observations
        ]


class MarkerScorer:
    candidate_local_deterministic = True

    def score(self, candidates) -> list[ClipScore]:
        return sorted(
            [
                ClipScore(
                    candidate,
                    candidate.metadata["score"],
                    passed_threshold=True,
                )
                for candidate in candidates
            ],
            key=lambda item: -item.overall_score,
        )


class RecordingRenderer:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.calls: list[tuple[ClipScore, int]] = []

    def render_one(self, score: ClipScore, identity: int) -> RenderJob:
        self.calls.append((score, identity))
        return RenderJob(
            score.candidate,
            self.output_dir / f"clip-{identity:03d}.mp4",
            metadata={"rank": identity},
        )


def feature_timeline(
    tmp_path: Path,
    markers: list[dict],
    *,
    source_id: str = "fixture-source",
    observer: str = "fixture",
) -> FeatureTimeline:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    observations = [
        Observation(
            timestamp_seconds=float(marker["seen_at"]),
            duration_seconds=float(marker.get("duration", 0.0)),
            observer=observer,
            type="candidate",
            value=marker,
        )
        for marker in markers
    ]
    result = ObserverResult(observer=observer, observations=observations)
    return FeatureTimeline(
        media_path=source,
        audio_path=tmp_path / "audio.wav",
        timeline_path=tmp_path / "timeline.json",
        timeline=AggregatedTimeline(
            groups=[TimelineGroup(item.timestamp_seconds, [item]) for item in observations],
            observer_results=[result],
        ),
        metadata={"source_id": source_id},
    )


def coordinator(
    tmp_path: Path,
    renderer: RecordingRenderer,
    *,
    generator=None,
    scorer=None,
    required_observers=("fixture",),
):
    return IncrementalPrerecordedCoordinator(
        candidate_generator=generator or MarkerGenerator(),
        candidate_scorer=scorer or MarkerScorer(),
        candidate_selector=CandidateSelector(),
        clip_renderer=renderer,
        config=IncrementalPipelineConfig(required_observers=required_observers),
    )


def watermarks(value: float, observer: str = "fixture") -> ObserverWatermarks:
    return ObserverWatermarks({observer: value})


def eof(value: float, observer: str = "fixture") -> IncrementalEOF:
    return IncrementalEOF(value, watermarks(value, observer))


def _write_incremental_audio_fixture(path: Path) -> None:
    samples = [0.45] * 8 + [0.95] + [0.45] * 21
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(10)
        wav_file.writeframes(
            b"".join(struct.pack("<h", round(sample * 32768)) for sample in samples)
        )


class CoordinatorWhisperModelSession:
    def __init__(self) -> None:
        self.calls = 0

    def transcribe(self, audio_path: Path, initial_prompt: str | None):
        self.calls += 1
        if self.calls == 1:
            return TranscriptionResult(
                segments=[
                    TranscriptionSegment(
                        0.2,
                        0.8,
                        "incremental speech",
                        confidence=0.9,
                    )
                ]
            )
        return TranscriptionResult()

    def close(self) -> None:
        pass


class CoordinatorWhisperBackend:
    def open_incremental_session(self, config):
        return CoordinatorWhisperModelSession()


def test_real_incremental_audio_drives_coordinator_across_watermarks(
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "audio.wav"
    media_path = tmp_path / "source.mp4"
    _write_incremental_audio_fixture(audio_path)
    media_path.write_bytes(b"source")
    observer = IncrementalWavAudioObserver(
        IncrementalAudioObserverConfig(
            chunk_frames=5,
            analysis=AudioObserverConfig(
                window_seconds=0.5,
                hop_seconds=0.5,
                peak_threshold=0.9,
                min_peak_distance_seconds=0.3,
                speaking_intensity_threshold_dbfs=-30.0,
            ),
        )
    )
    renderer = RecordingRenderer(tmp_path)
    incremental = IncrementalPrerecordedCoordinator(
        candidate_generator=CandidateGenerator(),
        candidate_scorer=CandidateScorer(
            CandidateScoringConfig(passing_score=0.0)
        ),
        candidate_selector=CandidateSelector(),
        clip_renderer=renderer,
        config=IncrementalPipelineConfig(required_observers=("audio",)),
    )
    accumulated = []
    observed_watermarks = []

    for batch in observer.batches(audio_path):
        accumulated.extend(batch.observations)
        result = ObserverResult(
            observer="audio",
            observations=list(accumulated),
            metadata={
                "sample_rate_hz": 10,
                "duration_seconds": batch.frames_processed / 10,
            },
        )
        timeline = FeatureTimeline(
            media_path=media_path,
            audio_path=audio_path,
            timeline_path=tmp_path / "timeline.json",
            timeline=FeatureAggregator().aggregate([result]),
            metadata={"source_id": "incremental-audio-fixture"},
        )
        if not batch.eof:
            incremental.advance(
                timeline,
                ObserverWatermarks({"audio": batch.watermark_seconds}),
            )
            observed_watermarks.append(incremental.watermark_seconds)
        else:
            completed = incremental.flush(
                timeline,
                IncrementalEOF(
                    batch.watermark_seconds,
                    ObserverWatermarks({"audio": batch.watermark_seconds}),
                ),
            )

    assert len(set(observed_watermarks)) >= 3
    assert observed_watermarks == sorted(observed_watermarks)
    assert completed.selected_scores
    assert len(completed.render_jobs) == len(renderer.calls)
    assert incremental.lifecycle is CoordinatorLifecycle.FLUSHED


def test_real_incremental_whisper_feeds_production_decision_components(
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "speech.wav"
    media_path = tmp_path / "source.mp4"
    _write_incremental_audio_fixture(audio_path)
    media_path.write_bytes(b"source")
    observer = IncrementalWavWhisperObserver(
        IncrementalWhisperObserverConfig(
            chunk_seconds=2.0,
            overlap_seconds=0.5,
        ),
        CoordinatorWhisperBackend(),
    )
    renderer = RecordingRenderer(tmp_path)
    incremental = IncrementalPrerecordedCoordinator(
        candidate_generator=CandidateGenerator(),
        candidate_scorer=CandidateScorer(
            CandidateScoringConfig(passing_score=0.0)
        ),
        candidate_selector=CandidateSelector(),
        clip_renderer=renderer,
        config=IncrementalPipelineConfig(required_observers=("audio", "whisper")),
    )
    speech = []

    for batch in observer.batches(audio_path):
        speech.extend(batch.observations)
        duration = batch.frames_processed / 10
        results = [
            ObserverResult(
                "audio",
                [],
                {"duration_seconds": duration},
            ),
            ObserverResult(
                "whisper",
                list(speech),
                {"duration_seconds": duration},
            ),
        ]
        timeline = FeatureTimeline(
            media_path=media_path,
            audio_path=audio_path,
            timeline_path=tmp_path / "timeline.json",
            timeline=FeatureAggregator().aggregate(results),
            metadata={"source_id": "incremental-whisper-fixture"},
        )
        watermarks = ObserverWatermarks(
            {
                "audio": duration,
                "whisper": batch.watermark_seconds,
            }
        )
        if batch.eof:
            completed = incremental.flush(
                timeline,
                IncrementalEOF(batch.watermark_seconds, watermarks),
            )
        else:
            incremental.advance(timeline, watermarks)

    assert completed.selected_scores
    assert completed.selected_scores[0].candidate.source_signals == [
        "whisper_speech"
    ]
    assert len(completed.render_jobs) == 1


def test_coordinator_uses_only_explicit_observer_watermark(tmp_path: Path) -> None:
    timeline = feature_timeline(
        tmp_path,
        [{"seen_at": 1, "start": 0, "end": 5, "score": 0.8, "name": "one"}],
    )
    renderer = RecordingRenderer(tmp_path)
    pipeline = coordinator(tmp_path, renderer)

    assert pipeline.advance(timeline, watermarks(9.9)) == []
    assert pipeline.advance(timeline, watermarks(10))
    assert pipeline.watermark_seconds == 10


def test_slower_whisper_watermark_blocks_audio_ahead(tmp_path: Path) -> None:
    timeline = feature_timeline(
        tmp_path,
        [{"seen_at": 1, "start": 0, "end": 5, "score": 0.8, "name": "speech"}],
    )
    renderer = RecordingRenderer(tmp_path)
    incremental = coordinator(
        tmp_path,
        renderer,
        required_observers=("audio", "whisper"),
    )

    assert incremental.advance(
        timeline,
        ObserverWatermarks({"audio": 20.0, "whisper": 9.0}),
    ) == []
    assert incremental.watermark_seconds == 9.0
    jobs = incremental.advance(
        timeline,
        ObserverWatermarks({"audio": 20.0, "whisper": 10.0}),
    )

    assert len(jobs) == 1
    assert incremental.watermark_seconds == 10.0


def test_combined_audio_and_whisper_eof_allows_final_flush(tmp_path: Path) -> None:
    timeline = feature_timeline(
        tmp_path,
        [{"seen_at": 1, "start": 0, "end": 5, "score": 0.8, "name": "speech"}],
    )
    renderer = RecordingRenderer(tmp_path)
    incremental = coordinator(
        tmp_path,
        renderer,
        required_observers=("audio", "whisper"),
    )

    result = incremental.flush(
        timeline,
        IncrementalEOF(
            12.0,
            ObserverWatermarks({"audio": 12.0, "whisper": 12.0}),
        ),
    )

    assert incremental.lifecycle is CoordinatorLifecycle.FLUSHED
    assert len(result.render_jobs) == 1


def test_replay_adapter_withholds_retroactive_observation(tmp_path: Path) -> None:
    timeline = feature_timeline(
        tmp_path,
        [
            {"seen_at": 5, "start": 0, "end": 10, "score": 0.6, "name": "weak"},
            {
                "seen_at": 0,
                "duration": 100,
                "start": 1,
                "end": 11,
                "score": 0.9,
                "name": "retroactive-strong",
            },
        ],
    )
    adapter = CompletedTimelineReplayAdapter()

    assert adapter.watermarks_at(timeline, 70).stable_through == {"fixture": 0}


class UnsafeGenerator(MarkerGenerator):
    maximum_backtrack_seconds = float("inf")


class UnboundedCompetitionGenerator(MarkerGenerator):
    maximum_competition_seconds = float("inf")


class GlobalStateScorer(MarkerScorer):
    candidate_local_deterministic = False


def test_unsafe_generator_backtracking_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        coordinator(tmp_path, RecordingRenderer(tmp_path), generator=UnsafeGenerator())


def test_unbounded_selector_competition_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="maximum_competition_seconds"):
        coordinator(
            tmp_path,
            RecordingRenderer(tmp_path),
            generator=UnboundedCompetitionGenerator(),
        )


def test_global_state_scorer_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="candidate-local"):
        coordinator(tmp_path, RecordingRenderer(tmp_path), scorer=GlobalStateScorer())


def test_fingerprint_is_cross_platform_path_independent() -> None:
    windows = ClipCandidate(Path(r"C:\media\source.mp4"), 0, 10, "same")
    linux = ClipCandidate(Path("/media/source.mp4"), 0, 10, "same")

    assert candidate_fingerprint(windows, "youtube:abc") == candidate_fingerprint(
        linux, "youtube:abc"
    )


@pytest.mark.parametrize("bad", [object(), float("nan"), float("inf")])
def test_fingerprint_rejects_unsupported_or_nonfinite_metadata(bad: object) -> None:
    candidate = ClipCandidate(Path("source.mp4"), 0, 10, "bad", metadata={"bad": bad})

    with pytest.raises((TypeError, ValueError)):
        candidate_fingerprint(candidate, "stable-source")


class FailingOnceRenderer(RecordingRenderer):
    def __init__(self, output_dir: Path, error: BaseException) -> None:
        super().__init__(output_dir)
        self.error = error
        self.attempted_identities: list[int] = []

    def render_one(self, score: ClipScore, identity: int) -> RenderJob:
        self.attempted_identities.append(identity)
        if len(self.attempted_identities) == 1:
            raise self.error
        return super().render_one(score, identity)


def test_render_retry_reuses_identity_and_output(tmp_path: Path) -> None:
    timeline = feature_timeline(
        tmp_path,
        [{"seen_at": 1, "start": 0, "end": 5, "score": 0.8, "name": "retry"}],
    )
    renderer = FailingOnceRenderer(tmp_path, RuntimeError("failed"))
    pipeline = coordinator(tmp_path, renderer)

    with pytest.raises(RuntimeError):
        pipeline.advance(timeline, watermarks(10))
    jobs = pipeline.advance(timeline, watermarks(10))

    assert renderer.attempted_identities == [1, 1]
    assert [job.output_path.name for job in jobs] == ["clip-001.mp4"]


def test_interruption_recovers_rendering_as_retryable(tmp_path: Path) -> None:
    timeline = feature_timeline(
        tmp_path,
        [{"seen_at": 1, "start": 0, "end": 5, "score": 0.8, "name": "retry"}],
    )
    renderer = FailingOnceRenderer(tmp_path, KeyboardInterrupt())
    pipeline = coordinator(tmp_path, renderer)
    score = MarkerScorer().score(MarkerGenerator().generate(timeline))[0]

    with pytest.raises(KeyboardInterrupt):
        pipeline.advance(timeline, watermarks(10))
    assert pipeline.render_state(score) is RenderLifecycleState.FAILED
    assert pipeline.advance(timeline, watermarks(10))
    assert pipeline.render_state(score) is RenderLifecycleState.RENDERED


def test_coordinator_lifecycle_is_explicitly_single_use(tmp_path: Path) -> None:
    timeline = feature_timeline(tmp_path, [])
    pipeline = coordinator(tmp_path, RecordingRenderer(tmp_path))

    pipeline.flush(timeline, eof(1))
    assert pipeline.lifecycle is CoordinatorLifecycle.FLUSHED
    with pytest.raises(RuntimeError, match="already been flushed"):
        pipeline.advance(timeline, watermarks(1))
    with pytest.raises(RuntimeError, match="already been flushed"):
        pipeline.flush(timeline, eof(1))


def test_coordinator_rejects_second_source(tmp_path: Path) -> None:
    first = feature_timeline(tmp_path, [], source_id="first")
    second = feature_timeline(tmp_path, [], source_id="second")
    pipeline = coordinator(tmp_path, RecordingRenderer(tmp_path))

    pipeline.advance(first, watermarks(0))
    with pytest.raises(RuntimeError, match="single-use"):
        pipeline.advance(second, watermarks(0))


def test_eof_waits_for_every_required_observer(tmp_path: Path) -> None:
    timeline = feature_timeline(tmp_path, [])
    pipeline = coordinator(
        tmp_path,
        RecordingRenderer(tmp_path),
        required_observers=("fixture", "other"),
    )

    with pytest.raises(ValueError, match="Missing required observer"):
        pipeline.flush(timeline, IncrementalEOF(10, ObserverWatermarks({"fixture": 10})))
    with pytest.raises(ValueError, match="final media duration"):
        pipeline.flush(
            timeline,
            IncrementalEOF(10, ObserverWatermarks({"fixture": 10, "other": 9})),
        )


def test_later_stronger_overlap_and_batch_selection_match(tmp_path: Path) -> None:
    timeline = feature_timeline(
        tmp_path,
        [
            {"seen_at": 1, "start": 0, "end": 10, "score": 0.6, "name": "weak"},
            {"seen_at": 2, "start": 1, "end": 11, "score": 0.9, "name": "strong"},
            {"seen_at": 3, "start": 20, "end": 25, "score": 0.7, "name": "later"},
        ],
    )
    renderer = RecordingRenderer(tmp_path)
    pipeline = coordinator(tmp_path, renderer)
    pipeline.advance(timeline, watermarks(30))
    result = pipeline.flush(timeline, eof(30))
    scores = MarkerScorer().score(MarkerGenerator().generate(timeline))
    batch = CandidateSelector().select(scores)

    assert [call[0].candidate.reason for call in renderer.calls] == ["strong", "later"]
    assert {item.candidate.reason for item in result.selected_scores} == {
        item.candidate.reason for item in batch.selected
    }
    assert {item.score.candidate.reason for item in result.suppressed} == {
        item.score.candidate.reason for item in batch.suppressed
    }


def test_real_components_replay_matches_batch_after_multiple_watermarks(
    tmp_path: Path,
) -> None:
    source = tmp_path / "realistic-source.mp4"
    source.write_bytes(b"fixture")

    def audio_cluster(timestamp: float, intensity: float, peak: float):
        return [
            Observation(
                timestamp - 2.0,
                "audio",
                "silence",
                {"loudness_dbfs": -55.0},
                duration_seconds=2.0,
            ),
            Observation(timestamp, "audio", "peak", {"amplitude": peak}),
            Observation(
                timestamp,
                "audio",
                "speaking_intensity",
                {"intensity": intensity, "loudness_dbfs": -20.0},
            ),
        ]

    audio = [
        *audio_cluster(10.0, 0.80, 0.95),
        *audio_cluster(12.1, 0.96, 0.99),
        *[
            Observation(
                float(timestamp),
                "audio",
                "speaking_intensity",
                {"intensity": 0.40, "loudness_dbfs": -30.0},
                duration_seconds=1.0,
            )
            for timestamp in range(13, 66)
        ],
        *audio_cluster(100.0, 0.85, 0.97),
    ]
    whisper = [
        Observation(10.0, "whisper", "speech", {"text": "ordinary"}, confidence=0.75),
        Observation(12.1, "whisper", "speech", {"text": "WOW!!!"}, confidence=0.75),
        Observation(100.0, "whisper", "speech", {"text": "later"}, confidence=0.75),
    ]
    observer_results = [
        ObserverResult("audio", audio, {"duration_seconds": 150.0}),
        ObserverResult("whisper", whisper, {"duration_seconds": 150.0}),
    ]
    grouped: dict[float, list[Observation]] = {}
    for result_item in observer_results:
        for observation in result_item.observations:
            grouped.setdefault(observation.timestamp_seconds, []).append(observation)
    timeline = FeatureTimeline(
        media_path=source,
        audio_path=tmp_path / "audio.wav",
        timeline_path=tmp_path / "timeline.json",
        timeline=AggregatedTimeline(
            groups=[
                TimelineGroup(timestamp, grouped[timestamp])
                for timestamp in sorted(grouped)
            ],
            observer_results=observer_results,
        ),
        metadata={"source_id": "fixture:real-components"},
    )

    batch_generator = CandidateGenerator()
    batch_scorer = CandidateScorer()
    batch_selector = CandidateSelector()
    batch_scores = batch_scorer.score(batch_generator.generate(timeline))
    assert all(
        score.candidate.end_seconds - score.candidate.start_seconds < 60.0
        for score in batch_scores
    )
    batch_selection = batch_selector.select(
        score for score in batch_scores if score.passed_threshold is True
    )

    class TrackingCoordinator(IncrementalPrerecordedCoordinator):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            self.advance_watermarks: list[float] = []

        def advance(self, timeline, observer_watermarks):
            self.advance_watermarks.append(
                min(observer_watermarks.stable_through.values())
            )
            return super().advance(timeline, observer_watermarks)

    renderer = RecordingRenderer(tmp_path)
    incremental = TrackingCoordinator(
        candidate_generator=CandidateGenerator(),
        candidate_scorer=CandidateScorer(),
        candidate_selector=CandidateSelector(),
        clip_renderer=renderer,
        config=IncrementalPipelineConfig(required_observers=("audio", "whisper")),
    )
    replay = CompletedTimelineReplayAdapter(
        CompletedTimelineReplayConfig(observation_batch_seconds=30.0)
    )
    result = replay.run(incremental, timeline, media_duration_seconds=150.0)

    identity = lambda score: (
        score.candidate.start_seconds,
        score.candidate.end_seconds,
    )
    assert [identity(score) for score in result.scores] == [
        identity(score) for score in batch_scores
    ]
    assert [score.overall_score for score in result.scores] == [
        score.overall_score for score in batch_scores
    ]
    editorial = EditorialStrengthEvaluator()
    assert editorial.evaluate(batch_scores, "fixture:real-components") == (
        editorial.evaluate(result.scores, "fixture:real-components")
    )
    assert {identity(score) for score in result.selected_scores} == {
        identity(score) for score in batch_selection.selected
    }
    incremental_suppressed = {
        identity(item.score): (
            identity(item.retained_score),
            item.overlap_seconds,
            item.overlap_ratio,
            item.reason,
        )
        for item in result.suppressed
    }
    batch_suppressed = {
        identity(item.score): (
            identity(item.retained_score),
            item.overlap_seconds,
            item.overlap_ratio,
            item.reason,
        )
        for item in batch_selection.suppressed
    }
    assert incremental_suppressed == batch_suppressed
    assert incremental.advance_watermarks == [30.0, 60.0, 90.0, 120.0]
    assert incremental.watermark_seconds == 150.0
    assert incremental.required_observers == ("audio", "whisper")


class CreatingRunner:
    def run(self, command):
        command = list(command)
        Path(command[-1]).write_bytes(b"clip")
        return subprocess.CompletedProcess(command, 0, "", "")


def test_batch_renderer_limit_remains_isolated(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    renderer = ClipRenderer(
        ClipRendererConfig(output_dir=tmp_path / "clips", maximum_clips=1),
        runner=CreatingRunner(),
        executable_locator=lambda _: "ffmpeg",
    )
    scores = [
        ClipScore(ClipCandidate(source, 0, 5, "first"), 0.8),
        ClipScore(ClipCandidate(source, 10, 15, "second"), 0.7),
    ]

    assert len(renderer.render(scores)) == 1
