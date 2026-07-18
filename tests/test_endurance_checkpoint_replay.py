"""Serialized Validation 05 continuation replay without media execution."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path

from aggregation import FeatureAggregator
from candidate_generation import CandidateGenerator
from candidate_scoring import CandidateScorer
from candidate_selection import CandidateSelector
from core import FeatureTimeline, Observation, ObserverResult, RenderJob
from pipeline import (
    IncrementalEOF,
    IncrementalPipelineConfig,
    IncrementalPrerecordedCoordinator,
    ObserverDeltaIdentity,
    ObserverWatermarks,
)
from pipeline.incremental import candidate_fingerprint, score_fingerprint
from whisper_observer import finalized_speech_segment_identity


ARTIFACT = (
    Path(__file__).parents[1]
    / "data"
    / "validation"
    / "youtube-UMjrTuMomlc-endurance-qsv-05-revision-safe"
)
MEDIA_DURATION_SECONDS = 8760.0
SESSION_ID = "validation-05-serialized-replay"


class RecordingRenderer:
    """Record immutable render plans without invoking a media backend."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.calls: list[tuple[object, int]] = []
        self.completions: dict[int, RenderJob] = {}

    def render_one(self, score, identity: int) -> RenderJob:
        self.calls.append((score, identity))
        job = RenderJob(
            score.candidate,
            self.output_dir / f"clip-{identity:03d}.mp4",
            metadata={"incremental_render_identity": identity},
        )
        self.completions[identity] = job
        return job

    def recover_render(self, score, identity: int) -> RenderJob | None:
        job = self.completions.get(identity)
        if job is not None and job.candidate != score.candidate:
            raise RuntimeError("Render identity changed candidate ownership.")
        return job


class RecordingCandidateGenerator(CandidateGenerator):
    """Capture closed-family emission order before coordinator scoring."""

    def __init__(self) -> None:
        super().__init__()
        self.closed_families = []
        self.transition_calls = 0

    def advance_incremental(self, *args, **kwargs):
        output = super().advance_incremental(*args, **kwargs)
        self.transition_calls += 1
        self.closed_families.extend(output.closed_families)
        return output

    def finalize_incremental(self, *args, **kwargs):
        output = super().finalize_incremental(*args, **kwargs)
        self.transition_calls += 1
        self.closed_families.extend(output.closed_families)
        return output


def _observations(values: list[dict[str, object]]) -> list[Observation]:
    return [
        Observation(
            float(item["timestamp_seconds"]),
            str(item["observer"]),
            str(item["type"]),
            deepcopy(item["value"]),
            item.get("duration_seconds"),
            item.get("confidence"),
            deepcopy(item.get("metadata", {})),
        )
        for item in values
    ]


def _timeline(
    media_path: Path,
    audio_path: Path,
    source_id: str,
    results: list[ObserverResult],
) -> FeatureTimeline:
    return FeatureTimeline(
        media_path,
        audio_path,
        ARTIFACT / "serialized-replay.json",
        FeatureAggregator().aggregate(results),
        metadata={"source_id": source_id},
    )


def _candidate_key(candidate) -> tuple[float, float, str]:
    return candidate.start_seconds, candidate.end_seconds, candidate.reason


def test_validation_04_incident_boundaries_use_real_generator_continuation(
    tmp_path: Path,
) -> None:
    report = json.loads(
        (ARTIFACT / "production-incremental-report.json").read_text("utf-8")
    )
    raw = [
        item
        for item in report["observations"]
        if float(item["timestamp_seconds"]) <= 160.0
    ]
    incremental_observations = _observations(raw)
    batch_observations = _observations(raw)
    source_id = report["source_id"]
    media_path = Path(report["source_video"])
    audio_path = Path(report["audio_source"])
    batch = CandidateGenerator().generate(
        _timeline(
            media_path,
            audio_path,
            source_id,
            [
                ObserverResult(
                    observer,
                    [
                        item
                        for item in batch_observations
                        if item.observer == observer
                    ],
                    {"duration_seconds": 160.0},
                )
                for observer in ("audio", "whisper")
            ],
        )
    )
    generator = CandidateGenerator()
    checkpoint = generator.start_incremental(
        source_id=source_id,
        media_path=media_path,
        required_observers=("audio", "whisper"),
    )
    chunks = [
        [item for item in incremental_observations if item.observer == "whisper"],
        [item for item in incremental_observations if item.observer == "audio"][:180],
        [item for item in incremental_observations if item.observer == "audio"][180:],
    ]
    closed = []
    for chunk in chunks:
        advance = generator.advance_incremental(
            checkpoint,
            chunk,
            160.0,
            {"audio": 0.0, "whisper": 0.0},
        )
        assert advance.checkpoint is not None
        assert advance.closed_families == ()
        generator.commit_incremental(checkpoint, advance)
        checkpoint = advance.checkpoint

    advance = generator.advance_incremental(
        checkpoint,
        (),
        160.0,
        {"audio": 160.0, "whisper": 160.0},
    )
    assert advance.checkpoint is not None
    closed.extend(advance.closed_families)
    generator.commit_incremental(checkpoint, advance)
    final = generator.finalize_incremental(advance.checkpoint, (), 160.0)
    closed.extend(final.closed_families)
    generator.commit_incremental(advance.checkpoint, final)

    incremental = [
        family.candidate
        for family in closed
        if family.candidate is not None
    ]
    assert incremental == batch
    boundaries = Counter(
        (candidate.start_seconds, candidate.end_seconds)
        for candidate in incremental
    )
    assert boundaries[(23.0, 55.52)] == 1
    assert boundaries[(73.0, 108.0)] == 1
    assert boundaries[(102.6035, 135.0)] == 1
    assert boundaries[(85.225, 119.0)] == 0
    assert boundaries[(123.0, 156.0)] == 0


def test_validation_05_serialized_observations_match_authoritative_batch(
    tmp_path: Path,
) -> None:
    summary = json.loads((ARTIFACT / "state-cache-summary.json").read_text("utf-8"))
    report = json.loads(
        (ARTIFACT / "production-incremental-report.json").read_text("utf-8")
    )
    observations = _observations(report["observations"])
    batch_observations = _observations(report["observations"])
    assert len(observations) == 18_939
    assert len(batch_observations) == len(observations)
    assert all(
        incremental is not completed
        for incremental, completed in zip(observations, batch_observations)
    )
    assert all(
        incremental.value is not completed.value
        for incremental, completed in zip(observations, batch_observations)
        if isinstance(incremental.value, dict | list)
    )
    assert summary["observation_totals"] == {"audio": 18_575, "whisper": 364}

    source_id = report["source_id"]
    media_path = Path(report["source_video"])
    audio_path = Path(report["audio_source"])
    by_observer = {
        name: [item for item in observations if item.observer == name]
        for name in ("audio", "whisper")
    }
    batch_by_observer = {
        name: [item for item in batch_observations if item.observer == name]
        for name in ("audio", "whisper")
    }
    complete = _timeline(
        media_path,
        audio_path,
        source_id,
        [
            ObserverResult(
                name,
                list(items),
                {"duration_seconds": MEDIA_DURATION_SECONDS},
            )
            for name, items in batch_by_observer.items()
        ],
    )
    batch_generator = CandidateGenerator()
    scorer = CandidateScorer()
    selector = CandidateSelector()
    batch_candidates = batch_generator.generate(complete)
    batch_scores = scorer.score(batch_candidates)
    batch_passing = [item for item in batch_scores if item.passed_threshold is True]
    batch_selection = selector.select(batch_passing)

    renderer = RecordingRenderer(tmp_path)
    incremental_generator = RecordingCandidateGenerator()
    incremental_scorer = CandidateScorer()
    incremental_selector = CandidateSelector()
    assert incremental_generator is not batch_generator
    assert incremental_scorer is not scorer
    assert incremental_selector is not selector
    coordinator = IncrementalPrerecordedCoordinator(
        incremental_generator,
        incremental_scorer,
        incremental_selector,
        renderer,
        IncrementalPipelineConfig(
            required_observers=("audio", "whisper"),
            session_id=SESSION_ID,
        ),
    )
    positions = {"audio": 0, "whisper": 0}
    sequences = {"audio": 0, "whisper": 0}
    frames = {"audio": 0, "whisper": 0}
    watermarks = {"audio": 0.0, "whisper": 0.0}
    batch_events = [
        item for item in summary["events"] if item["event"] == "observer_batch"
    ]
    assert len(batch_events) == 2_105

    for event in batch_events:
        observer = event["observer"]
        watermark = float(event["watermark_seconds"])
        watermarks[observer] = watermark
        items = by_observer[observer]
        end = positions[observer]
        if event["eof"]:
            end = len(items)
        else:
            while end < len(items) and items[end].timestamp_seconds <= watermark:
                end += 1
        delta = items[positions[observer] : end]
        processed_through = max(
            [
                watermark,
                *(
                    item.timestamp_seconds + (item.duration_seconds or 0.0)
                    for item in delta
                ),
            ]
        )
        frames[observer] = max(
            frames[observer] + 1,
            math.ceil(processed_through * 1000.0),
        )
        metadata: dict[str, object] = {
            "incremental_frames_processed": frames[observer],
            "sample_rate_hz": 1000,
        }
        if observer == "audio":
            metadata["finalized_peak_timestamps_seconds"] = tuple(
                item.timestamp_seconds for item in delta if item.type == "peak"
            )
        else:
            metadata["finalized_speech_segment_identities"] = tuple(
                finalized_speech_segment_identity(item)
                for item in delta
                if item.type == "speech"
            )
        coordinator.advance_delta(
            _timeline(
                media_path,
                audio_path,
                source_id,
                [ObserverResult(observer, delta, metadata)],
            ),
            ObserverWatermarks(dict(watermarks)),
            ObserverDeltaIdentity(
                source_id,
                SESSION_ID,
                observer,
                sequences[observer],
                bool(event["eof"]),
            ),
        )
        positions[observer] = end
        sequences[observer] += 1

    assert positions == {name: len(items) for name, items in by_observer.items()}
    pre_eof_render_count = len(renderer.calls)
    assert pre_eof_render_count == 172
    result = coordinator.flush_delta(
        _timeline(media_path, audio_path, source_id, []),
        IncrementalEOF(
            MEDIA_DURATION_SECONDS,
            ObserverWatermarks(
                {
                    "audio": MEDIA_DURATION_SECONDS,
                    "whisper": MEDIA_DURATION_SECONDS,
                }
            ),
        ),
    )

    batch_candidate_fingerprints = [
        candidate_fingerprint(item, source_id) for item in batch_candidates
    ]
    emitted_families = incremental_generator.closed_families
    emitted_candidates = [
        family.candidate
        for family in emitted_families
        if family.candidate is not None
    ]
    emitted_candidate_fingerprints = [
        candidate_fingerprint(item, source_id) for item in emitted_candidates
    ]
    assert incremental_generator.transition_calls == 2_106
    assert len(emitted_families) == len(batch_candidates)
    assert [family.family_id.ordinal for family in emitted_families] == list(
        range(len(batch_candidates))
    )
    assert len(emitted_candidates) == len(batch_candidates)
    assert emitted_candidate_fingerprints == batch_candidate_fingerprints
    assert emitted_candidates == batch_candidates
    assert Counter(emitted_candidate_fingerprints) == Counter(
        batch_candidate_fingerprints
    )
    incremental_candidates = [item.candidate for item in result.scores]
    incremental_candidate_fingerprints = [
        candidate_fingerprint(item, source_id) for item in incremental_candidates
    ]
    batch_score_fingerprints = [
        score_fingerprint(item, source_id) for item in batch_scores
    ]
    incremental_score_fingerprints = [
        score_fingerprint(item, source_id) for item in result.scores
    ]

    assert (len(batch_candidates), len(batch_passing)) == (176, 174)
    assert (len(batch_selection.selected), len(batch_selection.suppressed)) == (174, 0)
    assert (len(result.scores), len(result.selected_scores)) == (176, 174)
    assert sum(item.passed_threshold is True for item in result.scores) == 174
    assert len(result.suppressed) == 0
    assert incremental_candidate_fingerprints == [
        candidate_fingerprint(item.candidate, source_id) for item in batch_scores
    ]
    assert Counter(incremental_candidate_fingerprints) == Counter(
        batch_candidate_fingerprints
    )
    assert incremental_score_fingerprints == batch_score_fingerprints
    assert result.scores == batch_scores
    assert [
        score_fingerprint(item, source_id) for item in result.selected_scores
    ] == [
        score_fingerprint(item, source_id) for item in batch_selection.selected
    ]
    assert Counter(
        score_fingerprint(item, source_id) for item in result.selected_scores
    ) == Counter(
        score_fingerprint(item, source_id) for item in batch_selection.selected
    )

    batch_ordered = sorted(batch_candidates, key=_candidate_key)
    incremental_ordered = sorted(incremental_candidates, key=_candidate_key)
    assert [
        (
            item.start_seconds,
            item.end_seconds,
            item.reason,
            item.source_signals,
            item.metadata,
        )
        for item in incremental_ordered
    ] == [
        (
            item.start_seconds,
            item.end_seconds,
            item.reason,
            item.source_signals,
            item.metadata,
        )
        for item in batch_ordered
    ]

    boundaries = Counter(
        (item.start_seconds, item.end_seconds) for item in incremental_candidates
    )
    assert boundaries[(23.0, 55.52)] == 1
    assert boundaries[(73.0, 108.0)] == 1
    assert boundaries[(102.6035, 135.0)] == 1
    assert boundaries[(8748.0, 8760.0)] == 1
    assert boundaries[(85.225, 119.0)] == 0
    assert boundaries[(123.0, 156.0)] == 0
    assert boundaries[(8723.0, 8758.0)] == 0

    rendered_boundaries = Counter(
        (score.candidate.start_seconds, score.candidate.end_seconds)
        for score, _ in renderer.calls
    )
    assert len(renderer.calls) == 174
    assert [identity for _, identity in renderer.calls] == list(range(1, 175))
    assert rendered_boundaries[(23.0, 55.52)] == 1
    assert Counter(
        score_fingerprint(score, source_id) for score, _ in renderer.calls
    ) == Counter(
        score_fingerprint(item, source_id) for item in batch_selection.selected
    )

    metrics = coordinator.state_metrics
    assert metrics.generation_passes == 2_106
    assert metrics.peak_active_observations == 416
    assert metrics.peak_active_scores == 2
    assert metrics.scored_candidates <= 176
    assert metrics.candidate_fingerprints <= 176
    assert metrics.score_fingerprints <= 176
    assert metrics.active_observations == 0
    assert metrics.active_scores == 0
    assert metrics.finalized_scores == 176
    assert metrics.immutable_score_fingerprints == 176
    assert metrics.completed_render_jobs == 174
