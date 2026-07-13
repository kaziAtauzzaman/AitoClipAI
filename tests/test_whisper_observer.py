import json
import math
import shutil
import struct
import wave
from pathlib import Path
from types import ModuleType

import pytest

from aggregation import FeatureAggregator
from audio_observer import AudioObserver, FFmpegAudioExtractor, FFmpegAudioExtractorConfig
from observers import ObserverContext, ObserverEngine, ObserverRegistry
from pipeline import PipelineOrchestrator
from whisper_observer import (
    InvalidTranscriptionError,
    IncrementalWhisperAudioChunk,
    IncrementalWhisperEOF,
    IncrementalWhisperObserverConfig,
    IncrementalWhisperSessionCore,
    IncrementalWavWhisperObserver,
    OpenAIWhisperBackend,
    TranscriptionError,
    TranscriptionResult,
    TranscriptionSegment,
    WhisperObserver,
    WhisperObserverConfig,
    WhisperUnavailableError,
)


class FakeTranscriptionBackend:
    def __init__(self, result: TranscriptionResult) -> None:
        self.result = result
        self.calls: list[tuple[Path, WhisperObserverConfig]] = []

    def transcribe(
        self,
        audio_path: Path,
        config: WhisperObserverConfig,
    ) -> TranscriptionResult:
        self.calls.append((audio_path, config))
        return self.result


class FixtureAudioBackend:
    """Offline backend that derives a segment boundary from fixture WAV data."""

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
                    start_seconds=0.0,
                    end_seconds=duration,
                    text="offline fixture speech",
                    speaker="fixture-speaker",
                    confidence=0.97,
                )
            ],
            text="offline fixture speech",
            language="en",
            metadata={"backend": "fixture"},
        )


class ScriptedIncrementalModelSession:
    def __init__(self, results: list[TranscriptionResult]) -> None:
        self.results = list(results)
        self.calls: list[tuple[float, str | None]] = []
        self.close_calls = 0

    def transcribe(
        self,
        audio_path: Path,
        initial_prompt: str | None,
    ) -> TranscriptionResult:
        with wave.open(str(audio_path), "rb") as audio:
            duration = audio.getnframes() / audio.getframerate()
        self.calls.append((duration, initial_prompt))
        return self.results.pop(0) if self.results else TranscriptionResult()

    def close(self) -> None:
        self.close_calls += 1


class ScriptedIncrementalBackend:
    def __init__(self, results: list[TranscriptionResult]) -> None:
        self.results = results
        self.open_calls: list[WhisperObserverConfig] = []
        self.session: ScriptedIncrementalModelSession | None = None

    def open_incremental_session(
        self,
        config: WhisperObserverConfig,
    ) -> ScriptedIncrementalModelSession:
        self.open_calls.append(config)
        self.session = ScriptedIncrementalModelSession(self.results)
        return self.session


def write_audio_fixture(path: Path) -> None:
    sample_rate_hz = 8_000
    samples = [
        int(6_000 * math.sin(2 * math.pi * 220 * index / sample_rate_hz))
        for index in range(1_600)
    ]
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate_hz)
        audio.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))


def write_incremental_whisper_fixture(
    path: Path,
    duration_seconds: float,
    *,
    sample_rate_hz: int = 10,
) -> None:
    frame_count = round(duration_seconds * sample_rate_hz)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate_hz)
        audio.writeframes(b"\x00\x00" * frame_count)


def incremental_whisper_observer(
    results: list[TranscriptionResult],
    *,
    prompt_max_characters: int = 1_000,
):
    backend = ScriptedIncrementalBackend(results)
    observer = IncrementalWavWhisperObserver(
        IncrementalWhisperObserverConfig(
            chunk_seconds=4.0,
            overlap_seconds=1.0,
            prompt_max_characters=prompt_max_characters,
        ),
        backend,
    )
    return observer, backend


def test_whisper_observer_emits_timestamped_speech_observations(
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fixture")
    transcription = TranscriptionResult(
        segments=[
            TranscriptionSegment(
                start_seconds=1.25,
                end_seconds=2.75,
                text="Hello there",
                speaker="speaker-1",
                confidence=0.92,
                metadata={"segment_id": 7},
            ),
            TranscriptionSegment(
                start_seconds=3.0,
                end_seconds=4.0,
                text="General Kenobi",
            ),
        ],
        text="Hello there General Kenobi",
        language="en",
    )
    backend = FakeTranscriptionBackend(transcription)
    config = WhisperObserverConfig(model_name="small", language="en")
    observer = WhisperObserver(config=config, backend=backend)

    result = observer.observe(ObserverContext(source_path=audio_path))
    timeline = FeatureAggregator().aggregate([result])

    assert backend.calls == [(audio_path, config)]
    assert result.observer == "whisper"
    assert result.metadata["model_name"] == "small"
    assert result.metadata["language"] == "en"
    assert result.metadata["segment_count"] == 2
    assert [group.timestamp_seconds for group in timeline.groups] == [1.25, 3.0]
    first = timeline.groups[0].observations[0]
    assert first.type == "speech"
    assert first.duration_seconds == 1.5
    assert first.value == {"text": "Hello there", "speaker": "speaker-1"}
    assert first.confidence == 0.92
    assert first.metadata == {"segment_id": 7, "speaker": "speaker-1"}


def test_whisper_observer_rejects_invalid_segments(tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fixture")
    backend = FakeTranscriptionBackend(
        TranscriptionResult(
            segments=[
                TranscriptionSegment(
                    start_seconds=2.0,
                    end_seconds=1.0,
                    text="invalid",
                )
            ]
        )
    )

    with pytest.raises(InvalidTranscriptionError, match="timestamps"):
        WhisperObserver(backend=backend).observe(
            ObserverContext(source_path=audio_path)
        )


def test_whisper_failure_is_isolated_by_observer_engine(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.wav"
    engine = ObserverEngine(
        ObserverRegistry(observers=[WhisperObserver(backend=FixtureAudioBackend())])
    )

    result = engine.run(ObserverContext(source_path=missing_path))

    assert result.results == []
    assert len(result.failures) == 1
    assert result.failures[0].observer == "whisper"
    assert result.failures[0].error_type == "TranscriptionError"


def test_openai_backend_uses_configured_model_and_preserves_fields(
    tmp_path: Path,
) -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []

        def transcribe(self, path: str, **options: object) -> dict[str, object]:
            self.calls.append((path, options))
            return {
                "text": "hello",
                "language": "en",
                "segments": [
                    {
                        "start": 0.0,
                        "end": 0.5,
                        "text": "hello",
                        "speaker": "A",
                        "confidence": 0.88,
                        "avg_logprob": -0.2,
                    }
                ],
            }

    fake_model = FakeModel()
    loaded: list[tuple[str, dict[str, object]]] = []
    fake_module = ModuleType("whisper")

    def load_model(name: str, **options: object) -> FakeModel:
        loaded.append((name, options))
        return fake_model

    fake_module.load_model = load_model  # type: ignore[attr-defined]
    backend = OpenAIWhisperBackend(module_loader=lambda name: fake_module)
    config = WhisperObserverConfig(
        model_name="large-v3",
        language="en",
        task="transcribe",
        device="cpu",
        options={"temperature": 0.0},
    )
    audio_path = tmp_path / "fixture.wav"

    result = backend.transcribe(audio_path, config)

    assert loaded == [("large-v3", {"device": "cpu"})]
    assert fake_model.calls == [
        (
            str(audio_path),
            {
                "temperature": 0.0,
                "task": "transcribe",
                "condition_on_previous_text": True,
                "fp16": False,
                "language": "en",
            },
        )
    ]
    assert result.segments[0].speaker == "A"
    assert result.segments[0].confidence == 0.88
    assert result.segments[0].metadata["avg_logprob"] == -0.2


def test_openai_backend_enforces_deterministic_decoding_options(
    tmp_path: Path,
) -> None:
    class FakeDevice:
        type = "cpu"

    class FakeModel:
        device = FakeDevice()

        def __init__(self) -> None:
            self.options: list[dict[str, object]] = []

        def transcribe(self, path: str, **options: object) -> dict[str, object]:
            self.options.append(options)
            return {"text": "stable", "segments": []}

    model = FakeModel()
    module = ModuleType("whisper")
    module.load_model = lambda name, **options: model  # type: ignore[attr-defined]
    backend = OpenAIWhisperBackend(module_loader=lambda name: module)
    config = WhisperObserverConfig(
        options={
            "temperature": (0.0, 0.2, 0.4),
            "condition_on_previous_text": False,
            "fp16": True,
            "initial_prompt": "preserve this option",
        }
    )

    backend.transcribe(tmp_path / "audio.wav", config)

    assert model.options == [
        {
            "temperature": 0.0,
            "condition_on_previous_text": True,
            "fp16": False,
            "initial_prompt": "preserve this option",
            "task": "transcribe",
        }
    ]


def test_openai_backend_preserves_legacy_fallback_when_determinism_disabled(
    tmp_path: Path,
) -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.options: list[dict[str, object]] = []

        def transcribe(self, path: str, **options: object) -> dict[str, object]:
            self.options.append(options)
            return {"text": "legacy", "segments": []}

    model = FakeModel()
    module = ModuleType("whisper")
    module.load_model = lambda name, **options: model  # type: ignore[attr-defined]
    backend = OpenAIWhisperBackend(module_loader=lambda name: module)
    config = WhisperObserverConfig(
        deterministic=False,
        options={"temperature": (0.0, 0.2), "fp16": True},
    )

    backend.transcribe(tmp_path / "audio.wav", config)

    assert model.options == [
        {
            "temperature": (0.0, 0.2),
            "fp16": True,
            "task": "transcribe",
        }
    ]


def test_deterministic_backend_repeated_runs_are_equal(tmp_path: Path) -> None:
    class FallbackSensitiveModel:
        device = "cpu"

        def __init__(self) -> None:
            self.call_count = 0

        def transcribe(self, path: str, **options: object) -> dict[str, object]:
            self.call_count += 1
            deterministic = options.get("temperature") == 0.0
            suffix = "stable" if deterministic else f"sample-{self.call_count}"
            return {
                "text": suffix,
                "language": "en",
                "segments": [
                    {
                        "start": 1.0,
                        "end": 2.0,
                        "text": suffix,
                        "temperature": options.get("temperature"),
                    }
                ],
            }

    model = FallbackSensitiveModel()
    module = ModuleType("whisper")
    module.load_model = lambda name, **options: model  # type: ignore[attr-defined]
    backend = OpenAIWhisperBackend(module_loader=lambda name: module)
    config = WhisperObserverConfig()
    audio_path = tmp_path / "audio.wav"

    first = backend.transcribe(audio_path, config)
    second = backend.transcribe(audio_path, config)

    assert first == second
    assert first.text == "stable"
    assert first.segments[0].metadata["temperature"] == 0.0


def test_deterministic_backend_does_not_force_fp16_off_for_gpu(
    tmp_path: Path,
) -> None:
    class FakeModel:
        device = "cuda:0"

        def __init__(self) -> None:
            self.options: dict[str, object] = {}

        def transcribe(self, path: str, **options: object) -> dict[str, object]:
            self.options = options
            return {"text": "gpu", "segments": []}

    model = FakeModel()
    module = ModuleType("whisper")
    module.load_model = lambda name, **options: model  # type: ignore[attr-defined]
    backend = OpenAIWhisperBackend(module_loader=lambda name: module)

    backend.transcribe(tmp_path / "audio.wav", WhisperObserverConfig(device="cuda:0"))

    assert model.options["temperature"] == 0.0
    assert model.options["condition_on_previous_text"] is True
    assert "fp16" not in model.options


def test_openai_backend_reports_missing_optional_runtime(tmp_path: Path) -> None:
    def missing_module(name: str) -> ModuleType:
        raise ModuleNotFoundError(name)

    backend = OpenAIWhisperBackend(module_loader=missing_module)

    with pytest.raises(WhisperUnavailableError, match="not installed"):
        backend.transcribe(tmp_path / "audio.wav", WhisperObserverConfig())


def test_unexpected_backend_failure_becomes_typed_error(tmp_path: Path) -> None:
    class BrokenBackend:
        def transcribe(
            self,
            audio_path: Path,
            config: WhisperObserverConfig,
        ) -> TranscriptionResult:
            raise RuntimeError("backend crashed")

    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fixture")

    with pytest.raises(TranscriptionError, match="backend crashed"):
        WhisperObserver(backend=BrokenBackend()).observe(
            ObserverContext(source_path=audio_path)
        )


def test_incremental_whisper_rebases_chunk_timestamps_globally(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rebase.wav"
    write_incremental_whisper_fixture(path, 7.0)
    observer, _ = incremental_whisper_observer(
        [
            TranscriptionResult(
                segments=[TranscriptionSegment(1.0, 2.0, "first")]
            ),
            TranscriptionResult(
                segments=[TranscriptionSegment(0.5, 1.0, "second")]
            ),
        ]
    )

    observations = [
        item for batch in observer.batches(path) for item in batch.observations
    ]

    assert [
        (item.timestamp_seconds, item.duration_seconds, item.value["text"])
        for item in observations
    ] == [(1.0, 1.0, "first"), (3.5, 0.5, "second")]


def test_incremental_whisper_core_accepts_transport_neutral_pcm() -> None:
    core = IncrementalWhisperSessionCore(
        IncrementalWhisperObserverConfig(
            chunk_seconds=4.0,
            overlap_seconds=1.0,
        )
    )
    chunk = IncrementalWhisperAudioChunk(
        pcm_bytes=b"\x00\x00" * 40,
        sample_rate_hz=10,
        channels=1,
        sample_width_bytes=2,
        start_frame=0,
        end_frame=40,
        stable_through_frame=30,
    )

    batch = core.accept_chunk(
        chunk,
        TranscriptionResult(
            segments=[TranscriptionSegment(1.0, 2.0, "transport neutral")]
        ),
    )
    eof = core.flush(IncrementalWhisperEOF(final_frame=40, sample_rate_hz=10))

    assert [item.value["text"] for item in batch.observations] == [
        "transport neutral"
    ]
    assert batch.watermark_seconds == 3.0
    assert eof is not None and eof.watermark_seconds == 4.0
    assert core.flush(IncrementalWhisperEOF(40, 10)) is None


def test_incremental_whisper_deduplicates_overlap_segments(tmp_path: Path) -> None:
    path = tmp_path / "duplicates.wav"
    write_incremental_whisper_fixture(path, 7.0)
    observer, _ = incremental_whisper_observer(
        [
            TranscriptionResult(
                segments=[TranscriptionSegment(3.2, 3.8, "  Same   phrase ")]
            ),
            TranscriptionResult(
                segments=[TranscriptionSegment(0.2, 0.8, "same phrase")]
            ),
        ]
    )

    observations = [
        item for batch in observer.batches(path) for item in batch.observations
    ]

    assert len(observations) == 1
    assert observations[0].timestamp_seconds == 3.2
    assert observations[0].value["text"] == "same phrase"


@pytest.mark.parametrize(
    ("first_text", "revised_text"),
    [
        ("Wait, what?", "wait what"),
        ("hello world", "hello brave world"),
    ],
)
def test_incremental_whisper_reconciles_near_duplicate_overlap_phrases(
    tmp_path: Path,
    first_text: str,
    revised_text: str,
) -> None:
    path = tmp_path / "near-duplicates.wav"
    write_incremental_whisper_fixture(path, 7.0)
    observer, _ = incremental_whisper_observer(
        [
            TranscriptionResult(
                segments=[TranscriptionSegment(3.2, 3.8, first_text)]
            ),
            TranscriptionResult(
                segments=[TranscriptionSegment(0.2, 0.8, revised_text)]
            ),
        ]
    )

    observations = [
        item for batch in observer.batches(path) for item in batch.observations
    ]

    assert len(observations) == 1
    assert observations[0].value["text"] == revised_text


def test_incremental_whisper_holds_right_edge_until_next_chunk(
    tmp_path: Path,
) -> None:
    path = tmp_path / "provisional.wav"
    write_incremental_whisper_fixture(path, 7.0)
    observer, _ = incremental_whisper_observer(
        [
            TranscriptionResult(
                segments=[TranscriptionSegment(3.2, 3.8, "edge")]
            ),
            TranscriptionResult(
                segments=[TranscriptionSegment(0.2, 0.8, "edge")]
            ),
        ]
    )

    with observer.session(path) as session:
        first = session.read_batch()
        second = session.read_batch()

    assert first is not None and first.observations == ()
    assert first.metadata["provisional_segment_count"] == 1
    assert first.watermark_seconds <= 3.2
    assert second is not None
    assert [item.value["text"] for item in second.observations] == ["edge"]


def test_incremental_whisper_watermarks_are_monotonic_and_clamped(
    tmp_path: Path,
) -> None:
    path = tmp_path / "watermarks.wav"
    write_incremental_whisper_fixture(path, 10.0)
    observer, _ = incremental_whisper_observer(
        [
            TranscriptionResult(
                segments=[TranscriptionSegment(3.1, 3.9, "one")]
            ),
            TranscriptionResult(
                segments=[TranscriptionSegment(3.1, 3.9, "two")]
            ),
            TranscriptionResult(),
        ]
    )

    batches = list(observer.batches(path))
    watermarks = [batch.watermark_seconds for batch in batches]

    assert watermarks == sorted(watermarks)
    assert batches[0].watermark_seconds <= 3.1
    assert batches[1].watermark_seconds <= 6.1
    assert batches[-1].eof is True
    assert batches[-1].watermark_seconds == 10.0


def test_incremental_whisper_prompt_context_is_bounded(tmp_path: Path) -> None:
    path = tmp_path / "prompt.wav"
    write_incremental_whisper_fixture(path, 10.0)
    observer, backend = incremental_whisper_observer(
        [
            TranscriptionResult(
                segments=[TranscriptionSegment(0.1, 0.5, "first sentence")]
            ),
            TranscriptionResult(
                segments=[TranscriptionSegment(0.1, 0.5, "second sentence")]
            ),
            TranscriptionResult(),
        ],
        prompt_max_characters=12,
    )

    list(observer.batches(path))

    assert backend.session is not None
    prompts = [prompt for _, prompt in backend.session.calls]
    assert prompts[0] is None
    assert all(prompt is None or len(prompt) <= 12 for prompt in prompts)
    assert prompts[1] == "rst sentence"
    assert prompts[2] == "ond sentence"


def test_incremental_whisper_loads_model_once_per_session(tmp_path: Path) -> None:
    path = tmp_path / "model.wav"
    write_incremental_whisper_fixture(path, 10.0)
    observer, backend = incremental_whisper_observer(
        [TranscriptionResult(), TranscriptionResult(), TranscriptionResult()]
    )

    list(observer.batches(path))

    assert len(backend.open_calls) == 1
    assert backend.session is not None
    assert len(backend.session.calls) == 3
    assert backend.session.close_calls == 1


def test_incremental_whisper_retries_failed_chunk_without_skipping_or_duplication(
    tmp_path: Path,
) -> None:
    class RetryModelSession:
        def __init__(self) -> None:
            self.calls: list[float] = []

        def transcribe(
            self,
            audio_path: Path,
            initial_prompt: str | None,
        ) -> TranscriptionResult:
            with wave.open(str(audio_path), "rb") as audio:
                self.calls.append(audio.getnframes() / audio.getframerate())
            if len(self.calls) == 1:
                raise TranscriptionError("temporary failure")
            return TranscriptionResult(
                segments=[TranscriptionSegment(1.0, 2.0, "retry once")]
            )

        def close(self) -> None:
            pass

    class RetryBackend:
        def __init__(self) -> None:
            self.session = RetryModelSession()

        def open_incremental_session(self, config: WhisperObserverConfig):
            return self.session

    path = tmp_path / "retry.wav"
    write_incremental_whisper_fixture(path, 4.0)
    backend = RetryBackend()
    observer = IncrementalWavWhisperObserver(
        IncrementalWhisperObserverConfig(
            chunk_seconds=4.0,
            overlap_seconds=1.0,
        ),
        backend,
    )

    with observer.session(path) as session:
        with pytest.raises(TranscriptionError, match="temporary failure"):
            session.read_batch()
        successful = session.read_batch()
        eof = session.read_batch()

    assert backend.session.calls == [4.0, 4.0]
    assert successful is not None
    assert [item.value["text"] for item in successful.observations] == [
        "retry once"
    ]
    assert eof is not None and eof.observations == ()


def test_openai_incremental_session_reuses_one_loaded_model(tmp_path: Path) -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.calls = 0
            self.prompts: list[str | None] = []

        def transcribe(self, path: str, **options: object) -> dict[str, object]:
            self.calls += 1
            prompt = options.get("initial_prompt")
            self.prompts.append(prompt if isinstance(prompt, str) else None)
            return {"text": "", "segments": []}

    model = FakeModel()
    load_calls = 0

    def load_model(name: str, **options: object) -> FakeModel:
        nonlocal load_calls
        load_calls += 1
        return model

    module = ModuleType("whisper")
    module.load_model = load_model  # type: ignore[attr-defined]
    backend = OpenAIWhisperBackend(module_loader=lambda name: module)
    path = tmp_path / "openai-session.wav"
    write_incremental_whisper_fixture(path, 7.0)
    observer = IncrementalWavWhisperObserver(
        IncrementalWhisperObserverConfig(
            chunk_seconds=4.0,
            overlap_seconds=1.0,
            prompt_max_characters=5,
            analysis=WhisperObserverConfig(options={"initial_prompt": "unbounded"}),
        ),
        backend,
    )

    list(observer.batches(path))

    assert load_calls == 1
    assert model.calls == 2
    assert model.prompts == [None, None]


def test_incremental_whisper_flushes_provisional_at_eof_once(
    tmp_path: Path,
) -> None:
    path = tmp_path / "eof.wav"
    write_incremental_whisper_fixture(path, 2.0)
    observer, _ = incremental_whisper_observer(
        [
            TranscriptionResult(
                segments=[TranscriptionSegment(1.5, 2.0, "final words")]
            )
        ]
    )

    with observer.session(path) as session:
        first = session.read_batch()
        eof = session.read_batch()
        repeated = session.flush()

    assert first is not None and first.observations == ()
    assert eof is not None and eof.eof is True
    assert [item.value["text"] for item in eof.observations] == ["final words"]
    assert eof.watermark_seconds == 2.0
    assert repeated is None


def test_existing_whole_file_whisper_still_uses_single_full_source_call(
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "whole.wav"
    audio_path.write_bytes(b"fixture")
    backend = FakeTranscriptionBackend(
        TranscriptionResult(
            segments=[TranscriptionSegment(0.0, 1.0, "whole file")]
        )
    )
    config = WhisperObserverConfig()

    result = WhisperObserver(config=config, backend=backend).observe(
        ObserverContext(source_path=audio_path)
    )

    assert backend.calls == [(audio_path, config)]
    assert [item.value["text"] for item in result.observations] == ["whole file"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
def test_whisper_observer_offline_pipeline_integration(tmp_path: Path) -> None:
    media_path = tmp_path / "speech-fixture.wav"
    write_audio_fixture(media_path)
    observer_engine = ObserverEngine(
        ObserverRegistry(
            observers=[
                AudioObserver(),
                WhisperObserver(
                    config=WhisperObserverConfig(model_name="fixture-model"),
                    backend=FixtureAudioBackend(),
                ),
            ]
        )
    )
    orchestrator = PipelineOrchestrator(
        audio_extractor=FFmpegAudioExtractor(
            FFmpegAudioExtractorConfig(output_dir=tmp_path / "audio")
        ),
        observer_engine=observer_engine,
    )

    feature_timeline = orchestrator.analyze(media_path)

    assert feature_timeline.failures == []
    assert [
        result.observer for result in feature_timeline.timeline.observer_results
    ] == ["audio", "whisper"]
    speech = [
        observation
        for group in feature_timeline.timeline.groups
        for observation in group.observations
        if observation.observer == "whisper"
    ]
    assert len(speech) == 1
    assert speech[0].value["text"] == "offline fixture speech"
    assert speech[0].metadata["speaker"] == "fixture-speaker"
    assert speech[0].confidence == 0.97
    assert feature_timeline.timeline_path.is_file()
    persisted = json.loads(
        feature_timeline.timeline_path.read_text(encoding="utf-8")
    )
    persisted_observers = [
        result["observer"] for result in persisted["timeline"]["observer_results"]
    ]
    assert persisted_observers == ["audio", "whisper"]
