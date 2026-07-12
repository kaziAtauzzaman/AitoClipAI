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
            {"temperature": 0.0, "task": "transcribe", "language": "en"},
        )
    ]
    assert result.segments[0].speaker == "A"
    assert result.segments[0].confidence == 0.88
    assert result.segments[0].metadata["avg_logprob"] == -0.2


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
