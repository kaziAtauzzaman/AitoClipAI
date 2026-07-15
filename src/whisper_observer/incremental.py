"""Transport-neutral incremental Whisper state with a prerecorded WAV adapter."""

from collections.abc import Iterator
from difflib import SequenceMatcher
import hashlib
from pathlib import Path
import re
import tempfile
import wave

from core import Observation
from whisper_observer.backend import OpenAIWhisperBackend
from whisper_observer.config import IncrementalWhisperObserverConfig
from whisper_observer.contracts import (
    IncrementalTranscriptionBackend,
    IncrementalWhisperAudioChunk,
    IncrementalWhisperBatch,
    IncrementalWhisperEOF,
    IncrementalWhisperLifecycle,
    SegmentReconciliationPolicy,
    TranscriptionResult,
    TranscriptionSegment,
    finalized_speech_segment_identity,
)
from whisper_observer.errors import InvalidTranscriptionError, TranscriptionError


class TokenOverlapReconciliationPolicy:
    """Reconcile temporal overlap using exact text, then token similarity."""

    def reconcile(
        self,
        existing: TranscriptionSegment,
        candidate: TranscriptionSegment,
        *,
        timestamp_tolerance_seconds: float,
        similarity_threshold: float,
    ) -> TranscriptionSegment | None:
        overlap = min(existing.end_seconds, candidate.end_seconds) - max(
            existing.start_seconds,
            candidate.start_seconds,
        )
        timestamps_close = (
            abs(existing.start_seconds - candidate.start_seconds)
            <= timestamp_tolerance_seconds
            and abs(existing.end_seconds - candidate.end_seconds)
            <= timestamp_tolerance_seconds
        )
        if _deduplication_text(existing.text) == _deduplication_text(candidate.text):
            return candidate if overlap > 0 or timestamps_close else None
        if overlap <= 0:
            return None
        existing_tokens = _normalized_tokens(existing.text)
        candidate_tokens = _normalized_tokens(candidate.text)
        if not existing_tokens or not candidate_tokens:
            return None
        similarity = SequenceMatcher(
            None,
            existing_tokens,
            candidate_tokens,
            autojunk=False,
        ).ratio()
        return candidate if similarity >= similarity_threshold else None


class IncrementalWhisperSessionCore:
    """Own transcript stability independently of prerecorded or live transport."""

    def __init__(
        self,
        config: IncrementalWhisperObserverConfig,
        reconciliation_policy: SegmentReconciliationPolicy | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self._config = config
        self._policy = reconciliation_policy or TokenOverlapReconciliationPolicy()
        self._metadata = dict(metadata or {})
        self._pending: list[TranscriptionSegment] = []
        self._recent_emitted: list[TranscriptionSegment] = []
        self._prompt = ""
        self._watermark = 0.0
        self._frames_processed = 0
        self._chunk_index = 0
        self._sample_rate: int | None = None
        self._channels: int | None = None
        self._sample_width: int | None = None
        self._expected_chunk_start: int | None = None
        self._last_chunk_identity: str | None = None
        self._last_chunk_batch: IncrementalWhisperBatch | None = None
        self._lifecycle = IncrementalWhisperLifecycle.NEW

    @property
    def prompt(self) -> str:
        return self._prompt

    @property
    def lifecycle(self) -> IncrementalWhisperLifecycle:
        return self._lifecycle

    def accept_chunk(
        self,
        chunk: IncrementalWhisperAudioChunk,
        result: TranscriptionResult,
    ) -> IncrementalWhisperBatch:
        """Commit one successfully transcribed chronological PCM chunk."""

        self._require_active()
        identity = _chunk_identity(chunk)
        if identity == self._last_chunk_identity:
            assert self._last_chunk_batch is not None
            return self._last_chunk_batch
        self._validate_chunk_sequence(chunk)
        self._validate_result(result)
        candidates = [self._global_segment(item, chunk) for item in result.segments]
        chunk_start = chunk.start_frame / chunk.sample_rate_hz
        stable_edge = chunk.stable_through_frame / chunk.sample_rate_hz
        recent = self._pruned_recent(chunk_start)
        pending = list(self._pending)
        for candidate in candidates:
            if self._matching_index(recent, candidate) is not None:
                continue
            duplicate_index = self._matching_index(pending, candidate)
            if duplicate_index is None:
                pending.append(candidate)
            else:
                selected = self._reconcile(pending[duplicate_index], candidate)
                assert selected is not None
                pending[duplicate_index] = selected
        pending.sort(key=_segment_key)
        stable = [item for item in pending if item.end_seconds <= stable_edge]
        pending = [item for item in pending if item.end_seconds > stable_edge]
        stable.sort(key=_segment_key)
        watermark = stable_edge
        if pending:
            watermark = min(watermark, min(item.start_seconds for item in pending))
        if watermark < self._watermark:
            raise TranscriptionError("Incremental Whisper watermark moved backwards.")

        self._pending = pending
        self._recent_emitted = recent + stable
        self._append_prompt(stable)
        self._watermark = watermark
        self._frames_processed = chunk.end_frame
        self._chunk_index += 1
        self._sample_rate = chunk.sample_rate_hz
        self._channels = chunk.channels
        self._sample_width = chunk.sample_width_bytes
        self._expected_chunk_start = chunk.stable_through_frame
        self._lifecycle = IncrementalWhisperLifecycle.ACTIVE
        batch = self._batch(stable, eof=False)
        self._last_chunk_identity = identity
        self._last_chunk_batch = batch
        return batch

    def accepted_batch(
        self,
        chunk: IncrementalWhisperAudioChunk,
    ) -> IncrementalWhisperBatch | None:
        """Return the durable receipt for the most recently committed chunk."""

        if _chunk_identity(chunk) != self._last_chunk_identity:
            return None
        return self._last_chunk_batch

    def validate_chunk(self, chunk: IncrementalWhisperAudioChunk) -> None:
        """Validate transport ordering without mutating transcript state."""

        self._require_active()
        self._validate_chunk_sequence(chunk)

    def flush(self, eof: IncrementalWhisperEOF) -> IncrementalWhisperBatch | None:
        """Finalize provisional speech after an authoritative transport EOF."""

        if self._lifecycle is IncrementalWhisperLifecycle.FLUSHED:
            return None
        self._require_active()
        if eof.final_frame != self._frames_processed:
            raise TranscriptionError(
                "Incremental Whisper EOF must match the committed PCM frontier."
            )
        if self._sample_rate is not None and eof.sample_rate_hz != self._sample_rate:
            raise TranscriptionError("Incremental Whisper EOF sample rate changed.")
        stable = sorted(self._pending, key=_segment_key)
        self._pending = []
        self._recent_emitted.extend(stable)
        self._append_prompt(stable)
        self._sample_rate = eof.sample_rate_hz
        self._watermark = eof.final_frame / eof.sample_rate_hz
        self._lifecycle = IncrementalWhisperLifecycle.FLUSHED
        return self._batch(stable, eof=True)

    def close(self) -> None:
        if self._lifecycle is not IncrementalWhisperLifecycle.FLUSHED:
            self._lifecycle = IncrementalWhisperLifecycle.CLOSED

    def _require_active(self) -> None:
        if self._lifecycle in {
            IncrementalWhisperLifecycle.FLUSHED,
            IncrementalWhisperLifecycle.CLOSED,
        }:
            raise RuntimeError("Incremental Whisper core is no longer active.")

    def _validate_chunk_sequence(self, chunk: IncrementalWhisperAudioChunk) -> None:
        if self._expected_chunk_start is not None:
            if chunk.start_frame != self._expected_chunk_start:
                raise TranscriptionError(
                    "Incremental Whisper chunks must follow the confirmed stable edge."
                )
            if (
                chunk.sample_rate_hz != self._sample_rate
                or chunk.channels != self._channels
                or chunk.sample_width_bytes != self._sample_width
            ):
                raise TranscriptionError("Incremental Whisper PCM format changed.")
        elif chunk.start_frame != 0:
            raise TranscriptionError("Incremental Whisper input must begin at frame zero.")
        if chunk.end_frame < self._frames_processed:
            raise TranscriptionError("Incremental Whisper PCM frontier moved backwards.")

    def _validate_result(self, result: object) -> None:
        if not isinstance(result, TranscriptionResult):
            raise InvalidTranscriptionError(
                "Incremental Whisper backend must return TranscriptionResult."
            )
        for segment in result.segments:
            if not isinstance(segment, TranscriptionSegment):
                raise InvalidTranscriptionError(
                    "Incremental Whisper result contains a non-segment item."
                )
            if segment.start_seconds < 0 or segment.end_seconds < segment.start_seconds:
                raise InvalidTranscriptionError(
                    "Incremental Whisper segment timestamps are invalid."
                )
            if not _finite(segment.start_seconds) or not _finite(segment.end_seconds):
                raise InvalidTranscriptionError(
                    "Incremental Whisper segment timestamps must be finite."
                )

    def _global_segment(
        self,
        segment: TranscriptionSegment,
        chunk: IncrementalWhisperAudioChunk,
    ) -> TranscriptionSegment:
        duration = (chunk.end_frame - chunk.start_frame) / chunk.sample_rate_hz
        local_start = min(duration, max(0.0, segment.start_seconds))
        local_end = min(duration, max(0.0, segment.end_seconds))
        if local_end < local_start:
            raise InvalidTranscriptionError(
                "Incremental Whisper segment timestamps are invalid."
            )
        offset = chunk.start_frame / chunk.sample_rate_hz
        return TranscriptionSegment(
            start_seconds=round(offset + local_start, 6),
            end_seconds=round(offset + local_end, 6),
            text=_normalize_text(segment.text),
            speaker=_normalize_optional_text(segment.speaker),
            confidence=segment.confidence,
            metadata=dict(segment.metadata),
        )

    def _pruned_recent(self, chunk_start: float) -> list[TranscriptionSegment]:
        threshold = chunk_start - self._config.deduplication_tolerance_seconds
        return [
            item for item in self._recent_emitted if item.end_seconds >= threshold
        ]

    def _matching_index(
        self,
        segments: list[TranscriptionSegment],
        candidate: TranscriptionSegment,
    ) -> int | None:
        return next(
            (
                index
                for index, existing in enumerate(segments)
                if self._reconcile(existing, candidate) is not None
            ),
            None,
        )

    def _reconcile(
        self,
        existing: TranscriptionSegment,
        candidate: TranscriptionSegment,
    ) -> TranscriptionSegment | None:
        return self._policy.reconcile(
            existing,
            candidate,
            timestamp_tolerance_seconds=(
                self._config.deduplication_tolerance_seconds
            ),
            similarity_threshold=self._config.reconciliation_similarity_threshold,
        )

    def _append_prompt(self, segments: list[TranscriptionSegment]) -> None:
        prompt_text = " ".join(item.text for item in segments if item.text)
        if prompt_text and self._config.prompt_max_characters:
            self._prompt = f"{self._prompt} {prompt_text}".strip()[
                -self._config.prompt_max_characters :
            ]
        elif self._config.prompt_max_characters == 0:
            self._prompt = ""

    def _batch(
        self,
        segments: list[TranscriptionSegment],
        *,
        eof: bool,
    ) -> IncrementalWhisperBatch:
        assert self._sample_rate is not None
        observations = tuple(self._observation(item) for item in segments)
        metadata = {
            **self._metadata,
            "chunk_index": self._chunk_index,
            "provisional_segment_count": len(self._pending),
            "sample_rate_hz": self._sample_rate,
            "finalized_speech_segment_identities": tuple(
                finalized_speech_segment_identity(item) for item in observations
            ),
        }
        if eof:
            metadata["duration_seconds"] = self._watermark
        return IncrementalWhisperBatch(
            observer=self._config.analysis.observer_name,
            observations=observations,
            watermark_seconds=self._watermark,
            frames_processed=self._frames_processed,
            eof=eof,
            metadata=metadata,
        )

    def _observation(self, segment: TranscriptionSegment) -> Observation:
        metadata = dict(segment.metadata)
        if segment.speaker is not None:
            metadata["speaker"] = segment.speaker
        return Observation(
            timestamp_seconds=segment.start_seconds,
            duration_seconds=segment.end_seconds - segment.start_seconds,
            observer=self._config.analysis.observer_name,
            type="speech",
            value={"text": segment.text, "speaker": segment.speaker},
            confidence=segment.confidence,
            metadata=metadata,
        )


class IncrementalWavWhisperSession:
    """Thin WAV transport that retries a failed prepared chunk in place."""

    def __init__(
        self,
        source: Path,
        config: IncrementalWhisperObserverConfig,
        backend: IncrementalTranscriptionBackend,
        reconciliation_policy: SegmentReconciliationPolicy | None = None,
    ) -> None:
        self._source = Path(source)
        self._config = config
        try:
            self._wav = wave.open(str(self._source), "rb")
        except (OSError, wave.Error) as exc:
            raise TranscriptionError(
                f"Failed to open incremental Whisper WAV: {exc}"
            ) from exc
        self._channels = self._wav.getnchannels()
        self._sample_width = self._wav.getsampwidth()
        self._sample_rate = self._wav.getframerate()
        self._total_frames = self._wav.getnframes()
        if self._channels <= 0 or self._sample_rate <= 0:
            self._wav.close()
            raise TranscriptionError("Whisper WAV must define channels and sample rate.")
        if self._sample_width not in {1, 2, 4}:
            self._wav.close()
            raise TranscriptionError(
                f"Unsupported Whisper WAV sample width: {self._sample_width} bytes."
            )
        self._chunk_frames = max(1, round(config.chunk_seconds * self._sample_rate))
        requested_overlap = round(config.overlap_seconds * self._sample_rate)
        self._overlap_frames = max(0, min(requested_overlap, self._chunk_frames - 1))
        self._step_frames = self._chunk_frames - self._overlap_frames
        self._frame_size = self._channels * self._sample_width
        self._temporary_directory = tempfile.TemporaryDirectory(
            prefix="aitoclip-whisper-",
            dir=self._source.parent,
        )
        self._chunk_path = Path(self._temporary_directory.name) / "chunk.wav"
        try:
            self._model = backend.open_incremental_session(config.analysis)
        except Exception:
            self._temporary_directory.cleanup()
            self._wav.close()
            raise
        self._core = IncrementalWhisperSessionCore(
            config,
            reconciliation_policy,
            metadata={
                "source_path": str(self._source),
                "sample_rate_hz": self._sample_rate,
                "channels": self._channels,
                "model_name": config.analysis.model_name,
            },
        )
        self._first_chunk = True
        self._overlap_raw = b""
        self._next_chunk_start = 0
        self._retry_chunk: IncrementalWhisperAudioChunk | None = None
        self._eof_emitted = False
        self._closed = False

    @property
    def prompt(self) -> str:
        return self._core.prompt

    def read_batch(self) -> IncrementalWhisperBatch | None:
        if self._closed:
            return None
        chunk = self._retry_chunk
        if chunk is None:
            chunk = self._read_chunk()
            if chunk is None:
                return self.flush()
            self._retry_chunk = chunk
        self._write_chunk(chunk.pcm_bytes)
        try:
            result = self._model.transcribe(self._chunk_path, self._core.prompt or None)
            batch = self._core.accept_chunk(chunk, result)
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError(
                f"Incremental Whisper backend failed: {exc}"
            ) from exc
        self._commit_chunk(chunk)
        self._retry_chunk = None
        return batch

    def flush(self) -> IncrementalWhisperBatch | None:
        if self._eof_emitted:
            return None
        if self._retry_chunk is not None:
            raise TranscriptionError(
                "Cannot flush incremental Whisper while a chunk needs retry."
            )
        if self._wav.tell() < self._total_frames:
            raise TranscriptionError(
                "Cannot flush incremental Whisper before the WAV is exhausted."
            )
        batch = self._core.flush(
            IncrementalWhisperEOF(self._total_frames, self._sample_rate)
        )
        self._eof_emitted = True
        self.close()
        return batch

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._core.close()
        try:
            self._model.close()
        finally:
            try:
                self._wav.close()
            finally:
                self._temporary_directory.cleanup()

    def __enter__(self) -> "IncrementalWavWhisperSession":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _read_chunk(self) -> IncrementalWhisperAudioChunk | None:
        frames_to_read = self._chunk_frames if self._first_chunk else self._step_frames
        try:
            fresh_raw = self._wav.readframes(frames_to_read)
        except (OSError, wave.Error) as exc:
            self.close()
            raise TranscriptionError(
                f"Failed to read incremental Whisper WAV: {exc}"
            ) from exc
        if not fresh_raw:
            return None
        pcm = self._overlap_raw + fresh_raw
        frame_count = len(pcm) // self._frame_size
        end_frame = self._next_chunk_start + frame_count
        stable_frame = max(
            self._next_chunk_start,
            end_frame - self._overlap_frames,
        )
        return IncrementalWhisperAudioChunk(
            pcm_bytes=pcm,
            sample_rate_hz=self._sample_rate,
            channels=self._channels,
            sample_width_bytes=self._sample_width,
            start_frame=self._next_chunk_start,
            end_frame=end_frame,
            stable_through_frame=stable_frame,
        )

    def _commit_chunk(self, chunk: IncrementalWhisperAudioChunk) -> None:
        retained_frames = min(
            self._overlap_frames,
            len(chunk.pcm_bytes) // self._frame_size,
        )
        retained_bytes = retained_frames * self._frame_size
        self._overlap_raw = (
            chunk.pcm_bytes[-retained_bytes:] if retained_bytes else b""
        )
        self._next_chunk_start = chunk.end_frame - retained_frames
        self._first_chunk = False

    def _write_chunk(self, raw: bytes) -> None:
        try:
            with wave.open(str(self._chunk_path), "wb") as chunk:
                chunk.setnchannels(self._channels)
                chunk.setsampwidth(self._sample_width)
                chunk.setframerate(self._sample_rate)
                chunk.writeframes(raw)
        except (OSError, wave.Error) as exc:
            raise TranscriptionError(
                f"Failed to create Whisper chunk WAV: {exc}"
            ) from exc


class IncrementalWavWhisperObserver:
    """Create stable Whisper batches from an extracted WAV transport."""

    def __init__(
        self,
        config: IncrementalWhisperObserverConfig | None = None,
        backend: IncrementalTranscriptionBackend | None = None,
        reconciliation_policy: SegmentReconciliationPolicy | None = None,
    ) -> None:
        self._config = config or IncrementalWhisperObserverConfig()
        self._backend = backend or OpenAIWhisperBackend()
        self._reconciliation_policy = reconciliation_policy

    def session(self, source: Path) -> IncrementalWavWhisperSession:
        return IncrementalWavWhisperSession(
            Path(source),
            self._config,
            self._backend,
            self._reconciliation_policy,
        )

    def batches(self, source: Path) -> Iterator[IncrementalWhisperBatch]:
        with self.session(source) as session:
            while (batch := session.read_batch()) is not None:
                yield batch


def _normalize_text(value: str) -> str:
    return " ".join(str(value).split())


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = _normalize_text(value)
    return normalized or None


def _deduplication_text(value: str) -> str:
    return _normalize_text(value).casefold()


def _normalized_tokens(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"\w+", value.casefold(), flags=re.UNICODE))


def _segment_key(segment: TranscriptionSegment) -> tuple[float, float, str, str]:
    return (
        segment.start_seconds,
        segment.end_seconds,
        _deduplication_text(segment.text),
        segment.speaker or "",
    )


def _finite(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}


def _chunk_identity(chunk: IncrementalWhisperAudioChunk) -> str:
    """Return a deterministic identity for one exact PCM transport unit."""

    digest = hashlib.sha256()
    for value in (
        chunk.sample_rate_hz,
        chunk.channels,
        chunk.sample_width_bytes,
        chunk.start_frame,
        chunk.end_frame,
        chunk.stable_through_frame,
    ):
        digest.update(value.to_bytes(8, byteorder="big", signed=False))
    digest.update(chunk.pcm_bytes)
    return digest.hexdigest()
