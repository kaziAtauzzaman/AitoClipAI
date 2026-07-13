"""Live PCM transport adapter for the incremental Whisper session core."""

from pathlib import Path
import tempfile
import threading
import wave
import weakref
from contextlib import contextmanager
from collections.abc import Iterator

from whisper_observer.backend import OpenAIWhisperBackend
from whisper_observer.config import IncrementalWhisperObserverConfig
from whisper_observer.contracts import (
    IncrementalTranscriptionBackend,
    IncrementalWhisperAudioChunk,
    IncrementalWhisperBatch,
    IncrementalWhisperEOF,
    IncrementalWhisperLifecycle,
    SegmentReconciliationPolicy,
)
from whisper_observer.errors import TranscriptionError
from whisper_observer.incremental import IncrementalWhisperSessionCore


class LivePcmWhisperSession:
    """Push PCM through a guarded, single-caller Whisper session.

    Calls are synchronous. Concurrent or reentrant lifecycle operations are
    rejected instead of serialized; callers must retry them after the active
    operation finishes.
    """

    def __init__(
        self,
        config: IncrementalWhisperObserverConfig,
        backend: IncrementalTranscriptionBackend,
        reconciliation_policy: SegmentReconciliationPolicy | None = None,
    ) -> None:
        self._config = config
        self._temporary_directory = tempfile.TemporaryDirectory(
            prefix="aitoclip-live-whisper-"
        )
        self._chunk_path = Path(self._temporary_directory.name) / "chunk.wav"
        try:
            self._model = backend.open_incremental_session(config.analysis)
        except Exception:
            self._temporary_directory.cleanup()
            raise
        self._core = IncrementalWhisperSessionCore(
            config,
            reconciliation_policy,
            metadata={
                "transport": "live_pcm",
                "model_name": config.analysis.model_name,
            },
        )
        self._pending_chunk: IncrementalWhisperAudioChunk | None = None
        self._eof_emitted = False
        self._closed = False
        self._operation_lock = threading.Lock()
        self._resource_finalizer = weakref.finalize(
            self,
            _cleanup_abandoned_resources,
            self._model,
            self._temporary_directory,
        )

    @property
    def prompt(self) -> str:
        return self._core.prompt

    @property
    def lifecycle(self) -> IncrementalWhisperLifecycle:
        return self._core.lifecycle

    @property
    def has_pending_retry(self) -> bool:
        return self._pending_chunk is not None

    def submit_chunk(
        self,
        chunk: IncrementalWhisperAudioChunk,
    ) -> IncrementalWhisperBatch:
        """Transcribe and commit one chunk, retaining it if transcription fails."""

        with self._operation("submit_chunk"):
            self._require_open()
            if self._pending_chunk is not None:
                raise TranscriptionError(
                    "Retry the pending live Whisper chunk before submitting new audio."
                )
            self._core.validate_chunk(chunk)
            self._pending_chunk = chunk
            return self._transcribe_pending()

    def retry_pending(self) -> IncrementalWhisperBatch:
        """Retry exactly the retained failed chunk without accepting later input."""

        with self._operation("retry_pending"):
            self._require_open()
            if self._pending_chunk is None:
                raise TranscriptionError("No live Whisper chunk is pending retry.")
            return self._transcribe_pending()

    def flush(self, eof: IncrementalWhisperEOF) -> IncrementalWhisperBatch | None:
        """Finalize once after the live producer supplies authoritative EOF."""

        with self._operation("flush"):
            if self._eof_emitted:
                return None
            self._require_open()
            if self._pending_chunk is not None:
                raise TranscriptionError(
                    "Cannot flush live Whisper while a chunk needs retry."
                )
            batch = self._core.flush(eof)
            self._eof_emitted = True
            self._close_resources()
            return batch

    def close(self) -> None:
        with self._operation("close"):
            self._close_resources()

    def _close_resources(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._core.close()
        try:
            self._model.close()
        finally:
            try:
                self._temporary_directory.cleanup()
            finally:
                self._resource_finalizer.detach()

    def __enter__(self) -> "LivePcmWhisperSession":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _transcribe_pending(self) -> IncrementalWhisperBatch:
        chunk = self._pending_chunk
        assert chunk is not None
        accepted = self._core.accepted_batch(chunk)
        if accepted is not None:
            self._pending_chunk = None
            return accepted
        self._write_chunk(chunk)
        try:
            result = self._model.transcribe(self._chunk_path, self._core.prompt or None)
            batch = self._core.accept_chunk(chunk, result)
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError(
                f"Live incremental Whisper backend failed: {exc}"
            ) from exc
        self._pending_chunk = None
        return batch

    def _write_chunk(self, chunk: IncrementalWhisperAudioChunk) -> None:
        try:
            with wave.open(str(self._chunk_path), "wb") as output:
                output.setnchannels(chunk.channels)
                output.setsampwidth(chunk.sample_width_bytes)
                output.setframerate(chunk.sample_rate_hz)
                output.writeframes(chunk.pcm_bytes)
        except (OSError, wave.Error) as exc:
            raise TranscriptionError(
                f"Failed to create live Whisper chunk WAV: {exc}"
            ) from exc

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Live incremental Whisper session is closed.")

    @contextmanager
    def _operation(self, name: str) -> Iterator[None]:
        if not self._operation_lock.acquire(blocking=False):
            raise RuntimeError(
                f"Live incremental Whisper session is busy; cannot {name}."
            )
        try:
            yield
        finally:
            self._operation_lock.release()


def _cleanup_abandoned_resources(
    model: object,
    temporary_directory: tempfile.TemporaryDirectory[str],
) -> None:
    """Best-effort cleanup for a session abandoned without explicit close."""

    try:
        close = getattr(model, "close", None)
        if close is not None:
            close()
    except BaseException:
        pass
    try:
        temporary_directory.cleanup()
    except BaseException:
        pass


class LivePcmWhisperObserver:
    """Factory for live PCM sessions sharing Incremental Whisper semantics."""

    def __init__(
        self,
        config: IncrementalWhisperObserverConfig | None = None,
        backend: IncrementalTranscriptionBackend | None = None,
        reconciliation_policy: SegmentReconciliationPolicy | None = None,
    ) -> None:
        self._config = config or IncrementalWhisperObserverConfig()
        self._backend = backend or OpenAIWhisperBackend()
        self._reconciliation_policy = reconciliation_policy

    def session(self) -> LivePcmWhisperSession:
        return LivePcmWhisperSession(
            self._config,
            self._backend,
            self._reconciliation_policy,
        )
