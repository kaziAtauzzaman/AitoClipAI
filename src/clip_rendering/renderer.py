"""FFmpeg-backed deterministic clip rendering service."""

from collections import OrderedDict
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from enum import Enum
import hashlib
import json
import math
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import BinaryIO, Callable, Iterable, Iterator, Protocol, Sequence

if os.name == "nt":
    import msvcrt
else:
    import fcntl

from captioning import (
    CaptionArtifact,
    CandidateCaptionIdentity,
    candidate_caption_identity,
)
from clip_rendering.config import ClipRendererConfig, RendererBackend
from clip_rendering.errors import (
    ClipRenderingError,
    InvalidRenderInputError,
    IntelQSVUnavailableError,
    RenderingFFmpegNotFoundError,
    SubtitleRenderingError,
)
from core import ClipScore, RenderJob


class RenderCommandRunner(Protocol):
    """Execute a rendering command and capture diagnostics."""

    def run(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        """Run a command without raising for a non-zero return code."""


class SubprocessRenderCommandRunner:
    """Render command runner backed by the standard subprocess module."""

    def run(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                list(command),
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            raise ClipRenderingError(f"Failed to execute FFmpeg: {exc}") from exc


class ClipRenderer:
    """Render the highest-scoring candidates from their original source media."""

    _qsv_preflight_successes: set[str] = set()
    _qsv_preflight_lock = threading.Lock()
    _completion_schema = "aitoclip-render-completion-v1"

    def __init__(
        self,
        config: ClipRendererConfig | None = None,
        runner: RenderCommandRunner | None = None,
        executable_locator: Callable[[str], str | None] | None = None,
        qsv_capability_checker: Callable[[str], bool] | None = None,
    ) -> None:
        self._config = config or ClipRendererConfig()
        self._runner = runner or SubprocessRenderCommandRunner()
        self._executable_locator = executable_locator or shutil.which
        self._qsv_capability_checker = (
            qsv_capability_checker or self._runtime_qsv_preflight
        )
        # A fixed stripe table prevents per-render lock retention from growing
        # with a long-running stream while still serializing identical outputs.
        self._render_locks = tuple(threading.Lock() for _ in range(64))
        self._source_digests: OrderedDict[tuple[str, int, int], str] = OrderedDict()
        self._source_digest_lock = threading.Lock()
        self._validate_config()

    def render(
        self,
        scores: Iterable[ClipScore],
        caption_artifacts: Iterable[CaptionArtifact] | None = None,
    ) -> list[RenderJob]:
        """Sort scores, render the configured top candidates, and return jobs."""

        ranked = sorted(scores, key=self._ranking_key)
        if self._config.maximum_clips is not None:
            ranked = ranked[: self._config.maximum_clips]
        if not ranked:
            return []
        captions = self._caption_artifacts(caption_artifacts or [])
        return [
            self.render_one(
                score,
                rank,
                self._caption_for_score(score, captions),
            )
            for rank, score in enumerate(ranked, start=1)
        ]

    def render_one(
        self,
        score: ClipScore,
        identity: int,
        caption_artifact: CaptionArtifact | None = None,
    ) -> RenderJob:
        """Render one finalized score with an explicit monotonic identity.

        This entry point intentionally bypasses ``maximum_clips`` and score
        reranking.  It is used by chronological incremental coordinators after
        selection has already finalized one winner.
        """

        self._validate_render_identity(identity)
        self._validate_render_input(score, caption_artifact)
        self._create_output_directory()
        output_path = self._output_path(score, identity)
        with self._render_identity_lock(
            score.candidate.source_video_path,
            identity,
            output_path,
        ):
            recovered = self._recover_render_locked(
                score,
                identity,
                caption_artifact,
                output_path,
            )
            if recovered is not None:
                return recovered
            ffmpeg_path = self._executable_locator(self._config.ffmpeg_binary)
            if ffmpeg_path is None:
                raise RenderingFFmpegNotFoundError(
                    "FFmpeg executable was not found: "
                    f"{self._config.ffmpeg_binary!r}."
                )
            self._require_requested_backend(ffmpeg_path)
            return self._render_score(
                ffmpeg_path,
                score,
                identity,
                caption_artifact,
                output_path,
            )

    def recover_render(
        self,
        score: ClipScore,
        identity: int,
        caption_artifact: CaptionArtifact | None = None,
    ) -> RenderJob | None:
        """Recover a durable render completion without invoking the encoder."""

        self._validate_render_identity(identity)
        self._validate_render_input(score, caption_artifact)
        self._create_output_directory()
        output_path = self._output_path(score, identity)
        with self._render_identity_lock(
            score.candidate.source_video_path,
            identity,
            output_path,
        ):
            return self._recover_render_locked(
                score,
                identity,
                caption_artifact,
                output_path,
            )

    def _render_score(
        self,
        ffmpeg_path: str,
        score: ClipScore,
        rank: int,
        caption_artifact: CaptionArtifact | None,
        output_path: Path,
    ) -> RenderJob:
        temporary_path = self._temporary_output_path(output_path)
        prepared_manifest_path = self._prepared_manifest_path(
            score.candidate.source_video_path,
            rank,
        )
        self._remove_temporary_output(temporary_path)
        self._remove_temporary_output(prepared_manifest_path)

        try:
            command = self._build_command(
                ffmpeg_path,
                score,
                temporary_path,
                caption_artifact,
            )
            result = self._runner.run(command)
            if result.returncode != 0:
                diagnostic = result.stderr.strip() or result.stdout.strip()
                detail = f": {diagnostic}" if diagnostic else ""
                if (
                    self._config.renderer_backend is RendererBackend.INTEL_QSV
                    and self._is_qsv_initialization_failure(diagnostic)
                ):
                    raise IntelQSVUnavailableError(
                        "Intel QSV device, driver, session, or encoder "
                        f"initialization failed{detail}"
                    )
                error_type = (
                    SubtitleRenderingError
                    if caption_artifact is not None
                    and self._is_subtitle_failure(diagnostic)
                    else ClipRenderingError
                )
                raise error_type(
                    "FFmpeg clip rendering failed with exit code "
                    f"{result.returncode}{detail}"
                )
            if not temporary_path.is_file():
                raise ClipRenderingError(
                    "FFmpeg completed without creating rendered clip: "
                    f"{temporary_path}"
                )
            if temporary_path.stat().st_size <= 0:
                raise ClipRenderingError(
                    "FFmpeg created an empty rendered clip: "
                    f"{temporary_path}"
                )
            self._sync_file(temporary_path)
            job = self._render_job(
                score,
                output_path,
                rank,
                caption_artifact,
                reused=False,
            )
            manifest = self._completion_manifest(
                score,
                rank,
                caption_artifact,
                job,
                temporary_path,
            )
            self._render_attempt_step("after_completion_manifest_construction")
            self._write_manifest_atomic(prepared_manifest_path, manifest)
            self._render_attempt_step("after_encoder_success_before_final_rename")
            try:
                temporary_path.replace(output_path)
            except OSError as exc:
                raise ClipRenderingError(
                    f"Failed to promote rendered clip atomically: {exc}"
                ) from exc
            self._sync_directory(output_path.parent)
            self._render_attempt_step("after_final_artifact_publication")
            completion_manifest_path = self._completion_manifest_path(
                score.candidate.source_video_path,
                rank,
            )
            try:
                prepared_manifest_path.replace(completion_manifest_path)
            except OSError as exc:
                raise ClipRenderingError(
                    f"Failed to publish render completion manifest: {exc}"
                ) from exc
            self._sync_directory(completion_manifest_path.parent)
            self._render_attempt_step("after_manifest_publication")
            recovered = self._recover_render_locked(
                score,
                rank,
                caption_artifact,
                output_path,
            )
            if recovered is None:
                raise ClipRenderingError(
                    "Published render completion could not be recovered."
                )
            self._render_attempt_step("before_renderer_return")
            return recovered
        except BaseException:
            # A prepared manifest makes either the temporary or final artifact
            # recoverable. Without it, a temporary file is not an accepted render.
            if not prepared_manifest_path.is_file():
                self._remove_temporary_output(temporary_path)
            raise

    @staticmethod
    def _temporary_output_path(output_path: Path) -> Path:
        return output_path.with_name(f".{output_path.name}.rendering")

    def _completion_manifest_path(
        self,
        source_path: Path,
        identity: int,
    ) -> Path:
        source_key = hashlib.sha256(
            os.path.normcase(
                str(Path(source_path).resolve(strict=False))
            ).encode("utf-8")
        ).hexdigest()[:24]
        return (
            self._config.output_dir
            / ".render-completions"
            / f"{source_key}.render-{identity:012d}.json"
        )

    def _prepared_manifest_path(
        self,
        source_path: Path,
        identity: int,
    ) -> Path:
        final_path = self._completion_manifest_path(source_path, identity)
        return final_path.with_name(f"{final_path.stem}.prepared.json")

    @staticmethod
    def _manifest_write_path(manifest_path: Path) -> Path:
        return manifest_path.with_name(f".{manifest_path.name}.writing")

    @staticmethod
    def _remove_temporary_output(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            raise ClipRenderingError(
                f"Failed to clean temporary render output: {exc}"
            ) from exc

    def _validate_render_input(
        self,
        score: ClipScore,
        caption_artifact: CaptionArtifact | None,
    ) -> None:
        candidate = score.candidate
        source_path = Path(candidate.source_video_path)
        if not source_path.is_file():
            raise InvalidRenderInputError(f"Source media does not exist: {source_path}")
        if candidate.start_seconds < 0:
            raise InvalidRenderInputError("Clip start time cannot be negative.")
        if candidate.end_seconds <= candidate.start_seconds:
            raise InvalidRenderInputError("Clip end time must be after its start time.")
        if caption_artifact is not None and not caption_artifact.path.is_file():
            raise SubtitleRenderingError(
                f"Caption artifact does not exist: {caption_artifact.path}"
            )

    @staticmethod
    def _validate_render_identity(identity: int) -> None:
        if (
            isinstance(identity, bool)
            or not isinstance(identity, int)
            or identity <= 0
        ):
            raise InvalidRenderInputError(
                "Render identity must be a positive integer."
            )

    def _create_output_directory(self) -> None:
        try:
            self._config.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ClipRenderingError(
                f"Failed to create clip output directory: {exc}"
            ) from exc

    def _render_lock(self, source_path: Path, identity: int) -> threading.Lock:
        key = str(self._completion_manifest_path(source_path, identity))
        index = int.from_bytes(
            hashlib.sha256(key.encode("utf-8")).digest()[:2],
            "big",
        ) % len(self._render_locks)
        return self._render_locks[index]

    @contextmanager
    def _render_identity_lock(
        self,
        source_path: Path,
        identity: int,
        output_path: Path,
    ) -> Iterator[None]:
        """Serialize both immutable identity and final-path ownership.

        Identity manifests are source-scoped so independent sources may reuse a
        numeric identity. A custom filename template can nevertheless map two
        such identities onto one final path. The second filesystem lock closes
        that cross-identity collision without serializing unrelated outputs.
        """

        thread_lock = self._render_lock(source_path, identity)
        with thread_lock:
            manifest_path = self._completion_manifest_path(source_path, identity)
            identity_lock_path = manifest_path.with_suffix(".lock")
            output_lock_path = self._output_lock_path(output_path)
            with self._filesystem_lock(identity_lock_path):
                with self._filesystem_lock(output_lock_path):
                    yield

    def _output_lock_path(self, output_path: Path) -> Path:
        normalized = os.path.normcase(str(output_path.resolve(strict=False)))
        output_key = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return (
            self._config.output_dir
            / ".render-completions"
            / f"output-{output_key}.lock"
        )

    @contextmanager
    def _filesystem_lock(self, lock_path: Path) -> Iterator[None]:
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            stream = lock_path.open("a+b")
        except OSError as exc:
            raise ClipRenderingError(
                f"Failed to open durable render lock: {exc}"
            ) from exc
        with stream:
            self._acquire_file_lock(stream)
            try:
                yield
            finally:
                self._release_file_lock(stream)

    @staticmethod
    def _acquire_file_lock(stream: BinaryIO) -> None:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            while True:
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    return
                except OSError as exc:
                    if exc.errno not in (11, 13) and getattr(
                        exc, "winerror", None
                    ) not in (33, 36):
                        raise ClipRenderingError(
                            f"Failed to acquire render identity lock: {exc}"
                        ) from exc
                    time.sleep(0.05)
        else:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            except OSError as exc:
                raise ClipRenderingError(
                    f"Failed to acquire render identity lock: {exc}"
                ) from exc

    @staticmethod
    def _release_file_lock(stream: BinaryIO) -> None:
        try:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        except OSError as exc:
            raise ClipRenderingError(
                f"Failed to release render identity lock: {exc}"
            ) from exc

    def _recover_render_locked(
        self,
        score: ClipScore,
        identity: int,
        caption_artifact: CaptionArtifact | None,
        output_path: Path,
    ) -> RenderJob | None:
        manifest_path = self._completion_manifest_path(
            score.candidate.source_video_path,
            identity,
        )
        prepared_path = self._prepared_manifest_path(
            score.candidate.source_video_path,
            identity,
        )
        temporary_path = self._temporary_output_path(output_path)
        self._require_unclaimed_output_path(
            output_path,
            own_manifest_paths=(manifest_path, prepared_path),
        )
        expected_spec = self._render_spec_fingerprint(
            score,
            identity,
            caption_artifact,
            output_path,
        )

        manifest_exists = manifest_path.exists()
        manifest = self._load_manifest(manifest_path)
        if manifest_exists:
            if manifest is None:
                raise ClipRenderingError(
                    "Render identity has an unreadable durable completion record."
                )
            if not self._manifest_header_matches(
                manifest,
                expected_spec,
                identity,
                output_path,
            ):
                raise ClipRenderingError(
                    "Render identity is already owned by an incompatible render "
                    "specification."
                )
            if self._artifact_matches_manifest(output_path, manifest):
                self._remove_temporary_output(prepared_path)
                self._remove_temporary_output(temporary_path)
                self._remove_temporary_output(
                    self._manifest_write_path(manifest_path)
                )
                return self._job_from_manifest(
                    score,
                    manifest,
                    output_path,
                )

        prepared_exists = prepared_path.exists()
        prepared = self._load_manifest(prepared_path)
        if prepared_exists:
            if prepared is None:
                raise ClipRenderingError(
                    "Render identity has an unreadable prepared completion record."
                )
            if not self._manifest_header_matches(
                prepared,
                expected_spec,
                identity,
                output_path,
            ):
                raise ClipRenderingError(
                    "Render identity has an incompatible prepared render attempt."
                )
            if self._artifact_matches_manifest(output_path, prepared):
                self._sync_directory(output_path.parent)
                self._publish_prepared_manifest(prepared_path, manifest_path)
                return self._job_from_manifest(score, prepared, output_path)
            if self._artifact_matches_manifest(temporary_path, prepared):
                if output_path.exists() and not self._config.overwrite_existing:
                    raise ClipRenderingError(
                        "A final render artifact exists without a matching durable "
                        "completion manifest."
                    )
                try:
                    temporary_path.replace(output_path)
                except OSError as exc:
                    raise ClipRenderingError(
                        f"Failed to recover rendered clip atomically: {exc}"
                    ) from exc
                self._sync_directory(output_path.parent)
                self._publish_prepared_manifest(prepared_path, manifest_path)
                return self._job_from_manifest(score, prepared, output_path)

        if output_path.exists() and not self._config.overwrite_existing:
            if not prepared_path.exists():
                self._remove_temporary_output(temporary_path)
            self._remove_temporary_output(self._manifest_write_path(manifest_path))
            self._remove_temporary_output(self._manifest_write_path(prepared_path))
            raise ClipRenderingError(
                "Existing render state has no valid completion manifest for this "
                "render identity and configuration."
            )
        self._remove_temporary_output(manifest_path)
        self._remove_temporary_output(prepared_path)
        self._remove_temporary_output(self._manifest_write_path(manifest_path))
        self._remove_temporary_output(self._manifest_write_path(prepared_path))
        self._remove_temporary_output(temporary_path)
        return None

    def _require_unclaimed_output_path(
        self,
        output_path: Path,
        *,
        own_manifest_paths: tuple[Path, Path],
    ) -> None:
        """Reject a final path durably owned by another render identity."""

        completion_dir = self._config.output_dir / ".render-completions"
        if not completion_dir.is_dir():
            return
        expected = os.path.normcase(str(output_path.resolve(strict=False)))
        own_paths = {
            os.path.normcase(str(path.resolve(strict=False)))
            for path in own_manifest_paths
        }
        try:
            for path in completion_dir.glob("*.json"):
                if os.path.normcase(str(path.resolve(strict=False))) in own_paths:
                    continue
                manifest = self._load_manifest(path)
                if manifest is None:
                    continue
                claimed = manifest.get("output_path")
                if (
                    isinstance(claimed, str)
                    and os.path.normcase(claimed) == expected
                ):
                    raise ClipRenderingError(
                        "Final render output path is already owned by another durable "
                        "render identity."
                    )
        except OSError as exc:
            raise ClipRenderingError(
                f"Failed to inspect durable output ownership: {exc}"
            ) from exc

    def _completion_manifest(
        self,
        score: ClipScore,
        identity: int,
        caption_artifact: CaptionArtifact | None,
        job: RenderJob,
        artifact_path: Path,
    ) -> dict[str, object]:
        size = artifact_path.stat().st_size
        return {
            "schema": self._completion_schema,
            "render_identity": identity,
            "render_spec_sha256": self._render_spec_fingerprint(
                score,
                identity,
                caption_artifact,
                job.output_path,
            ),
            "output_path": str(job.output_path.resolve(strict=False)),
            "artifact_size_bytes": size,
            "artifact_sha256": _file_sha256(artifact_path),
            "job": {
                "aspect_ratio": job.aspect_ratio,
                "resolution": job.resolution,
                "captions_path": (
                    None
                    if job.captions_path is None
                    else str(job.captions_path)
                ),
                "preset": job.preset,
                "metadata": _canonical_manifest_value(job.metadata),
            },
        }

    def _render_spec_fingerprint(
        self,
        score: ClipScore,
        identity: int,
        caption_artifact: CaptionArtifact | None,
        output_path: Path,
    ) -> str:
        caption_path = None if caption_artifact is None else caption_artifact.path
        payload = {
            "schema": self._completion_schema,
            "render_identity": identity,
            "output_path": str(output_path.resolve(strict=False)),
            "source": self._source_identity(score.candidate.source_video_path),
            "score": score,
            "caption": (
                None
                if caption_path is None
                else {
                    "path": str(caption_path.resolve(strict=False)),
                    "size_bytes": caption_path.stat().st_size,
                    "sha256": _file_sha256(caption_path),
                }
            ),
            "renderer": {
                "filename_template": self._config.filename_template,
                "output_format": self._normalized_output_format(),
                "video_codec": self._config.video_codec,
                "renderer_backend": self._config.renderer_backend.value,
                "audio_codec": self._config.audio_codec,
                "burn_subtitles": self._config.burn_subtitles,
                "subtitle_character_encoding": (
                    self._config.subtitle_character_encoding
                ),
            },
        }
        encoded = json.dumps(
            _canonical_manifest_value(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _source_identity(self, source_path: Path) -> dict[str, object]:
        path = Path(source_path)
        try:
            stat = path.stat()
        except OSError as exc:
            raise InvalidRenderInputError(
                f"Could not inspect render source identity: {exc}"
            ) from exc
        resolved = str(path.resolve(strict=False))
        cache_key = (resolved, stat.st_size, stat.st_mtime_ns)
        with self._source_digest_lock:
            digest = self._source_digests.get(cache_key)
        if digest is None:
            try:
                digest = _file_sha256(path)
                current = path.stat()
            except OSError as exc:
                raise InvalidRenderInputError(
                    f"Render source changed while its identity was computed: {exc}"
                ) from exc
            if (
                current.st_size != stat.st_size
                or current.st_mtime_ns != stat.st_mtime_ns
            ):
                raise InvalidRenderInputError(
                    "Render source changed while its durable identity was computed."
                )
            with self._source_digest_lock:
                self._source_digests[cache_key] = digest
                self._source_digests.move_to_end(cache_key)
                while len(self._source_digests) > 4:
                    self._source_digests.popitem(last=False)
        return {
            "path": resolved,
            "size_bytes": stat.st_size,
            "sha256": digest,
        }

    @classmethod
    def _manifest_header_matches(
        cls,
        manifest: Mapping[str, object],
        expected_spec: str,
        identity: int,
        output_path: Path,
    ) -> bool:
        return (
            manifest.get("schema") == cls._completion_schema
            and manifest.get("render_identity") == identity
            and manifest.get("render_spec_sha256") == expected_spec
            and manifest.get("output_path")
            == str(output_path.resolve(strict=False))
            and isinstance(manifest.get("job"), dict)
        )

    @staticmethod
    def _artifact_matches_manifest(
        artifact_path: Path,
        manifest: Mapping[str, object],
    ) -> bool:
        size = manifest.get("artifact_size_bytes")
        digest = manifest.get("artifact_sha256")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or not artifact_path.is_file()
        ):
            return False
        try:
            return artifact_path.stat().st_size == size and _file_sha256(
                artifact_path
            ) == digest
        except OSError:
            return False

    @staticmethod
    def _load_manifest(path: Path) -> dict[str, object] | None:
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def _job_from_manifest(
        self,
        score: ClipScore,
        manifest: Mapping[str, object],
        output_path: Path,
    ) -> RenderJob:
        stored_job = manifest.get("job")
        if not isinstance(stored_job, dict):
            raise ClipRenderingError("Render completion manifest has no valid job.")
        metadata = stored_job.get("metadata")
        if not isinstance(metadata, dict):
            raise ClipRenderingError(
                "Render completion manifest has invalid job metadata."
            )
        captions_value = stored_job.get("captions_path")
        if captions_value is not None and not isinstance(captions_value, str):
            raise ClipRenderingError(
                "Render completion manifest has invalid caption ownership."
            )
        for optional in ("aspect_ratio", "resolution", "preset"):
            if stored_job.get(optional) is not None and not isinstance(
                stored_job.get(optional), str
            ):
                raise ClipRenderingError(
                    "Render completion manifest has invalid RenderJob fields."
                )
        return RenderJob(
            candidate=score.candidate,
            output_path=output_path,
            aspect_ratio=stored_job.get("aspect_ratio"),
            resolution=stored_job.get("resolution"),
            captions_path=(
                None if captions_value is None else Path(captions_value)
            ),
            preset=stored_job.get("preset"),
            metadata=metadata,
        )

    def _write_manifest_atomic(
        self,
        manifest_path: Path,
        manifest: Mapping[str, object],
    ) -> None:
        try:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ClipRenderingError(
                f"Failed to create render completion directory: {exc}"
            ) from exc
        writing_path = self._manifest_write_path(manifest_path)
        self._remove_temporary_output(writing_path)
        try:
            with writing_path.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(
                    manifest,
                    stream,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            writing_path.replace(manifest_path)
            self._sync_directory(manifest_path.parent)
        except OSError as exc:
            self._remove_temporary_output(writing_path)
            raise ClipRenderingError(
                f"Failed to write render completion manifest: {exc}"
            ) from exc

    def _publish_prepared_manifest(
        self,
        prepared_path: Path,
        final_path: Path,
    ) -> None:
        try:
            prepared_path.replace(final_path)
            self._sync_directory(final_path.parent)
        except OSError as exc:
            raise ClipRenderingError(
                f"Failed to publish recovered render completion: {exc}"
            ) from exc

    @staticmethod
    def _sync_directory(path: Path) -> None:
        try:
            descriptor = os.open(path, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        except OSError:
            pass
        finally:
            os.close(descriptor)

    @staticmethod
    def _sync_file(path: Path) -> None:
        try:
            with path.open("r+b") as stream:
                os.fsync(stream.fileno())
        except OSError as exc:
            raise ClipRenderingError(
                f"Failed to sync rendered artifact before publication: {exc}"
            ) from exc

    def _render_attempt_step(self, step: str) -> None:
        """Failure-injection seam around durable render publication."""

    def _output_path(self, score: ClipScore, rank: int) -> Path:
        candidate = score.candidate
        extension = self._normalized_output_format()
        values = {
            "stem": candidate.source_video_path.stem,
            "rank": rank,
            "start_ms": round(candidate.start_seconds * 1000),
            "end_ms": round(candidate.end_seconds * 1000),
            "score": f"{score.overall_score:.6f}",
            "score_millionths": round(score.overall_score * 1_000_000),
            "ext": extension,
        }
        try:
            filename = self._config.filename_template.format(**values)
        except (KeyError, ValueError) as exc:
            raise InvalidRenderInputError(
                f"Invalid clip filename template: {exc}"
            ) from exc
        if not filename or Path(filename).name != filename:
            raise InvalidRenderInputError(
                "Clip filename template must produce one non-empty filename."
            )
        if self._config.renderer_backend is RendererBackend.INTEL_QSV:
            path = Path(filename)
            filename = f"{path.stem}.intel_qsv{path.suffix}"
        return self._config.output_dir / filename

    def _build_command(
        self,
        ffmpeg_path: str,
        score: ClipScore,
        output_path: Path,
        caption_artifact: CaptionArtifact | None,
    ) -> list[str]:
        candidate = score.candidate
        start = f"{candidate.start_seconds:.6f}"
        duration = f"{candidate.end_seconds - candidate.start_seconds:.6f}"
        # FFmpeg rebases the primary input seek onto one shared timeline. Video
        # is aligned to its first decoded frame for a zero-based output. Audio
        # remains on the shared timeline; first_pts=0 fills any intentional
        # positive delay with silence instead of collapsing the content offset.
        video_filters = f"setpts=PTS-STARTPTS,trim=start=0:end={duration}"
        if caption_artifact is not None:
            escaped_path = escape_subtitle_filter_path(caption_artifact.path)
            character_encoding = escape_subtitle_filter_value(
                self._config.subtitle_character_encoding
            )
            video_filters += (
                f",subtitles=filename='{escaped_path}':charenc='{character_encoding}'"
            )
        filter_graph = (
            f"[0:v:0]{video_filters}[v];"
            f"[0:a:0]atrim=start=0:end={duration},"
            "aresample=async=1:first_pts=0[a]"
        )
        video_codec = (
            "h264_qsv"
            if self._config.renderer_backend is RendererBackend.INTEL_QSV
            else self._config.video_codec
        )
        command = [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y" if self._config.overwrite_existing else "-n",
            # Input options apply to the next input only. Keep this seek grouped
            # with the primary source when future secondary inputs are added.
            "-ss",
            start,
            "-i",
            str(candidate.source_video_path),
            "-t",
            duration,
            "-filter_complex",
            filter_graph,
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-c:v",
            video_codec,
            "-c:a",
            self._config.audio_codec,
            "-map_metadata",
            "-1",
            "-map_chapters",
            "-1",
            "-shortest",
            "-f",
            self._normalized_output_format(),
            str(output_path),
        ]
        if self._config.renderer_backend is RendererBackend.INTEL_QSV:
            codec_index = command.index("-c:v") + 2
            command[codec_index:codec_index] = [
                "-preset",
                "medium",
                "-global_quality",
                "23",
            ]
        return command

    def _render_job(
        self,
        score: ClipScore,
        output_path: Path,
        rank: int,
        caption_artifact: CaptionArtifact | None,
        *,
        reused: bool,
    ) -> RenderJob:
        candidate = score.candidate
        return RenderJob(
            candidate=candidate,
            output_path=output_path,
            captions_path=(
                caption_artifact.path if caption_artifact is not None else None
            ),
            metadata={
                "rank": rank,
                "overall_score": score.overall_score,
                "score_components": score.score_components,
                "score_rationale": score.rationale,
                "start_seconds": candidate.start_seconds,
                "end_seconds": candidate.end_seconds,
                "duration_seconds": candidate.end_seconds - candidate.start_seconds,
                "output_format": self._normalized_output_format(),
                "video_codec": (
                    "h264_qsv"
                    if self._config.renderer_backend is RendererBackend.INTEL_QSV
                    else self._config.video_codec
                ),
                "renderer_backend": self._config.renderer_backend.value,
                "audio_codec": self._config.audio_codec,
                "reused_existing": reused,
                "subtitles_burned_in": caption_artifact is not None,
            },
        )

    def _caption_artifacts(
        self,
        artifacts: Iterable[CaptionArtifact],
    ) -> dict[CandidateCaptionIdentity, CaptionArtifact]:
        indexed: dict[CandidateCaptionIdentity, CaptionArtifact] = {}
        for artifact in artifacts:
            identity = artifact.identity
            if identity in indexed:
                raise SubtitleRenderingError(
                    "Multiple caption artifacts share one source path and time window."
                )
            indexed[identity] = artifact
        return indexed

    def _caption_for_score(
        self,
        score: ClipScore,
        captions: dict[CandidateCaptionIdentity, CaptionArtifact],
    ) -> CaptionArtifact | None:
        if not self._config.burn_subtitles:
            return None
        identity = candidate_caption_identity(score.candidate)
        artifact = captions.get(identity)
        if artifact is None:
            raise SubtitleRenderingError(
                "No caption artifact matches the selected candidate source and window."
            )
        return artifact

    def _ranking_key(self, score: ClipScore) -> tuple[float, float, float, str, str]:
        candidate = score.candidate
        return (
            -score.overall_score,
            candidate.start_seconds,
            candidate.end_seconds,
            candidate.reason,
            str(candidate.source_video_path),
        )

    def _validate_config(self) -> None:
        config = self._config
        if not isinstance(config.renderer_backend, RendererBackend):
            raise InvalidRenderInputError(
                "Renderer backend must be a RendererBackend value."
            )
        if config.maximum_clips is not None and config.maximum_clips <= 0:
            raise InvalidRenderInputError("Maximum clips must be positive or None.")
        if not config.output_format.removeprefix("."):
            raise InvalidRenderInputError("Output format cannot be empty.")
        if not config.video_codec.strip() or not config.audio_codec.strip():
            raise InvalidRenderInputError("Video and audio codecs cannot be empty.")
        if not config.subtitle_character_encoding.strip():
            raise InvalidRenderInputError(
                "Subtitle character encoding cannot be empty."
            )

    def _require_requested_backend(self, ffmpeg_path: str) -> None:
        if (
            self._config.renderer_backend is RendererBackend.INTEL_QSV
            and not self._qsv_capability_checker(ffmpeg_path)
        ):
            raise IntelQSVUnavailableError(
                "Intel QSV device, driver, session, or encoder initialization failed."
            )

    @classmethod
    def _runtime_qsv_preflight(cls, ffmpeg_path: str) -> bool:
        cache_key = str(Path(ffmpeg_path).resolve(strict=False)).casefold()
        with cls._qsv_preflight_lock:
            if cache_key in cls._qsv_preflight_successes:
                return True
            command = [
                ffmpeg_path,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:size=64x64:rate=1:duration=1",
                "-frames:v",
                "1",
                "-an",
                "-c:v",
                "h264_qsv",
                "-preset",
                "medium",
                "-global_quality",
                "23",
                "-f",
                "null",
                "-",
            ]
            try:
                result = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                )
            except OSError as exc:
                raise ClipRenderingError(
                    f"Failed to execute Intel QSV runtime preflight: {exc}"
                ) from exc
            if result.returncode != 0:
                diagnostic = result.stderr.strip() or result.stdout.strip()
                if cls._is_qsv_initialization_failure(diagnostic):
                    return False
                detail = f": {diagnostic}" if diagnostic else ""
                raise ClipRenderingError(
                    "Intel QSV runtime preflight failed for a reason unrelated "
                    f"to device or encoder initialization{detail}"
                )
            cls._qsv_preflight_successes.add(cache_key)
            return True

    @staticmethod
    def _is_qsv_initialization_failure(diagnostic: str) -> bool:
        normalized = " ".join(diagnostic.casefold().split())
        return any(
            marker in normalized
            for marker in (
                "mfx",
                "quick sync",
                "qsv device",
                "qsv session",
                "h264_qsv encoder",
                "unknown encoder 'h264_qsv'",
                'unknown encoder "h264_qsv"',
                "no device available",
                "device creation failed",
                "device setup failed",
                "failed to initialise vaapi connection",
                "failed to initialize vaapi connection",
                "intel media driver",
            )
        )

    @staticmethod
    def _is_subtitle_failure(diagnostic: str) -> bool:
        normalized = diagnostic.casefold()
        return any(
            marker in normalized
            for marker in ("subtitle", "libass", "ass filter", "fontconfig")
        )

    def _normalized_output_format(self) -> str:
        return self._config.output_format.removeprefix(".")


def escape_subtitle_filter_path(path: Path) -> str:
    """Escape a subtitle filename for FFmpeg's filter-option parser."""

    normalized = str(path).replace("\\", "/")
    return escape_subtitle_filter_value(normalized)


def escape_subtitle_filter_value(value: str) -> str:
    """Escape characters significant inside a quoted FFmpeg filter value."""

    escaped = value.replace("\\", "\\\\")
    for character in (":", "'", ",", "[", "]", ";"):
        escaped = escaped.replace(character, f"\\{character}")
    return escaped


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_manifest_value(value: object) -> object:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidRenderInputError(
                "Render completion state cannot contain non-finite numbers."
            )
        return value
    if isinstance(value, Path):
        return str(value.resolve(strict=False))
    if isinstance(value, Enum):
        return _canonical_manifest_value(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _canonical_manifest_value(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_manifest_value(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_manifest_value(item) for item in value]
    raise InvalidRenderInputError(
        "Render completion state contains an unsupported value: "
        f"{type(value).__name__}."
    )
