# Incremental Whisper observer

`IncrementalWhisperSessionCore` owns provisional segments, reconciliation,
bounded prompts, watermarks, and EOF lifecycle independently of transport. It
accepts chronological PCM chunk contracts with explicit global frame bounds and
stable edges, followed by authoritative EOF. A future live adapter can feed the
same core without changing transcript semantics.

`IncrementalWavWhisperObserver` is the prerecorded transport adapter. It reads
an already extracted PCM WAV in chronological overlapping chunks, materializes
one reusable chunk WAV for the model, and never uses
`CompletedTimelineReplayAdapter` or future audio content. A failed transcription
retains the prepared PCM chunk and retries it before reading additional frames.

`LivePcmWhisperObserver` creates push-based sessions for an external live PCM
producer. Callers submit the same `IncrementalWhisperAudioChunk` contract used
by the core and finish with `IncrementalWhisperEOF`. The adapter holds at most
one failed chunk. While that chunk is pending, later input and EOF are rejected;
`retry_pending()` transcribes the identical PCM payload before progress resumes.
No live duration or EOF is inferred from buffered audio.

Live session methods are synchronous and guarded against concurrent or
reentrant use. The session is intentionally single-caller rather than a
thread-safe work queue: a conflicting `submit_chunk()`, `retry_pending()`,
`flush()`, or `close()` fails immediately and may be retried after the active
operation completes. This also prevents concurrent reuse or deletion of the
temporary chunk WAV. Explicit close remains preferred; an idempotent finalizer
closes the model handle and removes temporary resources if ownership is
abandoned.

The core retains only the identity and batch receipt for its most recently
committed chunk. A live retry can therefore recover an acceptance interrupted
between the core commit and adapter bookkeeping without retranscribing or
emitting the batch twice.

One model session is opened for the source and reused for every chunk. A chunk
contains `chunk_seconds` of audio and the next chunk starts
`chunk_seconds - overlap_seconds` later. Chunk-relative segment timestamps are
clamped to the chunk, rebased to source time, rounded to six decimal places,
and emitted in deterministic chronological order.

Segments ending inside the right overlap are provisional. The observer merges
the next chunk before emitting them. Equal case-folded, whitespace-normalized
text uses a fast exact-match path. Temporally overlapping text with sufficiently
similar normalized token sequences is also reconciled, covering punctuation
and minor wording changes. The later chunk representation replaces the
provisional one. The deterministic reconciliation policy is injectable, and a
small recent-emission window prevents an overlap duplicate from being emitted
again.

The Whisper watermark is the non-overlap chunk edge, clamped to the earliest
provisional segment start. At authoritative WAV EOF, all remaining provisional
segments are emitted exactly once and the watermark advances to the WAV
duration. The coordinator combines this value with Audio and other required
observer watermarks by taking their minimum.

Prompt continuity uses only emitted stable text. The retained `initial_prompt`
is truncated to `prompt_max_characters`, so transcript context remains bounded.

Incremental output is not expected to match whole-file Whisper byte for byte.
Chunk boundaries can change wording, punctuation, confidence, and segmentation.
Incremental mode additionally normalizes whitespace and rounds timestamps. Its
guarantees are chronological timestamps, deterministic normalization, no
duplicate emission across overlaps, provisional right-edge safety, and explicit
monotonic watermarks. The existing whole-file `WhisperObserver` is unchanged.
