# Downloader Module

The downloader module is the Phase 1 input stage for AitoClipAI. It downloads a
single source video with `yt-dlp`, stores the media under `data/downloads/`, and
writes JSON metadata beside the downloaded file.

It does not implement clipping, transcription, or AI analysis.

## Package Layout

- `src/downloader/config.py` - Downloader configuration and output template
  construction.
- `src/downloader/errors.py` - Expected exception types for graceful failure
  handling.
- `src/downloader/yt_dlp_client.py` - Adapter around `yt-dlp`.
- `src/downloader/metadata.py` - Metadata extraction, normalization, and JSON
  writing.
- `src/downloader/sanitization.py` - Deterministic conversion of arbitrary
  provider metadata into JSON-safe values.
- `src/downloader/downloader.py` - Download orchestration.

## Classes and Functions

### `DownloaderConfig`

Stores runtime settings for the downloader, including the downloads directory,
metadata filename suffix, yt-dlp filename template, and overwrite behavior.

- `output_template()` returns the full yt-dlp output template.
- `ensure_directories()` creates the configured downloads directory.

### `YtDlpClient`

Wraps direct `yt-dlp` usage so downloader services do not depend on yt-dlp
internals. This adapter boundary keeps future Twitch, Kick, or platform-specific
support easy to add.

- `extract_info(url, download)` calls yt-dlp and returns its raw metadata.
- `_build_options()` builds yt-dlp options from `DownloaderConfig`.

### `VideoMetadata`

Normalized metadata for one source video. It stores common fields used by future
pipeline stages and also preserves the full raw yt-dlp payload.

- `from_yt_dlp(info)` validates and converts raw yt-dlp metadata.
- `to_dict()` returns a JSON-serializable dictionary without deep-copying raw
  provider runtime objects.

### `JsonMetadataSanitizer`

Recursively preserves useful JSON values and common value objects while
replacing unsupported provider runtime objects with stable qualified-type
markers. It detects cycles, provides deterministic ordering for sets, handles
non-finite floats explicitly, and never mutates the original yt-dlp payload.
This prevents metadata persistence from failing on objects such as thread
locks while retaining the normalized metadata fields and usable raw values.

### `MetadataExtractor`

Extracts metadata without downloading media.

- `__init__(client)` injects any client that follows the metadata client
  protocol.
- `extract(url)` returns `VideoMetadata` for a URL.

### `MetadataWriter`

Writes normalized metadata beside a downloaded video.

- `__init__(config)` stores metadata persistence settings.
- `metadata_path_for(video_path)` returns the JSON path for a media file.
- `write(video_path, metadata)` writes metadata JSON and returns the JSON path.

### `DownloadResult`

Return object for a successful download. It contains the media path, metadata
path, and normalized metadata.

### `VideoDownloader`

Coordinates the full Phase 1 operation: validate URL, call the download client,
normalize metadata, resolve the downloaded media path, and write metadata JSON.

- `__init__(config, client, metadata_writer)` wires downloader collaborators and
  allows tests or future platform-specific clients to be injected.
- `download(url)` downloads one video and returns `DownloadResult`.
- `_resolve_downloaded_path(info)` extracts the downloaded media path from
  yt-dlp metadata.

### Protocols and Helpers

- `MetadataClient.extract_info(url, download)` defines the interface required by
  `MetadataExtractor`.
- `DownloadClient.extract_info(url, download)` defines the interface required by
  `VideoDownloader`.
- `_optional_str(value)` normalizes optional provider metadata into strings.
- `_optional_number(value)` keeps optional numeric metadata only when it is
  actually numeric.

### Exceptions

- `DownloaderError` is the base expected downloader exception.
- `MetadataExtractionError` represents metadata extraction or validation
  failures.
- `DownloadError` represents download, path resolution, or metadata write
  failures.
