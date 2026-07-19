"""Environment and file configuration for YouTube OAuth uploads."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Mapping

from uploading.errors import PermanentUploadError


YOUTUBE_UPLOAD_SCOPE = "https://www.googleapis.com/auth/youtube.upload"
YOUTUBE_READONLY_SCOPE = "https://www.googleapis.com/auth/youtube.readonly"
CLIENT_SECRETS_ENV = "AITOCLIP_YOUTUBE_CLIENT_SECRETS_PATH"
TOKEN_ENV = "AITOCLIP_YOUTUBE_TOKEN_PATH"
LEDGER_ENV = "AITOCLIP_UPLOAD_LEDGER_PATH"


@dataclass(frozen=True, slots=True)
class YouTubeUploadConfig:
    """Paths to local OAuth material and the upload ledger."""

    client_secrets_path: Path
    token_path: Path
    ledger_path: Path = Path("data") / "uploads" / "upload-ledger.json"
    scopes: tuple[str, ...] = (YOUTUBE_UPLOAD_SCOPE, YOUTUBE_READONLY_SCOPE)

    @classmethod
    def from_sources(
        cls,
        *,
        config_path: Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> "YouTubeUploadConfig":
        """Load non-secret paths from JSON, with environment overrides."""

        values: dict[str, object] = {}
        base = Path.cwd()
        if config_path is not None:
            path = Path(config_path)
            base = path.resolve(strict=False).parent
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise PermanentUploadError(
                    f"YouTube upload configuration could not be read: {exc}"
                ) from exc
            if not isinstance(raw, dict):
                raise PermanentUploadError(
                    "YouTube upload configuration must be a JSON object."
                )
            values.update(raw)
        environment = dict(os.environ if environ is None else environ)
        client_value = environment.get(CLIENT_SECRETS_ENV) or values.get(
            "client_secrets_path"
        )
        if not isinstance(client_value, str) or not client_value.strip():
            raise PermanentUploadError(
                "YouTube OAuth client secrets path is required via "
                f"{CLIENT_SECRETS_ENV} "
                "or client_secrets_path in the config file."
            )
        token_value = environment.get(TOKEN_ENV) or values.get(
            "token_path", "youtube-token.json"
        )
        ledger_value = environment.get(LEDGER_ENV) or values.get(
            "ledger_path", "../data/uploads/upload-ledger.json"
        )
        if not isinstance(token_value, str) or not token_value.strip():
            raise PermanentUploadError("YouTube OAuth token path must be a string.")
        if not isinstance(ledger_value, str) or not ledger_value.strip():
            raise PermanentUploadError("Upload ledger path must be a string.")
        return cls(
            client_secrets_path=_configured_path(client_value, base),
            token_path=_configured_path(token_value, base),
            ledger_path=_configured_path(ledger_value, base),
        )

    def validate_for_oauth(self) -> None:
        """Validate only material required before an interactive OAuth flow."""

        if not self.client_secrets_path.is_file():
            raise PermanentUploadError(
                f"YouTube OAuth client secrets file does not exist: "
                f"{self.client_secrets_path}"
            )


def _configured_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve(strict=False)
