"""Environment and file configuration for Facebook Page uploads."""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
from typing import Mapping

from uploading.errors import PermanentUploadError


DEFAULT_FACEBOOK_GRAPH_API_VERSION = "v25.0"
FACEBOOK_PAGE_ID_ENV = "AITOCLIP_FACEBOOK_PAGE_ID"
FACEBOOK_PAGE_ACCESS_TOKEN_ENV = "AITOCLIP_FACEBOOK_PAGE_ACCESS_TOKEN"
FACEBOOK_GRAPH_API_VERSION_ENV = "AITOCLIP_FACEBOOK_GRAPH_API_VERSION"
UPLOAD_LEDGER_ENV = "AITOCLIP_UPLOAD_LEDGER_PATH"
_GRAPH_VERSION_PATTERN = re.compile(r"v[1-9][0-9]*\.0\Z")


@dataclass(frozen=True, slots=True)
class FacebookUploadConfig:
    """One Facebook Page target and its environment-supplied credential."""

    page_id: str
    page_access_token: str = field(repr=False)
    graph_api_version: str = DEFAULT_FACEBOOK_GRAPH_API_VERSION
    ledger_path: Path = Path("data") / "uploads" / "upload-ledger.json"

    @classmethod
    def from_sources(
        cls,
        *,
        config_path: Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> "FacebookUploadConfig":
        """Load non-secret settings from JSON and the Page token from env."""

        values: dict[str, object] = {}
        base = Path.cwd()
        if config_path is not None:
            path = Path(config_path)
            base = path.resolve(strict=False).parent
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise PermanentUploadError(
                    f"Facebook upload configuration could not be read: {exc}"
                ) from exc
            if not isinstance(raw, dict):
                raise PermanentUploadError(
                    "Facebook upload configuration must be a JSON object."
                )
            values.update(raw)

        environment = dict(os.environ if environ is None else environ)
        page_id = environment.get(FACEBOOK_PAGE_ID_ENV) or values.get("page_id")
        token = environment.get(FACEBOOK_PAGE_ACCESS_TOKEN_ENV)
        version = environment.get(FACEBOOK_GRAPH_API_VERSION_ENV) or values.get(
            "graph_api_version", DEFAULT_FACEBOOK_GRAPH_API_VERSION
        )
        ledger = environment.get(UPLOAD_LEDGER_ENV) or values.get(
            "ledger_path", "../data/uploads/upload-ledger.json"
        )
        if not isinstance(page_id, str) or not page_id.strip():
            raise PermanentUploadError(
                f"Facebook Page ID is required via {FACEBOOK_PAGE_ID_ENV} "
                "or page_id in the config file."
            )
        if not isinstance(token, str) or not token.strip():
            raise PermanentUploadError(
                "A Facebook Page access token is required via "
                f"{FACEBOOK_PAGE_ACCESS_TOKEN_ENV}."
            )
        if not isinstance(version, str) or not version.strip():
            raise PermanentUploadError("Facebook Graph API version must be a string.")
        if not isinstance(ledger, str) or not ledger.strip():
            raise PermanentUploadError("Upload ledger path must be a string.")

        config = cls(
            page_id=page_id.strip(),
            page_access_token=token.strip(),
            graph_api_version=version.strip(),
            ledger_path=_configured_path(ledger, base),
        )
        config.validate_for_upload()
        return config

    def validate_for_upload(self) -> None:
        """Reject malformed Page targets before constructing an API client."""

        if not self.page_id.isdigit():
            raise PermanentUploadError("Facebook Page ID must contain only digits.")
        if not self.page_access_token.strip():
            raise PermanentUploadError("Facebook Page access token must not be empty.")
        if _GRAPH_VERSION_PATTERN.fullmatch(self.graph_api_version) is None:
            raise PermanentUploadError(
                "Facebook Graph API version must use the form v25.0."
            )


def _configured_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve(strict=False)
