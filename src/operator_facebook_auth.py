"""UI-facing Facebook credential management without eager uploader imports."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Protocol

from facebook_auth_contracts import (
    FacebookAuthenticationIssue,
    FacebookCredentialDiagnostic,
    FacebookCredentialState,
)

DEFAULT_FACEBOOK_DIAGNOSTIC_LOG = (
    Path(tempfile.gettempdir())
    / "AitoClipAI"
    / "facebook-credential-diagnostic.json"
)


class OperatorFacebookCredentialManager(Protocol):
    """Narrow credential surface consumed by the Tkinter operator."""

    def current_state(self) -> FacebookCredentialState:
        """Inspect local configuration without making a network request."""

    def replace(self, token: str) -> FacebookCredentialState:
        """Validate and atomically replace one Page credential."""


class ProductionFacebookCredentialManager:
    """Bind the operator to Windows Credential Manager on demand."""

    def __init__(
        self,
        config_path: Path = Path("config") / "facebook-upload.json",
        diagnostic_log_path: Path = DEFAULT_FACEBOOK_DIAGNOSTIC_LOG,
    ) -> None:
        self._config_path = Path(config_path)
        self._diagnostic_log_path = Path(diagnostic_log_path)

    @property
    def diagnostic_log_path(self) -> Path:
        return self._diagnostic_log_path

    def current_state(self) -> FacebookCredentialState:
        try:
            settings = self._settings()
            from uploading.facebook_credentials import (
                WindowsFacebookCredentialStore,
            )

            return (
                FacebookCredentialState.CONNECTED
                if WindowsFacebookCredentialStore().read(settings.page_id)
                else FacebookCredentialState.NOT_CONFIGURED
            )
        except FacebookAuthenticationIssue as exc:
            return exc.state
        except Exception:
            return FacebookCredentialState.UNAVAILABLE

    def replace(self, token: str) -> FacebookCredentialState:
        resolver = None
        try:
            settings = self._settings()
            from uploading.facebook_credentials import (
                create_facebook_credential_resolver,
            )

            resolver = create_facebook_credential_resolver(settings)
            resolver.replace(token)
        except FacebookAuthenticationIssue as exc:
            diagnostic = (
                exc.diagnostic
                or getattr(resolver, "last_diagnostic", None)
                or FacebookCredentialDiagnostic(stage="configuration")
            )
            exc.diagnostic = diagnostic
            exc.diagnostic_log_path = self._write_diagnostic(diagnostic)
            raise
        except Exception as exc:
            from uploading.errors import FacebookAuthenticationRequired

            diagnostic = (
                getattr(resolver, "last_diagnostic", None)
                or FacebookCredentialDiagnostic(
                    stage="configuration",
                    windows_error_code=getattr(
                        exc,
                        "windows_error_code",
                        None,
                    ),
                )
            )
            raise FacebookAuthenticationRequired(
                FacebookCredentialState.UNAVAILABLE,
                diagnostic=diagnostic,
                diagnostic_log_path=self._write_diagnostic(diagnostic),
            ) from exc
        diagnostic = (
            getattr(resolver, "last_diagnostic", None)
            or FacebookCredentialDiagnostic(
                stage="completed",
                validation_succeeded=True,
                cred_write_attempted=True,
                cred_write_succeeded=True,
            )
        )
        self._write_diagnostic(diagnostic)
        return FacebookCredentialState.CONNECTED

    def _settings(self):
        from uploading.facebook_config import FacebookUploadSettings

        return FacebookUploadSettings.from_sources(
            config_path=(
                self._config_path if self._config_path.is_file() else None
            )
        )

    def _write_diagnostic(
        self,
        diagnostic: FacebookCredentialDiagnostic,
    ) -> Path | None:
        path = self._diagnostic_log_path
        temporary = path.with_name(f".{path.name}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(
                json.dumps(
                    diagnostic.as_dict(),
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, path)
        except (OSError, TypeError, ValueError):
            return None
        return path
