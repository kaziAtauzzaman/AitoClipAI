"""Windows-backed Facebook Page credential resolution and validation."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import os
from typing import Any, Protocol

from facebook_auth_contracts import (
    FacebookCredentialDiagnostic,
    FacebookCredentialResolver,
    FacebookCredentialState,
    FacebookCredentialStore,
)
from uploading.errors import (
    FacebookAuthenticationRequired,
    PermanentUploadError,
)
from uploading.facebook_config import FacebookUploadSettings


_CREDENTIAL_TARGET_PREFIX = "AitoClipAI/Facebook/Page"
_CRED_TYPE_GENERIC = 1
_CRED_PERSIST_LOCAL_MACHINE = 2
_ERROR_NOT_FOUND = 1168
_RETRYABLE_HTTP_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
_AUTHENTICATION_GRAPH_CODES = frozenset({102, 190})
_PERMISSION_GRAPH_CODES = frozenset({10, 200, 299})
class FacebookCredentialStoreError(RuntimeError):
    """A credential-manager operation failed without exposing secret data."""

    def __init__(
        self,
        message: str,
        *,
        windows_error_code: int | None = None,
    ) -> None:
        self.windows_error_code = windows_error_code
        super().__init__(message)


class FacebookCredentialValidator(Protocol):
    """Read-only validation boundary for a Page token."""

    def validate(
        self,
        token: str,
        expected_page_id: str,
    ) -> "FacebookCredentialValidation":
        """Validate token identity and Page publishing capability."""


@dataclass(frozen=True, slots=True)
class FacebookCredentialValidation:
    """Non-secret proof returned by one successful credential validation."""

    page_id: str
    page_name: str | None
    can_publish: bool


class _WindowsCredentialBackend(Protocol):
    def read(self, target: str) -> str | None: ...

    def replace(self, target: str, username: str, secret: str) -> None: ...


class WindowsFacebookCredentialStore:
    """Store Page tokens as Windows Credential Manager generic credentials."""

    def __init__(
        self,
        backend: _WindowsCredentialBackend | None = None,
    ) -> None:
        self._backend = backend or _CtypesWindowsCredentialBackend()

    def read(self, page_id: str) -> str | None:
        return self._backend.read(_credential_target(page_id))

    def replace(self, page_id: str, token: str) -> None:
        if not isinstance(token, str) or not token.strip():
            raise FacebookCredentialStoreError(
                "Facebook credential cannot be empty."
            )
        self._backend.replace(
            _credential_target(page_id),
            f"facebook-page:{page_id}",
            token.strip(),
        )


class ValidatingFacebookCredentialResolver:
    """Resolve only validated Page tokens from an injected secret store."""

    def __init__(
        self,
        store: FacebookCredentialStore,
        validator: FacebookCredentialValidator,
        page_id: str,
    ) -> None:
        if not isinstance(page_id, str) or not page_id.isdigit():
            raise ValueError("Facebook credential resolver requires a numeric Page ID.")
        self._store = store
        self._validator = validator
        self._page_id = page_id
        self._last_diagnostic: FacebookCredentialDiagnostic | None = None

    @property
    def last_diagnostic(self) -> FacebookCredentialDiagnostic | None:
        return self._last_diagnostic

    def local_state(self) -> FacebookCredentialState:
        try:
            token = self._store.read(self._page_id)
        except Exception as exc:
            raise _unavailable() from exc
        return (
            FacebookCredentialState.CREDENTIAL_STORED
            if token
            else FacebookCredentialState.NOT_CONFIGURED
        )

    def resolve(self) -> str:
        try:
            token = self._store.read(self._page_id)
        except Exception as exc:
            raise _unavailable() from exc
        if not token:
            raise FacebookAuthenticationRequired(
                FacebookCredentialState.NOT_CONFIGURED
            )
        self._validator.validate(token, self._page_id)
        return token

    def replace(self, token: str) -> None:
        candidate = token.strip() if isinstance(token, str) else ""
        if not candidate:
            diagnostic = FacebookCredentialDiagnostic(stage="token_input")
            self._last_diagnostic = diagnostic
            raise FacebookAuthenticationRequired(
                FacebookCredentialState.REAUTHORIZATION_REQUIRED,
                diagnostic=diagnostic,
            )
        self._last_diagnostic = FacebookCredentialDiagnostic(
            stage="graph_validation"
        )
        try:
            self._validator.validate(candidate, self._page_id)
        except FacebookAuthenticationRequired as exc:
            self._last_diagnostic = (
                exc.diagnostic or self._last_diagnostic
            )
            raise
        except Exception:
            raise
        self._last_diagnostic = FacebookCredentialDiagnostic(
            stage="credential_write",
            validation_succeeded=True,
            cred_write_attempted=True,
        )
        try:
            self._store.replace(self._page_id, candidate)
        except Exception as exc:
            diagnostic = FacebookCredentialDiagnostic(
                stage="credential_write",
                windows_error_code=getattr(
                    exc,
                    "windows_error_code",
                    None,
                ),
                validation_succeeded=True,
                cred_write_attempted=True,
                cred_write_succeeded=False,
            )
            self._last_diagnostic = diagnostic
            raise _unavailable(diagnostic) from exc
        self._last_diagnostic = FacebookCredentialDiagnostic(
            stage="completed",
            validation_succeeded=True,
            cred_write_attempted=True,
            cred_write_succeeded=True,
        )


class FacebookGraphCredentialValidator:
    """Validate Page identity and publishing capability through Graph."""

    def __init__(
        self,
        session: Any,
        graph_api_version: str,
        *,
        transport_errors: tuple[type[BaseException], ...] = (
            ConnectionError,
            TimeoutError,
            OSError,
        ),
    ) -> None:
        self._session = session
        self._version = graph_api_version
        self._transport_errors = transport_errors

    @classmethod
    def from_settings(
        cls,
        settings: FacebookUploadSettings,
    ) -> "FacebookGraphCredentialValidator":
        settings.validate()
        try:
            import requests
        except ImportError as exc:
            raise PermanentUploadError(
                'Facebook authentication requires `pip install -e ".[facebook]"`.'
            ) from exc
        return cls(
            requests.Session(),
            settings.graph_api_version,
            transport_errors=(
                requests.Timeout,
                requests.ConnectionError,
                requests.RequestException,
            ),
        )

    def validate(
        self,
        token: str,
        expected_page_id: str,
    ) -> FacebookCredentialValidation:
        url = f"https://graph.facebook.com/{self._version}/me"
        try:
            response = self._session.get(
                url,
                params={
                    "fields": "id,name,can_post",
                    "access_token": token,
                },
                timeout=(10, 30),
            )
        except self._transport_errors as exc:
            raise _unavailable(
                FacebookCredentialDiagnostic(stage="graph_validation")
            ) from exc
        payload = _validation_payload(response, token=token)
        page_id = payload.get("id")
        if not isinstance(page_id, str) or page_id != expected_page_id:
            raise FacebookAuthenticationRequired(
                FacebookCredentialState.WRONG_PAGE,
                diagnostic=FacebookCredentialDiagnostic(
                    stage="page_identity_validation",
                    http_status=_http_status(response),
                ),
            )
        if payload.get("can_post") is not True:
            raise FacebookAuthenticationRequired(
                FacebookCredentialState.PERMISSION_ERROR,
                diagnostic=FacebookCredentialDiagnostic(
                    stage="page_permission_validation",
                    http_status=_http_status(response),
                ),
            )
        name = payload.get("name")
        return FacebookCredentialValidation(
            page_id=page_id,
            page_name=name if isinstance(name, str) and name else None,
            can_publish=True,
        )


def create_facebook_credential_resolver(
    settings: FacebookUploadSettings,
    *,
    store: FacebookCredentialStore | None = None,
    validator: FacebookCredentialValidator | None = None,
) -> FacebookCredentialResolver:
    """Build the current Page-token provider behind the stable resolver contract."""

    settings.validate()
    return ValidatingFacebookCredentialResolver(
        store or WindowsFacebookCredentialStore(),
        validator or FacebookGraphCredentialValidator.from_settings(settings),
        settings.page_id,
    )


def _validation_payload(
    response: Any,
    *,
    token: str,
) -> dict[str, Any]:
    status = _http_status(response)
    try:
        payload = response.json()
    except (TypeError, ValueError) as exc:
        raise _unavailable(
            FacebookCredentialDiagnostic(
                stage="graph_validation",
                http_status=status,
            )
        ) from exc
    if not isinstance(payload, dict):
        raise _unavailable(
            FacebookCredentialDiagnostic(
                stage="graph_validation",
                http_status=status,
            )
        )
    error = payload.get("error")
    if isinstance(error, dict):
        code = error.get("code")
        diagnostic = FacebookCredentialDiagnostic(
            stage="graph_validation",
            http_status=status,
            graph_error_code=code if isinstance(code, int) else None,
            graph_error_type=_optional_diagnostic_string(error.get("type")),
            graph_error_message=_redacted_graph_message(
                error.get("message"),
                token,
            ),
        )
        if code in _AUTHENTICATION_GRAPH_CODES or status == 401:
            raise FacebookAuthenticationRequired(
                FacebookCredentialState.REAUTHORIZATION_REQUIRED,
                diagnostic=diagnostic,
            )
        if code in _PERMISSION_GRAPH_CODES or status == 403:
            raise FacebookAuthenticationRequired(
                FacebookCredentialState.PERMISSION_ERROR,
                diagnostic=diagnostic,
            )
        if (
            bool(error.get("is_transient"))
            or status in _RETRYABLE_HTTP_STATUSES
        ):
            raise _unavailable(diagnostic)
        raise _unavailable(diagnostic)
    if status is not None and status >= 400:
        diagnostic = FacebookCredentialDiagnostic(
            stage="graph_validation",
            http_status=status,
        )
        if status == 401:
            raise FacebookAuthenticationRequired(
                FacebookCredentialState.REAUTHORIZATION_REQUIRED,
                diagnostic=diagnostic,
            )
        if status == 403:
            raise FacebookAuthenticationRequired(
                FacebookCredentialState.PERMISSION_ERROR,
                diagnostic=diagnostic,
            )
        raise _unavailable(diagnostic)
    return payload


def _credential_target(page_id: str) -> str:
    if not isinstance(page_id, str) or not page_id.isdigit():
        raise FacebookCredentialStoreError(
            "Facebook credential target requires a numeric Page ID."
        )
    return f"{_CREDENTIAL_TARGET_PREFIX}/{page_id}"


def _unavailable(
    diagnostic: FacebookCredentialDiagnostic | None = None,
) -> FacebookAuthenticationRequired:
    return FacebookAuthenticationRequired(
        FacebookCredentialState.UNAVAILABLE,
        diagnostic=diagnostic,
    )


def _http_status(response: Any) -> int | None:
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def _optional_diagnostic_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _redacted_graph_message(value: object, token: str) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return value.replace(token, "[REDACTED]") if token else value


class _CREDENTIALW(ctypes.Structure):
    _fields_ = [
        ("Flags", wintypes.DWORD),
        ("Type", wintypes.DWORD),
        ("TargetName", wintypes.LPWSTR),
        ("Comment", wintypes.LPWSTR),
        ("LastWritten", wintypes.FILETIME),
        ("CredentialBlobSize", wintypes.DWORD),
        ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
        ("Persist", wintypes.DWORD),
        ("AttributeCount", wintypes.DWORD),
        ("Attributes", ctypes.c_void_p),
        ("TargetAlias", wintypes.LPWSTR),
        ("UserName", wintypes.LPWSTR),
    ]


class _CtypesWindowsCredentialBackend:
    """Minimal ctypes wrapper around CredReadW/CredWriteW."""

    def __init__(self) -> None:
        if os.name != "nt":
            raise FacebookCredentialStoreError(
                "Windows Credential Manager is unavailable on this platform."
            )
        self._advapi32 = ctypes.WinDLL(  # type: ignore[attr-defined]
            "Advapi32.dll",
            use_last_error=True,
        )
        self._cred_read = self._advapi32.CredReadW
        self._cred_read.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.POINTER(ctypes.POINTER(_CREDENTIALW)),
        ]
        self._cred_read.restype = wintypes.BOOL
        self._cred_write = self._advapi32.CredWriteW
        self._cred_write.argtypes = [
            ctypes.POINTER(_CREDENTIALW),
            wintypes.DWORD,
        ]
        self._cred_write.restype = wintypes.BOOL
        self._cred_free = self._advapi32.CredFree
        self._cred_free.argtypes = [ctypes.c_void_p]
        self._cred_free.restype = None

    def read(self, target: str) -> str | None:
        pointer = ctypes.POINTER(_CREDENTIALW)()
        if not self._cred_read(
            target,
            _CRED_TYPE_GENERIC,
            0,
            ctypes.byref(pointer),
        ):
            error = ctypes.get_last_error()
            if error == _ERROR_NOT_FOUND:
                return None
            raise FacebookCredentialStoreError(
                f"Windows Credential Manager read failed ({error}).",
                windows_error_code=error,
            )
        try:
            credential = pointer.contents
            raw = ctypes.string_at(
                credential.CredentialBlob,
                credential.CredentialBlobSize,
            )
            try:
                value = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise FacebookCredentialStoreError(
                    "Stored Facebook credential has invalid encoding."
                ) from exc
            return value or None
        finally:
            self._cred_free(pointer)

    def replace(self, target: str, username: str, secret: str) -> None:
        raw = secret.encode("utf-8")
        buffer = (ctypes.c_ubyte * len(raw)).from_buffer_copy(raw)
        credential = _CREDENTIALW()
        credential.Type = _CRED_TYPE_GENERIC
        credential.TargetName = target
        credential.CredentialBlobSize = len(raw)
        credential.CredentialBlob = ctypes.cast(
            buffer,
            ctypes.POINTER(ctypes.c_ubyte),
        )
        credential.Persist = _CRED_PERSIST_LOCAL_MACHINE
        credential.UserName = username
        if not self._cred_write(ctypes.byref(credential), 0):
            error = ctypes.get_last_error()
            raise FacebookCredentialStoreError(
                f"Windows Credential Manager write failed ({error}).",
                windows_error_code=error,
            )
