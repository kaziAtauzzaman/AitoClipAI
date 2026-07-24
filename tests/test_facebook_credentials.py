import json
from pathlib import Path

import pytest

from facebook_auth_contracts import FacebookCredentialState
from uploading import (
    FacebookAuthenticationRequired,
    FacebookCredentialDiagnostic,
    FacebookCredentialStoreError,
    FacebookGraphCredentialValidator,
    FacebookUploadSettings,
    ValidatingFacebookCredentialResolver,
    WindowsFacebookCredentialStore,
)
from operator_facebook_auth import ProductionFacebookCredentialManager


PAGE_ID = "123456789012345"
TOKEN = "sensitive-page-token"


def test_operator_facebook_state_labels_are_stable() -> None:
    assert {
        state: state.label for state in FacebookCredentialState
    } == {
        FacebookCredentialState.CONNECTED: "Facebook Connected",
        FacebookCredentialState.CREDENTIAL_STORED: (
            "Facebook Credential Stored"
        ),
        FacebookCredentialState.NOT_CONFIGURED: "Facebook Not Configured",
        FacebookCredentialState.REAUTHORIZATION_REQUIRED: (
            "Facebook Reauthorization Required"
        ),
        FacebookCredentialState.PERMISSION_ERROR: "Facebook Permission Error",
        FacebookCredentialState.WRONG_PAGE: "Facebook Wrong Page",
        FacebookCredentialState.UNAVAILABLE: "Facebook Unavailable",
    }


class FakeStore:
    def __init__(self, token=None) -> None:
        self.token = token
        self.reads = []
        self.replacements = []

    def read(self, page_id):
        self.reads.append(page_id)
        return self.token

    def replace(self, page_id, token):
        self.replacements.append((page_id, token))
        self.token = token


class FailingStore(FakeStore):
    def read(self, page_id):
        raise OSError("credential manager unavailable")


class FakeValidator:
    def __init__(self, failure=None) -> None:
        self.failure = failure
        self.calls = []

    def validate(self, token, expected_page_id):
        self.calls.append((token, expected_page_id))
        if self.failure is not None:
            raise self.failure
        return object()


class FakeCredentialBackend:
    def __init__(self) -> None:
        self.values = {}
        self.writes = []

    def read(self, target):
        return self.values.get(target)

    def replace(self, target, username, secret):
        self.writes.append((target, username, secret))
        self.values[target] = secret


class FakeResponse:
    def __init__(self, status_code, payload) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, response=None, failure=None) -> None:
        self.response = response
        self.failure = failure
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self.failure is not None:
            raise self.failure
        return self.response


def test_first_time_configuration_validates_before_credential_write() -> None:
    store = FakeStore()
    validator = FakeValidator()
    resolver = ValidatingFacebookCredentialResolver(
        store,
        validator,
        PAGE_ID,
    )

    assert resolver.local_state() is FacebookCredentialState.NOT_CONFIGURED

    resolver.replace(TOKEN)

    assert validator.calls == [(TOKEN, PAGE_ID)]
    assert store.replacements == [(PAGE_ID, TOKEN)]
    assert (
        resolver.local_state()
        is FacebookCredentialState.CREDENTIAL_STORED
    )
    assert resolver.last_diagnostic == FacebookCredentialDiagnostic(
        stage="completed",
        validation_succeeded=True,
        cred_write_attempted=True,
        cred_write_succeeded=True,
    )


def test_invalid_replacement_does_not_overwrite_existing_credential() -> None:
    old_token = "existing-secret"
    store = FakeStore(old_token)
    failure = FacebookAuthenticationRequired(
        FacebookCredentialState.REAUTHORIZATION_REQUIRED
    )
    resolver = ValidatingFacebookCredentialResolver(
        store,
        FakeValidator(failure),
        PAGE_ID,
    )

    with pytest.raises(FacebookAuthenticationRequired) as captured:
        resolver.replace(TOKEN)

    assert captured.value.state is FacebookCredentialState.REAUTHORIZATION_REQUIRED
    assert store.token == old_token
    assert store.replacements == []


def test_missing_expired_and_store_unavailable_have_safe_states() -> None:
    missing = ValidatingFacebookCredentialResolver(
        FakeStore(),
        FakeValidator(),
        PAGE_ID,
    )
    with pytest.raises(FacebookAuthenticationRequired) as missing_error:
        missing.resolve()
    assert missing_error.value.state is FacebookCredentialState.NOT_CONFIGURED

    expired = ValidatingFacebookCredentialResolver(
        FakeStore(TOKEN),
        FakeValidator(
            FacebookAuthenticationRequired(
                FacebookCredentialState.REAUTHORIZATION_REQUIRED
            )
        ),
        PAGE_ID,
    )
    with pytest.raises(FacebookAuthenticationRequired) as expired_error:
        expired.resolve()
    assert (
        expired_error.value.state
        is FacebookCredentialState.REAUTHORIZATION_REQUIRED
    )

    unavailable = ValidatingFacebookCredentialResolver(
        FailingStore(),
        FakeValidator(),
        PAGE_ID,
    )
    with pytest.raises(FacebookAuthenticationRequired) as unavailable_error:
        unavailable.resolve()
    assert unavailable_error.value.state is FacebookCredentialState.UNAVAILABLE


def test_windows_store_uses_page_scoped_generic_credential_target() -> None:
    backend = FakeCredentialBackend()
    store = WindowsFacebookCredentialStore(backend)

    store.replace(PAGE_ID, TOKEN)

    assert backend.writes == [
        (
            f"AitoClipAI/Facebook/Page/{PAGE_ID}",
            f"facebook-page:{PAGE_ID}",
            TOKEN,
        )
    ]
    assert store.read(PAGE_ID) == TOKEN


def test_graph_validator_accepts_matching_page_with_publishing_capability() -> None:
    session = FakeSession(
        FakeResponse(
            200,
            {
                "id": PAGE_ID,
                "name": "JOINT",
                "can_post": True,
            },
        )
    )
    validator = FacebookGraphCredentialValidator(session, "v25.0")

    result = validator.validate(TOKEN, PAGE_ID)

    assert result.page_id == PAGE_ID
    assert result.page_name == "JOINT"
    assert result.can_publish is True
    assert len(session.calls) == 1
    url, request = session.calls[0]
    assert url == "https://graph.facebook.com/v25.0/me"
    assert request["params"]["fields"] == "id,name,can_post"
    assert request["params"]["access_token"] == TOKEN
    assert "tasks" not in request["params"]["fields"]


@pytest.mark.parametrize(
    ("response", "state"),
    [
        (
            FakeResponse(400, {"error": {"code": 190, "message": "expired"}}),
            FacebookCredentialState.REAUTHORIZATION_REQUIRED,
        ),
        (
            FakeResponse(403, {"error": {"code": 200, "message": "denied"}}),
            FacebookCredentialState.PERMISSION_ERROR,
        ),
        (
            FakeResponse(
                200,
                {
                    "id": PAGE_ID,
                    "name": "JOINT",
                    "can_post": False,
                },
            ),
            FacebookCredentialState.PERMISSION_ERROR,
        ),
        (
            FakeResponse(
                200,
                {
                    "id": "999999999999999",
                    "name": "Another Page",
                    "can_post": True,
                },
            ),
            FacebookCredentialState.WRONG_PAGE,
        ),
    ],
)
def test_graph_validator_classifies_actionable_authentication_states(
    response,
    state,
) -> None:
    validator = FacebookGraphCredentialValidator(FakeSession(response), "v25.0")

    with pytest.raises(FacebookAuthenticationRequired) as captured:
        validator.validate(TOKEN, PAGE_ID)

    assert captured.value.state is state
    assert TOKEN not in str(captured.value)
    assert TOKEN not in repr(captured.value)


def test_graph_validator_classifies_transport_failure_as_unavailable() -> None:
    validator = FacebookGraphCredentialValidator(
        FakeSession(failure=TimeoutError("request included a secret")),
        "v25.0",
    )

    with pytest.raises(FacebookAuthenticationRequired) as captured:
        validator.validate(TOKEN, PAGE_ID)

    assert captured.value.state is FacebookCredentialState.UNAVAILABLE
    assert TOKEN not in str(captured.value)


def test_graph_diagnostic_is_structured_and_redacts_token() -> None:
    response = FakeResponse(
        400,
        {
            "error": {
                "code": 100,
                "type": "OAuthException",
                "message": f"Rejected access token {TOKEN}",
            }
        },
    )
    validator = FacebookGraphCredentialValidator(FakeSession(response), "v25.0")

    with pytest.raises(FacebookAuthenticationRequired) as captured:
        validator.validate(TOKEN, PAGE_ID)

    assert captured.value.state is FacebookCredentialState.UNAVAILABLE
    assert captured.value.diagnostic == FacebookCredentialDiagnostic(
        stage="graph_validation",
        http_status=400,
        graph_error_code=100,
        graph_error_type="OAuthException",
        graph_error_message="Rejected access token [REDACTED]",
    )
    serialized = json.dumps(captured.value.diagnostic.as_dict())
    assert TOKEN not in serialized
    assert "access_token" not in serialized


def test_credential_write_failure_reports_attempt_and_windows_error() -> None:
    class WriteFailingStore(FakeStore):
        def replace(self, page_id, token):
            raise FacebookCredentialStoreError(
                "Windows Credential Manager write failed.",
                windows_error_code=5,
            )

    resolver = ValidatingFacebookCredentialResolver(
        WriteFailingStore(),
        FakeValidator(),
        PAGE_ID,
    )

    with pytest.raises(FacebookAuthenticationRequired) as captured:
        resolver.replace(TOKEN)

    assert captured.value.state is FacebookCredentialState.UNAVAILABLE
    assert captured.value.diagnostic == FacebookCredentialDiagnostic(
        stage="credential_write",
        windows_error_code=5,
        validation_succeeded=True,
        cred_write_attempted=True,
        cred_write_succeeded=False,
    )
    assert resolver.last_diagnostic == captured.value.diagnostic


def test_production_manager_reports_stored_until_fresh_validation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "facebook.json"
    config_path.write_text(
        json.dumps(
            {
                "page_id": PAGE_ID,
                "graph_api_version": "v25.0",
                "ledger_path": "ledger.json",
            }
        ),
        encoding="utf-8",
    )
    validations = []

    class StoredCredential:
        def read(self, page_id):
            assert page_id == PAGE_ID
            return TOKEN

    class ValidResolver:
        def resolve(self):
            validations.append(PAGE_ID)
            return TOKEN

    import uploading.facebook_credentials as credentials

    monkeypatch.setattr(
        credentials,
        "WindowsFacebookCredentialStore",
        StoredCredential,
    )
    monkeypatch.setattr(
        credentials,
        "create_facebook_credential_resolver",
        lambda settings: ValidResolver(),
    )
    manager = ProductionFacebookCredentialManager(config_path)

    assert (
        manager.current_state()
        is FacebookCredentialState.CREDENTIAL_STORED
    )
    assert manager.validate() is FacebookCredentialState.CONNECTED
    assert validations == [PAGE_ID]


@pytest.mark.parametrize(
    "state",
    [
        FacebookCredentialState.REAUTHORIZATION_REQUIRED,
        FacebookCredentialState.NOT_CONFIGURED,
        FacebookCredentialState.WRONG_PAGE,
        FacebookCredentialState.PERMISSION_ERROR,
        FacebookCredentialState.UNAVAILABLE,
    ],
)
def test_production_manager_preflight_preserves_authentication_state(
    monkeypatch,
    tmp_path: Path,
    state: FacebookCredentialState,
) -> None:
    config_path = tmp_path / "facebook.json"
    config_path.write_text(
        json.dumps(
            {
                "page_id": PAGE_ID,
                "graph_api_version": "v25.0",
                "ledger_path": "ledger.json",
            }
        ),
        encoding="utf-8",
    )

    class FailingResolver:
        def resolve(self):
            raise FacebookAuthenticationRequired(state)

    import uploading.facebook_credentials as credentials

    monkeypatch.setattr(
        credentials,
        "create_facebook_credential_resolver",
        lambda settings: FailingResolver(),
    )
    manager = ProductionFacebookCredentialManager(config_path)

    with pytest.raises(FacebookAuthenticationRequired) as captured:
        manager.validate()

    assert captured.value.state is state


def test_production_manager_writes_only_sanitized_diagnostic(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "facebook.json"
    config_path.write_text(
        json.dumps(
            {
                "page_id": PAGE_ID,
                "graph_api_version": "v25.0",
                "ledger_path": "ledger.json",
            }
        ),
        encoding="utf-8",
    )
    failure_diagnostic = FacebookCredentialDiagnostic(
        stage="graph_validation",
        http_status=400,
        graph_error_code=100,
        graph_error_type="OAuthException",
        graph_error_message="Unsupported Page field",
    )

    class FailingResolver:
        last_diagnostic = failure_diagnostic

        def replace(self, token):
            raise FacebookAuthenticationRequired(
                FacebookCredentialState.UNAVAILABLE,
                diagnostic=failure_diagnostic,
            )

    import uploading.facebook_credentials as credentials

    monkeypatch.setattr(
        credentials,
        "create_facebook_credential_resolver",
        lambda settings: FailingResolver(),
    )
    log_path = tmp_path / "diagnostic.json"
    manager = ProductionFacebookCredentialManager(
        config_path,
        diagnostic_log_path=log_path,
    )

    with pytest.raises(FacebookAuthenticationRequired) as captured:
        manager.replace(TOKEN)

    assert captured.value.diagnostic_log_path == log_path
    assert json.loads(log_path.read_text(encoding="utf-8")) == (
        failure_diagnostic.as_dict()
    )
    assert TOKEN not in log_path.read_text(encoding="utf-8")


def test_non_secret_config_ignores_legacy_token_environment(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "facebook.json"
    config_path.write_text(
        json.dumps(
            {
                "page_id": PAGE_ID,
                "graph_api_version": "v25.0",
                "ledger_path": "ledger.json",
            }
        ),
        encoding="utf-8",
    )

    settings = FacebookUploadSettings.from_sources(
        config_path=config_path,
        environ={"AITOCLIP_FACEBOOK_PAGE_ACCESS_TOKEN": TOKEN},
    )

    assert settings.page_id == PAGE_ID
    assert not hasattr(settings, "page_access_token")
    assert TOKEN not in repr(settings)
