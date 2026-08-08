"""Tests for OAuth token handling in GoogleCalendarPoller.

Regression cover for the audit's A2: a bare ``except Exception`` around
``creds.refresh()`` deleted a *valid* refresh token the first time the laptop
woke up offline. The token was destroyed on 2026-05-19 and calendar detection
was dead for three months, logging 2 309 warnings that nobody saw.

Two properties matter here:
  1. A transient failure must LEAVE THE TOKEN ALONE.
  2. Only a real ``invalid_grant`` may delete it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.detection.calendar_poller import CalendarAuthError, GoogleCalendarPoller


@pytest.fixture()
def paths(tmp_path: Path) -> tuple[Path, Path]:
    creds = tmp_path / "credentials.json"
    creds.write_text(json.dumps({"installed": {"client_id": "x"}}), encoding="utf-8")
    token = tmp_path / "google_token.json"
    token.write_text(json.dumps({"refresh_token": "keep-me"}), encoding="utf-8")
    return creds, token


def make_poller(creds: Path, token: Path) -> GoogleCalendarPoller:
    return GoogleCalendarPoller(credentials_path=creds, token_path=token)


class FakeCreds:
    """Stands in for google.oauth2.credentials.Credentials."""

    def __init__(self, raises: BaseException | None = None) -> None:
        self.valid = False
        self.expired = True
        self.refresh_token = "keep-me"
        self._raises = raises
        self.refreshed = False

    def refresh(self, _request: object) -> None:
        if self._raises is not None:
            raise self._raises
        self.refreshed = True
        self.valid = True

    def to_json(self) -> str:
        return json.dumps({"refresh_token": self.refresh_token})


def patch_creds(monkeypatch: pytest.MonkeyPatch, fake: FakeCreds) -> None:
    import google.oauth2.credentials as gcreds

    monkeypatch.setattr(
        gcreds.Credentials,
        "from_authorized_user_file",
        classmethod(lambda cls, *a, **k: fake),
    )


# ---------------------------------------------------------------------------
# The bug: transient errors must not destroy the token
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "exc",
    [
        ConnectionError("Network is unreachable"),
        TimeoutError("timed out"),
        OSError("[Errno 8] nodename nor servname provided"),
        RuntimeError("503 Service Unavailable"),
    ],
    ids=["offline", "timeout", "dns", "server-5xx"],
)
def test_transient_refresh_failure_keeps_the_token(paths, monkeypatch, exc) -> None:
    creds_path, token_path = paths
    patch_creds(monkeypatch, FakeCreds(raises=exc))
    poller = make_poller(creds_path, token_path)

    with pytest.raises(CalendarAuthError):
        poller.authenticate(headless=True)

    assert token_path.exists(), "a transient failure must never delete the token"
    assert json.loads(token_path.read_text())["refresh_token"] == "keep-me"


def test_invalid_grant_does_delete_the_token(paths, monkeypatch) -> None:
    from google.auth.exceptions import RefreshError

    creds_path, token_path = paths
    patch_creds(
        monkeypatch,
        FakeCreds(raises=RefreshError("invalid_grant: Token has been expired or revoked.")),
    )
    poller = make_poller(creds_path, token_path)

    with pytest.raises(CalendarAuthError):
        poller.authenticate(headless=True)

    assert not token_path.exists(), "a revoked token is genuinely useless; drop it"


def test_refresh_error_without_invalid_grant_keeps_the_token(paths, monkeypatch) -> None:
    """RefreshError is also raised for transport hiccups — only invalid_grant
    means the token itself is dead."""
    from google.auth.exceptions import RefreshError

    creds_path, token_path = paths
    patch_creds(monkeypatch, FakeCreds(raises=RefreshError("Connection reset by peer")))
    poller = make_poller(creds_path, token_path)

    with pytest.raises(CalendarAuthError):
        poller.authenticate(headless=True)

    assert token_path.exists()


def test_successful_refresh_persists_the_new_token(paths, monkeypatch) -> None:
    creds_path, token_path = paths
    fake = FakeCreds()
    patch_creds(monkeypatch, fake)
    poller = make_poller(creds_path, token_path)

    out = poller.authenticate(headless=True)

    assert out is fake
    assert fake.refreshed is True
    assert token_path.exists()


def test_missing_credentials_file_is_a_clear_error(tmp_path: Path) -> None:
    poller = GoogleCalendarPoller(
        credentials_path=tmp_path / "nope.json", token_path=tmp_path / "t.json"
    )
    with pytest.raises(CalendarAuthError, match="OAuth credentials not found"):
        poller.authenticate(headless=True)


# ---------------------------------------------------------------------------
# Reauth is announced once, not every 60 seconds
# ---------------------------------------------------------------------------
def test_reauth_callback_fires_once_per_failure_streak(paths, monkeypatch) -> None:
    creds_path, token_path = paths
    patch_creds(monkeypatch, FakeCreds(raises=ConnectionError("offline")))
    poller = make_poller(creds_path, token_path)

    calls: list[str] = []
    poller.on_needs_reauth(calls.append)

    for _ in range(5):
        assert poller.fetch_today_events() == []

    assert len(calls) == 1, f"expected one reauth announcement, got {len(calls)}"


def test_reauth_callback_rearms_after_a_success(paths, monkeypatch) -> None:
    creds_path, token_path = paths
    failing = FakeCreds(raises=ConnectionError("offline"))
    patch_creds(monkeypatch, failing)
    poller = make_poller(creds_path, token_path)

    calls: list[str] = []
    poller.on_needs_reauth(calls.append)

    poller.fetch_today_events()
    poller.fetch_today_events()
    assert len(calls) == 1

    # Auth recovers, then breaks again — that is a new event worth announcing.
    poller.notify_auth_recovered()
    poller.fetch_today_events()
    assert len(calls) == 2
