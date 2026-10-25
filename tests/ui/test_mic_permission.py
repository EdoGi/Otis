"""Microphone permission is settled at launch, not in the middle of a meeting.

Review finding (2026-10-25): after the native launcher was installed, the
stored Microphone grant still named the old applet's code hashes, so tccd
treated it as undecided (logged: "Failed to match existing code requirement
… kTCCServiceMicrophone"). The first mic open — the first recording, most
likely a real meeting — would have raised the prompt mid-call; a missed or
denied prompt meant a silent local side reported only as a WARN.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("rumps")

from src.audio import mic_permission
from src.config import Config
from src.detection.detector import MeetingDetector
from src.health import MICROPHONE, HealthRegistry, HealthState
from src.ui.menubar import MenuBarApp
from src.ui.notifications import NotificationManager


@pytest.fixture
def registry() -> HealthRegistry:
    return HealthRegistry()


def _app(tmp_path: Path, registry: HealthRegistry):
    sent: list[tuple[str, str, str]] = []
    app = MenuBarApp(
        config=Config({"app": {"working_days": [0, 1, 2, 3, 4, 5, 6]}}),
        detector=MeetingDetector(process_monitor=None, calendar_pollers=[]),
        recorder_factory=lambda _c: None,
        notifications=NotificationManager(backend=lambda t, s, m: sent.append((t, s, m))),
        icons_dir=tmp_path / "icons", user_config_path=tmp_path / "config.yaml",
        audio_dir=tmp_path / "audio", health=registry,
    )
    app._sent = sent  # type: ignore[attr-defined]
    return app


def _status(monkeypatch, status: str) -> list[object]:
    requests: list[object] = []
    monkeypatch.setattr(mic_permission, "microphone_authorization", lambda: status)
    monkeypatch.setattr(mic_permission, "request_microphone_access",
                        lambda cb: requests.append(cb) or True)
    return requests


def test_an_undecided_permission_is_asked_for_at_launch(tmp_path, registry, monkeypatch) -> None:
    requests = _status(monkeypatch, mic_permission.NOT_DETERMINED)
    app = _app(tmp_path, registry)
    app._check_microphone_permission()
    assert len(requests) == 1, "ask now, while the user is at the Mac"


def test_a_refused_prompt_becomes_a_failure_with_the_fix(tmp_path, registry, monkeypatch) -> None:
    requests = _status(monkeypatch, mic_permission.NOT_DETERMINED)
    app = _app(tmp_path, registry)
    app._check_microphone_permission()
    requests[0](False)  # the user clicked Don't Allow (on some other thread)
    app._drain_main_queue(None)
    rep = registry.get(MICROPHONE)
    assert rep.state is HealthState.FAIL
    assert "denied" in rep.detail
    assert any("Microphone" in t for t, _s, _m in app._sent)


def test_a_denied_permission_is_a_failure_not_a_quiet_mic(tmp_path, registry, monkeypatch) -> None:
    requests = _status(monkeypatch, mic_permission.DENIED)
    app = _app(tmp_path, registry)
    app._check_microphone_permission()
    assert requests == [], "macOS will not ask again; send the user to Settings"
    assert registry.get(MICROPHONE).state is HealthState.FAIL


def test_an_authorized_permission_changes_nothing(tmp_path, registry, monkeypatch) -> None:
    requests = _status(monkeypatch, mic_permission.AUTHORIZED)
    app = _app(tmp_path, registry)
    app._check_microphone_permission()
    assert requests == []
    assert registry.get(MICROPHONE) is None


def test_run_checks_the_permission(tmp_path, registry, monkeypatch) -> None:
    import rumps

    asked: list[bool] = []
    monkeypatch.setattr(MenuBarApp, "_check_microphone_permission", lambda self: asked.append(True))

    class NoLoopApp(rumps.App):
        def run(self, **kwargs):
            return None

    app = MenuBarApp(
        config=Config({"app": {"working_days": []}}),
        detector=MeetingDetector(process_monitor=None, calendar_pollers=[]),
        recorder_factory=lambda _c: None,
        notifications=NotificationManager(backend=lambda t, s, m: None),
        icons_dir=tmp_path / "icons", user_config_path=tmp_path / "config.yaml",
        audio_dir=tmp_path / "audio", health=registry, rumps_app_factory=NoLoopApp,
    )
    try:
        app.run()
    finally:
        rumps.events.before_quit.unregister(app._on_before_quit)
    assert asked == [True]
