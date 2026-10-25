"""`otis doctor` — the self-check that replaces scripts/check_phase2.py.

check_phase2 was red on this Mac for a reason unrelated to the Mac ("pytest
not found in PATH"), green on the things that were actually broken (routing,
via the structural BlackHole check), read only the bundled config, and
accepted Python 3.10, which cannot import src.
"""

from __future__ import annotations

import importlib.util
import socket
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.main as main_mod
from src import doctor
from src.audio.blackhole_check import RoutingState, RoutingStatus
from src.config import Config
from src.doctor import Check, Verdict

REPO = Path(__file__).resolve().parent.parent


def test_the_python_floor_matches_pyproject() -> None:
    spec = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["requires-python"]
    assert spec == ">=" + ".".join(map(str, doctor.MIN_PYTHON))


@pytest.mark.parametrize(("version", "verdict"), [
    ((3, 10, 14), Verdict.FAIL),  # check_phase2 said ✓ to this
    ((3, 11, 9), Verdict.FAIL),
    ((3, 12, 0), Verdict.OK),
    ((3, 13, 1), Verdict.OK),
])
def test_python_version(version, verdict) -> None:
    assert doctor.check_python(version).verdict is verdict


def test_the_users_config_is_what_gets_checked(tmp_path: Path, monkeypatch) -> None:
    """check_phase2 loaded only the bundled defaults, so this passed."""
    import src.config as config_mod

    bad = tmp_path / "config.yaml"
    bad.write_text("transcription:\n  model: not-a-model\n", encoding="utf-8")
    monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", bad)
    check, cfg = doctor.check_config(None)
    assert check.verdict is Verdict.FAIL
    assert "not-a-model" in check.detail
    assert cfg is None


def test_config_warnings_are_surfaced(tmp_path: Path, monkeypatch) -> None:
    import src.config as config_mod

    stale = tmp_path / "config.yaml"
    stale.write_text("detection:\n  mic_activation:\n    enabled: true\n", encoding="utf-8")
    monkeypatch.setattr(config_mod, "USER_CONFIG_PATH", stale)
    check, _ = doctor.check_config(None)
    assert check.verdict is Verdict.WARN
    assert "mic_activation" in check.detail


def test_unrouted_system_audio_fails() -> None:
    """The failure check_phase2's structural BlackHole check reported as ✓."""
    status = RoutingStatus(state=RoutingState.NOT_ROUTED, default_output_name="Speakers",
                           suggested_device="Otis")
    check = doctor.check_routing(lambda: status)
    assert check.verdict is Verdict.FAIL
    assert "Speakers" in check.detail


def test_inaudible_routing_warns() -> None:
    status = RoutingStatus(state=RoutingState.INAUDIBLE, default_output_name="Otis BT")
    assert doctor.check_routing(lambda: status).verdict is Verdict.WARN


def test_mcp_2_is_flagged() -> None:
    """mcp 2.x removed mcp.server.fastmcp, which the MCP server imports."""
    assert doctor.check_mcp_version(lambda name: "2.2.0").verdict is Verdict.FAIL
    assert doctor.check_mcp_version(lambda name: "1.30.0").verdict is Verdict.OK


def test_missing_modules_fail_without_importing_them() -> None:
    located: list[str] = []

    def find(name: str):
        located.append(name)
        return None if name == "rumps" else object()

    check = doctor.check_modules(find)
    assert check.verdict is Verdict.FAIL
    assert "rumps" in check.detail
    assert "mlx_whisper" in located  # located, never imported


@pytest.mark.parametrize(("model", "cached", "verdict"), [
    ("medium", ["medium", "small"], Verdict.OK),
    ("large", ["large-v3"], Verdict.OK),  # "large" resolves to the v3 repo
    ("medium", ["small", "tiny"], Verdict.WARN),
])
def test_model_cache(model, cached, verdict) -> None:
    assert doctor.check_model_cached(model, cached).verdict is verdict


def test_missing_notification_identity_warns(tmp_path: Path) -> None:
    (tmp_path / "bin").mkdir()
    assert doctor.check_notification_identity(tmp_path).verdict is Verdict.WARN
    (tmp_path / "bin" / "Info.plist").write_bytes(
        b'<?xml version="1.0"?><plist version="1.0"><dict><key>CFBundleIdentifier</key>'
        b"<string>rumps</string></dict></plist>"
    )
    assert doctor.check_notification_identity(tmp_path).verdict is Verdict.OK


def test_a_port_held_by_another_program_warns() -> None:
    with socket.socket() as other:
        other.bind(("127.0.0.1", 0))
        other.listen()
        port = other.getsockname()[1]
        assert doctor.check_port("127.0.0.1", port, otis_command=None).verdict is Verdict.WARN
        # The daemon never serves the web UI: a busy port is still someone else.
        assert doctor.check_port("127.0.0.1", port, otis_command="run").verdict is Verdict.WARN


def test_a_running_menu_bar_must_actually_answer_on_the_port() -> None:
    """Review finding: any lock holder made the port '✓ held by the running
    Otis', even a menu bar that lost the port to another program."""
    otis_page = "<html><head><title>Transcripts — Otis</title>"
    ok = doctor.check_port("127.0.0.1", 8765, otis_command="ui", fetch=lambda h, p: otis_page)
    assert ok.verdict is Verdict.OK
    foreign = doctor.check_port("127.0.0.1", 8765, otis_command="ui",
                                fetch=lambda h, p: '{"error": "not found"}')
    assert foreign.verdict is Verdict.WARN
    silent = doctor.check_port("127.0.0.1", 8765, otis_command="ui", fetch=lambda h, p: None)
    assert silent.verdict is Verdict.WARN


def test_a_check_that_raises_becomes_a_row_not_a_lost_report(tmp_path, monkeypatch) -> None:
    """Review finding: with sounddevice missing (or PortAudio failing to load)
    the BlackHole check raised and the whole report — including the row that
    named the missing package — was never printed."""
    import src.audio.blackhole_check as bc

    def no_portaudio(*a, **k):
        raise OSError("PortAudio library not found")

    monkeypatch.setattr(bc, "verify_blackhole_setup", no_portaudio)
    monkeypatch.setattr(doctor, "check_port", lambda *a, **k: Check("port", Verdict.OK))
    checks = doctor.run_checks(None, run_dir=tmp_path)
    row = next(c for c in checks if c.label == "BlackHole")
    assert row.verdict is Verdict.FAIL and "PortAudio" in row.detail
    assert any(c.label == "Disk" for c in checks), "the rest of the report still ran"


def test_a_running_otis_is_reported(tmp_path: Path) -> None:
    from src import single_instance

    assert doctor.check_running_instance(tmp_path).detail == "not running"
    holder = subprocess.Popen(
        [sys.executable, "-c",
         "import sys,time; sys.path.insert(0, sys.argv[1]);"
         "from pathlib import Path; from src.single_instance import acquire;"
         "l = acquire(Path(sys.argv[2]), command='ui'); print('held', flush=True); time.sleep(30)",
         str(REPO), str(tmp_path / single_instance.LOCK_NAME)],
        stdout=subprocess.PIPE, text=True,
    )
    try:
        assert holder.stdout is not None and holder.stdout.readline().strip() == "held"
        detail = doctor.check_running_instance(tmp_path).detail
        assert detail.startswith("running") and str(holder.pid) in detail
    finally:
        holder.kill()
        holder.wait(5)


def test_calendar_accounts_are_judged_by_their_token_files(tmp_path: Path) -> None:
    (tmp_path / "work_token.json").write_text("{}", encoding="utf-8")
    cfg = Config({"detection": {"calendar": {"accounts": [
        {"label": "personal", "credentials_path": str(tmp_path / "creds.json"),
         "token_path": str(tmp_path / "personal_token.json")},
        {"label": "work", "credentials_path": str(tmp_path / "creds.json"),
         "token_path": str(tmp_path / "work_token.json")},
    ]}}})
    checks = {c.label: c.verdict for c in doctor.check_calendar_accounts(cfg)}
    assert checks == {"Calendar (personal)": Verdict.WARN, "Calendar (work)": Verdict.OK}


def test_exit_code_follows_failures_not_warnings(monkeypatch, capsys) -> None:
    monkeypatch.setattr(doctor, "run_checks", lambda *a, **k: [Check("x", Verdict.WARN)])
    assert main_mod.main(["doctor"]) == 0
    monkeypatch.setattr(doctor, "run_checks", lambda *a, **k: [Check("x", Verdict.FAIL)])
    assert main_mod.main(["doctor"]) == 1
    assert "✗ x" in capsys.readouterr().out


def test_doctor_runs_beside_a_running_otis(tmp_path: Path, monkeypatch) -> None:
    """Like check-audio, it is most useful while Otis is up: no lock taken."""
    ran: list[bool] = []
    monkeypatch.setattr(doctor, "run_checks", lambda *a, **k: ran.append(True) or [])
    monkeypatch.setattr(main_mod, "_defer_to_running_instance",
                        lambda *a: pytest.fail("doctor deferred to the running copy"))
    from src import single_instance

    lock = single_instance.acquire(main_mod.LOG_DIR / single_instance.LOCK_NAME, command="ui")
    try:
        assert main_mod.main(["doctor"]) == 0
    finally:
        assert lock is not None
        lock.release()
    assert ran == [True]


# ---------------------------------------------------------------------------
# scripts/check_phase2.py is now a thin wrapper
# ---------------------------------------------------------------------------
def _load_check_phase2():
    spec = importlib.util.spec_from_file_location("check_phase2", REPO / "scripts" / "check_phase2.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_check_phase2_delegates_to_doctor(monkeypatch) -> None:
    mod = _load_check_phase2()
    monkeypatch.setattr(doctor, "main", lambda *a, **k: 0)
    ran: list[list[str]] = []
    monkeypatch.setattr(mod.subprocess, "run", lambda argv, **kw: ran.append(argv))
    assert mod.main([]) == 0
    assert ran == [], "tests must be opt-in, not a hidden 35 s suite run"


def test_check_phase2_runs_tests_with_this_interpreter(monkeypatch) -> None:
    """It used to look `pytest` up on PATH: a false ✗ outside an activated venv."""
    mod = _load_check_phase2()
    monkeypatch.setattr(doctor, "main", lambda *a, **k: 0)
    ran: list[list[str]] = []

    def fake_run(argv, **kw):
        ran.append(argv)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    assert mod.main(["--with-tests"]) == 0
    assert ran and ran[0][:3] == [sys.executable, "-m", "pytest"]


def test_the_web_ui_probe_never_goes_through_a_proxy(monkeypatch) -> None:
    """Review round 2: urllib honours http_proxy (and macOS proxy settings)
    and does not bypass 127.0.0.1, so a proxied shell turned a healthy web UI
    into "not serving" — and sent the probe to the proxy host."""
    import http.server
    import threading

    class Page(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = b"<html><head><title>Transcripts \\xe2\\x80\\x94 Otis</title></head></html>"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Page)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")  # nothing listens there
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    # A shell's NO_PROXY=127.0.0.1 would let a urllib revert slip through.
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.delenv("NO_PROXY", raising=False)
    try:
        page = doctor._fetch_page("127.0.0.1", server.server_address[1])
    finally:
        server.shutdown()
    assert page is not None and "Otis</title>" in page


def test_unused_models_are_reported_with_how_to_reclaim_them(tmp_path) -> None:
    """The disk was 97% full on 2026-10-25 with small and tiny (555 MB) cached
    but unused; prune-models existed and nothing pointed at it."""
    from src.storage.model_cache import CachedModel

    models = [CachedModel("medium", tmp_path, 1_525_000_000),
              CachedModel("small", tmp_path, 481_000_000),
              CachedModel("tiny", tmp_path, 74_000_000)]
    check = doctor.check_unused_models("medium", models, free_bytes=16 * 1000**3)
    assert check.verdict is Verdict.OK
    assert "small" in check.detail and "tiny" in check.detail and "555 MB" in check.detail
    assert "prune-models" in check.fix

    low = doctor.check_unused_models("medium", models, free_bytes=3 * 1000**3)
    assert low.verdict is Verdict.WARN, "with the disk low, reclaiming is worth a warning"

    none = doctor.check_unused_models("medium", models[:1], free_bytes=16 * 1000**3)
    assert none.verdict is Verdict.OK and none.detail == "none"


def test_prune_models_never_deletes_the_model_large_resolves_to(tmp_path) -> None:
    """Review: doctor understood that config 'large' means large-v3, but the
    prune-models command it recommends compared names literally and would have
    deleted large-v3 — the active model, ~3 GB to re-download."""
    from src.storage.model_cache import prune_models

    hub = tmp_path / "hub"
    for name in ("large-v3", "small"):
        snap = hub / f"models--mlx-community--whisper-{name}-mlx" / "snapshots" / "x"
        snap.mkdir(parents=True)
        (snap / "config.json").write_text("{}")
        (snap / "weights.npz").write_bytes(b"\0" * 10)
    victims = [m.name for m in prune_models("large", hub_dir=hub, dry_run=True)]
    assert victims == ["small"]


def test_the_unused_models_row_judges_the_disk_that_holds_the_cache(tmp_path, monkeypatch) -> None:
    import src.storage.model_cache as mc

    seen: list[object] = []
    monkeypatch.setattr(mc, "DEFAULT_HUB_DIR", tmp_path / "hub")
    monkeypatch.setattr(doctor, "_free_bytes_near", lambda p: seen.append(p) or 10**12)
    monkeypatch.setattr(doctor, "check_port", lambda *a, **k: Check("port", Verdict.OK))
    checks = doctor.run_checks(None, run_dir=tmp_path)
    assert any(c.label == "Unused Whisper models" for c in checks), "the row is part of the report"
    assert tmp_path / "hub" in seen
