"""Cover for audit findings E9, E10, E11 — the config was unchecked.

* A typo was simply ignored. ``transcription.deffer_while_in_call`` parses,
  merges, and does nothing, so the user believes they turned something off
  that is still on.
* An invalid Whisper model raised ValueError out of ``build_pipeline``,
  outside any handler — from a Login Item that means no menu bar appears and
  the traceback goes to a stdout nobody reads.
* Working hours could not cross midnight (``start <= cur <= end`` is never
  true for 22:00-02:00), and an unparseable time returned True, silently
  disabling the schedule.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from src.config_schema import ConfigError, unknown_keys, validate
from src.schedule import is_within_working_hours

WEEKDAYS = [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Unknown keys warn, never block
# ---------------------------------------------------------------------------
def test_a_typo_is_reported() -> None:
    """The exact typo the audit found in the wild."""
    warnings = validate({"transcription": {"deffer_while_in_call": True}})
    assert any("deffer_while_in_call" in w for w in warnings)


def test_unknown_keys_do_not_stop_the_app() -> None:
    """A config from a newer Otis must still launch this one."""
    assert validate({"future_feature": {"enabled": True}})  # warns, returns


def test_known_keys_are_silent() -> None:
    assert validate({
        "transcription": {"model": "small", "defer_while_in_call": True},
        "storage": {"audio_dir": "~/x", "audio_format": "flac"},
        "app": {"working_days": [0], "working_hours": {"start": "08:00",
                                                       "end": "20:00"}},
    }) == []


def test_calendar_accounts_are_the_users_own_business() -> None:
    """A list of accounts with arbitrary labels must not warn."""
    assert unknown_keys({
        "detection": {"calendar": {"accounts": [
            {"label": "work", "token_path": "~/x.json"},
        ]}}
    }) == []


# ---------------------------------------------------------------------------
# Values that cannot work
# ---------------------------------------------------------------------------
def test_a_misspelled_model_suggests_the_right_one() -> None:
    with pytest.raises(ConfigError, match="Did you mean 'medium'"):
        validate({"transcription": {"model": "mediumm"}})


def test_an_unrecognisable_model_lists_the_options() -> None:
    with pytest.raises(ConfigError, match="large-v3"):
        validate({"transcription": {"model": "huge"}})


@pytest.mark.parametrize("data,fragment", [
    ({"storage": {"audio_format": "mp3"}}, "audio_format"),
    ({"app": {"working_hours": {"start": "9am"}}}, "09:00"),
    ({"web": {"port": 99999}}, "1-65535"),
    ({"web": {"port": "http"}}, "must be a number"),
    ({"app": {"working_days": [0, 9]}}, "0-6"),
    ({"app": {"working_days": "weekdays"}}, "0-6"),
])
def test_unusable_values_are_refused_with_an_explanation(data, fragment) -> None:
    with pytest.raises(ConfigError, match=fragment):
        validate(data)


def test_a_valid_config_raises_nothing() -> None:
    validate({"transcription": {"model": "large-v3"},
              "storage": {"audio_format": "wav"},
              "web": {"port": 8765}})


# ---------------------------------------------------------------------------
# Working hours across midnight
# ---------------------------------------------------------------------------
def _in(dt: datetime, start: str, end: str, days=WEEKDAYS) -> bool:
    return is_within_working_hours(dt, working_days=days,
                                   start_hhmm=start, end_hhmm=end)


def test_an_overnight_window_covers_both_sides_of_midnight() -> None:
    """22:00-02:00 was never inside anything: detection simply never ran."""
    assert _in(datetime(2026, 8, 24, 23, 0), "22:00", "02:00") is True   # Mon
    assert _in(datetime(2026, 8, 25, 1, 0), "22:00", "02:00") is True    # Tue am


def test_an_overnight_window_still_excludes_the_middle_of_the_day() -> None:
    assert _in(datetime(2026, 8, 25, 12, 0), "22:00", "02:00") is False


def test_the_small_hours_belong_to_the_day_the_window_started_on() -> None:
    """01:00 Saturday is Friday's shift; 01:00 Sunday is Saturday's, which is off."""
    assert _in(datetime(2026, 8, 29, 1, 0), "22:00", "02:00") is True    # Sat am
    assert _in(datetime(2026, 8, 30, 1, 0), "22:00", "02:00") is False   # Sun am


def test_an_overnight_window_respects_the_working_day() -> None:
    assert _in(datetime(2026, 8, 29, 23, 0), "22:00", "02:00") is False  # Sat pm


def test_a_normal_window_is_unchanged() -> None:
    assert _in(datetime(2026, 8, 24, 12, 0), "08:00", "20:00") is True
    assert _in(datetime(2026, 8, 24, 22, 0), "08:00", "20:00") is False
    assert _in(datetime(2026, 8, 29, 12, 0), "08:00", "20:00") is False


def test_an_unparseable_time_does_not_silently_disable_the_schedule() -> None:
    """It returned True, which reads as 'always on' — but with a warning now."""
    assert _in(datetime(2026, 8, 24, 12, 0), "eight", "20:00") is True


# ---------------------------------------------------------------------------
# A bad config must not crash headlessly
# ---------------------------------------------------------------------------
def test_main_reports_a_bad_config_instead_of_raising(tmp_path, monkeypatch,
                                                      capsys) -> None:
    from src import main as main_mod

    # rumps.alert blocks until someone clicks it. Neutralised here so a
    # regression that opens one on the CLI path fails the run instead of
    # hanging it.
    try:
        import rumps

        monkeypatch.setattr(rumps, "alert", lambda **kw: 1)
    except Exception:  # pragma: no cover (non-macOS)
        pass

    bad = tmp_path / "config.yaml"
    bad.write_text("transcription:\n  model: nonsense\n", encoding="utf-8")
    monkeypatch.setattr(main_mod, "load_user_config",
                        lambda _p: __import__("src.config", fromlist=["Config"]
                                              ).Config({"transcription":
                                                        {"model": "nonsense"}}))

    code = main_mod.main(["--config", str(bad), "check-audio"])

    assert code == 2
    err = capsys.readouterr().err
    assert "nonsense" in err
    assert "~/.otis/config.yaml" in err


def test_a_cli_run_never_opens_a_modal(monkeypatch) -> None:
    """rumps.alert blocks until clicked; on a CLI that is worse than a trace.

    Asserted against rumps itself rather than the gui flag, because a version
    that ignores the flag hangs the run instead of failing it.
    """
    rumps = pytest.importorskip("rumps")
    from src import main as main_mod

    alerts: list[str] = []
    monkeypatch.setattr(rumps, "alert",
                        lambda **kw: alerts.append(kw.get("title", "")))

    assert main_mod._report_bad_config(ValueError("bad"), gui=False) == 2
    assert alerts == [], "a CLI run opened a blocking modal"

    assert main_mod._report_bad_config(ValueError("bad"), gui=True) == 2
    assert alerts == ["Otis could not start"]


def test_the_ui_path_is_the_one_that_gets_a_dialog(monkeypatch) -> None:
    """A Login Item has no terminal, which is the case the audit describes."""
    from src import main as main_mod

    seen: list[bool] = []
    monkeypatch.setattr(main_mod, "_report_bad_config",
                        lambda exc, gui=False: (seen.append(gui), 2)[1])
    monkeypatch.setattr(main_mod, "load_user_config",
                        lambda _p: __import__("src.config", fromlist=["Config"]
                                              ).Config({"transcription":
                                                        {"model": "nonsense"}}))

    assert main_mod.main(["check-audio"]) == 2
    assert main_mod.main(["ui"]) == 2
    assert seen == [False, True]
