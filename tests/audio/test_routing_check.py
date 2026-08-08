"""Tests for the system-audio routing check.

This is the guard that was missing: the old check only verified BlackHole and
some aggregate device *existed*, so it returned ok=True while the Mac's output
went straight to the built-in speakers and Otis recorded two months of silence.

The real-world failing configuration, taken from the audit:

    default output = 86 'Haut-parleurs MacBook Pro'   (not an aggregate)
    aggregate 56 'Otis' = [86 speakers, 98 BlackHole]  (exists, unused)
"""

from __future__ import annotations

from src.audio.blackhole_check import (
    RoutingProbe,
    RoutingState,
    check_output_routing,
)


def make_probe(
    *,
    default_output: int,
    names: dict[int, str],
    uids: dict[int, str],
    members: dict[int, list[int]] | None = None,
) -> RoutingProbe:
    members = members or {}
    return RoutingProbe(
        default_output_id=lambda: default_output,
        device_name=lambda d: names.get(d, f"<device {d}>"),
        device_uid=lambda d: uids.get(d, ""),
        aggregate_members=lambda d: list(members.get(d, [])),
        list_devices=lambda: sorted(names),
    )


# The exact configuration found on the user's Mac during the audit.
BROKEN_REAL_WORLD = dict(
    default_output=86,
    names={86: "Haut-parleurs MacBook Pro", 98: "BlackHole 2ch", 56: "Otis"},
    uids={86: "BuiltInSpeakerDevice", 98: "BlackHole2ch_UID", 56: "~:AMS2_StackedOutput:0"},
    members={56: [86, 98]},
)


def test_the_actual_bug_is_detected() -> None:
    status = check_output_routing(probe=make_probe(**BROKEN_REAL_WORLD))
    assert status.state is RoutingState.NOT_ROUTED
    assert status.ok is False
    assert status.default_output_name == "Haut-parleurs MacBook Pro"
    assert status.includes_blackhole is False
    # The fix is actionable: we know which aggregate they should pick.
    assert "Otis" in (status.suggested_device or "")


def test_routing_ok_when_default_output_is_the_aggregate() -> None:
    good = dict(BROKEN_REAL_WORLD)
    good["default_output"] = 56
    status = check_output_routing(probe=make_probe(**good))
    assert status.state is RoutingState.OK
    assert status.ok is True
    assert status.includes_blackhole is True
    assert "BlackHole 2ch" in status.member_names
    assert "Haut-parleurs MacBook Pro" in status.member_names


def test_blackhole_only_aggregate_warns_that_audio_is_inaudible() -> None:
    """'Otis BT' on the real Mac contains only BlackHole — capture works, but
    the user hears nothing, which they will read as a bug."""
    status = check_output_routing(probe=make_probe(
        default_output=60,
        names={60: "Otis BT", 98: "BlackHole 2ch"},
        uids={60: "~:AMS2_StackedOutput:1", 98: "BlackHole2ch_UID"},
        members={60: [98]},
    ))
    assert status.state is RoutingState.INAUDIBLE
    # Capture still works, so this is not a hard failure...
    assert status.includes_blackhole is True
    # ...but it is not "ok" either, because the user can't hear the call.
    assert status.ok is False


def test_default_output_is_blackhole_itself_is_inaudible() -> None:
    status = check_output_routing(probe=make_probe(
        default_output=98,
        names={98: "BlackHole 2ch"},
        uids={98: "BlackHole2ch_UID"},
    ))
    assert status.state is RoutingState.INAUDIBLE
    assert status.includes_blackhole is True


def test_aggregate_without_blackhole_is_not_routed() -> None:
    status = check_output_routing(probe=make_probe(
        default_output=70,
        names={70: "Studio Rig", 86: "Haut-parleurs MacBook Pro", 139: "ARZOPA"},
        uids={70: "~:AMS2_StackedOutput:9", 86: "BuiltInSpeakerDevice", 139: "ARZOPA-UID"},
        members={70: [86, 139]},
    ))
    assert status.state is RoutingState.NOT_ROUTED
    assert status.includes_blackhole is False


def test_matches_blackhole_by_name_when_uid_is_unhelpful() -> None:
    status = check_output_routing(probe=make_probe(
        default_output=56,
        names={56: "Multi-Output Device", 86: "Speakers", 98: "BlackHole 16ch"},
        uids={56: "agg", 86: "spk", 98: "SomeVendorUID"},
        members={56: [86, 98]},
    ))
    assert status.state is RoutingState.OK
    assert status.includes_blackhole is True


def test_no_suggestion_when_no_aggregate_contains_blackhole() -> None:
    status = check_output_routing(probe=make_probe(
        default_output=86,
        names={86: "Haut-parleurs MacBook Pro"},
        uids={86: "BuiltInSpeakerDevice"},
    ))
    assert status.state is RoutingState.NOT_ROUTED
    assert status.suggested_device is None


def test_probe_failure_reports_unknown_not_false_ok() -> None:
    """A CoreAudio failure must not masquerade as either healthy or broken."""

    def boom() -> int:
        raise RuntimeError("CoreAudio unavailable")

    probe = RoutingProbe(
        default_output_id=boom,
        device_name=lambda d: "",
        device_uid=lambda d: "",
        aggregate_members=lambda d: [],
    )
    status = check_output_routing(probe=probe)
    assert status.state is RoutingState.UNKNOWN
    assert status.ok is False
    assert status.error and "CoreAudio" in status.error


def test_status_message_is_actionable() -> None:
    status = check_output_routing(probe=make_probe(**BROKEN_REAL_WORLD))
    msg = status.message()
    assert "Haut-parleurs MacBook Pro" in msg
    assert "Otis" in msg
