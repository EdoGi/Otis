"""BlackHole installation and multi-output configuration verification.

BlackHole (https://existential.audio/blackhole/) is a virtual audio loopback
driver. To capture system audio while the user can still hear it, the user must
create a Multi-Output Device in Audio MIDI Setup that fans output to both their
real speakers/headphones AND BlackHole. We cannot introspect the members of an
aggregate device from sounddevice, so we report "probably configured" when an
aggregate output exists alongside an installed BlackHole.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from src.audio.devices import DeviceManager

logger = logging.getLogger(__name__)


SETUP_INSTRUCTIONS = """\
BlackHole setup steps:
  1. Install BlackHole 2ch:  brew install blackhole-2ch
     (or download from https://existential.audio/blackhole/)
  2. Open 'Audio MIDI Setup' (Applications > Utilities).
  3. Click the '+' bottom-left and choose 'Create Multi-Output Device'.
  4. In the new device, tick BOTH your real output (e.g. 'MacBook Pro Speakers')
     AND 'BlackHole 2ch'. Set the real output as the master/clock source.
  5. Right-click the multi-output device and choose 'Use This Device For Sound Output'
     when you want to record system audio. (Otis can do this for you later.)
  6. Re-run Otis; it will detect the configuration automatically.
"""


@dataclass
class BlackHoleStatus:
    installed: bool
    multi_output_configured: bool
    issues: list[str] = field(default_factory=list)
    blackhole_device_name: str | None = None
    multi_output_device_name: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "installed": self.installed,
            "multi_output_configured": self.multi_output_configured,
            "issues": list(self.issues),
            "blackhole_device_name": self.blackhole_device_name,
            "multi_output_device_name": self.multi_output_device_name,
        }

    @property
    def ok(self) -> bool:
        return self.installed and self.multi_output_configured


def verify_blackhole_setup(device_manager: DeviceManager | None = None) -> BlackHoleStatus:
    """Return a status object describing the BlackHole audio setup."""
    if device_manager is None:
        device_manager = DeviceManager()

    blackhole = device_manager.find_blackhole()
    multi_out = device_manager.find_multi_output_device()
    issues: list[str] = []

    if blackhole is None:
        issues.append(
            "BlackHole virtual driver not detected. Install with `brew install blackhole-2ch`."
        )
    if multi_out is None:
        issues.append(
            "No Multi-Output / Aggregate Device detected. "
            "Create one in Audio MIDI Setup combining your speakers and BlackHole."
        )

    status = BlackHoleStatus(
        installed=blackhole is not None,
        multi_output_configured=multi_out is not None,
        issues=issues,
        blackhole_device_name=blackhole.name if blackhole else None,
        multi_output_device_name=multi_out.name if multi_out else None,
    )

    if not status.ok:
        logger.warning("BlackHole setup incomplete: %s", "; ".join(issues))
    else:
        logger.info(
            "BlackHole OK (input=%s, multi-output=%s)",
            status.blackhole_device_name,
            status.multi_output_device_name,
        )
    return status


# ===========================================================================
# Output routing — is system audio ACTUALLY reaching BlackHole right now?
# ===========================================================================
# verify_blackhole_setup() above is a *structural* check: it asks whether the
# pieces exist. That is not the same question as whether they are wired up, and
# the difference cost two months of half-recorded meetings: BlackHole was
# installed, a Multi-Output device called "Otis" existed, the check returned
# ok=True — and the Mac's output was pointed at the built-in speakers the whole
# time, so BlackHole received pure digital silence.
#
# This check asks the real question: does the *current default output device*
# carry audio into BlackHole?


class RoutingState(str, Enum):
    OK = "ok"
    #: Audio reaches BlackHole, but nothing audible — the user hears nothing.
    INAUDIBLE = "inaudible"
    #: Audio does not reach BlackHole at all. System capture will be silent.
    NOT_ROUTED = "not_routed"
    #: CoreAudio wouldn't answer. Never claim health we couldn't verify.
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RoutingProbe:
    """The CoreAudio calls the routing check needs, injected so it's testable.

    Defaults are bound lazily in :func:`default_routing_probe` so importing
    this module on a non-macOS box (CI) stays safe.
    """

    default_output_id: Callable[[], int]
    device_name: Callable[[int], str]
    device_uid: Callable[[int], str]
    aggregate_members: Callable[[int], list[int]]
    list_devices: Callable[[], list[int]] = tuple  # type: ignore[assignment]


@dataclass(frozen=True)
class RoutingStatus:
    state: RoutingState
    default_output_id: int | None = None
    default_output_name: str = ""
    includes_blackhole: bool = False
    is_aggregate: bool = False
    member_names: list[str] = field(default_factory=list)
    #: Name of an aggregate that *does* contain BlackHole, when we found one.
    suggested_device: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        """True only when system audio is both captured *and* audible."""
        return self.state is RoutingState.OK

    def message(self) -> str:
        if self.state is RoutingState.OK:
            members = " + ".join(self.member_names) or self.default_output_name
            return f"System audio routed through {self.default_output_name!r} ({members})."
        if self.state is RoutingState.INAUDIBLE:
            return (
                f"Output {self.default_output_name!r} feeds BlackHole but has no "
                "speakers — Otis will record the call, but you won't hear it. "
                "Add your speakers or headphones to that device in Audio MIDI Setup."
            )
        if self.state is RoutingState.NOT_ROUTED:
            fix = (
                f"Set your Mac's output device to {self.suggested_device!r}."
                if self.suggested_device
                else "Create a Multi-Output device combining your speakers with "
                     "BlackHole 2ch, then select it as your output device."
            )
            return (
                f"Your Mac is playing to {self.default_output_name!r}, which does not "
                f"pass audio to BlackHole — system audio will record as silence. {fix}"
            )
        return f"Could not determine audio routing: {self.error or 'unknown error'}"


def _looks_like_blackhole(name: str, uid: str) -> bool:
    return "blackhole" in name.lower() or "blackhole" in uid.lower()


def default_routing_probe() -> RoutingProbe:
    """Bind the real CoreAudio probes (import is deferred for non-macOS)."""
    from src.audio import coreaudio_probe as cap

    return RoutingProbe(
        default_output_id=cap.get_default_output_device_id,
        device_name=cap.get_device_name,
        device_uid=cap.get_device_uid,
        aggregate_members=cap.get_aggregate_member_ids,
        list_devices=cap.list_audio_device_ids,
    )


#: An aggregate nested inside an aggregate is unusual but legal, and macOS
#: offers no depth limit. Bounded so a device that somehow contains itself
#: cannot spin the routing check.
_MAX_AGGREGATE_DEPTH = 4


def _expand_members(probe: RoutingProbe, device_id: int) -> list[int]:
    """Every leaf device the output eventually reaches.

    One level was walked before, so an aggregate containing another aggregate
    that contains BlackHole reported NOT_ROUTED — telling the user system
    audio was broken while it was working. A health panel that cries wolf is
    the failure P0 set out to fix, just pointed the other way.
    """
    seen: set[int] = set()
    leaves: list[int] = []

    def walk(did: int, depth: int) -> None:
        if did in seen or depth > _MAX_AGGREGATE_DEPTH:
            return
        seen.add(did)
        try:
            children = list(probe.aggregate_members(did))
        except Exception:  # pragma: no cover (defensive)
            children = []
        if not children:
            if depth > 0:
                leaves.append(did)
            return
        for child in children:
            walk(child, depth + 1)

    walk(device_id, 0)
    return leaves


def check_output_routing(probe: RoutingProbe | None = None) -> RoutingStatus:
    """Report whether the current default output actually feeds BlackHole."""
    if probe is None:
        try:
            probe = default_routing_probe()
        except Exception as exc:  # pragma: no cover (non-macOS import failure)
            return RoutingStatus(state=RoutingState.UNKNOWN, error=str(exc))

    try:
        out_id = probe.default_output_id()
        out_name = probe.device_name(out_id)
        members = _expand_members(probe, out_id)
    except Exception as exc:
        logger.warning("Audio routing probe failed: %s", exc)
        return RoutingStatus(state=RoutingState.UNKNOWN, error=str(exc))

    is_aggregate = bool(members)
    # A plain device is its own only "member" for the purposes of this check.
    candidates = members or [out_id]

    member_names: list[str] = []
    includes_blackhole = False
    has_audible_member = False
    for did in candidates:
        try:
            name = probe.device_name(did)
            uid = probe.device_uid(did)
        except Exception:  # pragma: no cover (defensive)
            continue
        member_names.append(name)
        if _looks_like_blackhole(name, uid):
            includes_blackhole = True
        else:
            has_audible_member = True

    suggestion = None if includes_blackhole else _suggest_aggregate(probe)

    if includes_blackhole and has_audible_member:
        state = RoutingState.OK
    elif includes_blackhole:
        state = RoutingState.INAUDIBLE
    else:
        state = RoutingState.NOT_ROUTED

    status = RoutingStatus(
        state=state,
        default_output_id=out_id,
        default_output_name=out_name,
        includes_blackhole=includes_blackhole,
        is_aggregate=is_aggregate,
        member_names=member_names,
        suggested_device=suggestion,
    )
    if state is RoutingState.OK:
        logger.info("Audio routing OK: %s", status.message())
    else:
        logger.warning("Audio routing degraded: %s", status.message())
    return status


def _suggest_aggregate(probe: RoutingProbe) -> str | None:
    """Find an aggregate that contains BlackHole *and* something audible."""
    try:
        device_ids = list(probe.list_devices())
    except Exception:  # pragma: no cover (defensive)
        return None

    fallback: str | None = None
    for did in device_ids:
        try:
            members = list(probe.aggregate_members(did))
        except Exception:  # pragma: no cover (defensive)
            continue
        if not members:
            continue
        has_bh = False
        has_audible = False
        for mid in members:
            try:
                if _looks_like_blackhole(probe.device_name(mid), probe.device_uid(mid)):
                    has_bh = True
                else:
                    has_audible = True
            except Exception:  # pragma: no cover (defensive)
                continue
        if has_bh and has_audible:
            return probe.device_name(did)
        if has_bh and fallback is None:
            fallback = probe.device_name(did)
    return fallback


def format_setup_instructions(status: BlackHoleStatus) -> str:
    """Render a user-facing message describing what to do next."""
    if status.ok:
        return "BlackHole is installed and a Multi-Output device is configured. ✓"
    lines = ["Otis needs BlackHole to capture system audio.", ""]
    lines.extend(f"• {issue}" for issue in status.issues)
    lines.append("")
    lines.append(SETUP_INSTRUCTIONS)
    return "\n".join(lines)
