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
from dataclasses import dataclass, field

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


def format_setup_instructions(status: BlackHoleStatus) -> str:
    """Render a user-facing message describing what to do next."""
    if status.ok:
        return "BlackHole is installed and a Multi-Output device is configured. ✓"
    lines = ["Otis needs BlackHole to capture system audio.", ""]
    lines.extend(f"• {issue}" for issue in status.issues)
    lines.append("")
    lines.append(SETUP_INSTRUCTIONS)
    return "\n".join(lines)
