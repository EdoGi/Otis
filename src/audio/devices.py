"""Audio device discovery and management.

Wraps ``sounddevice.query_devices()`` so the rest of the app can ask narrow
questions: "is BlackHole installed?", "what's the default mic?", "is there a
multi-output device that includes BlackHole?". All sounddevice access is funneled
through this module so tests can mock it in one place.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

BLACKHOLE_NAME_HINTS: tuple[str, ...] = ("blackhole", "black hole")

# Names that indicate a virtual/loopback device — never pick these as the
# user's "default mic" when CoreAudio doesn't supply one.
VIRTUAL_INPUT_HINTS: tuple[str, ...] = (
    "blackhole",
    "black hole",
    "soundflower",
    "loopback audio",
    "ishowu",
    "rogue amoeba",
    "aggregate",
    "multi-output",
)


@dataclass(frozen=True)
class AudioDevice:
    """A normalised view of a sounddevice device entry."""

    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_sample_rate: float
    hostapi: int

    @property
    def is_input(self) -> bool:
        return self.max_input_channels > 0

    @property
    def is_output(self) -> bool:
        return self.max_output_channels > 0

    @classmethod
    def from_sd(cls, index: int, raw: dict[str, Any]) -> "AudioDevice":
        return cls(
            index=index,
            name=str(raw.get("name", "")),
            max_input_channels=int(raw.get("max_input_channels", 0)),
            max_output_channels=int(raw.get("max_output_channels", 0)),
            default_sample_rate=float(raw.get("default_samplerate", 0.0) or 0.0),
            hostapi=int(raw.get("hostapi", 0)),
        )


def _looks_virtual(name: str) -> bool:
    """Heuristic: does this device name suggest a virtual/loopback driver?"""
    lower = name.lower()
    return any(hint in lower for hint in VIRTUAL_INPUT_HINTS)


class DeviceNotFoundError(RuntimeError):
    """Raised when a requested audio device cannot be located."""


class DeviceManager:
    """Discovers audio devices and answers questions about the audio setup.

    The constructor caches a snapshot. Call :meth:`refresh` to re-query the OS
    (e.g. after a Bluetooth headset connects mid-session).
    """

    def __init__(self) -> None:
        self._devices: list[AudioDevice] = []
        self._default_input_index: int | None = None
        self._default_output_index: int | None = None
        self.refresh()

    # ------------------------------------------------------------------ refresh
    def refresh(self) -> None:
        """Re-query sounddevice and update the cached device list."""
        import sounddevice as sd  # imported lazily so module imports cheaply

        raw_devices = sd.query_devices()
        self._devices = [
            AudioDevice.from_sd(idx, dict(d)) for idx, d in enumerate(raw_devices)
        ]

        default = sd.default.device  # (input_idx, output_idx) or single int
        if isinstance(default, (list, tuple)) and len(default) == 2:
            in_idx, out_idx = default
            self._default_input_index = int(in_idx) if in_idx is not None and in_idx != -1 else None
            self._default_output_index = (
                int(out_idx) if out_idx is not None and out_idx != -1 else None
            )
        elif isinstance(default, int):
            self._default_input_index = default if default != -1 else None
            self._default_output_index = default if default != -1 else None
        else:
            self._default_input_index = None
            self._default_output_index = None

        logger.debug(
            "DeviceManager refreshed: %d devices, default in=%s out=%s",
            len(self._devices),
            self._default_input_index,
            self._default_output_index,
        )

    # -------------------------------------------------------------- accessors
    @property
    def devices(self) -> list[AudioDevice]:
        return list(self._devices)

    def list_devices(self) -> list[AudioDevice]:
        """Return every device currently visible to CoreAudio (may be empty)."""
        return list(self._devices)

    def input_devices(self) -> list[AudioDevice]:
        return [d for d in self._devices if d.is_input]

    def output_devices(self) -> list[AudioDevice]:
        return [d for d in self._devices if d.is_output]

    def default_input(self) -> AudioDevice | None:
        """Return the system default input device, or the first physical input.

        On some Macs ``sd.default.device`` reports ``-1`` for the input slot
        even when a microphone is plugged in. We then fall back to the first
        device with input channels — but **skip virtual loopback devices**
        (BlackHole, Soundflower, Loopback, …) because picking BlackHole as the
        mic causes the system loopback to be recorded into the "mic" stream.
        """
        if self._default_input_index is not None:
            return self._devices[self._default_input_index]

        # Pass 1: physical (non-virtual) inputs only.
        for dev in self._devices:
            if dev.is_input and not _looks_virtual(dev.name):
                logger.info(
                    "CoreAudio reported no default input; falling back to physical mic %r.",
                    dev.name,
                )
                return dev
        # Pass 2: anything with input channels, even if virtual — last resort.
        for dev in self._devices:
            if dev.is_input:
                logger.warning(
                    "CoreAudio reported no default input and no physical mic was found; "
                    "falling back to %r (this is likely a loopback device).",
                    dev.name,
                )
                return dev
        return None

    def get_default_mic(self) -> AudioDevice | None:
        """Alias for :meth:`default_input` — the system default microphone."""
        return self.default_input()

    def default_output(self) -> AudioDevice | None:
        if self._default_output_index is None:
            return None
        return self._devices[self._default_output_index]

    def find_by_name(self, name: str, *, input_only: bool = False) -> AudioDevice | None:
        """Return first device whose name contains ``name`` (case-insensitive)."""
        needle = name.lower()
        for dev in self._devices:
            if input_only and not dev.is_input:
                continue
            if needle in dev.name.lower():
                return dev
        return None

    def require(self, name: str, *, input_only: bool = False) -> AudioDevice:
        dev = self.find_by_name(name, input_only=input_only)
        if dev is None:
            raise DeviceNotFoundError(
                f"Audio device matching {name!r} not found. "
                f"Available: {[d.name for d in self._devices]}"
            )
        return dev

    # ---------------------------------------------------------------- BlackHole
    def find_blackhole(self) -> AudioDevice | None:
        """Locate a BlackHole input device (looks for any name containing 'blackhole')."""
        for dev in self._devices:
            lower = dev.name.lower()
            if any(hint in lower for hint in BLACKHOLE_NAME_HINTS) and dev.is_input:
                return dev
        return None

    def is_blackhole_installed(self) -> bool:
        return self.find_blackhole() is not None

    def find_multi_output_device(self) -> AudioDevice | None:
        """Find any macOS aggregate / Multi-Output device.

        Two-tier detection so user-renamed devices ("Otis BT" etc.) still
        get picked up:

        1. **Authoritative**: ask CoreAudio for every device whose transport
           type is ``grup`` (aggregate). Match against our sounddevice list
           by name. This catches arbitrary user-chosen names.
        2. **Fallback**: substring match on ``Multi-Output`` / ``Aggregate``
           — used when the CoreAudio probe can't be imported (e.g. on Linux
           CI) or fails for some reason.
        """
        try:
            from src.audio.coreaudio_probe import list_aggregate_devices

            aggregate_names_lower = {name.lower() for _, name in list_aggregate_devices()}
        except Exception as exc:  # pragma: no cover (non-macOS / probe fail)
            logger.debug("CoreAudio aggregate probe unavailable: %s", exc)
            aggregate_names_lower = set()

        if aggregate_names_lower:
            for dev in self._devices:
                if dev.is_output and dev.name.lower() in aggregate_names_lower:
                    return dev

        for dev in self._devices:
            lower = dev.name.lower()
            if dev.is_output and ("multi-output" in lower or "aggregate" in lower):
                return dev
        return None
