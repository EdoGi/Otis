"""Lightweight probes into CoreAudio via ctypes.

Why ctypes and not pyobjc? On some installs the ``pyobjc-framework-CoreAudio``
package fails to expose a top-level ``CoreAudio`` module, so we'd need a fallback
anyway. Calling the C API directly is simpler, has no third-party dependencies,
and works on every macOS version we care about.

Public API:
    is_default_input_running() -> bool
        Returns True iff some process is currently capturing from the default
        input device. This is the signal Phase 2's ProcessMonitor uses to
        decide whether a whitelisted browser counts as "in a meeting".

    get_default_input_device_id() -> int
        The CoreAudio device id of the current system default input.

All functions raise :class:`CoreAudioProbeError` on failure. None of them are
hot — the ProcessMonitor calls them at most once per poll (every few seconds).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import sys
import threading

logger = logging.getLogger(__name__)


class CoreAudioProbeError(RuntimeError):
    """Raised when a CoreAudio call fails."""


# ----------------------------------------------------------------------------
# CoreAudio constants — four-character codes from <CoreAudio/AudioHardware.h>
# ----------------------------------------------------------------------------
def _fourcc(s: str) -> int:
    if len(s) != 4:
        raise ValueError(f"four-char code must be 4 ASCII chars, got {s!r}")
    return (ord(s[0]) << 24) | (ord(s[1]) << 16) | (ord(s[2]) << 8) | ord(s[3])


_kAudioObjectSystemObject = 1
_kAudioObjectPropertyScopeGlobal = _fourcc("glob")
_kAudioObjectPropertyElementMain = 0  # alias for kAudioObjectPropertyElementMaster
_kAudioHardwarePropertyDefaultInputDevice = _fourcc("dIn ")
_kAudioHardwarePropertyDevices = _fourcc("dev#")
_kAudioDevicePropertyDeviceIsRunningSomewhere = _fourcc("gone")
_kAudioDevicePropertyTransportType = _fourcc("tran")
_kAudioObjectPropertyName = _fourcc("lnam")
_kAudioDeviceTransportTypeAggregate = _fourcc("grup")
_kCFStringEncodingUTF8 = 0x08000100


class _AudioObjectPropertyAddress(ctypes.Structure):
    _fields_ = [
        ("mSelector", ctypes.c_uint32),
        ("mScope", ctypes.c_uint32),
        ("mElement", ctypes.c_uint32),
    ]


# ----------------------------------------------------------------------------
# Lazy framework loader: only resolved on first call. Importing this module on
# Linux (CI) must not fail.
# ----------------------------------------------------------------------------
_lib_lock = threading.Lock()
_lib: ctypes.CDLL | None = None
_cf: ctypes.CDLL | None = None


def _load_coreaudio() -> ctypes.CDLL:
    global _lib
    with _lib_lock:
        if _lib is not None:
            return _lib
        if sys.platform != "darwin":
            raise CoreAudioProbeError("CoreAudio is only available on macOS.")
        path = ctypes.util.find_library("CoreAudio")
        if path is None:
            path = "/System/Library/Frameworks/CoreAudio.framework/CoreAudio"
        try:
            lib = ctypes.CDLL(path)
        except OSError as exc:
            raise CoreAudioProbeError(f"Could not load CoreAudio: {exc}") from exc

        lib.AudioObjectGetPropertyData.restype = ctypes.c_int32
        lib.AudioObjectGetPropertyData.argtypes = [
            ctypes.c_uint32,
            ctypes.POINTER(_AudioObjectPropertyAddress),
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_void_p,
        ]
        lib.AudioObjectGetPropertyDataSize.restype = ctypes.c_int32
        lib.AudioObjectGetPropertyDataSize.argtypes = [
            ctypes.c_uint32,
            ctypes.POINTER(_AudioObjectPropertyAddress),
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        _lib = lib
        return _lib


def _load_corefoundation() -> ctypes.CDLL:
    """Load CoreFoundation for CFString → Python str conversion."""
    global _cf
    with _lib_lock:
        if _cf is not None:
            return _cf
        path = ctypes.util.find_library("CoreFoundation")
        if path is None:
            path = "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
        cf = ctypes.CDLL(path)
        cf.CFStringGetCStringPtr.restype = ctypes.c_char_p
        cf.CFStringGetCStringPtr.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        cf.CFStringGetLength.restype = ctypes.c_long
        cf.CFStringGetLength.argtypes = [ctypes.c_void_p]
        cf.CFStringGetMaximumSizeForEncoding.restype = ctypes.c_long
        cf.CFStringGetMaximumSizeForEncoding.argtypes = [ctypes.c_long, ctypes.c_uint32]
        cf.CFStringGetCString.restype = ctypes.c_bool
        cf.CFStringGetCString.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32
        ]
        cf.CFRelease.argtypes = [ctypes.c_void_p]
        _cf = cf
        return _cf


def _cfstring_to_str(cf_ref: int) -> str:
    """Convert a CFStringRef (as ``c_void_p`` value) to a Python ``str``."""
    if not cf_ref:
        return ""
    cf = _load_corefoundation()
    fast = cf.CFStringGetCStringPtr(cf_ref, _kCFStringEncodingUTF8)
    if fast:
        return fast.decode("utf-8", errors="replace")
    length = cf.CFStringGetLength(cf_ref)
    max_size = cf.CFStringGetMaximumSizeForEncoding(length, _kCFStringEncodingUTF8) + 1
    buf = ctypes.create_string_buffer(max_size)
    if cf.CFStringGetCString(cf_ref, buf, max_size, _kCFStringEncodingUTF8):
        return buf.value.decode("utf-8", errors="replace")
    return ""


def _get_property(
    object_id: int,
    selector: int,
    out_type: type,
) -> ctypes.Structure | ctypes.c_uint32:
    """Wrap ``AudioObjectGetPropertyData`` for a single fixed-size property."""
    lib = _load_coreaudio()
    addr = _AudioObjectPropertyAddress(
        selector,
        _kAudioObjectPropertyScopeGlobal,
        _kAudioObjectPropertyElementMain,
    )
    out = out_type()
    size = ctypes.c_uint32(ctypes.sizeof(out_type))
    err = lib.AudioObjectGetPropertyData(
        object_id,
        ctypes.byref(addr),
        0,
        None,
        ctypes.byref(size),
        ctypes.byref(out),
    )
    if err != 0:
        raise CoreAudioProbeError(
            f"AudioObjectGetPropertyData(selector={selector:#x}, object={object_id}) failed: {err}"
        )
    return out


# ----------------------------------------------------------------------------
# Public probes
# ----------------------------------------------------------------------------
def get_default_input_device_id() -> int:
    """Return the CoreAudio device id of the system default input."""
    out = _get_property(
        _kAudioObjectSystemObject,
        _kAudioHardwarePropertyDefaultInputDevice,
        ctypes.c_uint32,
    )
    return int(out.value)


def is_input_device_running(device_id: int) -> bool:
    """Return True iff some process is currently capturing from ``device_id``."""
    out = _get_property(
        device_id,
        _kAudioDevicePropertyDeviceIsRunningSomewhere,
        ctypes.c_uint32,
    )
    return bool(out.value)


def is_default_input_running() -> bool:
    """Return True iff some process is currently capturing from the default mic.

    Returns ``False`` (with a logged warning) on any CoreAudio failure so the
    caller can keep polling without crashing.
    """
    try:
        return is_input_device_running(get_default_input_device_id())
    except CoreAudioProbeError as exc:
        logger.warning("CoreAudio mic probe failed: %s", exc)
        return False


# ----------------------------------------------------------------------------
# Aggregate / multi-output device discovery
# ----------------------------------------------------------------------------
def list_audio_device_ids() -> list[int]:
    """Return CoreAudio device IDs for every device on the system."""
    lib = _load_coreaudio()
    addr = _AudioObjectPropertyAddress(
        _kAudioHardwarePropertyDevices,
        _kAudioObjectPropertyScopeGlobal,
        _kAudioObjectPropertyElementMain,
    )
    size = ctypes.c_uint32(0)
    err = lib.AudioObjectGetPropertyDataSize(
        _kAudioObjectSystemObject, ctypes.byref(addr), 0, None, ctypes.byref(size)
    )
    if err != 0 or size.value == 0:
        return []
    n = size.value // ctypes.sizeof(ctypes.c_uint32)
    Buffer = ctypes.c_uint32 * n
    buf = Buffer()
    err = lib.AudioObjectGetPropertyData(
        _kAudioObjectSystemObject,
        ctypes.byref(addr),
        0,
        None,
        ctypes.byref(size),
        buf,
    )
    if err != 0:
        raise CoreAudioProbeError(f"failed to enumerate devices: {err}")
    return list(buf)


def get_device_name(device_id: int) -> str:
    """Return the human-readable name of a CoreAudio device id."""
    lib = _load_coreaudio()
    addr = _AudioObjectPropertyAddress(
        _kAudioObjectPropertyName,
        _kAudioObjectPropertyScopeGlobal,
        _kAudioObjectPropertyElementMain,
    )
    cf_ref = ctypes.c_void_p()
    size = ctypes.c_uint32(ctypes.sizeof(cf_ref))
    err = lib.AudioObjectGetPropertyData(
        device_id, ctypes.byref(addr), 0, None, ctypes.byref(size), ctypes.byref(cf_ref)
    )
    if err != 0:
        return f"<device {device_id}>"
    try:
        return _cfstring_to_str(cf_ref.value or 0)
    finally:
        if cf_ref.value:
            _load_corefoundation().CFRelease(cf_ref.value)


def get_transport_type(device_id: int) -> int:
    """Return the four-char transport type code for a device.

    Common values: ``bltn`` built-in, ``blue`` Bluetooth, ``usb_`` USB,
    ``grup`` aggregate (Multi-Output / Aggregate Device).
    """
    out = _get_property(device_id, _kAudioDevicePropertyTransportType, ctypes.c_uint32)
    return int(out.value)


def is_aggregate_device(device_id: int) -> bool:
    """Return True iff ``device_id`` is a Multi-Output / Aggregate device."""
    try:
        return get_transport_type(device_id) == _kAudioDeviceTransportTypeAggregate
    except CoreAudioProbeError:
        return False


def list_aggregate_devices() -> list[tuple[int, str]]:
    """Return ``(device_id, name)`` for every aggregate / multi-output device."""
    out: list[tuple[int, str]] = []
    try:
        ids = list_audio_device_ids()
    except CoreAudioProbeError as exc:
        logger.warning("Could not enumerate CoreAudio devices: %s", exc)
        return out
    for did in ids:
        if is_aggregate_device(did):
            out.append((did, get_device_name(did)))
    return out
