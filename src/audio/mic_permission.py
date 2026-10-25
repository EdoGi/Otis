"""Microphone permission, asked for at launch rather than mid-meeting.

macOS ties an ad-hoc-signed app's Microphone grant to its code hash. After
the native launcher replaced the AppleScript applet (2026-10-25) the stored
grant still named the applet's hashes, so tccd treated it as undecided — and
the first mic open, the first *recording*, would have raised the prompt in the
middle of a meeting. Until it is answered the mic delivers zeros; if it is
missed or denied, the local side of every later meeting is silent, which the
recorder only reports as "a silent mic usually just means nobody spoke".

So the menu bar checks at launch: undecided -> ask now; denied -> say so as
a failure, with the fix.

Imported lazily: AVFoundation is macOS-only and costs ~0.1 s to load.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)

AUTHORIZED = "authorized"
DENIED = "denied"
RESTRICTED = "restricted"
NOT_DETERMINED = "not_determined"
UNKNOWN = "unknown"

_STATUS_NAMES = {0: NOT_DETERMINED, 1: RESTRICTED, 2: DENIED, 3: AUTHORIZED}

#: Where the user fixes a denial.
SETTINGS_URL = "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone"


_warned_unavailable = False


def microphone_authorization() -> str:
    """This app's microphone authorization, as one of the constants above."""
    global _warned_unavailable
    try:
        import AVFoundation as av

        status = av.AVCaptureDevice.authorizationStatusForMediaType_(av.AVMediaTypeAudio)
    except Exception as exc:  # not macOS, or a venv set up before the dependency
        if not _warned_unavailable:
            _warned_unavailable = True
            logger.warning(
                "Cannot check microphone permission (%s); after a rebuild the "
                "prompt will appear at the first recording. Run scripts/setup.sh.", exc,
            )
        return UNKNOWN
    return _STATUS_NAMES.get(int(status), UNKNOWN)


def request_microphone_access(on_result: Callable[[bool], None]) -> bool:
    """Show the system prompt now. ``on_result(granted)`` runs on an arbitrary
    thread once the user answers. Returns False if the request could not be made."""
    try:
        import AVFoundation as av

        av.AVCaptureDevice.requestAccessForMediaType_completionHandler_(
            av.AVMediaTypeAudio, lambda granted: on_result(bool(granted))
        )
    except Exception:  # pragma: no cover
        logger.debug("Could not request microphone access", exc_info=True)
        return False
    return True
