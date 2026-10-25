"""Keep the menu-bar tests independent of this Mac's audio routing.

Starting a recording now re-checks the output route (src/ui/menubar.py
_start_recorder_worker), and launch has always checked it. Unstubbed, both
query real CoreAudio, so what a test saw — and which toasts it counted —
depended on where the developer's Mac happened to be playing. Tests that care
about routing patch it explicitly (test_modal_freeze) or inject
``routing_check`` (test_routing_rechecks).
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _routing_is_unknown(monkeypatch):
    import src.audio.blackhole_check as bc

    monkeypatch.setattr(
        bc, "check_output_routing",
        lambda probe=None: bc.RoutingStatus(state=bc.RoutingState.UNKNOWN,
                                            error="not checked under test"),
    )
