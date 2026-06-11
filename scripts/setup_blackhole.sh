#!/usr/bin/env bash
# setup_blackhole.sh - install BlackHole 2ch and walk through multi-output setup.
#
# This script handles the Homebrew install step automatically; the multi-output
# device must be created manually in Audio MIDI Setup (no public API for it).

set -euo pipefail

BLACKHOLE_FORMULA="blackhole-2ch"
BLACKHOLE_DEVICE_NAME="BlackHole 2ch"

color() {
    local code="$1"; shift
    printf '\033[%sm%s\033[0m\n' "$code" "$*"
}
info()   { color "1;34" "==> $*"; }
ok()     { color "1;32" "✓  $*"; }
warn()   { color "1;33" "!  $*"; }
fatal()  { color "1;31" "✗  $*"; exit 1; }

# ---------------------------------------------------------------------- macOS?
if [[ "$(uname -s)" != "Darwin" ]]; then
    fatal "Otis is macOS-only; BlackHole has no Linux/Windows equivalent."
fi

# -------------------------------------------------------------------- Homebrew
if ! command -v brew >/dev/null 2>&1; then
    fatal "Homebrew not found. Install it from https://brew.sh and re-run."
fi

# ------------------------------------------------------------ Install BlackHole
already_installed=false
if brew list --cask 2>/dev/null | grep -q "^${BLACKHOLE_FORMULA}\$"; then
    already_installed=true
fi

if $already_installed; then
    ok "BlackHole already installed (brew cask: ${BLACKHOLE_FORMULA})."
else
    info "Installing ${BLACKHOLE_FORMULA} via Homebrew (you may be prompted for your password)..."
    brew install --cask "${BLACKHOLE_FORMULA}"
    ok "BlackHole installed."
fi

# ------------------------------------------------------ Verify driver presence
info "Verifying BlackHole appears in CoreAudio device list..."
if system_profiler SPAudioDataType 2>/dev/null | grep -qi "blackhole"; then
    ok "BlackHole detected by CoreAudio."
else
    warn "BlackHole was installed but is not yet visible to CoreAudio."
    warn "You may need to log out/in (or reboot) before continuing."
fi

# --------------------------------------------------------- Manual multi-output
cat <<'EOF'

------------------------------------------------------------------------------
NEXT STEP — create a Multi-Output Device (manual, ~30 seconds):
------------------------------------------------------------------------------
  1. Open 'Audio MIDI Setup' (Applications > Utilities, or Spotlight: ⌘+Space).
  2. Click the '+' button bottom-left → 'Create Multi-Output Device'.
  3. In the right panel, tick BOTH:
        ✓ Your real output (e.g. 'MacBook Pro Speakers' or your headphones)
        ✓ BlackHole 2ch
  4. Set your real output as the 'Master Device' (clock source).
  5. (Optional) Rename it to "Otis Output" for clarity.
  6. When you want Otis to record system audio, set this multi-output
     device as your Mac's audio output (System Settings > Sound > Output, or
     option-click the volume in the menu bar).

After creating the device, you can verify with:
    .venv/bin/python -c "from src.audio.blackhole_check import verify_blackhole_setup; \
                print(verify_blackhole_setup())"
------------------------------------------------------------------------------
EOF

# ------------------------------------------------------------ List audio devices
info "Current audio devices:"
# Prefer the project venv — sounddevice is installed there, not system-wide.
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
[[ -x "${PYTHON}" ]] || PYTHON="python3"
if command -v "${PYTHON}" >/dev/null 2>&1; then
    "${PYTHON}" - <<'PY' || warn "Could not list devices via sounddevice (is it installed?)."
try:
    import sounddevice as sd
    for i, d in enumerate(sd.query_devices()):
        flag = []
        if d["max_input_channels"]:  flag.append("in")
        if d["max_output_channels"]: flag.append("out")
        print(f"  [{i:>2}] {d['name']!r}  ({'/'.join(flag) or '?'})")
except Exception as exc:
    print(f"  (sounddevice unavailable: {exc})")
PY
else
    warn "python3 not found; skipping device listing."
fi

ok "BlackHole setup script finished."
