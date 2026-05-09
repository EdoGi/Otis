#!/usr/bin/env bash
# setup.sh — full bootstrap for a fresh clone of Otis.
#
# What it does:
#   1. Creates ~/.otis/ (mode 700) for credentials / tokens.
#   2. Creates a Python 3.12 venv at .venv/.
#   3. Installs the project (editable) and all runtime + dev deps.
#   4. Pre-generates the menu-bar icon PNGs into ~/.otis/icons/.
#   5. Walks the user through BlackHole installation.
#   6. Walks the user through Google Calendar OAuth.
#
# Re-running it is idempotent — existing venv / icons / tokens are preserved.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"

color() {
    local code="$1"; shift
    printf '\033[%sm%s\033[0m\n' "$code" "$*"
}
info()  { color "1;34" "==> $*"; }
ok()    { color "1;32" "✓  $*"; }
warn()  { color "1;33" "!  $*"; }
fatal() { color "1;31" "✗  $*"; exit 1; }

# ---------------------------------------------------------------------- macOS
if [[ "$(uname -s)" != "Darwin" ]]; then
    fatal "Otis is macOS-only."
fi

# ---------------------------------------------------------------------- Python
PYTHON="${PYTHON:-/opt/homebrew/bin/python3.12}"
if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="$(command -v python3.12 || true)"
fi
if [[ -z "${PYTHON}" || ! -x "${PYTHON}" ]]; then
    fatal "Python 3.12 not found. Install with 'brew install python@3.12' and retry."
fi
info "Using ${PYTHON} ($(${PYTHON} --version))"

# ---------------------------------------------------------------- ~/.otis
mkdir -p "${HOME}/.otis"
chmod 700 "${HOME}/.otis"
ok "${HOME}/.otis ready."

# ---------------------------------------------------------------- venv
if [[ -d ".venv" ]]; then
    info ".venv already exists; reusing."
else
    info "Creating .venv..."
    "${PYTHON}" -m venv .venv
    ok ".venv created."
fi
# shellcheck source=/dev/null
source .venv/bin/activate
python -m pip install --quiet --upgrade pip

info "Installing Otis in editable mode (this can take a minute on first run)..."
pip install --quiet -e ".[dev]"
ok "Dependencies installed."

# -------------------------------------------------------------- icon priming
info "Generating menu-bar icons under ~/.otis/icons/ ..."
python - <<'PY'
from pathlib import Path
from src.ui.icons import ensure_icons
icons = ensure_icons(Path("~/.otis/icons").expanduser())
print(f"  {len(icons)} icons:", ", ".join(sorted(icons.keys())))
PY
ok "Icons ready."

# -------------------------------------------------------------- BlackHole
info "BlackHole virtual audio driver — required to capture system audio."
if system_profiler SPAudioDataType 2>/dev/null | grep -qi "blackhole"; then
    ok "BlackHole already detected."
else
    warn "BlackHole not detected. Running scripts/setup_blackhole.sh ..."
    "${PROJECT_ROOT}/scripts/setup_blackhole.sh" || warn "BlackHole setup needs your attention; re-run when ready."
fi

# -------------------------------------------------------------- Google Cal
info "Google Calendar — optional but powers the 2-min upcoming alerts."
if [[ -f "${HOME}/.otis/credentials.json" ]]; then
    ok "credentials.json found at ~/.otis/credentials.json — skipping."
    if [[ -f "${HOME}/.otis/google_token.json" ]]; then
        ok "Personal account already authenticated."
    else
        warn "Run ./scripts/setup_google_cal.sh whenever you're ready to authenticate."
    fi
else
    warn "No credentials.json yet. Run ./scripts/setup_google_cal.sh to get started."
fi

# -------------------------------------------------------------- final
echo
ok "Setup complete."
echo
info "Try it:"
echo "  ./scripts/run.sh                       # menu bar"
echo "  ./scripts/run.sh check-audio           # verify BlackHole"
echo "  ./scripts/run.sh run                   # headless daemon"
echo
info "To launch automatically at login:"
echo "  System Settings → General → Login Items → '+' → choose:"
echo "    ${PROJECT_ROOT}/scripts/run.sh"
