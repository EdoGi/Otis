#!/usr/bin/env bash
# run.sh — activate the project venv and start the menu-bar app.
# Used as the Login Item in System Settings → General → Login Items.
#
# Usage:
#     ./scripts/run.sh                # menu bar (default)
#     ./scripts/run.sh run             # headless daemon
#     ./scripts/run.sh check-audio     # one-shot audio diagnostic

set -euo pipefail

# Otis.app runs this script with stdout and stderr sent to /dev/null, so a
# launch that dies before Python logging starts — no venv, an import error, a
# native crash — used to leave nothing anywhere. Off a terminal, keep that
# output in ~/.otis/launch.log (rotated at 1 MB); on a terminal, leave it be.
if [[ ! -t 2 ]]; then
    LAUNCH_LOG="${HOME}/.otis/launch.log"
    mkdir -p "${HOME}/.otis"
    if [[ -f "${LAUNCH_LOG}" ]] && (( $(stat -f %z "${LAUNCH_LOG}") > 1000000 )); then
        mv -f "${LAUNCH_LOG}" "${LAUNCH_LOG}.1"
    fi
    exec >>"${LAUNCH_LOG}" 2>&1
    echo "------ Otis launching $(date '+%Y-%m-%d %H:%M:%S') pid=$$ args: $* ------"
fi

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ ! -d ".venv" ]]; then
    echo "No .venv in ${PROJECT_ROOT}. Run scripts/setup.sh first." >&2
    exit 1
fi

# shellcheck source=/dev/null
source .venv/bin/activate

exec python -m src.main "$@"
