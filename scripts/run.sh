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
# native crash — used to leave nothing anywhere. When both stdout and stderr
# are /dev/null (the applet, a Login Item), keep that output in
# ~/.otis/launch.log, rotated at 1 MB. Terminals, pipes, files and a user's own
# `2>/dev/null` keep their output. Every step fails open: a logging problem
# must never be the thing that stops Otis from starting.
if [[ /dev/fd/1 -ef /dev/null && /dev/fd/2 -ef /dev/null ]]; then
    LAUNCH_LOG="${HOME}/.otis/launch.log"
    if mkdir -p "${HOME}/.otis" 2>/dev/null; then
        if [[ -f "${LAUNCH_LOG}" ]] && (( $(stat -f %z "${LAUNCH_LOG}" 2>/dev/null || echo 0) > 1000000 )); then
            mv -f "${LAUNCH_LOG}" "${LAUNCH_LOG}.1" 2>/dev/null || true
        fi
        if [[ ! -e "${LAUNCH_LOG}" || -w "${LAUNCH_LOG}" ]]; then
            exec >>"${LAUNCH_LOG}" 2>&1 || true
            echo "------ Otis launching $(date '+%Y-%m-%d %H:%M:%S') pid=$$ args: $* ------" || true
        fi
    fi
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
