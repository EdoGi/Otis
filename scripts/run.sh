#!/usr/bin/env bash
# run.sh — activate the project venv and start the menu-bar app.
# Used as the Login Item in System Settings → General → Login Items.
#
# Usage:
#     ./scripts/run.sh                # menu bar (default)
#     ./scripts/run.sh run             # headless daemon
#     ./scripts/run.sh check-audio     # one-shot audio diagnostic

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ ! -d ".venv" ]]; then
    echo "No .venv in ${PROJECT_ROOT}. Run scripts/setup.sh first." >&2
    exit 1
fi

# shellcheck source=/dev/null
source .venv/bin/activate

exec python -m src.main "$@"
