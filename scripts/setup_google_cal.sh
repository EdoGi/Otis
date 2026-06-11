#!/usr/bin/env bash
# setup_google_cal.sh — guide the user through provisioning Google Calendar API
# credentials for Otis. With an optional ``label`` argument, the OAuth flow
# is run for an additional Google account (e.g. a work Workspace account):
#
#   ./scripts/setup_google_cal.sh             # personal account (default)
#   ./scripts/setup_google_cal.sh work        # second account, token saved as
#                                             #   ~/.otis/google_token_work.json
#
# The same credentials.json (OAuth client) is shared across all accounts —
# only the cached token differs.

set -euo pipefail

LABEL="${1:-personal}"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

CRED_DIR="${HOME}/.otis"
CRED_FILE="${CRED_DIR}/credentials.json"
if [[ "${LABEL}" == "personal" || "${LABEL}" == "default" || "${LABEL}" == "primary" ]]; then
    TOKEN_FILE="${CRED_DIR}/google_token.json"
else
    TOKEN_FILE="${CRED_DIR}/google_token_${LABEL}.json"
fi

color() {
    local code="$1"; shift
    printf '\033[%sm%s\033[0m\n' "$code" "$*"
}
info()  { color "1;34" "==> $*"; }
ok()    { color "1;32" "✓  $*"; }
warn()  { color "1;33" "!  $*"; }

mkdir -p "${CRED_DIR}"
chmod 700 "${CRED_DIR}"

cat <<EOF

Otis — Google Calendar credentials setup
=============================================

Account label:  ${LABEL}
Token will be saved to:  ${TOKEN_FILE}

Otis polls your Google Calendar to detect upcoming meetings. To do that
it needs an OAuth client id from a Google Cloud project YOU control.

If this is your FIRST account, follow steps 1–5 below once (~5 minutes).
If you've done this for another account already, you can REUSE the same
credentials.json — just skip to step 6.

  1. Open https://console.cloud.google.com/projectcreate
     • Create a new project (any name, e.g. "Otis").

  2. Enable the Google Calendar API:
     https://console.cloud.google.com/apis/library/calendar-json.googleapis.com
     • Click 'Enable'.

  3. Configure the OAuth consent screen:
     https://console.cloud.google.com/apis/credentials/consent
     • User type: 'External' → Create.
     • App name: Otis
     • Support / dev email: your Gmail.
     • Scopes: add '.../auth/calendar.readonly'.
     • Test users: add ALL the Google accounts you intend to use
       (personal + work + …). Each one needs to be listed here or
       OAuth will refuse them.
     • Save.

  4. Create OAuth client credentials:
     https://console.cloud.google.com/apis/credentials
     • '+ Create Credentials' → 'OAuth client ID'
     • Application type: 'Desktop app'
     • Name: 'Otis CLI'
     • Click Create, then 'Download JSON'.

  5. Save the downloaded file to:
       ${CRED_FILE}
     (One credentials.json works for every account you set up — you only do
     steps 1–5 once.)

  6. The browser will open. Pick the Google account matching label
     '${LABEL}' (NOT another account, or you'll have to delete the token
     and start over). Grant access. The token is cached at:
       ${TOKEN_FILE}

EOF

if [[ -f "${CRED_FILE}" ]]; then
    ok "Found existing credentials at ${CRED_FILE}"
else
    warn "No credentials file at ${CRED_FILE} yet. Drop the downloaded JSON there, then re-run this script."
    exit 0
fi

# ---------------------------------------------------------------- auth test
echo
info "Running an interactive OAuth test (a browser tab will open)."
info "Sign in with the Google account you want to label '${LABEL}'."

# The Google API libraries live in the project venv — bare python3 would
# fail with ModuleNotFoundError on a fresh machine.
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
    warn "No venv at ${PROJECT_ROOT}/.venv — run scripts/setup.sh first. Falling back to python3."
    PYTHON="python3"
fi

# `if <command>` keeps set -e from aborting on failure, so the hints below
# actually print when the auth test fails.
if PYTHONPATH="${PROJECT_ROOT}" "${PYTHON}" - <<PY
from src.detection.calendar_poller import GoogleCalendarPoller, CalendarAuthError
import sys

try:
    poller = GoogleCalendarPoller(
        credentials_path="${CRED_FILE}",
        token_path="${TOKEN_FILE}",
    )
    poller.authenticate(headless=False)
    events = poller.fetch_today_events()
except CalendarAuthError as e:
    print(f"Auth error: {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Calendar fetch failed: {e}", file=sys.stderr)
    sys.exit(2)

print(f"OK — fetched {len(events)} event(s) for today on account '${LABEL}'.")
for ev in events[:5]:
    line = f"  • {ev.start.strftime('%H:%M')}  {ev.title!r}"
    if ev.meeting_link:
        line += f"  → {ev.meeting_link}"
    print(line)
PY
then
    ok "Account '${LABEL}' is authenticated."
    ok "Token cached at ${TOKEN_FILE} (chmod 600)."
    echo
    info "Next steps:"
    echo "  • To add another account, run:  ./scripts/setup_google_cal.sh <label>"
    echo "  • To list calendars on this account:"
    echo "      python scripts/list_calendars.py --token ${TOKEN_FILE}"
    echo "  • To enable in Otis, edit config/default_config.yaml and add an entry"
    echo "    under detection.calendar.accounts with label: '${LABEL}'."
else
    warn "Auth test failed. Check that:"
    warn "  • the OAuth client is type 'Desktop app'"
    warn "  • the Google account you chose is added as a Test User"
    warn "  • the calendar.readonly scope is enabled"
fi
