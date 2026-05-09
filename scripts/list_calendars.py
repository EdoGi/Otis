"""List every calendar visible to a cached OAuth token.

Run this after ``setup_google_cal.sh`` has succeeded — it prints the calendar
IDs you can drop into ``calendar_ids`` in your config.

Usage:
    python scripts/list_calendars.py
    python scripts/list_calendars.py --token ~/.otis/google_token_work.json
    python scripts/list_calendars.py --label work
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detection.calendar_poller import (
    GoogleCalendarPoller,
    _default_token_path_for_label,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="List Google calendars accessible to a token.")
    parser.add_argument(
        "--credentials",
        default="~/.otis/credentials.json",
        help="Path to the OAuth client (credentials.json).",
    )
    g = parser.add_mutually_exclusive_group()
    g.add_argument(
        "--token",
        default=None,
        help="Path to a specific cached token JSON (e.g. ~/.otis/google_token_work.json).",
    )
    g.add_argument(
        "--label",
        default=None,
        help="Account label (e.g. 'personal', 'work'). Resolves to ~/.otis/google_token_<label>.json.",
    )
    args = parser.parse_args()

    if args.token is not None:
        token_path = args.token
    elif args.label is not None:
        token_path = _default_token_path_for_label(args.label)
    else:
        token_path = "~/.otis/google_token.json"

    poller = GoogleCalendarPoller(
        credentials_path=args.credentials,
        token_path=token_path,
    )
    creds = poller.authenticate(headless=False)

    from googleapiclient.discovery import build

    service = build("calendar", "v3", credentials=creds, cache_discovery=False)
    response = service.calendarList().list().execute()
    items = response.get("items", [])
    if not items:
        print("(no calendars found — is the calendar.readonly scope granted?)")
        return 1

    print(f"{len(items)} calendar(s) accessible to token {token_path}:\n")
    for cal in items:
        primary = " [PRIMARY]" if cal.get("primary") else ""
        access = cal.get("accessRole", "?")
        cid = cal.get("id", "?")
        summary = cal.get("summary", "(no name)")
        print(f"  • {summary!r}{primary}")
        print(f"      id:     {cid}")
        print(f"      access: {access}")
        print()

    print("To watch on this account, edit config/default_config.yaml and add IDs under")
    print("detection.calendar.accounts[*].calendar_ids.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
