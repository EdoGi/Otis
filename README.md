# Otis

**Local macOS meeting transcriber.** Records mic + system audio, transcribes
with Whisper on-device, exposes transcripts to Claude via MCP. Nothing leaves
your Mac.

[![status](https://img.shields.io/badge/phase-6%2F6-blue)](#phases) [![tests](https://img.shields.io/badge/tests-315%2B%20passing-brightgreen)](#tests) [![python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org) [![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> Status: **Phase 6 of 6** — all phases shipped: recording, detection, transcription, storage, web UI, and the MCP server for Claude.

## Why

Meeting transcribers are everywhere, and they all upload your audio somewhere.
Otis doesn't. The recording, transcription, storage, and search index all live
on your Mac. The MCP server lets Claude search and summarise your meetings
without sending the contents to anyone.

## Phases

1. **Project scaffold + audio engine** ✅
2. **Meeting detection (Google Calendar + process monitoring)** ✅
3. **Menu-bar UI + notifications** ✅
4. **Transcription pipeline (mlx-whisper, post-meeting batch)** ✅
5. **Storage (markdown + YAML frontmatter, retention policies)** ✅
6. **Web UI + MCP server** ✅

## Requirements

- macOS 13+ on Apple Silicon (mlx-whisper, Phase 4)
- Python 3.12 (3.10+ should work — `setup.sh` defaults to `python3.12`)
- [BlackHole 2ch](https://existential.audio/blackhole/) virtual audio driver
- Homebrew

## Install — one command

```bash
git clone https://github.com/EdoGi/Otis.git
cd Otis
./scripts/setup.sh
```

`setup.sh` creates the venv, installs dependencies, generates the menu-bar
icons, and walks you through BlackHole + Google Calendar OAuth. It is
idempotent — re-run it any time to verify your environment.

## Run

### Daily use — double-click an `.app` (recommended)

After running `./scripts/setup.sh` once, build the bundle:

```bash
./scripts/build_app.sh
mv dist/Otis.app /Applications/
```

**First launch only — bypass Gatekeeper.** Otis is locally built (no Apple
Developer ID), so macOS blocks `open` by default. You need to right-click the
.app once:

1. Open `/Applications/` in Finder.
2. **Right-click** (or Ctrl-click) on `Otis.app` → **Open**.
3. Click **Open** in the "unidentified developer" dialog.

After that, double-click and Spotlight work normally.

Now Otis is a real first-class macOS app:

- Open it from **Spotlight** (`⌘+Space` → "Otis" ⏎) or **Launchpad**.
- The Otis face shows up in the About dialog, force-quit window, and Notification Center.
- Auto-launch at login: **System Settings → General → Login Items → `+` → /Applications/Otis.app**.

Re-run `./scripts/build_app.sh` if you move the project folder. The `.app`
itself is per-machine — not committed to the repo.

### From the terminal

```bash
./scripts/run.sh                # menu-bar app (default)
./scripts/run.sh check-audio    # one-shot: BlackHole + audio device list
./scripts/run.sh run            # headless daemon: auto-records AND transcribes
```

The headless daemon (`otis run`) uses the same pipeline as the menu bar: it
reads your `~/.otis/config.yaml` overrides, honours working days/hours, and
transcribes each recording when the meeting ends.

In the menu bar you'll get an Otis mic icon. Click it for the menu:

```
Otis
─────
Start Recording          (visible in IDLE / APPROACHING / DETECTED)
Pause Recording          (visible while RECORDING)
Resume Recording         (visible while PAUSED)
Stop & Transcribe        (visible while RECORDING / PAUSED)
─────
Language: Auto-detect    (Auto / English / French / Italian / Portuguese / Spanish / German)
─────
Open Transcripts         → http://127.0.0.1:8765 (local web UI)
Open Transcripts Folder  → ~/Otis/transcripts in Finder
─────
Settings                 (Whisper model · Working days · Working hours · App whitelist)
─────
Generate Transcript      (retry orphaned / failed recordings)
Recent Transcripts       (last 5 — clickable)
─────
About Otis · Quit
```

While a transcription runs, the menu-bar title shows live progress (`47%`)
next to the blue PROCESSING icon.

### Icon states

| State | Icon | When |
|---|---|---|
| Idle | gray mic | nothing happening |
| Approaching | orange mic | calendar event in <2 min |
| Detected | blinking orange/gray mic | a meeting app started |
| Recording | red dot | actively capturing |
| Paused | yellow `\|\|` | recording paused |
| Processing | blue ring | transcription in progress |
| Off-hours | gray crescent moon | outside `working_hours` |

### Notifications

macOS Notification Center for: `meeting_approaching`, `meeting_detected`,
`recording_started`, `recording_paused`, `process_disappeared`,
`transcription_complete`, `error`. Rate-limited to one notification per type
per 30 s.

## Web UI

`otis ui` automatically serves a local transcript browser at
**http://127.0.0.1:8765** (configurable via `web.host` / `web.port`):
list + filter, full-text search with snippets, and a per-meeting reader.
"Open Transcripts" in the menu bar takes you straight there. No JavaScript,
nothing leaves the machine; if the port is busy Otis keeps running and tells
you via a notification.

Run it standalone (without the menu bar):

```bash
.venv/bin/python -m src.web.server
```

## MCP server (Claude integration)

The MCP server gives Claude read-only access to your transcripts — search,
list, and fetch — over **stdio**, launched by the client itself (nothing to
start or keep running):

```bash
# Claude Code:
claude mcp add otis -- /path/to/Otis/.venv/bin/python -m src.mcp.server
```

Or in Claude Desktop's `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "otis": {
      "command": "/path/to/Otis/.venv/bin/python",
      "args": ["-m", "src.mcp.server"]
    }
  }
}
```

Three read-only tools are exposed:

| Tool | What it does |
|---|---|
| `list_transcripts` | Browse metadata, filtered by date range / participant / tag / language / title. |
| `search_transcripts` | Full-text search across bodies; returns snippets per hit. |
| `get_transcript` | Fetch one meeting in full (frontmatter + Markdown body) by id. |

## Auto-launch at login

After running `./scripts/build_app.sh`:

System Settings → General → **Login Items** → `+` → `/Applications/Otis.app`.

(If you prefer not to build the bundle, you can also point Login Items at
`~/Documents/Otis/scripts/run.sh` directly — works the same, just lacks the
icon and Spotlight entry.)

## Permissions

On first run macOS will prompt for:

1. **Microphone access** — required. Grant via *System Settings → Privacy & Security → Microphone*.
2. **Notifications** — required for menu-bar pop-ups (*Privacy & Security → Notifications*).

If denied, the recorder raises `PermissionDeniedError` with instructions.

## Configuration

Three layers, deep-merged:

1. `config/default_config.yaml` — bundled defaults.
2. `~/.otis/config.yaml` — your overrides; written automatically when you
   toggle Settings in the menu bar (Whisper model, working days, app whitelist).
3. `--config /path/to/your.yaml` — explicit override on the command line.

The most relevant knobs:

| Key | Default | Notes |
|---|---|---|
| `audio.sample_rate` | `16000` | Optimal for Whisper. Don't bump it. |
| `audio.channels` | `1` | Mono per stream. |
| `audio.system_audio_device` | `BlackHole 2ch` | Substring match. |
| `app.working_days` | `[0,1,2,3,4]` | Mon–Fri (`datetime.weekday()`, 0 = Mon). |
| `app.working_hours` | `08:00 → 20:00` | Outside this, detection is paused. |
| `storage.audio_dir` | `~/Otis/audio` | WAVs + metadata land here. |
| `storage.transcript_dir` | `~/Otis/transcripts` | Phase 4. |
| `transcription.model` | `small` | `tiny` / `base` / `small` / `medium` / `large-v3`. |

Multi-account Google Calendar lives under `detection.calendar.accounts` —
see the inline comment in the YAML.

## Tests

```bash
pytest
```

315+ tests across audio, detection, transcription, storage, daemon, web, MCP,
and UI, plus an aggressive end-to-end harness (`python scripts/stress_test.py`,
22 checks). The suite uses a fake `sounddevice` module and never touches a
real CoreAudio stack, so it runs anywhere — the only exception is the
menu-bar behavioural tests, which need macOS + rumps and skip themselves
elsewhere.

## Layout

```
otis/
├── config/default_config.yaml
├── src/
│   ├── audio/         # capture engine                           (Phase 1)
│   ├── detection/     # process + calendar + state machine        (Phase 2)
│   ├── ui/            # rumps menu bar + notifications + icons    (Phase 3)
│   ├── transcription/ # mlx-whisper                               (Phase 4)
│   ├── storage/       # transcript + retention                    (Phase 5)
│   ├── web/           # Flask UI                                   (Phase 6)
│   ├── mcp/           # MCP server                                 (Phase 6)
│   ├── daemon.py      # headless `otis run` daemon
│   ├── config.py      # YAML loader
│   └── main.py        # CLI entry point
├── scripts/
│   ├── setup.sh                  # full bootstrap
│   ├── run.sh                    # daily launcher
│   ├── build_app.sh              # build the double-clickable Otis.app
│   ├── setup_blackhole.sh
│   ├── setup_google_cal.sh
│   ├── retranscribe.py           # redo old sessions (bigger model, crash recovery)
│   ├── stress_test.py            # aggressive end-to-end harness (22 checks)
│   ├── regenerate_icons.py
│   ├── list_calendars.py
│   ├── list_devices.py
│   ├── probe_mic.py
│   ├── smoke_record.py
│   ├── smoke_process_monitor.py
│   └── check_phase2.py
└── tests/
```

## Privacy

Everything is on-device:

- Audio capture runs locally via `sounddevice` + BlackHole.
- Transcription runs locally via `mlx-whisper` (Apple Silicon GPU).
- Transcripts are written as Markdown files in `~/Otis/transcripts/`.
- The web UI binds to `127.0.0.1` only; the MCP server runs over stdio
  (no network socket at all) and is read-only.
- The Google Calendar token is OAuth-2 with the **read-only** scope —
  Otis can't write to or delete your calendars.
- No telemetry. No analytics. No phone-home.

The OAuth client (`credentials.json`) and tokens you generate live in
`~/.otis/` — outside the project tree, never staged for git.

## Contributing

This is a solo / hobby project I use daily; PRs welcome but expect a slow
review cadence. Issue reports with full Mac model + macOS version + the log
output (`./scripts/run.sh --log-level DEBUG`) are most useful.

## License

[MIT](LICENSE) — see the LICENSE file.
