# Otis rework — handoff

Branch `rework/p0-p5-audit-followup`, 21 commits, nothing merged to `main`.
State at handoff: **621 tests pass, 22/22 stress checks pass, working tree clean,
repo lint at the session baseline of 103** (all pre-existing).

Validate with:

    .venv/bin/python -m pytest -q && python scripts/stress_test.py

The plan being executed is `~/.claude/plans/quizzical-nibbling-toast.md`
(findings + P0–P5). This document records only what the plan does **not** say:
what was verified against real hardware, what turned out to be wrong, and the
traps waiting in the phases that are still open.

---

## Status

| Phase | State |
|---|---|
| **P0** — stop losing data | **Complete** (8/8), verified live |
| **P1** — trustworthy detection | **Complete** (8/8), verified live |
| **P2** — recording robustness | items 17, 18, 19, 20, 21 done · **item 22 open** |
| **P3** — footprint (FLAC) | not started |
| **P4** — performance | not started |
| **P5** — UI / config / code health | not started |

A verification pass (two adversarial review agents + live checks) ran after
P0/P1 and found **11 defects**, five of which were regressions introduced by the
P0/P1 work itself. All 11 are fixed, each with a permanent regression test in
`tests/test_verification_regressions.py`. Do not assume "the plan said it was
done" means it works — that assumption is exactly what the verification pass
caught, repeatedly.

---

## Verified on the user's real machine

Not inferred from tests. These were measured against the live install.

**System audio — the audit's headline bug — is fixed.**

    2026-08-07  system.wav  peak=0.0000   (broken: 2 months of silence)
    2026-08-21  system.wav  peak=0.2043   SIGNAL

The transcript from that session carries `system_audio: ok` and 35
`Participant:` lines against 31 `Me:` — the first two-sided transcript since
June. The user's Mac output is now set to the `Otis` Multi-Output device.

**Calendar is reconnected**, both accounts, fetching real events. The Status
panel reads:

    ✓ System audio — routed via Otis
    ✓ Calendar (personal) — 1 calendar(s)
    ✓ Calendar (work) — 1 calendar(s)

**Idle detection cost**, 200 real poll cycles with the user's config:

    wall median   0.80 ms    (audit baseline 24.8 ms)
    at 5s cadence 0.016 %    (audit baseline 0.50 %)
    per hour      0.58 s CPU (audit baseline 17.9 s)
    false detections: []

---

## Things the plan got wrong — measured, don't re-derive

**1. Anchor jitter is not writer-thread scheduling.** The plan blamed the
5–99 ms mic-vs-system skew on stamping the anchor in the writer thread. Measured
all three candidate sources side by side, six runs:

    writer thread        spread 173.28 ms, stdev 59.09
    callback monotonic   spread 173.21 ms, stdev 59.12
    callback ADC         spread 173.05 ms, stdev 59.02

The hand-off contributes under a millisecond. The spread is the two devices
genuinely starting at different times, which an accurate anchor should *record*,
not erase. Item 18 still landed (hardware capture time is the right source) but
it buys ~1 ms, not an order of magnitude.

**2. A startup barrier makes things worse.** The obvious analogue of item 17's
teardown barrier — starting both streams back to back — is a trap:

    start() 50 ms apart     skew spread   6.89 ms
    start() back to back    skew spread 504.97 ms

The devices contend during CoreAudio spin-up. Streams are opened sequentially
**on purpose**. Don't "fix" this.

**3. NSWorkspace is stale without a main run loop.** `runningApplications` is
only refreshed while the run loop runs. Verified by launching TextEdit from a
background thread with no `NSApplication`: it never appeared, and never
disappeared on quit. Pumping `CFRunLoopRunInMode` from that thread does not
help. `NSRunningApplication.runningApplicationsWithBundleIdentifier_` **is**
live, if a per-entry query is ever wanted.

So the fast source is **opt-in**: `main.py` passes
`process_iter=nsworkspace_process_iter` because rumps drives a real run loop;
`otis run` keeps psutil, because it is headless and would otherwise go blind to
any app launched after startup. Measured: 0.70 ms vs 13.40 ms per poll.

**4. Echo-dedup timing evidence is misleading.** On the real transcript the mic
"leads" on 21 of 25 duplicated lines, which looks like the system copy is the
echo. It is not — that is the alignment offset pushing the system stream later.
The structural argument is what holds: BlackHole records system *output*, which
cannot contain your microphone, so a duplicate always means the mic heard the
speakers. Dedup runs mic-against-system and keeps the system copy.

---

## P3 — read this before starting (data migration)

**Hard ordering: the FLAC decoder must land before the encoder.**

`WhisperEngine._wav_to_float32_mono16k` fast-paths *only* 16 kHz / mono /
16-bit WAV via `wave.open`, and otherwise falls back to
`mlx_whisper.audio.load_audio`, which shells out to **ffmpeg — which is not
installed on this machine** (`which ffmpeg` → nothing). `soundfile` is not
installed either. Encoding to FLAC before adding a decode path would silently
break `retranscribe.py` and orphan recovery.

`/usr/bin/afconvert` is present and does FLAC natively via AudioToolbox — no new
dependency.

**Measured FLAC ratios on the user's actual files** (not estimates):

    mic (speech)      1.73x - 2.20x
    system (silence)  770x
    total             1073 MB -> ~271 MB  (-75%)

Note the plan's original "~110 MB (−90%)" figure was optimistic and has been
corrected in the plan document. Now that system audio works, both streams carry
speech, so the forward-looking steady state is ~115 MB per meeting-hour.

**Back up first.** `cp -R ~/Otis ~/Otis.backup` before the one-shot migration,
and land the migration in its own commit on a clean tree.

---

## Open defects from the verification pass

Real, confirmed, not yet fixed. Roughly in severity order. Each names where it
naturally belongs.

- **`duration_seconds` is wrong when there are no segments** (`processor.py`,
  ~line 274). Duration comes from the last segment, not the recording length, so
  a silent meeting with 30 minutes of pauses is filed as zero seconds. Belongs
  with the P0 pause work; fix as
  `max(segment_extent, pause_timeline.to_wall(audio_duration_from_mic_bytes))`.
- **The idle timer stalls transcription** (`whisper_engine.py` ~439).
  `_release_mlx_model_cache()` runs *inside* `_lock` — it does a cold
  `importlib` and a Metal call, measured at 1.0 s of blocking against a running
  job and everything queued behind `_run_lock`. Move the release outside the
  lock, guarded by a dedicated release lock. **P4.**
- **`shutdown()` does not shut down** (`whisper_engine.py` ~316). An in-flight
  transcription's `finally` calls `_reset_idle_timer()` afterwards and re-arms a
  300 s timer that fires after the app claimed to stop. Needs a `_shutdown` flag
  checked in `_reset_idle_timer`. **P4.**
- **Poller keyspace still uses `ev.id`** (`calendar_poller.py` ~436, ~550).
  The detector was fixed to compare `canonical_key`, but `fetch_today_events`
  dedupes by `id` and `_alerted_upcoming` / `_alerted_ended` key by `id`. Not
  user-visible today only because `detector._alerted_uids` absorbs it — but
  `upcoming_within()` (used by `scripts/check_phase2.py`) still returns
  duplicates. **P1 leftover, cheap.**
- **`_state_entered_at` is bypassed on calendar paths** (`detector.py` ~298,
  ~303, ~394, ~239). Those assign `self._context` directly instead of going
  through `_transition()`, leaving a stale timestamp. Latent today because
  `APPROACHING_MAX_SECONDS` is only read when `end_time is None`, which
  `parse_event` guarantees never happens — i.e. that branch is dead in
  production. Either route every state change through one choke point or delete
  the dead branch.
- **Titling still leaks proper nouns as topics** (`titling.py`). Names are no
  longer guessed as *participants*, but a frequent capitalised first name can
  still become a *topic* (`Call about marie & budget`). `suggest_title` only
  runs for ad-hoc recordings, which by construction have `participants == []`,
  so the participant branch is effectively dead in production. Filter
  capitalised-mid-sentence tokens out of `topic_counts` too.
- **Health callbacks are delivered out of order** under contention
  (`health.py` ~154). The transition is computed under the lock and delivered
  outside it with no ordering, so a slow notifier can deliver a stale FAIL after
  a newer OK. ~13/200 trials. Needs a `_notify_lock` around the fan-out.
- **Nested aggregates aren't expanded** (`blackhole_check.py` ~205). A
  Multi-Output containing another aggregate that contains BlackHole reports
  NOT_ROUTED, with a "fix" pointing at a device already downstream. Recurse with
  a `visited` set and a depth cap.
- **`_suggest_aggregate` runs on the rumps main thread** and makes ~135
  synchronous CoreAudio round-trips (30-device system). A wedged Bluetooth
  driver freezes the menu bar. Memoise name/uid, or move it off the main thread.
- **`RoutingState.UNKNOWN` is invisible in the UI.** It is correctly not
  *claimed* as healthy, but `is_degraded()` excludes it, no notification branch
  handles it, and "Fix System Audio…" stays hidden — so a CoreAudio probe
  failure is silently swallowed.
- **`TRANSCRIPTION` and `DISK` health rows are never reported** by anything in
  `src/`, so two of five panel rows can never render. Either report them or drop
  them from `CANONICAL_ORDER`.
- **`report_health=False` is dead code** — no caller in `src/` or `scripts/`
  passes it, and the test justifying it claims `retranscribe.py` uses it, which
  it does not.
- **RECORDING has no max-duration ceiling.** Close the lid mid-meeting and the
  daemon records until the disk fills. Needs recorder coordination; belongs with
  the remaining P2 work.

---

## Testing lessons worth keeping

The verification pass found a consistent failure mode, and it recurred **inside
this session** even after being named:

> A test asserts the *shape* of the code rather than its *behaviour*, and keeps
> passing when the behaviour breaks.

Concrete examples from this branch:

- `test_authorized_http_carries_a_timeout` checked that `_authorized_http`
  returned a transport with the right timeout — and never checked that a client
  could be built from it. The HTTP-timeout commit passed both `http` and
  `credentials` to `googleapiclient.build()`, which are mutually exclusive, so
  **every** calendar service build failed from P1 until the user's log caught it
  four commits later. Fixed in `164ef62`; the new test mirrors googleapiclient's
  own guard.
- The daemon and menu-bar fixtures both hard-disabled `defer_while_in_call`, so
  every other test in those files exercised the pre-`b321c0d` path.
- `test_back_to_back_meeting_correlates_after_transcription` passed only because
  the fixture defaulted `ical_uid=None`, making the suite structurally blind to
  the id-vs-canonical_key bug.

Prefer a test that drives the real call path, even with fakes, over one that
inspects an intermediate object.

Also: **`~/.otis/config.yaml` still contains a stale `mic_activation` block.**
It is correctly ignored. There is no config schema validation, so unknown keys
are silently accepted — that is P5 item 39.
