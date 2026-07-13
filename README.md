# Agent activity → WakaTime

A standalone Python synchronizer that converts Codex and Claude Code transcript lifecycles into privacy-safe WakaTime `ai coding` heartbeats.

Runtime requirements are Python 3.10+ and a WakaTime API key in `~/.wakatime.cfg`. Transcript roots are ordinary filesystem directories. Any tool that writes compatible Codex or Claude JSONL files can be added through configuration.

## What it measures

The synchronizer reads turn boundaries:

- Codex: `task_started` through `task_complete` or `turn_aborted`.
- Claude Code: a visible user prompt through the terminal assistant event.
- Active turns: the latest transcript event plus a bounded 15-minute grace window.

It emits a heartbeat at the turn start, every two minutes, and at the turn end. Overlapping turns form one foreground timeline; the most recently started turn owns the overlap. Identical transcripts found under several roots are deduplicated.

This matches WakaTime's heartbeat model, where nearby heartbeats are joined into durations according to the account's keystroke timeout.

## Privacy boundary

Each outbound heartbeat contains this fixed allowlist:

```text
entity, type, category, time, project, plugin, ai_session
```

`project` and `ai_session` are opaque HMAC identifiers generated with a random local key. Prompt text, responses, commands, filenames, paths, repository names, branches, native session IDs, and native turn IDs stay local. Logs contain aggregate counters. SQLite stores delivery fingerprint, timestamp, send time, and mode.

The first v1 run removes the legacy `sent_events` table, enables SQLite secure deletion, and compacts the database. This clears raw session paths stored by older releases.

Keep `sources.conf`, `privacy.key`, `state.db`, and cron logs owner-readable only. They live outside the repository.

## Install

From a checkout:

```bash
./scripts/install.sh
```

The installer places the executable files in `~/.local/share/codex-wakatime` and creates `~/.config/agent-wakatime/sources.conf` on first install.

For a remote install:

```bash
git clone https://github.com/Latand/codex-wakatime.git
cd codex-wakatime
./scripts/install.sh
```

## Configure transcript roots

Edit `~/.config/agent-wakatime/sources.conf`:

```ini
codex=~/.codex/sessions
claude=~/.claude/projects

# Add any additional filesystem roots. Globs are accepted.
codex=~/.local/share/agent-accounts/codex/*/sessions
claude=~/.local/share/agent-accounts/claude/*/projects
```

Blank lines and `#` comments are accepted. Repeated roots are collapsed after path resolution. CLI sources can be appended for one run:

```bash
~/.local/share/codex-wakatime/codex-wakatime-sync.sh 2h \
  --source 'codex:~/.another-codex/sessions' \
  --source 'claude:~/.another-claude/projects'
```

## Validate and sync

Privacy-safe preview:

```bash
~/.local/share/codex-wakatime/codex-wakatime-sync.sh 45m --dry-run
```

Live sync:

```bash
~/.local/share/codex-wakatime/codex-wakatime-sync.sh 45m
```

The API key is read directly from the `[settings]` section of `~/.wakatime.cfg`; it stays out of process arguments and logs.

## Cron

Run every 15 minutes with a lock:

```cron
*/15 * * * * PATH=$PATH:$HOME/.local/bin flock -n $HOME/.codex-wakatime/sync.lock $HOME/.local/share/codex-wakatime/codex-wakatime-sync.sh 45m --log-level WARNING >> $HOME/.codex-wakatime/cron.log 2>&1
```

The 45-minute scan window and privacy-safe receipt database make retries harmless. Failed batches remain pending for the next run.

## Development

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python -m py_compile codex_wakatime_sync.py
bash -n scripts/codex-wakatime-sync.sh scripts/install.sh
```

MIT License
