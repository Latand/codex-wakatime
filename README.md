# Codex → WakaTime Sync

Drop-in script for converting OpenAI Codex Code shell sessions into WakaTime heartbeats labeled as AI coding.

## Install

```bash
mkdir -p ~/.local/share/codex-wakatime
curl -L https://raw.githubusercontent.com/latand/codex-wakatime/main/codex_wakatime_sync.py \
  -o ~/.local/share/codex-wakatime/codex_wakatime_sync.py
curl -L https://raw.githubusercontent.com/latand/codex-wakatime/main/scripts/codex-wakatime-sync.sh \
  -o ~/.local/share/codex-wakatime/codex-wakatime-sync.sh
chmod +x ~/.local/share/codex-wakatime/codex_wakatime_sync.py \
          ~/.local/share/codex-wakatime/codex-wakatime-sync.sh
```

(Replace the `curl` URLs with your own fork while iterating locally.)

## Requirements

- Python 3.10+
- `wakatime-cli` installed and configured with your API key in `~/.wakatime.cfg`
- Codex logs at `~/.codex/sessions/` (default Codex CLI location)

## Manual Sync

```bash
python ~/.local/share/codex-wakatime/codex_wakatime_sync.py \
  sync --since 2h --log-level INFO
```

Useful flags:
- `--project` to override the project name
- `--mode api --api-key <key>` to bypass the CLI
- `--dry-run` to preview heartbeats without sending

## Cron Example

```
*/15 * * * * PATH=$PATH:$HOME/.local/bin \
  $HOME/.local/share/codex-wakatime/codex-wakatime-sync.sh 45m --log-level WARNING \
  >> $HOME/.codex-wakatime/cron.log 2>&1
```

The shell wrapper runs the Python sync and immediately flushes `wakatime-cli --sync-offline-activity 0` to push queued heartbeats.

## Notes

- Command contents are never sent to WakaTime; the script stores only a SHA-256 hash and token count for troubleshooting.
- Heartbeats are tagged as `ai coding` so they surface under WakaTime's AI category.
- Deduplication happens via `~/.codex-wakatime/state.db`; delete rows to backfill past sessions.

MIT License
