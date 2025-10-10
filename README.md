# Codex → WakaTime Sync

Drop-in script for converting OpenAI Codex/Claude Code shell sessions into WakaTime heartbeats labeled as AI coding.

## Install

```bash
mkdir -p ~/.local/share/codex-wakatime
curl -L https://raw.githubusercontent.com/latand/codex-wakatime/main/codex_wakatime_sync.py \
  -o ~/.local/share/codex-wakatime/codex_wakatime_sync.py
chmod +x ~/.local/share/codex-wakatime/codex_wakatime_sync.py
```

(Replace the `curl` URL with your own fork while iterating locally.)

## Requirements

- Python 3.10+
- `wakatime-cli` installed and configured with your API key in `~/.wakatime.cfg`
- Codex or Claude logs at `~/.codex/sessions/` (default Codex CLI location)

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
  python ~/.local/share/codex-wakatime/codex_wakatime_sync.py sync --since 45m \
  >> $HOME/.codex-wakatime/cron.log 2>&1
```

To flush any queued heartbeats after each run add `wakatime-cli --sync-offline-activity 0`.

## Notes

- Heartbeats are tagged as `ai coding` so they surface under WakaTime's AI category.
- Deduplication happens via `~/.codex-wakatime/state.db`; delete rows to backfill past sessions.

MIT License
