#!/usr/bin/env python3
"""
Codex→WakaTime Bridge: sync Codex shell activity to WakaTime heartbeats.

Parses Codex rollout logs from sessions directory and sends coding activity
to WakaTime via CLI or API mode with deduplication and retry logic.
"""

import argparse
import base64
import dataclasses
import datetime
import hashlib
import itertools
import json
import logging
import os
import re
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================================
# Data Structures
# ============================================================================


@dataclasses.dataclass
class SessionContext:
    """Maintains session state from session_meta events."""
    cwd: Optional[str] = None
    git_branch: Optional[str] = None


@dataclasses.dataclass
class Heartbeat:
    """Normalized heartbeat ready to send to WakaTime."""
    call_id: str
    timestamp: float  # epoch seconds
    entity: str  # command truncated to 200 chars
    entity_type: str  # always 'app'
    category: str  # e.g., 'ai coding'
    project: str
    session_path: str
    metadata: Dict[str, Any]  # extra context (workdir, command, branch)


@dataclasses.dataclass
class Stats:
    """Tracks processing statistics."""
    scanned: int = 0
    eligible: int = 0
    duplicates: int = 0
    sent: int = 0
    failed: int = 0


# ============================================================================
# Time Parsing
# ============================================================================


def parse_since(since: str) -> datetime.datetime:
    """Parse --since argument to datetime (UTC).

    Supports:
    - Relative: 15m, 2h, 1d
    - ISO date: 2025-10-10
    - ISO datetime: 2025-10-10T14:51:40Z or 2025-10-10T14:51:40
    """
    now = datetime.datetime.now(datetime.timezone.utc)

    # Try relative formats first
    if since.endswith('m'):
        minutes = int(since[:-1])
        return now - datetime.timedelta(minutes=minutes)
    elif since.endswith('h'):
        hours = int(since[:-1])
        return now - datetime.timedelta(hours=hours)
    elif since.endswith('d'):
        days = int(since[:-1])
        return now - datetime.timedelta(days=days)

    # Try ISO formats
    try:
        # ISO datetime with Z
        if since.endswith('Z'):
            return datetime.datetime.fromisoformat(since[:-1]).replace(tzinfo=datetime.timezone.utc)
        # ISO datetime without timezone (assume UTC)
        elif 'T' in since:
            return datetime.datetime.fromisoformat(since).replace(tzinfo=datetime.timezone.utc)
        # ISO date only
        else:
            return datetime.datetime.fromisoformat(since).replace(tzinfo=datetime.timezone.utc)
    except ValueError:
        raise ValueError(f"Invalid --since format: {since}. Use 15m, 2h, 1d, ISO date, or ISO datetime")


# ============================================================================
# State Database
# ============================================================================


class StateDB:
    """SQLite database for tracking sent events."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self._init_schema()

    def _init_schema(self):
        """Create sent_events table if not exists."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS sent_events (
                call_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                session_path TEXT NOT NULL,
                sent_at TEXT NOT NULL,
                mode TEXT NOT NULL
            )
        """)
        self.conn.commit()

    def is_sent(self, call_id: str) -> bool:
        """Check if call_id was already sent."""
        cursor = self.conn.execute(
            "SELECT 1 FROM sent_events WHERE call_id = ? LIMIT 1",
            (call_id,)
        )
        return cursor.fetchone() is not None

    def mark_sent(self, heartbeat: Heartbeat, mode: str):
        """Mark heartbeat as sent."""
        self.conn.execute(
            """
            INSERT INTO sent_events (call_id, timestamp, session_path, sent_at, mode)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                heartbeat.call_id,
                datetime.datetime.fromtimestamp(heartbeat.timestamp, datetime.timezone.utc).isoformat(),
                heartbeat.session_path,
                datetime.datetime.now(datetime.timezone.utc).isoformat(),
                mode
            )
        )
        self.conn.commit()

    def close(self):
        """Close database connection."""
        self.conn.close()


# ============================================================================
# Codex Log Parser
# ============================================================================


def _parse_iso_timestamp(value: str) -> Optional[datetime.datetime]:
    """Parse ISO8601 value, accepting trailing Z."""
    if not value:
        return None
    try:
        if value.endswith('Z'):
            return datetime.datetime.fromisoformat(value[:-1]).replace(tzinfo=datetime.timezone.utc)
        return datetime.datetime.fromisoformat(value).replace(tzinfo=datetime.timezone.utc)
    except ValueError:
        return None


def _session_file_start(session_file: Path) -> datetime.datetime:
    """Approximate session start from filename or file mtime."""
    match = re.search(r"rollout-(\d{4}-\d{2}-\d{2}T\d{2})-(\d{2})-(\d{2})", session_file.name)
    if match:
        iso_prefix = match.group(1)
        mins = match.group(2)
        secs = match.group(3)
        iso_value = f"{iso_prefix}:{mins}:{secs}"
        parsed = _parse_iso_timestamp(iso_value)
        if parsed:
            return parsed
    return datetime.datetime.fromtimestamp(session_file.stat().st_mtime, datetime.timezone.utc)


def parse_codex_logs(
    sessions_dir: Path,
    since: datetime.datetime,
    max_events: Optional[int],
    logger: logging.Logger
) -> List[Tuple[Dict[str, Any], Path, datetime.datetime]]:
    """Scan and parse Codex rollout logs.

    Returns list of (event_dict, session_path, event_time) tuples for events >= since.
    """
    events: List[Tuple[Dict[str, Any], Path, datetime.datetime]] = []
    session_files = sorted(sessions_dir.rglob("rollout-*.jsonl"))

    for session_file in session_files:
        try:
            base_time = _session_file_start(session_file)
            fallback_offset = 0

            with open(session_file, 'r', encoding='utf-8') as handle:
                for line_num, line in enumerate(handle, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError as exc:
                        logger.warning(f"JSON decode error in {session_file}:{line_num}: {exc}")
                        continue

                    raw_ts = event.get('timestamp') or event.get('time')
                    event_time = _parse_iso_timestamp(raw_ts)
                    if event_time is None:
                        event_time = base_time + datetime.timedelta(seconds=fallback_offset)
                        fallback_offset += 1

                    if event_time < since:
                        continue

                    events.append((event, session_file, event_time))

                    if max_events and len(events) >= max_events:
                        logger.info(f"Reached max_events limit ({max_events})")
                        return sorted(events, key=lambda item: item[2])

        except Exception as exc:
            logger.warning(f"Error reading {session_file}: {exc}")
            continue

    events.sort(key=lambda item: item[2])
    return events


def normalize_to_heartbeats(
    events: List[Tuple[Dict[str, Any], Path, datetime.datetime]],
    project_override: Optional[str],
    logger: logging.Logger
) -> List[Heartbeat]:
    """Convert parsed events to normalized Heartbeat objects.

    Handles both modern (response_item) and legacy (type=function_call) formats.
    Maintains session context from session_meta events.
    """
    heartbeats = []
    session_contexts: Dict[str, SessionContext] = {}

    for event, session_path, event_time in events:
        # Update session context from session_meta
        if event.get('type') == 'session_meta':
            payload = event.get('payload', {})
            ctx = session_contexts.setdefault(str(session_path), SessionContext())
            ctx.cwd = payload.get('cwd') or ctx.cwd
            git_payload = payload.get('git') or {}
            ctx.git_branch = git_payload.get('branch') or ctx.git_branch
            continue

        # Modern format: response_item with payload.type == function_call
        if event.get('type') == 'response_item':
            payload = event.get('payload', {})
            if payload.get('type') != 'function_call':
                continue

            func_name = payload.get('name', '')
            call_id = payload.get('call_id', '')
            args_json = payload.get('arguments', '{}')

        # Legacy format: type == function_call
        elif event.get('type') == 'function_call':
            func_name = event.get('name', '')
            call_id = event.get('call_id', '')
            args_json = event.get('arguments', '{}')

        else:
            continue

        # Skip non-shell calls (allow shell, bash, ignore update_plan gracefully)
        if func_name not in ('shell', 'bash', 'Bash'):
            if func_name == 'update_plan':
                continue  # Skip gracefully
            continue

        # Parse arguments
        try:
            args = json.loads(args_json)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse arguments for {call_id}")
            continue

        command_list = args.get('command', [])
        if not command_list:
            continue

        workdir = args.get('workdir')

        # Build entity (command truncated to 200 chars)
        command_str = ' '.join(command_list)
        entity = command_str[:200]

        # Determine project
        ctx = session_contexts.get(str(session_path), SessionContext())
        project_candidates = []
        if project_override:
            project_candidates.append(project_override)
        if workdir:
            project_candidates.append(Path(workdir).name)
        if ctx.cwd:
            project_candidates.append(Path(ctx.cwd).name)
        project_candidates.append(session_path.parent.name)

        project = next(
            (
                candidate
                for candidate in project_candidates
                if candidate and candidate not in ('.', os.sep)
            ),
            'codex'
        )

        # Build metadata
        metadata = {
            'command': command_list,
            'workdir': workdir or ctx.cwd,
        }
        if ctx.git_branch:
            metadata['branch'] = ctx.git_branch

        heartbeats.append(Heartbeat(
            call_id=call_id,
            timestamp=event_time.timestamp(),
            entity=entity,
            entity_type='app',
            category='ai coding',
            project=project,
            session_path=str(session_path),
            metadata=metadata
        ))

    return heartbeats


# ============================================================================
# Senders
# ============================================================================


def send_cli_mode(
    heartbeat: Heartbeat,
    wakatime_bin: str,
    dry_run: bool,
    logger: logging.Logger
) -> bool:
    """Send heartbeat via wakatime-cli.

    Returns True on success, False on failure.
    Retries once on failure after 2s delay.
    """
    if dry_run:
        logger.info(f"[DRY-RUN] Would send via CLI: {heartbeat.entity} @ {heartbeat.timestamp}")
        return True

    cmd = [
        wakatime_bin,
        '--entity', heartbeat.entity,
        '--entity-type', 'app',
        '--category', heartbeat.category,
        '--time', str(heartbeat.timestamp),
        '--project', heartbeat.project,
        '--plugin', 'codex-bridge/0.1',
        '--write'
    ]

    for attempt in range(2):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                logger.debug(f"CLI sent: {heartbeat.call_id}")
                return True
            else:
                logger.warning(
                    f"CLI failed (attempt {attempt + 1}/2) for {heartbeat.call_id}: "
                    f"exit={result.returncode}, stderr={result.stderr[:200]}"
                )
                if attempt == 0:
                    time.sleep(2)

        except subprocess.TimeoutExpired:
            logger.warning(f"CLI timeout (attempt {attempt + 1}/2) for {heartbeat.call_id}")
            if attempt == 0:
                time.sleep(2)
        except Exception as e:
            logger.warning(f"CLI error (attempt {attempt + 1}/2) for {heartbeat.call_id}: {e}")
            if attempt == 0:
                time.sleep(2)

    return False


def send_api_mode_batch(
    heartbeats: List[Heartbeat],
    api_key: str,
    api_url: str,
    dry_run: bool,
    logger: logging.Logger
) -> Tuple[int, int]:
    """Send batch of heartbeats via WakaTime API.

    Returns (success_count, failure_count).
    Implements exponential backoff on 429/5xx errors.
    """
    if dry_run:
        for hb in heartbeats:
            logger.info(f"[DRY-RUN] Would send via API: {hb.entity} @ {hb.timestamp}")
        return len(heartbeats), 0

    # Build payload
    payload = {
        'heartbeats': [
            {
                'entity': hb.entity,
                'type': hb.entity_type,
                'category': hb.category,
                'time': hb.timestamp,
                'project': hb.project,
                'plugin': 'codex-bridge/0.1',
                'metadata': json.dumps(hb.metadata)
            }
            for hb in heartbeats
        ]
    }

    # Prepare request
    url = f"{api_url.rstrip('/')}/api/v1/users/current/heartbeats.bulk"
    auth_str = base64.b64encode(f"{api_key}:".encode()).decode()
    headers = {
        'Authorization': f'Basic {auth_str}',
        'Content-Type': 'application/json'
    }
    data = json.dumps(payload).encode('utf-8')

    # Exponential backoff retry
    max_retries = 5
    backoff = 1.0

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method='POST')
            with urllib.request.urlopen(req, timeout=30) as response:
                if response.status == 201 or response.status == 202:
                    logger.debug(f"API batch sent: {len(heartbeats)} heartbeats")
                    return len(heartbeats), 0
                else:
                    logger.warning(f"API unexpected status {response.status}")
                    return 0, len(heartbeats)

        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:
                logger.warning(
                    f"API HTTP {e.code} (attempt {attempt + 1}/{max_retries}), "
                    f"retrying in {backoff}s..."
                )
                if attempt < max_retries - 1:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 30)
                    continue

            logger.error(f"API HTTP error {e.code}: {e.reason}")
            return 0, len(heartbeats)

        except urllib.error.URLError as e:
            logger.error(f"API URL error: {e.reason}")
            return 0, len(heartbeats)

        except Exception as e:
            logger.error(f"API error: {e}")
            return 0, len(heartbeats)

    logger.error(f"API failed after {max_retries} retries")
    return 0, len(heartbeats)


# ============================================================================
# Main Sync Logic
# ============================================================================


def sync(
    sessions_dir: Path,
    state_db: StateDB,
    since: datetime.datetime,
    mode: str,
    dry_run: bool,
    batch_size: int,
    wakatime_bin: str,
    api_key: Optional[str],
    api_url: str,
    project: Optional[str],
    max_events: Optional[int],
    logger: logging.Logger
) -> Stats:
    """Main sync pipeline: scan → parse → normalize → dedupe → send."""
    stats = Stats()

    # 1. Scan and parse
    logger.info(f"Scanning sessions from {sessions_dir} since {since.isoformat()}")
    events = parse_codex_logs(sessions_dir, since, max_events, logger)
    stats.scanned = len(events)
    logger.info(f"Scanned {stats.scanned} events")

    # 2. Normalize to heartbeats
    heartbeats = normalize_to_heartbeats(events, project, logger)
    stats.eligible = len(heartbeats)
    logger.info(f"Normalized {stats.eligible} eligible heartbeats")

    # 3. Deduplicate
    unique_heartbeats = []
    for hb in heartbeats:
        if state_db.is_sent(hb.call_id):
            stats.duplicates += 1
        else:
            unique_heartbeats.append(hb)

    logger.info(f"Filtered out {stats.duplicates} duplicates, {len(unique_heartbeats)} remain")

    # 4. Send
    if mode == 'cli':
        for hb in unique_heartbeats:
            success = send_cli_mode(hb, wakatime_bin, dry_run, logger)
            if success:
                if not dry_run:
                    state_db.mark_sent(hb, mode)
                stats.sent += 1
            else:
                stats.failed += 1

    elif mode == 'api':
        # Send in batches
        for i in range(0, len(unique_heartbeats), batch_size):
            batch = unique_heartbeats[i:i + batch_size]
            sent, failed = send_api_mode_batch(batch, api_key, api_url, dry_run, logger)

            if not dry_run:
                # Mark successfully sent heartbeats
                for hb in batch[:sent]:
                    state_db.mark_sent(hb, mode)

            stats.sent += sent
            stats.failed += failed

    return stats


# ============================================================================
# CLI Entry Point
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description='Sync Codex shell activity to WakaTime heartbeats',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', required=True)

    sync_parser = subparsers.add_parser('sync', help='Sync Codex logs to WakaTime')
    sync_parser.add_argument(
        '--since',
        default='2h',
        help='Process events since this time (15m, 2h, 1d, ISO date, ISO datetime). Default: 2h'
    )
    sync_parser.add_argument(
        '--sessions-dir',
        type=Path,
        default=Path.home() / '.codex' / 'sessions',
        help='Path to Codex sessions directory. Default: ~/.codex/sessions'
    )
    sync_parser.add_argument(
        '--state-db',
        type=Path,
        default=Path.home() / '.codex-wakatime' / 'state.db',
        help='Path to state database. Default: ~/.codex-wakatime/state.db'
    )
    sync_parser.add_argument(
        '--mode',
        choices=['cli', 'api'],
        default='cli',
        help='Send mode: cli (wakatime-cli) or api (HTTP API). Default: cli'
    )
    sync_parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Log heartbeats without sending or recording to DB'
    )
    sync_parser.add_argument(
        '--batch-size',
        type=int,
        default=25,
        help='Batch size for API mode. Default: 25'
    )
    sync_parser.add_argument(
        '--wakatime-bin',
        default='wakatime-cli',
        help='Path to wakatime-cli binary. Default: wakatime-cli'
    )
    sync_parser.add_argument(
        '--api-key',
        help='WakaTime API key (for API mode). Can also use WAKATIME_API_KEY env var'
    )
    sync_parser.add_argument(
        '--api-url',
        default='https://api.wakatime.com',
        help='WakaTime API base URL. Default: https://api.wakatime.com'
    )
    sync_parser.add_argument(
        '--project',
        help='Override project name (default: derive from workdir/cwd)'
    )
    sync_parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level. Default: INFO'
    )
    sync_parser.add_argument(
        '--max-events',
        type=int,
        help='Maximum events to process (for testing)'
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)

    if args.command == 'sync':
        # Parse since
        try:
            since = parse_since(args.since)
        except ValueError as e:
            logger.error(str(e))
            sys.exit(1)

        # Validate sessions directory
        if not args.sessions_dir.exists():
            logger.error(f"Sessions directory not found: {args.sessions_dir}")
            sys.exit(1)

        # Get API key from env if not provided
        api_key = args.api_key or os.environ.get('WAKATIME_API_KEY')
        if args.mode == 'api' and not api_key:
            logger.error("API mode requires --api-key or WAKATIME_API_KEY env var")
            sys.exit(1)

        # Open state database
        state_db = StateDB(args.state_db)

        try:
            stats = sync(
                sessions_dir=args.sessions_dir,
                state_db=state_db,
                since=since,
                mode=args.mode,
                dry_run=args.dry_run,
                batch_size=args.batch_size,
                wakatime_bin=args.wakatime_bin,
                api_key=api_key,
                api_url=args.api_url,
                project=args.project,
                max_events=args.max_events,
                logger=logger
            )

            # Print summary
            logger.info("=" * 60)
            logger.info("SYNC SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Scanned events:      {stats.scanned}")
            logger.info(f"Eligible heartbeats: {stats.eligible}")
            logger.info(f"Duplicates skipped:  {stats.duplicates}")
            logger.info(f"Successfully sent:   {stats.sent}")
            logger.info(f"Failed:              {stats.failed}")
            logger.info("=" * 60)

            # Exit with error if any failures
            if stats.failed > 0 and not args.dry_run:
                sys.exit(1)

        finally:
            state_db.close()


if __name__ == "__main__":
    main()
