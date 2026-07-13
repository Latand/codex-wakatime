#!/usr/bin/env python3
"""Privacy-preserving Codex and Claude activity sync for WakaTime."""

from __future__ import annotations

import argparse
import base64
import configparser
import dataclasses
import datetime
import glob
import hashlib
import hmac
import json
import logging
import os
import secrets
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional, Sequence


VERSION = "1.0.0"
PLUGIN = f"agent-wakatime/{VERSION}"
SUPPORTED_ENGINES = {"claude", "codex"}
DEFAULT_SOURCE_FILE = Path.home() / ".config" / "agent-wakatime" / "sources.conf"
DEFAULT_STATE_DB = Path.home() / ".codex-wakatime" / "state.db"
DEFAULT_PRIVACY_KEY = Path.home() / ".codex-wakatime" / "privacy.key"
DEFAULT_WAKATIME_CONFIG = Path.home() / ".wakatime.cfg"


@dataclasses.dataclass(frozen=True)
class Source:
    """A transcript directory assigned to an engine adapter."""

    engine: str
    directory: Path

    def __post_init__(self) -> None:
        if self.engine not in SUPPORTED_ENGINES:
            raise ValueError("unsupported source engine")


@dataclasses.dataclass(frozen=True)
class ActivityInterval:
    engine: str
    session_key: str
    turn_key: str
    start: float
    end: float
    project_key: str


@dataclasses.dataclass(frozen=True)
class ActivityHeartbeat:
    fingerprint: str
    timestamp: float
    engine: str
    project: str
    session_key: str


@dataclasses.dataclass
class CollectionStats:
    files: int = 0
    inaccessible_files: int = 0
    malformed_lines: int = 0
    intervals: int = 0
    duplicate_intervals: int = 0


@dataclasses.dataclass
class SyncStats:
    files: int = 0
    intervals: int = 0
    generated: int = 0
    duplicates: int = 0
    sent: int = 0
    failed: int = 0
    malformed_lines: int = 0
    inaccessible_files: int = 0


@dataclasses.dataclass
class ParseResult:
    intervals: list[ActivityInterval] = dataclasses.field(default_factory=list)
    malformed_lines: int = 0
    inaccessible: bool = False


def _private_hash(value: str, privacy_key: bytes, length: int = 16) -> str:
    digest = hmac.new(privacy_key, value.encode("utf-8"), hashlib.sha256).hexdigest()
    return digest[:length]


def _parse_iso_timestamp(value: Any) -> Optional[datetime.datetime]:
    if isinstance(value, (int, float)):
        return datetime.datetime.fromtimestamp(float(value), datetime.timezone.utc)
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.timezone.utc)
    return parsed.astimezone(datetime.timezone.utc)


def parse_since(value: str, now: Optional[datetime.datetime] = None) -> datetime.datetime:
    """Parse a relative duration or an ISO-8601 timestamp as UTC."""
    current = now or datetime.datetime.now(datetime.timezone.utc)
    units = {"m": 60, "h": 3600, "d": 86400}
    if len(value) > 1 and value[-1] in units:
        try:
            amount = int(value[:-1])
        except ValueError as exc:
            raise ValueError("invalid relative time") from exc
        if amount <= 0:
            raise ValueError("relative time must be positive")
        return current - datetime.timedelta(seconds=amount * units[value[-1]])
    parsed = _parse_iso_timestamp(value)
    if parsed is None:
        raise ValueError("invalid time; use 45m, 2h, 1d, or ISO-8601")
    return parsed


def parse_duration(value: str) -> int:
    units = {"s": 1, "m": 60, "h": 3600}
    if len(value) < 2 or value[-1] not in units:
        raise ValueError("invalid duration; use 30s, 15m, or 1h")
    try:
        amount = int(value[:-1])
    except ValueError as exc:
        raise ValueError("invalid duration") from exc
    if amount <= 0:
        raise ValueError("duration must be positive")
    return amount * units[value[-1]]


_PROJECT_SEED_CACHE: dict[str, str] = {}


def _run_git(path: Path, args: Sequence[str]) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), *args],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode:
        return None
    value = result.stdout.strip()
    return value or None


def _project_seed(path_value: Any, repository_hint: Any = None) -> str:
    if isinstance(repository_hint, str) and repository_hint:
        return f"repository:{repository_hint}"
    if not isinstance(path_value, str) or not path_value:
        return "unknown-project"
    cached = _PROJECT_SEED_CACHE.get(path_value)
    if cached:
        return cached

    original = Path(path_value).expanduser()
    candidate = original
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if candidate.exists():
        top = _run_git(candidate, ["rev-parse", "--show-toplevel"])
        remote = _run_git(candidate, ["config", "--get", "remote.origin.url"])
        if remote:
            seed = f"repository:{remote}"
        elif top:
            seed = f"repository-root:{top}"
        else:
            seed = f"workspace:{original}"
    else:
        seed = f"workspace:{original}"
    _PROJECT_SEED_CACHE[path_value] = seed
    return seed


def _make_interval(
    engine: str,
    session_id: str,
    turn_id: str,
    start: float,
    end: float,
    project_seed: str,
    privacy_key: bytes,
) -> ActivityInterval:
    return ActivityInterval(
        engine=engine,
        session_key=_private_hash(f"session:{engine}:{session_id}", privacy_key),
        turn_key=_private_hash(f"turn:{engine}:{turn_id}", privacy_key),
        start=start,
        end=end,
        project_key=_private_hash(f"project:{project_seed}", privacy_key, 12),
    )


def _codex_intervals(
    session_file: Path,
    since: float,
    now: float,
    active_grace_seconds: int,
    privacy_key: bytes,
) -> ParseResult:
    result = ParseResult()
    session_id = session_file.stem
    project_seed = "unknown-project"
    starts: dict[str, tuple[float, str]] = {}
    latest_event = 0.0

    try:
        handle = session_file.open("r", encoding="utf-8")
    except OSError:
        result.inaccessible = True
        return result

    with handle:
        for line in handle:
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                result.malformed_lines += 1
                continue
            if not isinstance(event, dict):
                continue

            event_time = _parse_iso_timestamp(event.get("timestamp") or event.get("time"))
            timestamp = event_time.timestamp() if event_time else None
            if timestamp is not None:
                latest_event = max(latest_event, timestamp)

            if event.get("type") == "session_meta":
                payload = event.get("payload") or {}
                if isinstance(payload, dict):
                    session_id = str(payload.get("id") or payload.get("session_id") or session_id)
                    git_data = payload.get("git") or {}
                    repository_hint = git_data.get("repository_url") if isinstance(git_data, dict) else None
                    project_seed = _project_seed(payload.get("cwd"), repository_hint)
                continue
            if timestamp is None:
                continue

            payload = event.get("payload") or {}
            if not isinstance(payload, dict) or event.get("type") != "event_msg":
                continue
            event_kind = payload.get("type")
            turn_id = str(payload.get("turn_id") or "")
            if event_kind == "task_started" and turn_id:
                for previous_turn, (previous_start, previous_project) in starts.items():
                    if timestamp >= since and timestamp >= previous_start:
                        result.intervals.append(
                            _make_interval(
                                "codex",
                                session_id,
                                previous_turn,
                                previous_start,
                                timestamp,
                                previous_project,
                                privacy_key,
                            )
                        )
                starts.clear()
                starts[turn_id] = (timestamp, project_seed)
            elif event_kind in {"task_complete", "turn_aborted"} and turn_id in starts:
                start, turn_project = starts.pop(turn_id)
                if timestamp >= since and timestamp >= start:
                    result.intervals.append(
                        _make_interval(
                            "codex",
                            session_id,
                            turn_id,
                            start,
                            timestamp,
                            turn_project,
                            privacy_key,
                        )
                    )

    if starts and latest_event:
        open_end = min(now, latest_event + active_grace_seconds)
        for turn_id, (start, turn_project) in starts.items():
            if open_end >= since and open_end >= start:
                result.intervals.append(
                    _make_interval(
                        "codex",
                        session_id,
                        turn_id,
                        start,
                        open_end,
                        turn_project,
                        privacy_key,
                    )
                )
    return result


def _is_claude_prompt(event: dict[str, Any]) -> bool:
    if event.get("type") != "user" or event.get("isMeta") is True:
        return False
    if event.get("toolUseResult") is not None or event.get("sourceToolAssistantUUID") is not None:
        return False
    if event.get("isCompactSummary") is True or event.get("isVisibleInTranscriptOnly") is True:
        return False
    content = (event.get("message") or {}).get("content")
    if isinstance(content, str):
        return True
    if isinstance(content, list):
        return any(
            isinstance(item, dict) and item.get("type") in {"text", "image"}
            for item in content
        )
    return False


def _claude_intervals(
    session_file: Path,
    since: float,
    now: float,
    active_grace_seconds: int,
    privacy_key: bytes,
) -> ParseResult:
    result = ParseResult()
    fallback_session_id = session_file.stem
    current: Optional[tuple[str, str, float, str]] = None
    latest_event = 0.0

    try:
        handle = session_file.open("r", encoding="utf-8")
    except OSError:
        result.inaccessible = True
        return result

    def close_current(end: float) -> None:
        nonlocal current
        if current is None:
            return
        session_id, turn_id, start, project_seed = current
        current = None
        if end >= since and end >= start:
            result.intervals.append(
                _make_interval(
                    "claude",
                    session_id,
                    turn_id,
                    start,
                    end,
                    project_seed,
                    privacy_key,
                )
            )

    with handle:
        for line in handle:
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                result.malformed_lines += 1
                continue
            if not isinstance(event, dict):
                continue
            event_time = _parse_iso_timestamp(event.get("timestamp"))
            if event_time is None:
                continue
            timestamp = event_time.timestamp()
            latest_event = max(latest_event, timestamp)

            if _is_claude_prompt(event):
                close_current(timestamp)
                session_id = str(event.get("sessionId") or fallback_session_id)
                turn_id = str(event.get("promptId") or event.get("uuid") or f"prompt-{timestamp:.6f}")
                current = (
                    session_id,
                    turn_id,
                    timestamp,
                    _project_seed(event.get("cwd")),
                )
                continue

            message = event.get("message") or {}
            assistant_end = (
                event.get("type") == "assistant"
                and isinstance(message, dict)
                and message.get("stop_reason") in {"end_turn", "stop_sequence"}
            )
            duration_end = event.get("type") == "system" and event.get("subtype") == "turn_duration"
            if assistant_end or duration_end:
                close_current(timestamp)

    if current is not None and latest_event:
        close_current(min(now, latest_event + active_grace_seconds))
    return result


def _timeline_heartbeats(
    intervals: Sequence[ActivityInterval],
    since: float,
    now: float,
    cadence_seconds: int,
    privacy_key: bytes,
) -> list[ActivityHeartbeat]:
    relevant = [
        interval
        for interval in intervals
        if interval.end >= since and interval.start <= now and interval.end > interval.start
    ]
    boundaries = {since, now}
    for interval in relevant:
        boundaries.add(max(since, interval.start))
        boundaries.add(min(now, interval.end))
    ordered_boundaries = sorted(boundaries)

    segments: list[tuple[float, float, ActivityInterval]] = []
    for start, end in zip(ordered_boundaries, ordered_boundaries[1:]):
        if end <= start:
            continue
        active = [
            interval
            for interval in relevant
            if interval.start <= start and interval.end >= end
        ]
        if not active:
            continue
        foreground = max(
            active,
            key=lambda interval: (
                interval.start,
                interval.engine,
                interval.session_key,
                interval.turn_key,
            ),
        )
        if segments and segments[-1][1] == start and segments[-1][2] == foreground:
            previous_start, _, previous_interval = segments[-1]
            segments[-1] = (previous_start, end, previous_interval)
        else:
            segments.append((start, end, foreground))

    heartbeats: list[ActivityHeartbeat] = []
    for index, (start, end, foreground) in enumerate(segments):
        if start == since and foreground.start < since:
            elapsed = since - foreground.start
            steps = int(elapsed // cadence_seconds)
            point = foreground.start + steps * cadence_seconds
            if point < since:
                point += cadence_seconds
        else:
            point = start
        points: list[float] = []
        while point < end:
            points.append(point)
            point += cadence_seconds
        next_start = segments[index + 1][0] if index + 1 < len(segments) else None
        if next_start is None or next_start > end:
            points.append(end)

        for heartbeat_time in points:
            fingerprint_seed = (
                f"heartbeat:{foreground.engine}:{foreground.session_key}:"
                f"{foreground.turn_key}:{heartbeat_time:.6f}"
            )
            heartbeats.append(
                ActivityHeartbeat(
                    fingerprint=_private_hash(fingerprint_seed, privacy_key, 32),
                    timestamp=heartbeat_time,
                    engine=foreground.engine,
                    project=f"project-{foreground.project_key}",
                    session_key=foreground.session_key,
                )
            )
    return heartbeats


def collect_heartbeats(
    sources: Sequence[Source],
    since: datetime.datetime,
    now: datetime.datetime,
    privacy_key: bytes,
    cadence_seconds: int = 120,
    active_grace_seconds: int = 900,
) -> tuple[list[ActivityHeartbeat], CollectionStats]:
    """Collect a single privacy-safe activity timeline from transcript roots."""
    if cadence_seconds <= 0 or active_grace_seconds <= 0:
        raise ValueError("cadence and active grace must be positive")
    if len(privacy_key) < 16:
        raise ValueError("privacy key is too short")

    since_epoch = since.timestamp()
    now_epoch = now.timestamp()
    stats = CollectionStats()
    intervals: list[ActivityInterval] = []
    for source in sources:
        if not source.directory.is_dir():
            continue
        try:
            session_files = list(source.directory.rglob("*.jsonl"))
        except OSError:
            stats.inaccessible_files += 1
            continue
        for session_file in session_files:
            try:
                modified_at = session_file.stat().st_mtime
            except OSError:
                stats.inaccessible_files += 1
                continue
            if modified_at < since_epoch - active_grace_seconds:
                continue
            stats.files += 1
            parser = _codex_intervals if source.engine == "codex" else _claude_intervals
            parsed = parser(
                session_file,
                since_epoch,
                now_epoch,
                active_grace_seconds,
                privacy_key,
            )
            stats.malformed_lines += parsed.malformed_lines
            stats.inaccessible_files += int(parsed.inaccessible)
            intervals.extend(parsed.intervals)

    unique_intervals: list[ActivityInterval] = []
    seen: set[tuple[Any, ...]] = set()
    for interval in sorted(
        intervals,
        key=lambda item: (item.start, item.end, item.engine, item.session_key, item.turn_key),
    ):
        identity = (
            interval.engine,
            interval.session_key,
            interval.turn_key,
            interval.start,
            interval.end,
        )
        if identity in seen:
            stats.duplicate_intervals += 1
            continue
        seen.add(identity)
        unique_intervals.append(interval)

    stats.intervals = len(unique_intervals)
    return (
        _timeline_heartbeats(
            unique_intervals,
            since_epoch,
            now_epoch,
            cadence_seconds,
            privacy_key,
        ),
        stats,
    )


def _parse_source_spec(spec: str) -> tuple[str, str]:
    separator = "=" if "=" in spec else ":"
    if separator not in spec:
        raise ValueError("source must use ENGINE:DIR or ENGINE=DIR")
    engine, pattern = spec.split(separator, 1)
    engine = engine.strip().lower()
    pattern = pattern.strip()
    if engine not in SUPPORTED_ENGINES or not pattern:
        raise ValueError("invalid source specification")
    return engine, pattern


def load_sources(
    source_specs: Sequence[str],
    source_file: Optional[Path] = None,
) -> list[Source]:
    """Load source specs, expand directory globs, and remove duplicate roots."""
    specs: list[str] = []
    if source_file and source_file.exists():
        try:
            lines = source_file.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ValueError("unable to read source configuration") from exc
        for line_number, raw_line in enumerate(lines, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                _parse_source_spec(line)
            except ValueError as exc:
                raise ValueError(f"invalid source configuration at line {line_number}") from exc
            specs.append(line)
    specs.extend(source_specs)
    if not specs:
        specs = [
            f"codex:{Path.home() / '.codex' / 'sessions'}",
            f"claude:{Path.home() / '.claude' / 'projects'}",
        ]

    sources: dict[tuple[str, Path], Source] = {}
    for spec in specs:
        engine, raw_pattern = _parse_source_spec(spec)
        expanded_pattern = os.path.expandvars(os.path.expanduser(raw_pattern))
        matches = glob.glob(expanded_pattern)
        for match in matches:
            directory = Path(match)
            if directory.is_dir():
                resolved = directory.resolve()
                sources[(engine, resolved)] = Source(engine, resolved)
    return sorted(sources.values(), key=lambda source: (source.engine, str(source.directory)))


def load_or_create_privacy_key(path: Path) -> bytes:
    """Load a local random key, creating it with owner-only permissions."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        path.parent.chmod(0o700)
    except OSError:
        pass
    try:
        key = path.read_bytes()
    except FileNotFoundError:
        key = secrets.token_bytes(32)
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            key = path.read_bytes()
        else:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(key)
    except OSError as exc:
        raise ValueError("unable to read privacy key") from exc
    if len(key) < 32:
        raise ValueError("privacy key must contain at least 32 bytes")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return key


class StateDB:
    """Privacy-safe delivery receipts keyed by opaque heartbeat fingerprints."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            db_path.parent.chmod(0o700)
        except OSError:
            pass
        self.conn = sqlite3.connect(db_path)
        self._init_schema()
        try:
            db_path.chmod(0o600)
        except OSError:
            pass

    def _init_schema(self) -> None:
        self.conn.execute("PRAGMA secure_delete=ON")
        self.conn.execute("PRAGMA journal_mode=DELETE")
        tables = {
            row[0]
            for row in self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        scrubbed = False
        if "sent_events" in tables:
            self.conn.execute("DROP TABLE sent_events")
            scrubbed = True
        if "sent_heartbeats" in tables:
            columns = {
                row[1] for row in self.conn.execute("PRAGMA table_info(sent_heartbeats)")
            }
            expected = {"fingerprint", "timestamp", "sent_at", "mode"}
            if columns != expected:
                self.conn.execute("DROP TABLE sent_heartbeats")
                scrubbed = True
        self.conn.commit()
        if scrubbed:
            self.conn.execute("VACUUM")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sent_heartbeats (
                fingerprint TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                sent_at TEXT NOT NULL,
                mode TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def is_sent(self, fingerprint: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM sent_heartbeats WHERE fingerprint = ? LIMIT 1",
            (fingerprint,),
        ).fetchone()
        return row is not None

    def mark_sent(self, heartbeats: Sequence[ActivityHeartbeat], mode: str) -> None:
        sent_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.conn.executemany(
            """
            INSERT OR IGNORE INTO sent_heartbeats (fingerprint, timestamp, sent_at, mode)
            VALUES (?, ?, ?, ?)
            """,
            [
                (heartbeat.fingerprint, heartbeat.timestamp, sent_at, mode)
                for heartbeat in heartbeats
            ],
        )
        self.conn.commit()

    def prune(self, before: float) -> None:
        self.conn.execute("DELETE FROM sent_heartbeats WHERE timestamp < ?", (before,))
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


def heartbeat_payload(heartbeat: ActivityHeartbeat) -> dict[str, Any]:
    """Build the complete allowlisted WakaTime payload for one heartbeat."""
    return {
        "entity": f"agent:{heartbeat.engine}",
        "type": "app",
        "category": "ai coding",
        "time": heartbeat.timestamp,
        "project": heartbeat.project,
        "plugin": PLUGIN,
        "ai_session": heartbeat.session_key,
    }


def read_wakatime_api_key(config_path: Path) -> str:
    parser = configparser.RawConfigParser()
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            parser.read_file(handle)
        key = parser.get("settings", "api_key")
    except (OSError, configparser.Error) as exc:
        raise ValueError("unable to read WakaTime configuration") from exc
    key = key.strip()
    if not key:
        raise ValueError("WakaTime API key is missing")
    return key


def make_api_sender(
    api_key: str,
    api_url: str,
    logger: logging.Logger,
) -> Callable[[Sequence[ActivityHeartbeat]], bool]:
    if not api_url.startswith("https://"):
        raise ValueError("WakaTime API URL must use HTTPS")
    endpoint = f"{api_url.rstrip('/')}/users/current/heartbeats.bulk"
    authorization = base64.b64encode(f"{api_key}:".encode("utf-8")).decode("ascii")

    def send(heartbeats: Sequence[ActivityHeartbeat]) -> bool:
        data = json.dumps([heartbeat_payload(item) for item in heartbeats]).encode("utf-8")
        headers = {
            "Authorization": f"Basic {authorization}",
            "Content-Type": "application/json",
            "User-Agent": PLUGIN,
        }
        backoff = 1.0
        for attempt in range(5):
            request = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    if response.status in {200, 201, 202}:
                        return True
                    logger.warning("WakaTime returned an unexpected status")
                    return False
            except urllib.error.HTTPError as exc:
                retryable = exc.code == 429 or exc.code >= 500
                if retryable and attempt < 4:
                    logger.warning("WakaTime request will be retried after HTTP %d", exc.code)
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 30)
                    continue
                logger.error("WakaTime rejected a heartbeat batch with HTTP %d", exc.code)
                return False
            except (urllib.error.URLError, TimeoutError, OSError):
                if attempt < 4:
                    logger.warning("WakaTime network request will be retried")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 30)
                    continue
                logger.error("WakaTime network request failed")
                return False
        return False

    return send


def sync_activity(
    sources: Sequence[Source],
    state: StateDB,
    since: datetime.datetime,
    now: datetime.datetime,
    privacy_key: bytes,
    sender: Callable[[Sequence[ActivityHeartbeat]], bool],
    logger: logging.Logger,
    batch_size: int = 25,
    cadence_seconds: int = 120,
    active_grace_seconds: int = 900,
    dry_run: bool = False,
) -> SyncStats:
    if batch_size < 1 or batch_size > 25:
        raise ValueError("batch size must be between 1 and 25")
    heartbeats, collected = collect_heartbeats(
        sources,
        since,
        now,
        privacy_key,
        cadence_seconds,
        active_grace_seconds,
    )
    pending = [heartbeat for heartbeat in heartbeats if not state.is_sent(heartbeat.fingerprint)]
    stats = SyncStats(
        files=collected.files,
        intervals=collected.intervals,
        generated=len(heartbeats),
        duplicates=len(heartbeats) - len(pending) + collected.duplicate_intervals,
        malformed_lines=collected.malformed_lines,
        inaccessible_files=collected.inaccessible_files,
    )
    if dry_run:
        logger.info(
            "sync preview: sources=%d files=%d intervals=%d heartbeats=%d pending=%d",
            len(sources),
            stats.files,
            stats.intervals,
            stats.generated,
            len(pending),
        )
        return stats

    for offset in range(0, len(pending), batch_size):
        batch = pending[offset : offset + batch_size]
        if sender(batch):
            state.mark_sent(batch, "api")
            stats.sent += len(batch)
        else:
            stats.failed += len(batch)
    logger.info(
        "sync complete: sources=%d files=%d intervals=%d generated=%d duplicates=%d sent=%d failed=%d malformed=%d inaccessible=%d",
        len(sources),
        stats.files,
        stats.intervals,
        stats.generated,
        stats.duplicates,
        stats.sent,
        stats.failed,
        stats.malformed_lines,
        stats.inaccessible_files,
    )
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sync privacy-safe Codex and Claude activity to WakaTime"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    sync_parser = subparsers.add_parser("sync", help="scan configured transcript roots")
    sync_parser.add_argument("--since", default="45m")
    sync_parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="ENGINE:DIR",
        help="add a Codex or Claude transcript root; directory globs are accepted",
    )
    sync_parser.add_argument("--source-file", type=Path)
    sync_parser.add_argument(
        "--sessions-dir",
        type=Path,
        action="append",
        default=[],
        help=argparse.SUPPRESS,
    )
    sync_parser.add_argument("--state-db", type=Path, default=DEFAULT_STATE_DB)
    sync_parser.add_argument("--privacy-key-file", type=Path, default=DEFAULT_PRIVACY_KEY)
    sync_parser.add_argument("--wakatime-config", type=Path, default=DEFAULT_WAKATIME_CONFIG)
    sync_parser.add_argument(
        "--api-url",
        default="https://api.wakatime.com/api/v1",
    )
    sync_parser.add_argument("--batch-size", type=int, default=25)
    sync_parser.add_argument("--cadence-seconds", type=int, default=120)
    sync_parser.add_argument("--active-grace", default="15m")
    sync_parser.add_argument("--dry-run", action="store_true")
    sync_parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("agent-wakatime")
    now = datetime.datetime.now(datetime.timezone.utc)

    try:
        since = parse_since(args.since, now)
        active_grace = parse_duration(args.active_grace)
        source_specs = list(args.source)
        source_specs.extend(f"codex:{directory}" for directory in args.sessions_dir)
        source_file = args.source_file
        if source_file is None and DEFAULT_SOURCE_FILE.exists():
            source_file = DEFAULT_SOURCE_FILE
        sources = load_sources(source_specs, source_file)
        if not sources:
            raise ValueError("no readable transcript source directories were found")
        privacy_key = load_or_create_privacy_key(args.privacy_key_file)
        state = StateDB(args.state_db)
        try:
            if args.dry_run:
                sender: Callable[[Sequence[ActivityHeartbeat]], bool] = lambda _batch: True
            else:
                api_key = read_wakatime_api_key(args.wakatime_config)
                sender = make_api_sender(api_key, args.api_url, logger)
            stats = sync_activity(
                sources,
                state,
                since,
                now,
                privacy_key,
                sender,
                logger,
                batch_size=args.batch_size,
                cadence_seconds=args.cadence_seconds,
                active_grace_seconds=active_grace,
                dry_run=args.dry_run,
            )
            state.prune((now - datetime.timedelta(days=90)).timestamp())
        finally:
            state.close()
    except ValueError as exc:
        logger.error("sync configuration failed: %s", exc)
        return 2
    except (OSError, sqlite3.Error):
        logger.error("sync configuration failed during local storage access")
        return 2
    except Exception:
        logger.error("sync failed with an internal error")
        return 1
    return 1 if stats.failed else 0


if __name__ == "__main__":
    sys.exit(main())
