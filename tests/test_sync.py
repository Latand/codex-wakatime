import datetime as dt
import io
import json
import logging
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from codex_wakatime_sync import (
    ActivityHeartbeat,
    Source,
    StateDB,
    collect_heartbeats,
    heartbeat_payload,
    load_sources,
    make_api_sender,
    sync_activity,
)


UTC = dt.timezone.utc
PRIVACY_KEY = b"test-privacy-key-with-enough-entropy"


def write_jsonl(path: Path, rows: list[dict], trailing: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows) + trailing,
        encoding="utf-8",
    )


def codex_rows(
    *,
    session_id: str = "session-1",
    turn_id: str = "turn-1",
    start: str = "2026-01-02T10:00:00Z",
    end: str = "2026-01-02T10:05:00Z",
    cwd: str = "/private/example/project",
) -> list[dict]:
    return [
        {
            "timestamp": start,
            "type": "session_meta",
            "payload": {"id": session_id, "cwd": cwd},
        },
        {
            "timestamp": start,
            "type": "event_msg",
            "payload": {"type": "task_started", "turn_id": turn_id},
        },
        {
            "timestamp": end,
            "type": "event_msg",
            "payload": {"type": "task_complete", "turn_id": turn_id},
        },
    ]


class CollectionBehaviorTests(unittest.TestCase):
    def collect(self, sources: list[Source], **kwargs):
        return collect_heartbeats(
            sources,
            since=kwargs.pop("since", dt.datetime(2026, 1, 2, 9, 59, tzinfo=UTC)),
            now=kwargs.pop("now", dt.datetime(2026, 1, 2, 10, 10, tzinfo=UTC)),
            privacy_key=PRIVACY_KEY,
            **kwargs,
        )

    def test_complete_codex_turn_becomes_regular_heartbeats(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_jsonl(root / "rollout.jsonl", codex_rows())

            heartbeats, _ = self.collect([Source("codex", root)], cadence_seconds=120)

            self.assertEqual(
                [heartbeat.timestamp for heartbeat in heartbeats],
                [1767348000.0, 1767348120.0, 1767348240.0, 1767348300.0],
            )
            self.assertTrue(all(heartbeat.engine == "codex" for heartbeat in heartbeats))

    def test_same_codex_session_in_multiple_sources_is_counted_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            rows = codex_rows(
                session_id="shared-session",
                turn_id="shared-turn",
                end="2026-01-02T10:02:00Z",
            )
            write_jsonl(base / "one" / "rollout.jsonl", rows)
            write_jsonl(base / "two" / "rollout.jsonl", rows)

            heartbeats, stats = self.collect(
                [Source("codex", base / "one"), Source("codex", base / "two")]
            )

            self.assertEqual(len(heartbeats), 2)
            self.assertEqual(stats.duplicate_intervals, 1)

    def test_complete_claude_turn_becomes_regular_heartbeats(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_jsonl(
                root / "workspace" / "session.jsonl",
                [
                    {
                        "timestamp": "2026-01-02T10:00:00Z",
                        "type": "user",
                        "uuid": "prompt-1",
                        "sessionId": "claude-session",
                        "cwd": "/private/example/project",
                        "message": {"role": "user", "content": "private prompt"},
                    },
                    {
                        "timestamp": "2026-01-02T10:03:00Z",
                        "type": "assistant",
                        "sessionId": "claude-session",
                        "message": {
                            "role": "assistant",
                            "stop_reason": "end_turn",
                            "content": [{"type": "text", "text": "private response"}],
                        },
                    },
                ],
            )

            heartbeats, _ = self.collect([Source("claude", root)])

            self.assertEqual(
                [heartbeat.timestamp for heartbeat in heartbeats],
                [1767348000.0, 1767348120.0, 1767348180.0],
            )
            self.assertTrue(all(heartbeat.engine == "claude" for heartbeat in heartbeats))

    def test_parallel_turns_form_one_foreground_timeline(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_jsonl(
                root / "codex.jsonl",
                codex_rows(end="2026-01-02T10:06:00Z"),
            )
            write_jsonl(
                root / "claude.jsonl",
                [
                    {
                        "timestamp": "2026-01-02T10:02:00Z",
                        "type": "user",
                        "uuid": "prompt-2",
                        "sessionId": "claude-2",
                        "cwd": "/private/example/project",
                        "message": {"content": "sentinel prompt"},
                    },
                    {
                        "timestamp": "2026-01-02T10:04:00Z",
                        "type": "assistant",
                        "message": {"stop_reason": "end_turn", "content": []},
                    },
                ],
            )

            heartbeats, _ = self.collect(
                [Source("codex", root), Source("claude", root)],
                cadence_seconds=120,
            )

            self.assertEqual(
                [(heartbeat.timestamp, heartbeat.engine) for heartbeat in heartbeats],
                [
                    (1767348000.0, "codex"),
                    (1767348120.0, "claude"),
                    (1767348240.0, "codex"),
                    (1767348360.0, "codex"),
                ],
            )

    def test_open_turn_is_capped_by_last_activity_grace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rows = codex_rows()[:2]
            rows.append(
                {
                    "timestamp": "2026-01-02T10:08:00Z",
                    "type": "event_msg",
                    "payload": {"type": "token_count", "turn_id": "turn-1"},
                }
            )
            write_jsonl(root / "open.jsonl", rows)

            heartbeats, _ = self.collect(
                [Source("codex", root)],
                cadence_seconds=120,
                active_grace_seconds=900,
            )

            self.assertEqual(heartbeats[-1].timestamp, 1767348600.0)
            self.assertEqual(len(heartbeats), 6)

    def test_new_codex_turn_closes_an_abandoned_open_turn(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rows = codex_rows()[:2]
            rows.extend(
                [
                    {
                        "timestamp": "2026-01-02T10:03:00Z",
                        "type": "event_msg",
                        "payload": {"type": "task_started", "turn_id": "turn-2"},
                    },
                    {
                        "timestamp": "2026-01-02T10:05:00Z",
                        "type": "event_msg",
                        "payload": {"type": "task_complete", "turn_id": "turn-2"},
                    },
                ]
            )
            write_jsonl(root / "abandoned.jsonl", rows)

            heartbeats, _ = self.collect([Source("codex", root)], cadence_seconds=120)

            self.assertEqual(
                [heartbeat.timestamp for heartbeat in heartbeats],
                [1767348000.0, 1767348120.0, 1767348180.0, 1767348300.0],
            )

    def test_partial_jsonl_keeps_complete_events(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_jsonl(root / "partial.jsonl", codex_rows(), trailing='{"unfinished":')

            heartbeats, stats = self.collect([Source("codex", root)])

            self.assertEqual(len(heartbeats), 4)
            self.assertEqual(stats.malformed_lines, 1)


class PrivacyAndStateTests(unittest.TestCase):
    def test_outgoing_payload_has_a_small_privacy_allowlist(self) -> None:
        heartbeat = ActivityHeartbeat(
            fingerprint="f" * 32,
            timestamp=1767348000.0,
            engine="codex",
            project="project-opaque",
            session_key="session-opaque",
        )

        payload = heartbeat_payload(heartbeat)
        encoded = json.dumps(payload, sort_keys=True)

        self.assertEqual(
            set(payload),
            {"entity", "type", "category", "time", "project", "plugin", "ai_session"},
        )
        for sentinel in (
            "/private/example/project",
            "private prompt",
            "private response",
            "session-native-id",
            "turn-native-id",
            "rm -rf private-command",
        ):
            self.assertNotIn(sentinel, encoded)

    def test_legacy_state_is_scrubbed_and_new_state_has_no_private_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "state.db"
            sentinel = "/private/sentinel/workspace"
            connection = sqlite3.connect(db_path)
            connection.execute(
                "CREATE TABLE sent_events "
                "(call_id TEXT PRIMARY KEY, timestamp TEXT, session_path TEXT, sent_at TEXT, mode TEXT)"
            )
            connection.execute(
                "INSERT INTO sent_events VALUES (?, ?, ?, ?, ?)",
                ("call", "time", sentinel, "time", "cli"),
            )
            connection.commit()
            connection.close()

            state = StateDB(db_path)
            heartbeat = ActivityHeartbeat("f" * 32, 1767348000.0, "codex", "p", "s")
            state.mark_sent([heartbeat], "api")
            self.assertTrue(state.is_sent(heartbeat.fingerprint))
            state.close()

            connection = sqlite3.connect(db_path)
            tables = {
                row[0]
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(sent_heartbeats)")
            }
            connection.close()
            self.assertNotIn("sent_events", tables)
            self.assertEqual(columns, {"fingerprint", "timestamp", "sent_at", "mode"})
            self.assertNotIn(sentinel.encode(), db_path.read_bytes())

    def test_failed_delivery_remains_retryable_and_logs_hide_transcript_data(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sentinel = "private-prompt-sentinel"
            rows = codex_rows(cwd=f"/private/{sentinel}", end="2026-01-02T10:02:00Z")
            rows.insert(
                2,
                {
                    "timestamp": "2026-01-02T10:01:00Z",
                    "type": "event_msg",
                    "payload": {"type": "user_message", "message": sentinel},
                },
            )
            write_jsonl(root / "session.jsonl", rows)
            state = StateDB(root / "state.db")
            stream = io.StringIO()
            logger = logging.getLogger(f"test-{id(self)}")
            logger.handlers = [logging.StreamHandler(stream)]
            logger.setLevel(logging.INFO)
            logger.propagate = False

            failed = sync_activity(
                [Source("codex", root)],
                state=state,
                since=dt.datetime(2026, 1, 2, 9, 59, tzinfo=UTC),
                now=dt.datetime(2026, 1, 2, 10, 10, tzinfo=UTC),
                privacy_key=PRIVACY_KEY,
                sender=lambda _batch: False,
                logger=logger,
            )
            succeeded = sync_activity(
                [Source("codex", root)],
                state=state,
                since=dt.datetime(2026, 1, 2, 9, 59, tzinfo=UTC),
                now=dt.datetime(2026, 1, 2, 10, 10, tzinfo=UTC),
                privacy_key=PRIVACY_KEY,
                sender=lambda _batch: True,
                logger=logger,
            )
            state.close()

            self.assertGreater(failed.failed, 0)
            self.assertEqual(failed.sent, 0)
            self.assertEqual(succeeded.failed, 0)
            self.assertEqual(succeeded.sent, failed.failed)
            self.assertNotIn(sentinel, stream.getvalue())
            self.assertNotIn(sentinel.encode(), (root / "state.db").read_bytes())

    def test_api_sender_posts_an_array_with_only_allowlisted_fields(self) -> None:
        heartbeat = ActivityHeartbeat(
            fingerprint="f" * 32,
            timestamp=1767348000.0,
            engine="claude",
            project="project-opaque",
            session_key="session-opaque",
        )
        captured = []

        class Response:
            status = 202

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        def open_request(request, timeout):
            captured.append((request, timeout))
            return Response()

        logger = logging.getLogger(f"api-test-{id(self)}")
        with mock.patch("urllib.request.urlopen", side_effect=open_request):
            sender = make_api_sender(
                "secret-api-key",
                "https://api.wakatime.com/api/v1",
                logger,
            )
            self.assertTrue(sender([heartbeat]))

        request, timeout = captured[0]
        body = json.loads(request.data)
        self.assertEqual(timeout, 30)
        self.assertIsInstance(body, list)
        self.assertEqual(body, [heartbeat_payload(heartbeat)])
        self.assertEqual(
            request.full_url,
            "https://api.wakatime.com/api/v1/users/current/heartbeats.bulk",
        )


class SourceConfigTests(unittest.TestCase):
    def test_source_file_and_cli_globs_expand_and_deduplicate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            codex = root / "codex"
            claude_one = root / "claude-one"
            claude_two = root / "claude-two"
            for directory in (codex, claude_one, claude_two):
                directory.mkdir()
            config = root / "sources.conf"
            config.write_text(
                f"# transcript roots\ncodex={codex}\nclaude={root}/claude-*\n",
                encoding="utf-8",
            )

            sources = load_sources([f"codex:{codex}"], config)

            self.assertEqual(
                [(source.engine, source.directory) for source in sources],
                [
                    ("claude", claude_one.resolve()),
                    ("claude", claude_two.resolve()),
                    ("codex", codex.resolve()),
                ],
            )


if __name__ == "__main__":
    unittest.main()
