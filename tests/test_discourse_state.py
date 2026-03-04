"""Tests for discourse state persistence."""

from datetime import datetime, timezone, timedelta

import pytest

from athena.discourse.state import (
    DiscourseThreadState,
    BotRateLimitState,
    load_state,
    save_state,
    count_recent_replies,
    record_reply,
)


class TestLoadSaveState:
    """Tests for round-trip serialization of thread and rate limit state."""

    def test_roundtrip_empty(self, tmp_path):
        """Empty state round-trips correctly."""
        state_file = tmp_path / "state.json"
        threads: dict = {}
        rate = BotRateLimitState()
        save_state(state_file, threads, rate)
        loaded_threads, loaded_rate = load_state(state_file)
        assert loaded_threads == {}
        assert loaded_rate.reply_timestamps == []

    def test_roundtrip_with_thread(self, tmp_path):
        """A thread with populated fields survives serialization."""
        state_file = tmp_path / "state.json"
        thread = DiscourseThreadState(
            topic_id=42,
            topic_title="Test Thread",
            last_seen_post_number=7,
            our_post_ids=[101, 102],
            triage_score=8,
        )
        threads = {42: thread}
        rate = BotRateLimitState()
        save_state(state_file, threads, rate)
        loaded_threads, _ = load_state(state_file)
        assert 42 in loaded_threads
        assert loaded_threads[42].topic_title == "Test Thread"
        assert loaded_threads[42].last_seen_post_number == 7
        assert loaded_threads[42].our_post_ids == [101, 102]

    def test_roundtrip_with_conversation_messages(self, tmp_path):
        """Conversation messages are preserved through round-trip."""
        state_file = tmp_path / "state.json"
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        thread = DiscourseThreadState(
            topic_id=1,
            topic_title="Chat",
            last_seen_post_number=2,
            conversation_messages=msgs,
        )
        save_state(state_file, {1: thread}, BotRateLimitState())
        loaded_threads, _ = load_state(state_file)
        assert loaded_threads[1].conversation_messages == msgs

    def test_missing_file_returns_empty(self, tmp_path):
        """A non-existent state file returns empty structures."""
        loaded_threads, loaded_rate = load_state(tmp_path / "nonexistent.json")
        assert loaded_threads == {}
        assert loaded_rate.reply_timestamps == []

    def test_invalid_json_returns_empty(self, tmp_path):
        """A corrupted state file returns empty structures."""
        bad = tmp_path / "bad.json"
        bad.write_text("not json{{{")
        loaded_threads, loaded_rate = load_state(bad)
        assert loaded_threads == {}
        assert loaded_rate.reply_timestamps == []


class TestRateLimiting:
    """Tests for reply rate limiting helpers."""

    def test_count_recent_replies_empty(self):
        """An empty rate limit state returns zero."""
        rate = BotRateLimitState()
        assert count_recent_replies(rate, window_hours=1) == 0

    def test_record_and_count(self):
        """Recording replies increments the count."""
        rate = BotRateLimitState()
        record_reply(rate)
        record_reply(rate)
        assert count_recent_replies(rate, window_hours=1) == 2
        assert count_recent_replies(rate, window_hours=24) == 2

    def test_pruning_old_timestamps(self):
        """Timestamps older than 48 hours are pruned on record_reply."""
        rate = BotRateLimitState()
        old_ts = (datetime.now(timezone.utc) - timedelta(hours=49)).isoformat()
        rate.reply_timestamps.append(old_ts)
        record_reply(rate)
        assert old_ts not in rate.reply_timestamps
        assert len(rate.reply_timestamps) == 1

    def test_count_filters_by_window(self):
        """Only timestamps within the window are counted."""
        rate = BotRateLimitState()
        # Add a timestamp from 2 hours ago
        two_hours_ago = (
            datetime.now(timezone.utc) - timedelta(hours=2)
        ).isoformat()
        rate.reply_timestamps.append(two_hours_ago)
        # Add a fresh timestamp
        record_reply(rate)
        assert count_recent_replies(rate, window_hours=1) == 1
        assert count_recent_replies(rate, window_hours=24) == 2
