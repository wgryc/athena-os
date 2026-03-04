"""Thread state persistence for the Discourse bot."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class DiscourseThreadState:
    """State for a single Discourse topic the bot has seen or participated in.

    Args:
        topic_id: Discourse topic ID.
        topic_title: Human-readable topic title.
        last_seen_post_number: Highest post_number we have processed.
        our_post_ids: Discourse post IDs created by the bot in this thread.
        conversation_messages: LLM conversation history for this thread.
        first_seen: ISO timestamp when this topic was first seen.
        last_engaged: ISO timestamp of the bot's most recent reply, or None.
        triage_score: Most recent triage score (0-10).
    """

    topic_id: int
    topic_title: str
    last_seen_post_number: int
    our_post_ids: list[int] = field(default_factory=list)
    conversation_messages: list[dict] = field(default_factory=list)
    first_seen: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    last_engaged: str | None = None
    triage_score: int = 0


@dataclass
class BotRateLimitState:
    """Tracks reply counts for rate limiting.

    Args:
        reply_timestamps: ISO timestamps of replies posted. Used to calculate
            hourly and daily counts by filtering within time windows.
    """

    reply_timestamps: list[str] = field(default_factory=list)


def load_state(
    state_file: str | Path,
) -> tuple[dict[int, DiscourseThreadState], BotRateLimitState]:
    """Load thread state and rate limit state from a JSON file.

    Args:
        state_file: Path to the state JSON file.

    Returns:
        Tuple of (threads_dict, rate_limit_state). ``threads_dict`` is keyed
        by ``topic_id`` (int). Returns empty structures if the file does not
        exist or cannot be parsed.
    """
    path = Path(state_file)
    if not path.exists():
        return {}, BotRateLimitState()

    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}, BotRateLimitState()

    threads: dict[int, DiscourseThreadState] = {}
    for topic_id_str, thread_data in data.get("threads", {}).items():
        topic_id = int(topic_id_str)
        threads[topic_id] = DiscourseThreadState(**thread_data)

    rate_data = data.get("rate_limit", {})
    rate_limit = BotRateLimitState(
        reply_timestamps=rate_data.get("reply_timestamps", [])
    )

    return threads, rate_limit


def save_state(
    state_file: str | Path,
    threads: dict[int, DiscourseThreadState],
    rate_limit: BotRateLimitState,
) -> None:
    """Persist thread state and rate limit state to a JSON file.

    Args:
        state_file: Path to write the state JSON file.
        threads: Dict of ``topic_id`` -> ``DiscourseThreadState``.
        rate_limit: Current rate limit tracking state.
    """
    path = Path(state_file)
    data = {
        "threads": {str(tid): asdict(ts) for tid, ts in threads.items()},
        "rate_limit": asdict(rate_limit),
    }
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")


def count_recent_replies(rate_limit: BotRateLimitState, window_hours: int) -> int:
    """Count replies within the last N hours.

    Args:
        rate_limit: Current rate limit state.
        window_hours: How many hours back to count.

    Returns:
        Number of replies within the window.
    """
    now = datetime.now(timezone.utc)
    count = 0
    for ts_str in rate_limit.reply_timestamps:
        try:
            ts = datetime.fromisoformat(ts_str)
            if (now - ts).total_seconds() <= window_hours * 3600:
                count += 1
        except (ValueError, TypeError):
            continue
    return count


def record_reply(rate_limit: BotRateLimitState) -> None:
    """Record a new reply timestamp and prune old entries.

    Adds the current UTC time to ``reply_timestamps`` and removes any entries
    older than 48 hours to prevent unbounded growth.

    Args:
        rate_limit: Rate limit state to update in place.
    """
    now = datetime.now(timezone.utc)
    rate_limit.reply_timestamps.append(now.isoformat())
    rate_limit.reply_timestamps = [
        ts
        for ts in rate_limit.reply_timestamps
        if (now - datetime.fromisoformat(ts)).total_seconds() <= 48 * 3600
    ]
