"""Discourse bot module for AthenaOS."""

from .api import DiscourseClient, DiscoursePost, DiscourseTopic
from .state import (
    DiscourseThreadState,
    BotRateLimitState,
    load_state,
    save_state,
    count_recent_replies,
    record_reply,
)
from .triage import score_thread, TriageResult
from .agent import build_system_prompt, generate_reply

__all__ = [
    "DiscourseClient",
    "DiscoursePost",
    "DiscourseTopic",
    "DiscourseThreadState",
    "BotRateLimitState",
    "load_state",
    "save_state",
    "count_recent_replies",
    "record_reply",
    "score_thread",
    "TriageResult",
    "build_system_prompt",
    "generate_reply",
]
