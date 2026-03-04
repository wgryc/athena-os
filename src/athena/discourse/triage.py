"""Stage 1 triage: score a Discourse thread for relevance and value-add."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import litellm
from jinja2 import Environment, FileSystemLoader

TEMPLATES_DIR = Path(__file__).parent / "templates"
_jinja_env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))

DEFAULT_TRIAGE_MODEL = os.getenv("DEFAULT_LLM_MODEL", "anthropic/claude-haiku-4-5-20251001")


@dataclass
class TriageResult:
    """Result of a triage call.

    Args:
        score: Overall engagement score 0-10.
        relevance: Relevance sub-score 0-10.
        value_add: Value-add sub-score 0-10.
        reason: Human-readable explanation.
        raw_response: The raw LLM response text (for debugging).
    """

    score: int
    relevance: int
    value_add: int
    reason: str
    raw_response: str


def score_thread(
    topic_title: str,
    recent_posts: list[dict[str, str]],
    personality_summary: str,
    category_name: str = "",
    model: str = DEFAULT_TRIAGE_MODEL,
) -> TriageResult:
    """Score a Discourse thread for whether the bot should engage.

    Calls a cheap LLM (default: Haiku) with a triage prompt rendered from
    ``triage_prompt.j2``. Parses the JSON response into a ``TriageResult``.
    Falls back to ``score=0`` if parsing fails.

    Args:
        topic_title: Title of the Discourse topic.
        recent_posts: List of dicts with ``username`` and ``content`` keys,
            representing the most recent posts (up to ~5).
        personality_summary: Short description of the bot's domain/expertise.
        category_name: Human-readable name of the Discourse category.
        model: LiteLLM model string for the triage call.

    Returns:
        ``TriageResult`` with score and reasoning.
    """
    template = _jinja_env.get_template("triage_prompt.j2")
    prompt = template.render(
        personality_summary=personality_summary,
        category_name=category_name,
        topic_title=topic_title,
        recent_posts=recent_posts,
    )

    try:
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
        )
        choice = response.choices[0]
        raw = choice.message.content or ""

        # Some models return empty content — log for debugging
        if not raw.strip():
            print(
                f"[Triage] Warning: empty content from {model}. "
                f"finish_reason={choice.finish_reason}, "
                f"message={choice.message}",
                flush=True,
            )
    except Exception as e:
        return TriageResult(
            score=0,
            relevance=0,
            value_add=0,
            reason=f"Triage LLM call failed: {e}",
            raw_response="",
        )

    return _parse_triage_response(raw)


def _parse_triage_response(raw: str) -> TriageResult:
    """Parse the JSON triage response from the LLM.

    Handles the case where the LLM wraps the JSON in a markdown code block.
    Falls back to ``score=0`` on any parse failure.

    Args:
        raw: Raw LLM response string.

    Returns:
        ``TriageResult`` parsed from the JSON, or a zero-score fallback.
    """
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

    try:
        data = json.loads(cleaned)
        return TriageResult(
            score=int(data.get("score", 0)),
            relevance=int(data.get("relevance", 0)),
            value_add=int(data.get("value_add", 0)),
            reason=str(data.get("reason", "")),
            raw_response=raw,
        )
    except (json.JSONDecodeError, ValueError, TypeError):
        return TriageResult(
            score=0,
            relevance=0,
            value_add=0,
            reason=f"Failed to parse triage response: {raw[:200]}",
            raw_response=raw,
        )
