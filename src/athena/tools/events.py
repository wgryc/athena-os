"""Emerging Trajectories events feed tool."""

from __future__ import annotations

import json
import os
from typing import Any

import litellm
import requests
import tiktoken

from athena.tools import Tool, VisualTool

_ET_REQUEST_URL = "https://v2.emergingtrajectories.com/p/api/v0/get_events"

_ALL_PROJECT_CODES = "tracker_usa,commodities_tracker"

_MAX_TOKENS = 50_000


class ETEventsSearch(Tool):
    """Fetch recent events from the Emerging Trajectories events feed."""

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.environ.get("ET_API_KEY", "")

    @property
    def name(self) -> str:
        return "et_events_search"

    @property
    def label(self) -> str:
        return "Events Search"

    @property
    def description(self) -> str:
        return (
            "Fetch recent events from Emerging Trajectories. "
            "Returns events covering US stock market news and commodities news "
            "from the past 36 hours. Each event includes an ID, timestamp, and content."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": (
                        "Optional keyword to filter events. Only events whose content "
                        "contains this keyword (case-insensitive) will be returned."
                    ),
                },
            },
            "required": [],
        }

    def execute(self, **kwargs: Any) -> str:
        keyword: str | None = kwargs.get("keyword")

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "project_codes": _ALL_PROJECT_CODES,
            "hours": 36,
        }

        response = requests.post(_ET_REQUEST_URL, headers=headers, json=payload)

        if not response.ok:
            return f"Failed to fetch events (HTTP {response.status_code})."

        data = response.json().get("data", [])

        if not data:
            return "No events found."

        # Filter by keyword if provided
        if keyword:
            kw_lower = keyword.lower()
            data = [e for e in data if kw_lower in str(e).lower()]

        if not data:
            return f"No events matching '{keyword}'."

        enc = tiktoken.get_encoding("cl100k_base")
        header = f"Emerging Trajectories Events ({len(data)} results):\n"
        token_count = len(enc.encode(header))
        parts: list[str] = [header]
        included = 0

        for i, event in enumerate(data, 1):
            if isinstance(event, str):
                block = f"--- Event {i} ---\nContent:\n{event}\n"
            elif isinstance(event, dict):
                event_id = event.get("id", "N/A")
                timestamp = event.get("timestamp", event.get("created_at", "N/A"))
                content = event.get("content", event.get("text", "N/A"))
                block = (
                    f"--- Event {i} ---\n"
                    f"ID: {event_id}\n"
                    f"Timestamp: {timestamp}\n"
                    f"Content:\n{content}\n"
                )
            else:
                continue

            block_tokens = len(enc.encode(block))
            if token_count + block_tokens > _MAX_TOKENS:
                parts.append(
                    f"\n[Truncated: showing {included} of {len(data)} events "
                    f"to stay within {_MAX_TOKENS:,} token limit]"
                )
                break

            parts.append(block)
            token_count += block_tokens
            included += 1

        return "\n".join(parts)


_LLM_MODEL = "anthropic/claude-sonnet-4-5"


def _parse_event_line(event) -> tuple[str, str]:
    """Extract timestamp and content from an event.

    Handles both the string format ``"{id} on {timestamp}: {content}"``
    and dict format with ``timestamp``/``content`` keys.

    Args:
        event: An event as a dict (with ``timestamp`` and ``content`` keys)
            or a raw string in the ``"{id} on {timestamp}: {content}"`` format.

    Returns:
        A ``(timestamp, content)`` tuple. Either value may be an empty string
        if parsing fails.
    """
    if isinstance(event, dict):
        ts = event.get("timestamp", event.get("created_at", ""))
        content = event.get("content", event.get("text", ""))
        return ts, content
    text = str(event)
    # Format: "{id} on {timestamp}: {content}"
    on_idx = text.find(" on ")
    if on_idx != -1:
        rest = text[on_idx + 4:]  # after " on "
        colon_idx = rest.find(": ")
        if colon_idx != -1:
            return rest[:colon_idx].strip(), rest[colon_idx + 2:].strip()
    return "", text


class ETEventsWidget(VisualTool):
    """Dashboard widget that fetches ET events and uses an LLM to pick the most relevant ones."""

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.environ.get("ET_API_KEY", "")
        self._events: list[dict] = []
        self._description: str = ""
        self._count: int = 5

    @property
    def name(self) -> str:
        return "et_events_widget"

    @property
    def label(self) -> str:
        return "Top Events"

    @property
    def description(self) -> str:
        return "Display top relevant events from Emerging Trajectories, curated by an LLM."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Short description of events of interest (e.g. 'tech earnings surprises').",
                },
                "count": {
                    "type": "integer",
                    "description": "Number of top events to display (default 5).",
                },
            },
            "required": ["description"],
        }

    def _fetch_events(self) -> list:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "project_codes": _ALL_PROJECT_CODES,
            "hours": 36,
        }
        response = requests.post(_ET_REQUEST_URL, headers=headers, json=payload)
        if not response.ok:
            return []
        data = response.json().get("data", [])
        # The API returns a single string with one event per line,
        # formatted as: "{id} on {timestamp}: {content}"
        if isinstance(data, str):
            return [e.strip() for e in data.split("\n") if e.strip()]
        return data

    def _rank_events(self, raw_events: list, description: str, count: int) -> list[dict]:
        """Ask an LLM to pick the top events matching the description.

        Args:
            raw_events: List of event dicts or strings from the ET API.
            description: User-provided topic of interest for ranking.
            count: Maximum number of top events to return.

        Returns:
            A list of dicts with keys ``timestamp``, ``headline``,
            ``relevance``, and ``content``. Falls back to the first *count*
            events without curation if the LLM call fails.
        """
        # Cap events sent to the LLM to keep the prompt manageable
        _MAX_EVENTS_FOR_LLM = 200
        events_for_llm = raw_events[:_MAX_EVENTS_FOR_LLM]

        # Build a compact representation of events for the LLM
        event_summaries: list[str] = []
        for i, event in enumerate(events_for_llm):
            if isinstance(event, str):
                event_summaries.append(f"[{i}] {event[:400]}")
            elif isinstance(event, dict):
                ts = event.get("timestamp", event.get("created_at", ""))
                content = event.get("content", event.get("text", ""))
                event_summaries.append(f"[{i}] ({ts}) {content[:400]}")

        if not event_summaries:
            return []

        prompt = (
            f"You are an events curator. Below are {len(event_summaries)} recent events.\n"
            f"The user is interested in: \"{description}\"\n\n"
            f"Pick the top {count} most relevant and interesting events for this interest.\n"
            f"Return ONLY a JSON array of objects, each with:\n"
            f'  - "index": the [N] index from the list\n'
            f'  - "headline": a concise 1-sentence headline you write summarizing the event\n'
            f'  - "relevance": a short phrase explaining why this is relevant\n\n'
            f"Events:\n" + "\n".join(event_summaries)
        )

        try:
            response = litellm.completion(
                model=_LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
            )
            text = response.choices[0].message.content.strip()
            # Strip markdown fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                if text.endswith("```"):
                    text = text[: text.rfind("```")]
            picks = json.loads(text)
        except Exception as exc:
            # Fallback: return first N events without LLM curation
            print(f"[ETEventsWidget] LLM ranking failed: {exc}", flush=True)
            picks = []
            for i, event in enumerate(events_for_llm[:count]):
                ts, content = _parse_event_line(event)
                picks.append({
                    "index": i,
                    "headline": content[:120],
                    "relevance": "recent event",
                })

        # Resolve full event data for each pick
        result: list[dict] = []
        for pick in picks[:count]:
            idx = pick.get("index", 0)
            if idx < 0 or idx >= len(events_for_llm):
                continue
            ts, content = _parse_event_line(events_for_llm[idx])
            result.append({
                "timestamp": ts,
                "headline": pick.get("headline", content[:120]),
                "relevance": pick.get("relevance", ""),
                "content": content,
            })

        return result

    def execute(self, **kwargs: Any) -> str:
        self._description = kwargs.get("description", "")
        self._count = int(kwargs.get("count", 5))

        raw_events = self._fetch_events()
        if not raw_events:
            self._events = []
            return self.to_context()

        self._events = self._rank_events(raw_events, self._description, self._count)
        return self.to_context()

    def to_context(self) -> str:
        if not self._events:
            return f"(no events for \"{self._description}\")"
        lines = [f"Top {len(self._events)} curated events for \"{self._description}\":"]
        for i, e in enumerate(self._events, 1):
            ts = f" ({e['timestamp']})" if e.get("timestamp") else ""
            lines.append(
                f"\n  {i}. {e['headline']}{ts}\n"
                f"     Relevance: {e['relevance']}\n"
                f"     Detail: {e['content']}"
            )
        return "\n".join(lines)

    def to_html(self) -> str:
        if not self._events:
            return '<div class="widget-card widget-error">No events found</div>'

        desc_html = self._description.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        items = ""
        for e in self._events:
            headline = e["headline"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            relevance = e["relevance"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            ts = e.get("timestamp", "")
            ts_html = f'<span class="event-ts">{ts}</span>' if ts else ""
            items += (
                '<div class="event-item">'
                f'<div class="event-headline">{headline}</div>'
                f'<div class="event-meta">'
                f'<span class="event-relevance">{relevance}</span>'
                f'{ts_html}'
                f'</div>'
                '</div>'
            )

        return (
            '<div class="widget-card">'
            f'<h3 class="widget-title">Top Events</h3>'
            f'<div class="event-description">{desc_html}</div>'
            f'<div class="event-list">{items}</div>'
            '</div>'
        )
