"""Discourse REST API client."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class DiscoursePost:
    """A single post within a Discourse topic.

    Args:
        post_id: Discourse-assigned post ID.
        post_number: Sequential post number within the topic (1-indexed).
        username: Author's Discourse username.
        raw: Markdown source of the post (may fall back to HTML ``cooked``).
        created_at: ISO timestamp string from the API.
        topic_id: Parent topic ID.
    """

    post_id: int
    post_number: int
    username: str
    raw: str
    created_at: str
    topic_id: int


@dataclass
class DiscourseTopic:
    """Metadata and posts for a Discourse topic.

    Args:
        topic_id: Discourse-assigned topic ID.
        title: Topic title.
        posts_count: Total number of posts in the topic.
        last_posted_at: ISO timestamp of the most recent post.
        posts: List of posts (populated by ``get_topic``; empty from ``list_topics``).
    """

    topic_id: int
    title: str
    posts_count: int
    last_posted_at: str
    posts: list[DiscoursePost]


class DiscourseClient:
    """Authenticated Discourse REST API client.

    Args:
        base_url: Base URL of the Discourse instance (no trailing slash).
        api_key: Discourse API key (typically from ``DISCOURSE_API_KEY`` env var).
        bot_username: Username the bot posts as.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        bot_username: str,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.headers.update({
            "Api-Key": api_key,
            "Api-Username": bot_username,
            "Content-Type": "application/json",
        })

    def _get(self, path: str, params: dict | None = None) -> dict[str, Any]:
        """Execute a GET request and return parsed JSON.

        Args:
            path: URL path (e.g. ``/c/5.json``).
            params: Optional query parameters.

        Returns:
            Parsed JSON response body.

        Raises:
            requests.HTTPError: On non-2xx responses.
        """
        url = f"{self._base_url}{path}"
        response = self._session.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        """Execute a POST request and return parsed JSON.

        Args:
            path: URL path (e.g. ``/posts``).
            body: Request body dict (will be JSON-encoded).

        Returns:
            Parsed JSON response body.

        Raises:
            requests.HTTPError: On non-2xx responses.
        """
        url = f"{self._base_url}{path}"
        response = self._session.post(url, json=body, timeout=30)
        response.raise_for_status()
        return response.json()

    def list_topics(self, category_id: int) -> list[DiscourseTopic]:
        """Fetch recent topics in a category (metadata only, no posts).

        Calls ``GET /c/{category_id}.json``. Returns topics sorted by
        ``last_posted_at`` descending (Discourse default).

        Args:
            category_id: Discourse category ID.

        Returns:
            List of ``DiscourseTopic`` objects with empty ``posts`` lists.
        """
        data = self._get(f"/c/{category_id}.json")
        topics: list[DiscourseTopic] = []
        for t in data.get("topic_list", {}).get("topics", []):
            topics.append(DiscourseTopic(
                topic_id=t["id"],
                title=t["title"],
                posts_count=t.get("posts_count", 0),
                last_posted_at=t.get("last_posted_at", ""),
                posts=[],
            ))
        return topics

    def get_topic(self, topic_id: int) -> DiscourseTopic:
        """Fetch a topic with all its posts.

        Calls ``GET /t/{topic_id}.json``. The ``raw`` field is preferred for
        post content; falls back to ``cooked`` (HTML) when ``raw`` is absent.

        Args:
            topic_id: Discourse topic ID.

        Returns:
            ``DiscourseTopic`` with populated ``posts`` list.
        """
        data = self._get(f"/t/{topic_id}.json")
        posts: list[DiscoursePost] = []
        for p in data.get("post_stream", {}).get("posts", []):
            posts.append(DiscoursePost(
                post_id=p["id"],
                post_number=p["post_number"],
                username=p["username"],
                raw=p.get("raw", p.get("cooked", "")),
                created_at=p.get("created_at", ""),
                topic_id=topic_id,
            ))
        return DiscourseTopic(
            topic_id=topic_id,
            title=data.get("title", ""),
            posts_count=data.get("posts_count", 0),
            last_posted_at=data.get("last_posted_at", ""),
            posts=posts,
        )

    def create_topic(
        self,
        title: str,
        raw: str,
        category_id: int,
    ) -> DiscourseTopic:
        """Create a new topic in a category.

        Calls ``POST /posts`` with a ``title`` and ``category`` to start a
        new thread.

        Args:
            title: Topic title.
            raw: Markdown body of the opening post.
            category_id: Discourse category ID to create the topic in.

        Returns:
            ``DiscourseTopic`` with the opening post in its ``posts`` list.
        """
        body: dict[str, Any] = {
            "title": title,
            "raw": raw,
            "category": category_id,
        }
        data = self._post("/posts", body)
        topic_id = data["topic_id"]
        opening_post = DiscoursePost(
            post_id=data["id"],
            post_number=data["post_number"],
            username=data["username"],
            raw=raw,
            created_at=data.get("created_at", ""),
            topic_id=topic_id,
        )
        return DiscourseTopic(
            topic_id=topic_id,
            title=title,
            posts_count=1,
            last_posted_at=opening_post.created_at,
            posts=[opening_post],
        )

    def create_reply(
        self,
        topic_id: int,
        raw: str,
        reply_to_post_number: int | None = None,
    ) -> DiscoursePost:
        """Post a reply to a topic.

        Calls ``POST /posts``.

        Args:
            topic_id: Topic to reply to.
            raw: Markdown body of the reply.
            reply_to_post_number: Specific post number to reply to (optional).

        Returns:
            The created ``DiscoursePost``.
        """
        body: dict[str, Any] = {"topic_id": topic_id, "raw": raw}
        if reply_to_post_number is not None:
            body["reply_to_post_number"] = reply_to_post_number
        data = self._post("/posts", body)
        return DiscoursePost(
            post_id=data["id"],
            post_number=data["post_number"],
            username=data["username"],
            raw=raw,
            created_at=data.get("created_at", ""),
            topic_id=topic_id,
        )
