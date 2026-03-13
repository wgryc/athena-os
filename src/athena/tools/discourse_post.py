"""Tools for interacting with Discourse forums from within a chat session."""

from __future__ import annotations

from typing import Any

from . import Tool
from ..discourse.api import DiscourseClient


class CreateDiscoursePost(Tool):
    """Create a new topic on the Discourse forum.

    Args:
        client: Authenticated Discourse API client.
        category_id: Category to post into.
    """

    def __init__(self, client: DiscourseClient, category_id: int) -> None:
        self._client = client
        self._category_id = category_id

    @property
    def name(self) -> str:
        return "create_discourse_post"

    @property
    def description(self) -> str:
        return (
            "Create a new topic (post) on the Discourse forum. "
            "Provide a title and a markdown body. The post will be "
            "published to the configured forum category."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "The topic title. Should be concise and descriptive.",
                },
                "body": {
                    "type": "string",
                    "description": "The full markdown body of the post.",
                },
            },
            "required": ["title", "body"],
        }

    def execute(self, **kwargs: Any) -> str:
        title = kwargs.get("title", "")
        body = kwargs.get("body", "")

        if not title or not body:
            return "Error: both 'title' and 'body' are required."

        try:
            topic = self._client.create_topic(
                title=title,
                raw=body,
                category_id=self._category_id,
            )
            return (
                f"Successfully created topic #{topic.topic_id}: \"{topic.title}\" "
                f"with {topic.posts_count} post(s)."
            )
        except Exception as e:
            return f"Error creating Discourse topic: {e}"


class ListDiscourseTopics(Tool):
    """List recent topics in the configured Discourse category.

    Args:
        client: Authenticated Discourse API client.
        category_id: Category to list topics from.
    """

    def __init__(self, client: DiscourseClient, category_id: int) -> None:
        self._client = client
        self._category_id = category_id

    @property
    def name(self) -> str:
        return "list_discourse_topics"

    @property
    def description(self) -> str:
        return (
            "List recent topics in the Discourse forum category. "
            "Returns topic IDs, titles, post counts, and last activity timestamps. "
            "Use a topic_id with read_discourse_topic to see the full posts."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    def execute(self, **kwargs: Any) -> str:
        try:
            topics = self._client.list_topics(self._category_id)
            if not topics:
                return "No topics found in this category."
            lines = []
            for t in topics:
                lines.append(
                    f"- Topic #{t.topic_id}: \"{t.title}\" "
                    f"({t.posts_count} posts, last activity: {t.last_posted_at})"
                )
            return "\n".join(lines)
        except Exception as e:
            return f"Error listing Discourse topics: {e}"


class ReadDiscourseTopic(Tool):
    """Read all posts in a specific Discourse topic.

    Args:
        client: Authenticated Discourse API client.
    """

    def __init__(self, client: DiscourseClient) -> None:
        self._client = client

    @property
    def name(self) -> str:
        return "read_discourse_topic"

    @property
    def description(self) -> str:
        return (
            "Read all posts in a Discourse topic by its topic_id. "
            "Returns each post's author, content, and timestamp. "
            "Use list_discourse_topics first to find topic IDs."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "topic_id": {
                    "type": "integer",
                    "description": "The Discourse topic ID to read.",
                },
            },
            "required": ["topic_id"],
        }

    def execute(self, **kwargs: Any) -> str:
        topic_id = kwargs.get("topic_id")
        if not topic_id:
            return "Error: 'topic_id' is required."

        try:
            topic = self._client.get_topic(int(topic_id))
            lines = [f"Topic #{topic.topic_id}: \"{topic.title}\" ({topic.posts_count} posts)\n"]
            for p in topic.posts:
                lines.append(
                    f"--- Post #{p.post_number} by {p.username} ({p.created_at}) ---\n"
                    f"{p.raw}\n"
                )
            return "\n".join(lines)
        except Exception as e:
            return f"Error reading Discourse topic: {e}"
