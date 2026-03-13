"""Tests for Discourse REST API client."""

from unittest.mock import patch, MagicMock

import pytest

from athena.discourse.api import DiscourseClient, DiscoursePost, DiscourseTopic


@pytest.fixture
def client():
    """Create a DiscourseClient with test credentials."""
    return DiscourseClient(
        base_url="https://forum.example.com",
        api_key="test-api-key",
        bot_username="athena-bot",
    )


class TestDiscourseClient:
    """Tests for the DiscourseClient HTTP methods."""

    def test_headers_set(self, client):
        """Auth and content-type headers are set on the session."""
        headers = client._session.headers
        assert headers["Api-Key"] == "test-api-key"
        assert headers["Api-Username"] == "athena-bot"
        assert headers["Content-Type"] == "application/json"

    def test_trailing_slash_stripped(self):
        """Trailing slash on base_url is stripped."""
        c = DiscourseClient(
            base_url="https://forum.example.com/",
            api_key="k",
            bot_username="u",
        )
        assert c._base_url == "https://forum.example.com"

    def test_list_topics(self, client):
        """list_topics parses the topic_list response."""
        mock_data = {
            "topic_list": {
                "topics": [
                    {
                        "id": 10,
                        "title": "First Topic",
                        "posts_count": 3,
                        "last_posted_at": "2025-01-01T12:00:00Z",
                    },
                    {
                        "id": 20,
                        "title": "Second Topic",
                        "posts_count": 1,
                        "last_posted_at": "2025-01-02T12:00:00Z",
                    },
                ]
            }
        }
        with patch.object(client, "_get", return_value=mock_data):
            topics = client.list_topics(category_id=5)
        assert len(topics) == 2
        assert topics[0].topic_id == 10
        assert topics[0].title == "First Topic"
        assert topics[0].posts == []
        assert topics[1].topic_id == 20

    def test_get_topic(self, client):
        """get_topic parses posts from the post_stream."""
        mock_data = {
            "title": "Test Thread",
            "posts_count": 2,
            "last_posted_at": "2025-01-01T12:00:00Z",
            "post_stream": {
                "posts": [
                    {
                        "id": 100,
                        "post_number": 1,
                        "username": "alice",
                        "raw": "Hello world",
                        "created_at": "2025-01-01T10:00:00Z",
                    },
                    {
                        "id": 101,
                        "post_number": 2,
                        "username": "bob",
                        "cooked": "<p>Hi Alice</p>",
                        "created_at": "2025-01-01T11:00:00Z",
                    },
                ]
            },
        }
        with patch.object(client, "_get", return_value=mock_data):
            topic = client.get_topic(topic_id=42)
        assert topic.title == "Test Thread"
        assert len(topic.posts) == 2
        assert topic.posts[0].raw == "Hello world"
        # Falls back to cooked when raw is absent
        assert topic.posts[1].raw == "<p>Hi Alice</p>"

    def test_create_reply(self, client):
        """create_reply sends correct body and parses the response."""
        mock_data = {
            "id": 200,
            "post_number": 3,
            "username": "athena-bot",
            "created_at": "2025-01-01T12:00:00Z",
        }
        with patch.object(client, "_post", return_value=mock_data) as mock_post:
            post = client.create_reply(
                topic_id=42,
                raw="My reply",
                reply_to_post_number=2,
            )
        mock_post.assert_called_once_with(
            "/posts",
            {"topic_id": 42, "raw": "My reply", "reply_to_post_number": 2},
        )
        assert post.post_id == 200
        assert post.post_number == 3
        assert post.raw == "My reply"

    def test_create_reply_without_reply_to(self, client):
        """create_reply omits reply_to_post_number when None."""
        mock_data = {
            "id": 201,
            "post_number": 1,
            "username": "athena-bot",
            "created_at": "2025-01-01T12:00:00Z",
        }
        with patch.object(client, "_post", return_value=mock_data) as mock_post:
            client.create_reply(topic_id=42, raw="First post")
        mock_post.assert_called_once_with(
            "/posts",
            {"topic_id": 42, "raw": "First post"},
        )

    def test_create_topic(self, client):
        """create_topic sends title, raw, and category and parses the response."""
        mock_data = {
            "id": 300,
            "topic_id": 55,
            "post_number": 1,
            "username": "athena-bot",
            "created_at": "2025-01-03T10:00:00Z",
        }
        with patch.object(client, "_post", return_value=mock_data) as mock_post:
            topic = client.create_topic(
                title="New Analysis",
                raw="Here is my analysis...",
                category_id=5,
            )
        mock_post.assert_called_once_with(
            "/posts",
            {"title": "New Analysis", "raw": "Here is my analysis...", "category": 5},
        )
        assert topic.topic_id == 55
        assert topic.title == "New Analysis"
        assert len(topic.posts) == 1
        assert topic.posts[0].post_id == 300
        assert topic.posts[0].raw == "Here is my analysis..."
