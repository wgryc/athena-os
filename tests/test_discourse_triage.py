"""Tests for discourse triage scoring."""

from unittest.mock import patch, MagicMock

from athena.discourse.triage import score_thread, _parse_triage_response, TriageResult


class TestParseTriageResponse:
    """Tests for parsing the JSON triage response."""

    def test_valid_json(self):
        """A clean JSON response parses correctly."""
        raw = '{"relevance": 8, "value_add": 7, "score": 8, "reason": "Relevant topic"}'
        result = _parse_triage_response(raw)
        assert result.score == 8
        assert result.relevance == 8
        assert result.value_add == 7
        assert "Relevant" in result.reason

    def test_json_in_code_fence(self):
        """JSON wrapped in markdown code fences is handled."""
        raw = '```json\n{"relevance": 5, "value_add": 5, "score": 5, "reason": "Neutral"}\n```'
        result = _parse_triage_response(raw)
        assert result.score == 5

    def test_json_in_bare_code_fence(self):
        """JSON wrapped in bare triple backticks is handled."""
        raw = '```\n{"relevance": 3, "value_add": 2, "score": 3, "reason": "Low relevance"}\n```'
        result = _parse_triage_response(raw)
        assert result.score == 3

    def test_invalid_json_returns_zero(self):
        """Unparseable responses fall back to score=0."""
        result = _parse_triage_response("This is not JSON at all")
        assert result.score == 0
        assert "Failed to parse" in result.reason

    def test_missing_fields_default_to_zero(self):
        """Missing sub-scores default to 0."""
        raw = '{"score": 6}'
        result = _parse_triage_response(raw)
        assert result.score == 6
        assert result.relevance == 0
        assert result.value_add == 0

    def test_raw_response_preserved(self):
        """The raw LLM response is stored on the result."""
        raw = '{"relevance": 9, "value_add": 9, "score": 9, "reason": "Perfect"}'
        result = _parse_triage_response(raw)
        assert result.raw_response == raw


class TestScoreThread:
    """Tests for the full score_thread function with mocked LLM."""

    def test_successful_score(self):
        """A successful LLM call returns the parsed score."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = (
            '{"relevance": 9, "value_add": 8, "score": 9, "reason": "Highly relevant"}'
        )
        with patch(
            "athena.discourse.triage.litellm.completion",
            return_value=mock_response,
        ):
            result = score_thread(
                topic_title="Apple stock analysis Q4",
                recent_posts=[
                    {"username": "user1", "content": "What about AAPL?"}
                ],
                personality_summary="Financial analyst specializing in equities",
            )
        assert result.score == 9
        assert result.relevance == 9

    def test_llm_failure_returns_zero(self):
        """An LLM error falls back to score=0."""
        with patch(
            "athena.discourse.triage.litellm.completion",
            side_effect=Exception("API error"),
        ):
            result = score_thread(
                topic_title="Some topic",
                recent_posts=[],
                personality_summary="Financial analyst",
            )
        assert result.score == 0
        assert "failed" in result.reason.lower()


class TestParseTopicResponse:
    """Tests for parsing the TITLE:/--- format from the post command."""

    def test_standard_format(self):
        """Standard TITLE:/---/body format parses correctly."""
        from athena.discourse.worker import _parse_topic_response

        text = "TITLE: My Great Topic\n---\nHere is the body.\n\nSecond paragraph."
        title, body = _parse_topic_response(text)
        assert title == "My Great Topic"
        assert body == "Here is the body.\n\nSecond paragraph."

    def test_no_separator(self):
        """TITLE: without --- separator still works."""
        from athena.discourse.worker import _parse_topic_response

        text = "TITLE: Quick Note\nJust a short body."
        title, body = _parse_topic_response(text)
        assert title == "Quick Note"
        assert body == "Just a short body."

    def test_fallback_no_title_prefix(self):
        """Without TITLE: prefix, first line becomes the title."""
        from athena.discourse.worker import _parse_topic_response

        text = "Some heading\nSome body text."
        title, body = _parse_topic_response(text)
        assert title == "Some heading"
        assert body == "Some body text."
