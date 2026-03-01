"""Tests for frontend app helper functions."""

import pytest

from athena.frontend.app import (
    _wrap_user_message,
    _parse_llm_response,
    _estimate_token_count,
)


class TestWrapUserMessage:
    """Tests for _wrap_user_message()."""

    def test_basic_wrapping(self):
        result = _wrap_user_message("What's AAPL at?", "web")
        assert "[SYSTEM INFO]" in result
        assert "Source: web" in result
        assert "[MESSAGE]" in result
        assert "What's AAPL at?" in result

    def test_contains_datetime(self):
        result = _wrap_user_message("Hello", "web")
        assert "Current Date/Time:" in result

    def test_telegram_source(self):
        result = _wrap_user_message("Hello", "telegram")
        assert "Source: telegram" in result

    def test_scheduler_source(self):
        result = _wrap_user_message("Run task", "scheduler")
        assert "Source: scheduler" in result

    def test_multiline_message(self):
        result = _wrap_user_message("Line 1\nLine 2\nLine 3", "web")
        assert "Line 1\nLine 2\nLine 3" in result


class TestParseLlmResponse:
    """Tests for _parse_llm_response()."""

    def test_full_structured_response(self):
        text = (
            "<INTERNAL_THINKING>\n"
            "Let me analyze the data.\n"
            "</INTERNAL_THINKING>\n\n"
            "<RESPONSE_TO_USER>\n"
            "AAPL is trading at $150.\n"
            "</RESPONSE_TO_USER>"
        )
        thinking, response, is_silent = _parse_llm_response(text)
        assert "analyze the data" in thinking
        assert "AAPL is trading at $150" in response
        assert not is_silent

    def test_silent_response(self):
        text = (
            "<INTERNAL_THINKING>\n"
            "Task completed.\n"
            "</INTERNAL_THINKING>\n\n"
            "<RESPONSE_TO_USER>NO_VISIBLE_MESSAGE</RESPONSE_TO_USER>"
        )
        thinking, response, is_silent = _parse_llm_response(text)
        assert "Task completed" in thinking
        assert is_silent

    def test_unstructured_fallback(self):
        text = "Just a plain response without tags."
        thinking, response, is_silent = _parse_llm_response(text)
        assert thinking == ""
        assert response == text.strip()
        assert not is_silent

    def test_empty_thinking(self):
        text = (
            "<INTERNAL_THINKING></INTERNAL_THINKING>\n"
            "<RESPONSE_TO_USER>Hello!</RESPONSE_TO_USER>"
        )
        thinking, response, is_silent = _parse_llm_response(text)
        assert thinking == ""
        assert response == "Hello!"
        assert not is_silent

    def test_multiline_thinking_and_response(self):
        text = (
            "<INTERNAL_THINKING>\n"
            "Step 1: Check portfolio.\n"
            "Step 2: Analyze trends.\n"
            "Step 3: Formulate response.\n"
            "</INTERNAL_THINKING>\n\n"
            "<RESPONSE_TO_USER>\n"
            "Your portfolio is performing well.\n"
            "\n"
            "Key highlights:\n"
            "- AAPL up 5%\n"
            "- MSFT up 3%\n"
            "</RESPONSE_TO_USER>"
        )
        thinking, response, is_silent = _parse_llm_response(text)
        assert "Step 1" in thinking
        assert "Step 3" in thinking
        assert "Key highlights" in response
        assert "AAPL up 5%" in response
        assert not is_silent


class TestEstimateTokenCount:
    """Tests for _estimate_token_count()."""

    def test_basic_estimate(self):
        messages = [
            {"role": "user", "content": "Hello world"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        count = _estimate_token_count(messages, "You are a helpful assistant.")
        assert count > 0

    def test_empty_conversation(self):
        count = _estimate_token_count([], "System prompt")
        assert count > 0

    def test_larger_conversation_has_more_tokens(self):
        small = [{"role": "user", "content": "Hi"}]
        large = [
            {"role": "user", "content": "Hi " * 1000},
            {"role": "assistant", "content": "Hello " * 1000},
        ]
        small_count = _estimate_token_count(small, "System")
        large_count = _estimate_token_count(large, "System")
        assert large_count > small_count

    def test_with_tool_calls(self):
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "get_stock_price", "arguments": '{"symbol": "AAPL"}'},
                    }
                ],
            },
        ]
        count = _estimate_token_count(messages, "System prompt")
        assert count > 0
