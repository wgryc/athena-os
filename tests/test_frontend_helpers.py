"""Tests for frontend app helper functions."""

import pytest

from athena.frontend.app import (
    _wrap_user_message,
    _parse_llm_response,
    _estimate_token_count,
    _safe_split_index,
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


class TestSafeSplitIndex:
    """Tests for _safe_split_index()."""

    def test_no_tool_messages(self):
        """When there are no tool messages, split normally."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "Good!"},
            {"role": "user", "content": "Bye"},
        ]
        assert _safe_split_index(messages, 4) == 1

    def test_tool_result_at_split_boundary(self):
        """Tool result at the split boundary should pull the assistant msg in."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "tc_1"}]},
            {"role": "tool", "tool_call_id": "tc_1", "content": "result"},
            {"role": "assistant", "content": "Here you go."},
            {"role": "user", "content": "Thanks"},
            {"role": "assistant", "content": "Welcome!"},
        ]
        # desired_keep=4 → naive split at index 2 → messages[2] is tool
        # should back up to index 1 (the assistant with tool_calls)
        assert _safe_split_index(messages, 4) == 1

    def test_multiple_tool_results_at_boundary(self):
        """Multiple consecutive tool results should all be preserved."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "tc_1"}, {"id": "tc_2"}]},
            {"role": "tool", "tool_call_id": "tc_1", "content": "result1"},
            {"role": "tool", "tool_call_id": "tc_2", "content": "result2"},
            {"role": "assistant", "content": "Done."},
            {"role": "user", "content": "Ok"},
        ]
        # desired_keep=4 → naive split at index 2 → messages[2] is tool
        # messages[1] is assistant (not tool), so split = 1
        assert _safe_split_index(messages, 4) == 1

    def test_no_adjustment_needed(self):
        """When the split point is clean, no adjustment is needed."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "tc_1"}]},
            {"role": "tool", "tool_call_id": "tc_1", "content": "result"},
            {"role": "assistant", "content": "Here you go."},
            {"role": "user", "content": "Thanks"},
        ]
        # desired_keep=2 → split at index 3 → messages[3] is assistant, no adjustment
        assert _safe_split_index(messages, 2) == 3

    def test_desired_keep_exceeds_length(self):
        """When desired_keep >= len(messages), return 0."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        assert _safe_split_index(messages, 5) == 0
        assert _safe_split_index(messages, 2) == 0

    def test_all_messages_are_tool_results(self):
        """Edge case: if walking back reaches index 0, return 0 (keep all)."""
        messages = [
            {"role": "tool", "tool_call_id": "tc_1", "content": "r1"},
            {"role": "tool", "tool_call_id": "tc_2", "content": "r2"},
            {"role": "tool", "tool_call_id": "tc_3", "content": "r3"},
        ]
        assert _safe_split_index(messages, 2) == 0
