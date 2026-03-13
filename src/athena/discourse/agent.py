"""Stage 2 reply generation with full tool-calling loop.

Adapted from the ``process_gateway_message`` loop in
``athena.frontend.app`` but without Flask dependencies. LLM utility
functions are copied locally to avoid importing ``app.py`` (which pulls
in Flask and triggers ``load_dotenv()`` at module scope).
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import litellm
from jinja2 import Environment, FileSystemLoader

from ..tools import Tool

TEMPLATES_DIR = Path(__file__).parent / "templates"
_jinja_env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))

DEFAULT_REPLY_MODEL = os.getenv("DEFAULT_LLM_MODEL", "anthropic/claude-haiku-4-5-20251001")
DEFAULT_MAX_TOKENS = 4000
COMPACT_ON_NUM_TOKENS = 60000

# ---------------------------------------------------------------------------
# LLM utility functions (copied from frontend/app.py to avoid Flask import)
# ---------------------------------------------------------------------------

_LITELLM_FIELDS = frozenset(("role", "content", "tool_calls", "tool_call_id"))


def _strip_extra_fields(msg: dict) -> dict:
    """Return a copy of *msg* keeping only fields recognized by litellm.

    Args:
        msg: A conversation message dict that may contain extra keys.

    Returns:
        New dict containing only the keys in ``_LITELLM_FIELDS``.
    """
    return {k: v for k, v in msg.items() if k in _LITELLM_FIELDS}


def _estimate_token_count(
    messages: list[dict],
    system_prompt: str = "",
    model: str = DEFAULT_REPLY_MODEL,
) -> int:
    """Estimate the total token count of the conversation.

    Args:
        messages: The conversation message list.
        system_prompt: The system prompt string.
        model: LiteLLM model string (used for tokenizer selection).

    Returns:
        Estimated total token count.
    """
    try:
        all_messages = [{"role": "system", "content": system_prompt}] + [
            _strip_extra_fields(m) for m in messages
        ]
        return litellm.token_counter(model=model, messages=all_messages)
    except Exception:
        total_chars = len(system_prompt)
        for m in messages:
            content = m.get("content") or ""
            total_chars += len(content)
            if m.get("tool_calls"):
                total_chars += len(json.dumps(m["tool_calls"]))
        return total_chars // 4


def _safe_split_index(messages: list[dict], desired_keep: int) -> int:
    """Find a safe split index that never orphans tool_result messages.

    Args:
        messages: Full conversation message list.
        desired_keep: How many trailing messages we'd like to preserve.

    Returns:
        The index into *messages* at which to split.
    """
    if desired_keep >= len(messages):
        return 0

    split = len(messages) - desired_keep

    while split > 0 and messages[split].get("role") == "tool":
        split -= 1

    return split


def _compact_conversation(
    messages: list[dict],
    system_prompt: str,
    model: str = DEFAULT_REPLY_MODEL,
) -> list[dict]:
    """Compact the conversation by asking the LLM to summarize it.

    Args:
        messages: Current conversation messages.
        system_prompt: The system prompt (for context in summarization).
        model: LiteLLM model string for the summarization call.

    Returns:
        New compacted message list.
    """
    preserve_count = 4
    if len(messages) <= preserve_count:
        return messages

    split = _safe_split_index(messages, preserve_count)
    if split == 0:
        return messages

    messages_to_summarize = messages[:split]
    preserved = messages[split:]

    summary_request = [
        {
            "role": "system",
            "content": (
                "You are a conversation summarizer. Summarize the following "
                "conversation into a concise but comprehensive summary. Preserve "
                "all important facts, decisions, tool results, and context. This "
                "summary will replace the original messages to manage context "
                "window size.\n\nFormat your summary as a clear, structured recap."
            ),
        },
        {
            "role": "user",
            "content": (
                "Summarize this conversation:\n\n"
                + json.dumps(
                    [_strip_extra_fields(m) for m in messages_to_summarize],
                    indent=2,
                )
            ),
        },
    ]

    try:
        response = litellm.completion(
            model=model,
            messages=summary_request,
            max_tokens=4000,
        )
        summary_text = response.choices[0].message.content

        compacted = [
            {
                "role": "assistant",
                "content": f"[CONVERSATION SUMMARY]\n{summary_text}",
                "_is_compaction": True,
            }
        ] + preserved

        print(
            f"[Compaction] Reduced {len(messages)} messages to {len(compacted)} "
            f"(summary + {len(preserved)} preserved)",
            flush=True,
        )
        return compacted

    except Exception as e:
        print(f"[Compaction] Failed: {e}", flush=True)
        fallback_split = _safe_split_index(messages, 20)
        return messages[fallback_split:]


# Regex patterns for parsing structured LLM responses
_THINKING_PATTERN = re.compile(
    r"<INTERNAL_THINKING>(.*?)</INTERNAL_THINKING>", re.DOTALL
)
_RESPONSE_PATTERN = re.compile(
    r"<RESPONSE_TO_USER>(.*?)</RESPONSE_TO_USER>", re.DOTALL
)
# Fallback: opening tag present but closing tag missing (LLM truncated)
_RESPONSE_OPEN_PATTERN = re.compile(
    r"<RESPONSE_TO_USER>(.*)", re.DOTALL
)
_THINKING_OPEN_PATTERN = re.compile(
    r"<INTERNAL_THINKING>(.*)", re.DOTALL
)
_NO_VISIBLE_MESSAGE = "NO_VISIBLE_MESSAGE"


def _parse_llm_response(full_text: str) -> tuple[str, str, bool]:
    """Parse structured LLM response into thinking and user-visible parts.

    Handles cases where the LLM omits closing tags by falling back to
    open-ended capture from the opening tag to end-of-string.

    Args:
        full_text: The complete LLM response text.

    Returns:
        Tuple of ``(thinking_text, response_text, is_silent)``.
    """
    # Extract thinking
    thinking_match = _THINKING_PATTERN.search(full_text)
    if thinking_match:
        thinking = thinking_match.group(1).strip()
    else:
        open_match = _THINKING_OPEN_PATTERN.search(full_text)
        thinking = open_match.group(1).strip() if open_match else ""

    # Extract response — try closed tags first, then open tag fallback
    response_match = _RESPONSE_PATTERN.search(full_text)
    if response_match:
        response = response_match.group(1).strip()
    else:
        open_match = _RESPONSE_OPEN_PATTERN.search(full_text)
        if open_match:
            response = open_match.group(1).strip()
        else:
            # No tags at all — strip any thinking block and use the rest
            response = re.sub(
                r"<INTERNAL_THINKING>.*?</INTERNAL_THINKING>",
                "",
                full_text,
                flags=re.DOTALL,
            ).strip()

    is_silent = response == _NO_VISIBLE_MESSAGE
    return thinking, response, is_silent


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_system_prompt(
    personality: str,
    topic_title: str,
    category_name: str = "",
    memory: str = "",
    todo: str = "",
    chat_mode: bool = False,
    bot_username: str = "",
) -> str:
    """Render the system prompt for a specific thread context.

    Args:
        personality: Bot personality string from config.
        topic_title: Title of the Discourse topic.
        category_name: Human-readable name of the forum category.
        memory: Contents of the bot's memory file.
        todo: Contents of the bot's todo file.
        chat_mode: If ``True``, render for direct CLI chat instead of forum.
        bot_username: The bot's Discourse username (e.g. ``test_bot_1``).

    Returns:
        Rendered system prompt string.
    """
    template = _jinja_env.get_template("system_prompt.j2")
    return template.render(
        personality=personality,
        topic_title=topic_title,
        category_name=category_name,
        memory=memory,
        todo=todo,
        chat_mode=chat_mode,
        bot_username=bot_username,
        current_datetime=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )


def generate_reply(
    conversation_messages: list[dict],
    system_prompt: str,
    tools_by_name: dict[str, Tool],
    model: str = DEFAULT_REPLY_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    verbose: bool = False,
) -> tuple[str, list[dict]]:
    """Run the full tool-calling loop and return the final reply text.

    This is a direct adaptation of the ``process_gateway_message`` loop in
    ``app.py``. The loop:

    1. Builds ``llm_messages = [system_prompt] + stripped conversation``
    2. Calls ``litellm.completion`` with ``stream=True``
    3. Accumulates tool calls from streaming chunks by index
    4. If tool calls: execute via ``tool.handle_tool_call()``, append
       results, loop again
    5. If no tool calls: break, parse ``INTERNAL_THINKING``/``RESPONSE_TO_USER``

    Token compaction is applied before the first LLM call if the
    conversation exceeds ``COMPACT_ON_NUM_TOKENS``.

    Args:
        conversation_messages: Mutable list of LLM conversation messages for
            this thread. Will be extended in place with tool calls and results.
        system_prompt: Rendered system prompt for this thread.
        tools_by_name: Dict mapping tool name to ``Tool`` instance.
        model: LiteLLM model string.
        max_tokens: Max tokens for the reply.
        verbose: If ``True``, print tool usage to stdout.

    Returns:
        Tuple of ``(reply_text, updated_conversation_messages)``.
        ``reply_text`` is the ``RESPONSE_TO_USER`` section (or full text if
        tags are absent). The ``conversation_messages`` list is extended in
        place and also returned.
    """
    # Compact if needed
    token_count = _estimate_token_count(conversation_messages, system_prompt, model)
    if token_count > COMPACT_ON_NUM_TOKENS:
        if verbose:
            print(
                f"[Agent] Compacting conversation ({token_count} tokens)",
                flush=True,
            )
        conversation_messages[:] = _compact_conversation(
            conversation_messages, system_prompt, model
        )

    tools_json = [t.to_json() for t in tools_by_name.values()] or None

    llm_messages = [{"role": "system", "content": system_prompt}] + [
        _strip_extra_fields(m) for m in conversation_messages
    ]

    full_response = ""

    try:
        while True:
            response = litellm.completion(
                model=model,
                messages=llm_messages,
                max_tokens=max_tokens,
                tools=tools_json,
                stream=True,
            )

            tool_calls_acc: dict[int, dict] = {}
            chunk_content = ""

            for chunk in response:
                delta = chunk.choices[0].delta

                if delta.content:
                    chunk_content += delta.content
                    full_response += delta.content

                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        if tc.id:
                            tool_calls_acc[idx]["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                tool_calls_acc[idx]["function"][
                                    "name"
                                ] += tc.function.name
                            if tc.function.arguments:
                                tool_calls_acc[idx]["function"][
                                    "arguments"
                                ] += tc.function.arguments

            # No tool calls → final response
            if not tool_calls_acc:
                break

            # Process tool calls
            tc_list = [tool_calls_acc[i] for i in sorted(tool_calls_acc)]
            assistant_msg: dict = {
                "role": "assistant",
                "content": chunk_content or None,
                "tool_calls": tc_list,
            }
            llm_messages.append(assistant_msg)
            conversation_messages.append(assistant_msg)

            if verbose:
                names = [tc["function"]["name"] for tc in tc_list]
                print(f"[Agent] Tool calls: {names}", flush=True)

            for tc in tc_list:
                tool = tools_by_name.get(tc["function"]["name"])
                if tool:
                    try:
                        result = tool.handle_tool_call(tc["function"]["arguments"])
                    except Exception as exc:
                        result = (
                            f"Error executing {tc['function']['name']}: {exc}"
                        )
                else:
                    result = f"Error: unknown tool '{tc['function']['name']}'"

                if verbose:
                    print(
                        f"[Agent] Tool result ({tc['function']['name']}): "
                        f"{result[:200]}",
                        flush=True,
                    )

                tool_msg: dict = {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                }
                llm_messages.append(tool_msg)
                conversation_messages.append(tool_msg)

            # Reset for next iteration
            full_response = ""

    except Exception as e:
        return f"Error generating reply: {e}", conversation_messages

    # Parse structured response
    thinking, reply_text, is_silent = _parse_llm_response(full_response)

    if verbose and thinking:
        print(f"[Agent] Thinking: {thinking[:500]}", flush=True)

    # Append final assistant message to conversation
    conversation_messages.append(
        {
            "role": "assistant",
            "content": full_response,
            "_thinking": thinking,
        }
    )

    if is_silent:
        return "", conversation_messages

    return reply_text, conversation_messages
