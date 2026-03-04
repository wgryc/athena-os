"""Discourse bot worker process: poll, triage, reply."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from .api import DiscourseClient, DiscourseTopic, DiscoursePost
from .state import (
    DiscourseThreadState,
    BotRateLimitState,
    load_state,
    save_state,
    count_recent_replies,
    record_reply,
)
from .triage import score_thread
from .agent import build_system_prompt, generate_reply

from ..tools import Tool
from ..tools.prices import GetStockPrice, GetOptionPrice, StockPriceHistory
from ..tools.web import NewsSearch, CrawlPage
from ..tools.events import ETEventsSearch
from ..tools.filings import SECFilings, SECFilingContent, InsiderTrades
from ..tools.agent_files import ReadAgentFile, UpdateAgentFile

# Config defaults
_DEFAULT_POLL_INTERVAL = 300
_DEFAULT_TRIAGE_MODEL = os.getenv("DEFAULT_LLM_MODEL", "anthropic/claude-haiku-4-5-20251001")
_DEFAULT_REPLY_MODEL = os.getenv("DEFAULT_LLM_MODEL", "anthropic/claude-haiku-4-5-20251001")
_DEFAULT_MAX_REPLY_TOKENS = 4000
_DEFAULT_ENGAGEMENT_THRESHOLD = 7
_DEFAULT_MAX_REPLIES_PER_HOUR = 5
_DEFAULT_MAX_REPLIES_PER_DAY = 30

# Tool name -> constructor mapping
_TOOL_REGISTRY: dict[str, type] = {
    "get_stock_price": GetStockPrice,
    "get_option_price": GetOptionPrice,
    "stock_price_history": StockPriceHistory,
    "news_search": NewsSearch,
    "crawl_page": CrawlPage,
    "et_events_search": ETEventsSearch,
    "sec_filings": SECFilings,
    "sec_filing_content": SECFilingContent,
    "insider_trades": InsiderTrades,
}


def load_config(config_path: str | Path) -> dict:
    """Load and validate the bot's JSON config file.

    Required fields: ``discourse_base_url``, ``discourse_api_key``,
    ``category_id``, ``bot_username``, ``personality_file``. All other
    fields fall back to module-level defaults.

    Args:
        config_path: Path to the JSON config file.

    Returns:
        Config dict with defaults applied.

    Raises:
        SystemExit: If the config file is missing or required fields are absent.
    """
    path = Path(config_path)
    if not path.exists():
        print(f"Error: config file not found: {path}", file=sys.stderr)
        sys.exit(1)

    try:
        config = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON in config: {e}", file=sys.stderr)
        sys.exit(1)

    required = [
        "discourse_base_url",
        "discourse_api_key",
        "category_id",
        "bot_username",
        "personality_file",
    ]
    for field_name in required:
        if field_name not in config:
            print(
                f"Error: missing required config field: {field_name}",
                file=sys.stderr,
            )
            sys.exit(1)

    # Load personality from file
    personality_path = Path(config["personality_file"])
    if not personality_path.is_absolute():
        personality_path = path.parent / personality_path
    if not personality_path.exists():
        print(
            f"Error: personality file not found: {personality_path}",
            file=sys.stderr,
        )
        sys.exit(1)
    config["personality"] = personality_path.read_text().strip()
    config["_personality_path"] = str(personality_path.resolve())

    # Load memory file (create if missing)
    memory_file = config.get("memory_file", "memory.md")
    memory_path = Path(memory_file)
    if not memory_path.is_absolute():
        memory_path = path.parent / memory_path
    if not memory_path.exists():
        memory_path.write_text("# Memory\n\n_No memories recorded yet._\n")
    config["memory"] = memory_path.read_text().strip()
    config["_memory_path"] = str(memory_path.resolve())

    # Load todo file (create if missing)
    todo_file = config.get("todo_file", "todo.md")
    todo_path = Path(todo_file)
    if not todo_path.is_absolute():
        todo_path = path.parent / todo_path
    if not todo_path.exists():
        todo_path.write_text("# Todo\n\n_No tasks yet._\n")
    config["todo"] = todo_path.read_text().strip()
    config["_todo_path"] = str(todo_path.resolve())

    # Apply defaults
    config.setdefault("poll_interval_seconds", _DEFAULT_POLL_INTERVAL)
    config.setdefault("triage_model", _DEFAULT_TRIAGE_MODEL)
    config.setdefault("reply_model", _DEFAULT_REPLY_MODEL)
    config.setdefault("max_reply_tokens", _DEFAULT_MAX_REPLY_TOKENS)
    config.setdefault("engagement_threshold", _DEFAULT_ENGAGEMENT_THRESHOLD)
    config.setdefault("max_replies_per_hour", _DEFAULT_MAX_REPLIES_PER_HOUR)
    config.setdefault("max_replies_per_day", _DEFAULT_MAX_REPLIES_PER_DAY)
    config.setdefault("state_file", f"discourse_{config['category_id']}_state.json")
    config.setdefault("tools", list(_TOOL_REGISTRY.keys()))

    return config


def build_tools(tool_names: list[str], config: dict | None = None) -> dict[str, Tool]:
    """Instantiate the configured tools, including agent-file tools.

    Args:
        tool_names: List of tool name strings from the config ``tools`` list.
        config: Bot configuration dict. When provided, agent-file tools
            (read/update personality, memory, todo) are always included.

    Returns:
        Dict mapping tool name to ``Tool`` instance. Unknown tool names are
        skipped with a warning.
    """
    tools: dict[str, Tool] = {}
    for name in tool_names:
        cls = _TOOL_REGISTRY.get(name)
        if cls is None:
            print(f"Warning: unknown tool '{name}', skipping", flush=True)
            continue
        tools[name] = cls()

    # Always add agent-file tools when config provides resolved paths
    if config:
        for file_key, label in [
            ("_personality_path", "personality"),
            ("_memory_path", "memory"),
            ("_todo_path", "todo"),
        ]:
            fpath = config.get(file_key)
            if fpath:
                read_name = f"read_{label}"
                update_name = f"update_{label}"
                tools[read_name] = ReadAgentFile(fpath, label, read_name)
                tools[update_name] = UpdateAgentFile(fpath, label, update_name)

    return tools


def _should_engage(
    topic: DiscourseTopic,
    thread_state: DiscourseThreadState | None,
    bot_username: str,
) -> tuple[bool, list[DiscoursePost]]:
    """Determine whether there are new posts worth triaging.

    Returns ``False`` if:
    - All posts are from the bot itself
    - No posts are newer than ``last_seen_post_number``
    - The last post is from the bot (cooldown: wait for a human reply)

    Args:
        topic: Fully populated ``DiscourseTopic``.
        thread_state: Existing state for this thread, or ``None`` if first visit.
        bot_username: The bot's Discourse username.

    Returns:
        Tuple of ``(should_triage, new_posts_from_humans)``.
    """
    last_seen = thread_state.last_seen_post_number if thread_state else 0

    new_posts = [
        p
        for p in topic.posts
        if p.post_number > last_seen and p.username != bot_username
    ]

    if not new_posts:
        return False, []

    # Cooldown: don't reply if the most recent post overall is from the bot
    if topic.posts and topic.posts[-1].username == bot_username:
        return False, []

    return True, new_posts


def process_thread(
    topic: DiscourseTopic,
    thread_state: DiscourseThreadState | None,
    config: dict,
    tools_by_name: dict[str, Tool],
    client: DiscourseClient,
    dry_run: bool = False,
    verbose: bool = False,
) -> DiscourseThreadState:
    """Triage and optionally reply to a single Discourse thread.

    Creates a new ``DiscourseThreadState`` if ``thread_state`` is ``None``
    (first encounter). Always updates ``last_seen_post_number``. Only calls
    the reply model if triage passes the ``engagement_threshold``.

    Args:
        topic: Fully populated ``DiscourseTopic`` with all posts.
        thread_state: Existing state for this thread, or ``None``.
        config: Bot configuration dict.
        tools_by_name: Instantiated tools dict.
        client: Authenticated Discourse API client.
        dry_run: If ``True``, generate reply but do not POST.
        verbose: If ``True``, print triage and reply details.

    Returns:
        Updated ``DiscourseThreadState`` (caller must persist to disk).
    """
    bot_username = config["bot_username"]

    # Initialize state on first encounter
    if thread_state is None:
        thread_state = DiscourseThreadState(
            topic_id=topic.topic_id,
            topic_title=topic.title,
            last_seen_post_number=0,
        )

    should_triage, new_human_posts = _should_engage(
        topic, thread_state, bot_username
    )

    if not should_triage:
        if topic.posts:
            thread_state.last_seen_post_number = max(
                p.post_number for p in topic.posts
            )
        return thread_state

    # Build triage input from new human posts (cap at 5 most recent)
    recent_posts_for_triage = [
        {"username": p.username, "content": p.raw[:1000]}
        for p in new_human_posts[-5:]
    ]

    triage_result = score_thread(
        topic_title=topic.title,
        recent_posts=recent_posts_for_triage,
        personality_summary=config["personality"][:500],
        model=config["triage_model"],
    )

    thread_state.triage_score = triage_result.score

    if verbose:
        print(
            f"[Triage] Topic {topic.topic_id} '{topic.title[:60]}' "
            f"score={triage_result.score} reason={triage_result.reason[:100]}",
            flush=True,
        )

    if triage_result.score < config["engagement_threshold"]:
        thread_state.last_seen_post_number = max(
            p.post_number for p in topic.posts
        )
        return thread_state

    # Append new human posts as user messages into the thread's conversation
    for post in new_human_posts:
        thread_state.conversation_messages.append(
            {
                "role": "user",
                "content": f"[{post.username}]: {post.raw}",
            }
        )

    system_prompt = build_system_prompt(
        personality=config["personality"],
        topic_title=topic.title,
        memory=config.get("memory", ""),
        todo=config.get("todo", ""),
    )

    reply_text, updated_messages = generate_reply(
        conversation_messages=thread_state.conversation_messages,
        system_prompt=system_prompt,
        tools_by_name=tools_by_name,
        model=config["reply_model"],
        max_tokens=config["max_reply_tokens"],
        verbose=verbose,
    )
    thread_state.conversation_messages = updated_messages

    if verbose:
        print(f"[Reply] {reply_text[:500]}", flush=True)

    if not reply_text:
        # Silent response (NO_VISIBLE_MESSAGE) — skip posting
        thread_state.last_seen_post_number = max(
            p.post_number for p in topic.posts
        )
        return thread_state

    if not dry_run:
        reply_to_post_number = new_human_posts[-1].post_number
        try:
            created_post = client.create_reply(
                topic_id=topic.topic_id,
                raw=reply_text,
                reply_to_post_number=reply_to_post_number,
            )
            thread_state.our_post_ids.append(created_post.post_id)
            thread_state.last_engaged = datetime.now(timezone.utc).isoformat()
            print(
                f"[Worker] Posted reply to topic {topic.topic_id} "
                f"(post #{created_post.post_number})",
                flush=True,
            )
        except Exception as e:
            print(
                f"[Worker] Failed to post reply to topic {topic.topic_id}: {e}",
                flush=True,
            )
    else:
        print(
            f"[DryRun] Would post reply to topic {topic.topic_id}", flush=True
        )

    thread_state.last_seen_post_number = max(
        p.post_number for p in topic.posts
    )
    return thread_state


def run_poll_cycle(
    config: dict,
    client: DiscourseClient,
    tools_by_name: dict[str, Tool],
    threads: dict[int, DiscourseThreadState],
    rate_limit: BotRateLimitState,
    dry_run: bool = False,
    verbose: bool = False,
) -> None:
    """Execute one full poll cycle: fetch topics, triage, reply, persist state.

    Modifies ``threads`` and ``rate_limit`` in place. Caller is responsible
    for calling ``save_state()`` after this returns.

    Args:
        config: Bot configuration dict.
        client: Authenticated Discourse API client.
        tools_by_name: Instantiated tools dict.
        threads: Mutable dict of known thread states.
        rate_limit: Mutable rate limit tracker.
        dry_run: If ``True``, do not POST replies.
        verbose: If ``True``, print progress details.
    """
    category_id = config["category_id"]
    max_per_hour = config["max_replies_per_hour"]
    max_per_day = config["max_replies_per_day"]

    # Check global rate limits before doing any work
    hourly_count = count_recent_replies(rate_limit, window_hours=1)
    daily_count = count_recent_replies(rate_limit, window_hours=24)

    if hourly_count >= max_per_hour:
        print(
            f"[Worker] Hourly limit reached ({hourly_count}/{max_per_hour}), "
            "skipping cycle",
            flush=True,
        )
        return
    if daily_count >= max_per_day:
        print(
            f"[Worker] Daily limit reached ({daily_count}/{max_per_day}), "
            "skipping cycle",
            flush=True,
        )
        return

    # Fetch topics
    try:
        recent_topics = client.list_topics(category_id)
    except Exception as e:
        print(f"[Worker] Failed to list topics: {e}", flush=True)
        return

    if verbose:
        print(
            f"[Worker] Found {len(recent_topics)} topics in category {category_id}",
            flush=True,
        )

    for topic_meta in recent_topics:
        # Re-check rate limits per topic
        hourly_count = count_recent_replies(rate_limit, window_hours=1)
        daily_count = count_recent_replies(rate_limit, window_hours=24)
        if hourly_count >= max_per_hour or daily_count >= max_per_day:
            print("[Worker] Rate limit reached mid-cycle, stopping", flush=True)
            break

        existing_state = threads.get(topic_meta.topic_id)

        # Fetch full topic if: we've never seen it, OR it has new posts
        should_fetch = (
            existing_state is None
            or topic_meta.posts_count > existing_state.last_seen_post_number
        )

        if not should_fetch:
            continue

        try:
            full_topic = client.get_topic(topic_meta.topic_id)
        except Exception as e:
            print(
                f"[Worker] Failed to fetch topic {topic_meta.topic_id}: {e}",
                flush=True,
            )
            continue

        updated_state = process_thread(
            topic=full_topic,
            thread_state=existing_state,
            config=config,
            tools_by_name=tools_by_name,
            client=client,
            dry_run=dry_run,
            verbose=verbose,
        )

        # If we actually replied (detected by last_engaged changing)
        if (
            updated_state.last_engaged
            != (existing_state.last_engaged if existing_state else None)
            and updated_state.last_engaged is not None
        ):
            record_reply(rate_limit)

        threads[topic_meta.topic_id] = updated_state


def _init_client_and_tools(config: dict) -> tuple[DiscourseClient, dict[str, Tool]]:
    """Create the Discourse client and tool registry from config.

    Args:
        config: Validated bot configuration dict.

    Returns:
        Tuple of ``(client, tools_by_name)``.

    Raises:
        SystemExit: If ``ANTHROPIC_API_KEY`` is not set.
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        print(
            "Error: ANTHROPIC_API_KEY environment variable not set",
            file=sys.stderr,
        )
        sys.exit(1)

    client = DiscourseClient(
        base_url=config["discourse_base_url"],
        api_key=config["discourse_api_key"],
        bot_username=config["bot_username"],
    )
    tools_by_name = build_tools(config["tools"], config=config)
    return client, tools_by_name


def _cmd_run(args: argparse.Namespace) -> None:
    """Handle the ``run`` subcommand — poll, triage, reply loop."""
    config = load_config(args.config)
    client, tools_by_name = _init_client_and_tools(config)
    threads, rate_limit = load_state(config["state_file"])

    print(
        f"[Worker] Starting Discourse bot: category={config['category_id']} "
        f"username={config['bot_username']} tools={list(tools_by_name.keys())}",
        flush=True,
    )

    while True:
        print(
            f"[Worker] Poll cycle at {datetime.now(timezone.utc).isoformat()}",
            flush=True,
        )

        run_poll_cycle(
            config=config,
            client=client,
            tools_by_name=tools_by_name,
            threads=threads,
            rate_limit=rate_limit,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )

        save_state(config["state_file"], threads, rate_limit)

        if args.once:
            print(
                "[Worker] --once flag set, exiting after one cycle", flush=True
            )
            break

        print(
            f"[Worker] Sleeping {config['poll_interval_seconds']}s until next cycle",
            flush=True,
        )
        time.sleep(config["poll_interval_seconds"])


def _cmd_post(args: argparse.Namespace) -> None:
    """Handle the ``post`` subcommand — create a new topic.

    Generates a topic title and body using the LLM with tools, then posts
    it to the configured Discourse category.
    """
    config = load_config(args.config)
    client, tools_by_name = _init_client_and_tools(config)

    prompt = args.prompt
    verbose = args.verbose

    if verbose:
        print(f"[Post] Generating topic from prompt: {prompt[:200]}", flush=True)

    system_prompt = build_system_prompt(
        personality=config["personality"],
        topic_title="(new topic — you are writing the opening post)",
        memory=config.get("memory", ""),
        todo=config.get("todo", ""),
    )

    # Override system prompt to ask for a title + body
    system_prompt += (
        "\n\n## Special Instructions for New Topic\n\n"
        "You are creating a NEW forum topic, not replying to an existing one.\n"
        "Your response inside <RESPONSE_TO_USER> must be formatted as:\n\n"
        "TITLE: Your Topic Title Here\n"
        "---\n"
        "The body of your post in markdown.\n\n"
        "The first line MUST start with 'TITLE: ' followed by a concise, "
        "descriptive title. Then a line with just '---', then the post body."
    )

    conversation: list[dict] = [{"role": "user", "content": prompt}]

    reply_text, _ = generate_reply(
        conversation_messages=conversation,
        system_prompt=system_prompt,
        tools_by_name=tools_by_name,
        model=config["reply_model"],
        max_tokens=config["max_reply_tokens"],
        verbose=verbose,
    )

    if not reply_text:
        print("[Post] LLM returned no content, aborting", flush=True)
        return

    # Parse title from response
    title, body = _parse_topic_response(reply_text)

    if verbose:
        print(f"[Post] Title: {title}", flush=True)
        print(f"[Post] Body:\n{body[:500]}", flush=True)

    if args.dry_run:
        print(f"[DryRun] Would create topic: {title}", flush=True)
        print(f"[DryRun] Body:\n{body}", flush=True)
        return

    try:
        topic = client.create_topic(
            title=title,
            raw=body,
            category_id=config["category_id"],
        )
        print(
            f"[Post] Created topic {topic.topic_id}: {title}",
            flush=True,
        )
    except Exception as e:
        print(f"[Post] Failed to create topic: {e}", flush=True)


def _reload_agent_files(config: dict, config_path: str | Path) -> None:
    """Re-read memory and todo files from disk into the config dict.

    Called before each LLM turn in chat mode so the system prompt always
    reflects the latest file contents (which tools may have modified).

    Args:
        config: Bot configuration dict (mutated in place).
        config_path: Original config file path (for resolving relative paths).
    """
    for key, internal_key in [("_memory_path", "memory"), ("_todo_path", "todo")]:
        fpath = config.get(key)
        if fpath and Path(fpath).exists():
            config[internal_key] = Path(fpath).read_text().strip()


def _cmd_chat(args: argparse.Namespace) -> None:
    """Handle the ``chat`` subcommand — interactive CLI chat with the bot."""
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    from ..cli.main import ATHENA_LOGO, INVESTING_WARNING

    config = load_config(args.config)
    config_path = args.config

    if not os.getenv("ANTHROPIC_API_KEY"):
        print(
            "Error: ANTHROPIC_API_KEY environment variable not set",
            file=sys.stderr,
        )
        sys.exit(1)

    tools_by_name = build_tools(config["tools"], config=config)

    # Print branded header
    print(ATHENA_LOGO)
    print(INVESTING_WARNING)
    print()
    print(f" Model:       {config['reply_model']}")
    print(f" Personality:  {config.get('personality_file', 'N/A')}")
    print(f" Memory:       {config.get('memory_file', 'memory.md')}")
    print(f" Todo:         {config.get('todo_file', 'todo.md')}")
    print(f" Tools:        {len(tools_by_name)} loaded")
    print()
    print(" Type your message and press Enter. Type 'exit' or 'quit' to leave.")
    print(" " + "─" * 60)
    print()

    conversation_messages: list[dict] = []
    verbose = args.verbose

    while True:
        try:
            user_input = input("\033[36mYou:\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("\nGoodbye!")
            break

        conversation_messages.append({"role": "user", "content": user_input})

        # Reload agent files so system prompt reflects any tool edits
        _reload_agent_files(config, config_path)

        system_prompt = build_system_prompt(
            personality=config["personality"],
            topic_title="",
            memory=config.get("memory", ""),
            todo=config.get("todo", ""),
            chat_mode=True,
        )

        reply_text, conversation_messages = generate_reply(
            conversation_messages=conversation_messages,
            system_prompt=system_prompt,
            tools_by_name=tools_by_name,
            model=config["reply_model"],
            max_tokens=config["max_reply_tokens"],
            verbose=verbose,
        )

        if reply_text:
            print(f"\n\033[33mAthena:\033[0m {reply_text}\n")
        else:
            if verbose:
                print("[Chat] (silent response — no visible message)", flush=True)


def _parse_topic_response(text: str) -> tuple[str, str]:
    """Parse a TITLE:/--- formatted response into title and body.

    Args:
        text: LLM response text expected to contain ``TITLE: ...`` on the
            first line, then ``---``, then the body.

    Returns:
        Tuple of ``(title, body)``. Falls back to using the first line as
        the title if the format is not followed.
    """
    lines = text.strip().split("\n")
    if lines and lines[0].startswith("TITLE:"):
        title = lines[0][len("TITLE:"):].strip()
        # Find the --- separator
        body_start = 1
        if len(lines) > 1 and lines[1].strip() == "---":
            body_start = 2
        body = "\n".join(lines[body_start:]).strip()
        return title, body

    # Fallback: first line as title, rest as body
    title = lines[0] if lines else "New Topic"
    body = "\n".join(lines[1:]).strip() if len(lines) > 1 else text
    return title, body


def main() -> None:
    """CLI entry point for the Discourse bot.

    Usage::

        python -m athena.discourse run --config discourse_finance.json
        python -m athena.discourse run --config discourse_finance.json --dry-run --once -v
        python -m athena.discourse post --config discourse_finance.json --prompt "Write about Q4 earnings"
    """
    parser = argparse.ArgumentParser(
        description="AthenaOS Discourse bot",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- run subcommand ---
    run_parser = subparsers.add_parser(
        "run",
        help="Poll the forum, triage threads, and reply",
    )
    run_parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to the bot JSON config file",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Triage and generate replies but do not post to Discourse",
    )
    run_parser.add_argument(
        "--once",
        action="store_true",
        help="Run one poll cycle and exit (useful for cron and testing)",
    )
    run_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print triage scores, tool calls, and reply text to stdout",
    )
    run_parser.set_defaults(func=_cmd_run)

    # --- post subcommand ---
    post_parser = subparsers.add_parser(
        "post",
        help="Create a new topic in the configured category",
    )
    post_parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to the bot JSON config file",
    )
    post_parser.add_argument(
        "--prompt",
        "-p",
        required=True,
        help="Prompt describing the topic to create (e.g. 'Write about Q4 AAPL earnings')",
    )
    post_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate the topic but do not post to Discourse",
    )
    post_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print tool calls and generated content to stdout",
    )
    post_parser.set_defaults(func=_cmd_post)

    # --- chat subcommand ---
    chat_parser = subparsers.add_parser(
        "chat",
        help="Interactive CLI chat with the bot (for testing)",
    )
    chat_parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to the bot JSON config file",
    )
    chat_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print tool calls and thinking to stdout",
    )
    chat_parser.set_defaults(func=_cmd_chat)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
