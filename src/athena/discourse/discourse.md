# Discourse Bot: Strategy & Plan

## Vision

Bring AthenaOS to our Discourse forum so we can interact with Athena and show
how humans and AI can collaborate together, and how AIs can also collaborate
with each other. The bot monitors specific forum categories, decides when it has
something substantive to contribute, and posts thoughtful replies grounded in
real financial data via AthenaOS tools.

---

## 1. Requirements

### From the User

1. **Category-scoped**: Each bot instance monitors a single Discourse category
   (e.g., "Finance", "Geopolitics"). The category is specified in its config.
2. **Follow-up on existing threads**: The bot must revisit discussions it has
   already participated in to check for new replies and continue the
   conversation.
3. **Full tool access**: The bot has access to the same LLM-callable tools as
   the frontend chat (stock prices, news search, SEC filings, events, options,
   web crawl, etc.).
4. **Per-bot config file**: Each bot's personality, category, engagement rules,
   and model settings are defined in a standalone JSON config file (not the main
   `config.json`). You can run multiple bots with different configs.
5. **API keys in `.env`**: Discourse API credentials and all other keys
   (`ANTHROPIC_API_KEY`, `SERPAPI_KEY`, etc.) are read from environment
   variables / `.env`, not stored in the config file.

### Architectural

6. **Separate process, not a Gateway**: The existing `Gateway` ABC in
   `gateways.py` is designed for 1:1 conversational channels that share a
   single `conversation_messages` list. A forum bot needs per-thread
   conversation isolation, selective engagement, and forum-native threading.
   This is a standalone worker process.
7. **Per-thread conversation state**: Each Discourse topic the bot participates
   in maintains its own conversation history, tool call results, and last-seen
   post tracking. This is essential so context from Thread A doesn't leak into
   Thread B.
8. **Two-stage LLM pipeline**: A cheap triage pass decides whether to engage;
   a full tool-equipped pass generates the reply. This keeps costs manageable.
9. **Reuse existing AthenaOS components**: Tools, LiteLLM integration, token
   compaction logic, and Jinja2 template patterns are reused directly.

---

## 2. Config File Format

Each bot instance is driven by a JSON config file, e.g. `discourse_finance.json`:

```json
{
    "discourse_base_url": "https://forum.example.com",
    "discourse_api_key": "your_discourse_api_key_here",
    "category_id": 5,
    "bot_username": "athena-finance",
    "personality_file": "personality.md",
    "memory_file": "memory.md",
    "todo_file": "todo.md",
    "poll_interval_seconds": 300,
    "triage_model": "anthropic/claude-haiku-4-5-20251001",
    "reply_model": "anthropic/claude-sonnet-4-5",
    "max_reply_tokens": 4000,
    "engagement_threshold": 7,
    "max_replies_per_hour": 5,
    "max_replies_per_day": 30,
    "state_file": "discourse_finance_state.json",
    "tools": [
        "get_stock_price",
        "stock_price_history",
        "news_search",
        "crawl_page",
        "et_events_search",
        "sec_filings",
        "sec_filing_content",
        "insider_trades",
        "get_option_price"
    ]
}
```

### Field Reference

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `discourse_base_url` | string | yes | Base URL of the Discourse instance |
| `discourse_api_key` | string | yes | Discourse API key for this bot user |
| `category_id` | int | yes | Discourse category ID to monitor |
| `bot_username` | string | yes | Discourse username the bot posts as |
| `personality_file` | string | yes | Path to a markdown file containing the bot's personality/instructions (relative to config file or absolute) |
| `memory_file` | string | no | Path to a markdown file for the bot's persistent memory (default: `memory.md` relative to config) |
| `todo_file` | string | no | Path to a markdown file for the bot's task list (default: `todo.md` relative to config) |
| `poll_interval_seconds` | int | no | Polling frequency (default: 300) |
| `triage_model` | string | no | Model for triage pass (default: Haiku) |
| `reply_model` | string | no | Model for reply generation (default: Sonnet) |
| `max_reply_tokens` | int | no | Max tokens per reply (default: 4000) |
| `engagement_threshold` | int | no | Triage score (0-10) needed to reply (default: 7) |
| `max_replies_per_hour` | int | no | Rate limit per hour (default: 5) |
| `max_replies_per_day` | int | no | Rate limit per day (default: 30) |
| `state_file` | string | no | Path for thread state persistence (default: `discourse_{category_id}_state.json`) |
| `tools` | list[str] | no | Tool names to enable (default: all available) |

### Environment Variables

The Discourse API key is stored per-bot in the JSON config file (not in
`.env`), since each bot user may have a different key. All other keys
remain in `.env` as usual:

```
# LLM (already in .env for AthenaOS)
ANTHROPIC_API_KEY=...

# Tools (already in .env for AthenaOS)
SERPAPI_KEY=...
SCRAPING_BEE_API_KEY=...
ET_API_KEY=...
MASSIVE_API_KEY=...
DATABENTO_API_KEY=...
```

---

## 3. Architecture

### Why Not a Gateway

The `Gateway` ABC (`gateways.py`) assumes:
- All gateways share one `conversation_messages` list
- Every incoming message gets a response
- A single ongoing conversation (no threading)

A forum bot needs:
- **Per-thread isolation** — each Discourse topic is its own conversation
- **Selective engagement** — skip threads where it can't add value
- **Forum-native threading** — reply to specific posts, quote context

### Process Architecture

```
discourse_worker.py (standalone process)
│
├── Load config JSON
├── Initialize tools (same registry as app.py)
├── Load thread state from state_file
│
└── Poll loop (every poll_interval_seconds):
    │
    ├── Fetch new/updated topics in category_id
    │   GET /c/{category_id}.json
    │
    ├── For topics we've participated in:
    │   GET /t/{topic_id}.json
    │   Check for new posts since last_seen_post_number
    │
    ├── For each candidate thread (new topic OR new reply in existing):
    │   │
    │   ├── Stage 1: TRIAGE (cheap model)
    │   │   "Score 0-10: should we engage? Is this in our domain?
    │   │    Can we add value beyond what's already been said?"
    │   │   → Skip if score < engagement_threshold
    │   │
    │   ├── Stage 2: REPLY (full model + tools)
    │   │   System prompt = personality + thread context + tools
    │   │   Conversation = this thread's history only
    │   │   → LLM generates reply, may call tools
    │   │   → Tool results fed back into conversation loop
    │   │
    │   ├── POST reply to Discourse
    │   │   POST /posts {topic_id, raw, reply_to_post_number}
    │   │
    │   └── Update thread state (last_seen, our_posts, conversation)
    │
    ├── Persist state to state_file
    └── Sleep until next poll
```

### Thread State Model

Each thread the bot has seen or participated in gets a state entry:

```python
@dataclass
class DiscourseThreadState:
    topic_id: int
    topic_title: str
    last_seen_post_number: int
    our_post_ids: list[int]
    conversation_messages: list[dict]  # LLM conversation for this thread
    first_seen: str                    # ISO timestamp
    last_engaged: str | None           # ISO timestamp of last reply
    triage_score: int                  # Last triage score
```

Persisted to the `state_file` as a JSON dict keyed by `topic_id`.

---

## 4. Two-Stage LLM Pipeline

### Stage 1: Triage

A fast, cheap LLM call (Haiku) that decides whether to engage.

**Input**: The thread title, latest posts (or full thread if short), and the
bot's domain/personality description.

**Prompt template** (`triage_prompt.j2`):

```
You are evaluating whether to participate in a forum discussion.

Your domain: {{ personality_summary }}
Category: {{ category_name }}

Thread title: {{ topic_title }}
Recent posts:
{% for post in recent_posts %}
[{{ post.username }}]: {{ post.content }}
{% endfor %}

Score this thread 0-10 on two dimensions, then give a final score:
1. Relevance: Is this in your domain of expertise?
2. Value-add: Can you contribute something new/useful beyond what's been said?

Respond ONLY with JSON: {"relevance": N, "value_add": N, "score": N, "reason": "..."}
```

**Decision**: If `score >= engagement_threshold`, proceed to Stage 2.

### Stage 2: Reply Generation

Full model (Sonnet/Opus) with all configured tools.

**System prompt** (`system_prompt.j2`):

```
{{ personality }}

You are participating in a Discourse forum discussion. Write a reply that:
- Is conversational and forum-appropriate
- Backs up claims with data when possible (use your tools)
- Directly addresses the points raised in the most recent posts
- Does not repeat what others have already said
- Uses markdown formatting (Discourse supports full markdown)

Do NOT include any meta-commentary about being an AI or about your tools.
When you use a tool, incorporate the results naturally into your response.
```

**Conversation**: The thread's stored `conversation_messages`, with new posts
appended as user messages. Tool calls and results are included in the
conversation loop just like the frontend chat.

---

## 5. Reused AthenaOS Components

| Component | Location | Reuse |
|-----------|----------|-------|
| LLM-callable tools | `tools/prices.py`, `tools/web.py`, `tools/events.py`, `tools/filings.py` | Direct — instantiate the same classes, register subset per config |
| Tool base class + JSON schema | `tools/__init__.py` | Direct — `Tool.to_json()`, `Tool.handle_tool_call()` |
| LiteLLM integration | `litellm.completion()` | Direct — same call pattern with tool support |
| Token compaction | `app.py` `_compact_conversation()` | Extract into shared utility or duplicate (small function) |
| Jinja2 template pattern | `agents/templates/` | Follow same pattern, new templates in `discourse/templates/` |
| Pricing data manager | `pricingdata.py` | Direct — tools that need it accept it as constructor arg |
| `.env` loading | `dotenv` | Direct — `load_dotenv()` at startup |

### New Code Required

| Component | Purpose |
|-----------|---------|
| `discourse/worker.py` | Main polling loop and orchestration |
| `discourse/triage.py` | Stage 1 triage scoring |
| `discourse/agent.py` | Stage 2 reply generation with tool loop |
| `discourse/api.py` | Thin Discourse REST API client |
| `discourse/state.py` | Thread state persistence (load/save JSON) |
| `discourse/templates/` | Jinja2 templates for system prompt and triage prompt |

---

## 6. Discourse API Details

Authentication is via HTTP headers on every request:

```
Api-Key: {DISCOURSE_API_KEY}
Api-Username: {bot_username from config}
```

### Key Endpoints

| Action | Method | Endpoint |
|--------|--------|----------|
| List topics in category | GET | `/c/{category_id}.json` |
| Get topic with posts | GET | `/t/{topic_id}.json` |
| Get specific posts | GET | `/t/{topic_id}/posts.json?post_ids[]=...` |
| Create a reply | POST | `/posts` with `{topic_id, raw, reply_to_post_number}` |

### Polling Strategy

1. **Category scan**: `GET /c/{category_id}.json` returns recent topics with
   `last_posted_at` timestamps. Compare against stored state to find topics
   with new activity.
2. **Thread fetch**: For active topics, `GET /t/{topic_id}.json` returns all
   posts. Filter to posts with `post_number > last_seen_post_number`.
3. **Rate limiting**: Discourse API has default rate limits of 60 requests per
   minute. With a 5-minute poll interval, we're well within limits.

No new Python packages needed — `requests` is already a transitive dependency.

---

## 7. Rate Limiting & Safety

- **`max_replies_per_hour` / `max_replies_per_day`**: Hard caps tracked in the
  state file. The bot will skip engagement if it has hit its limit even if
  triage scores are high.
- **No self-replies**: The bot never replies to its own posts.
- **Cooldown per thread**: After replying to a thread, the bot waits for at
  least one human reply before considering replying again (prevents monologuing).
- **Token compaction**: Long threads will be compacted using the same
  summarization approach as the frontend chat, so cost stays bounded.

---

## 8. File Structure

```
src/athena/discourse/
    __init__.py
    worker.py           # Main entry point + polling loop
    triage.py           # Stage 1 relevance scoring
    agent.py            # Stage 2 reply generation (per-thread, with tool loop)
    api.py              # Discourse REST API client
    state.py            # Thread state persistence
    templates/
        system_prompt.j2
        triage_prompt.j2
    discourse.md        # This file
```

---

## 9. CLI Usage

The CLI uses two subcommands: **`run`** (poll a category and reply to threads)
and **`post`** (create a new topic). Both require a `--config` pointing to a
bot JSON config file.

### `run` — Poll and Reply

Continuously polls the configured Discourse category for new or updated
threads, triages them for relevance, and replies when the bot has something
to contribute.

```bash
# Start the polling loop (runs forever, polls every poll_interval_seconds)
python -m athena.discourse run -c discourse_finance.json

# Dry-run a single poll cycle — triage and generate replies, but don't post
python -m athena.discourse run -c discourse_finance.json --dry-run --once -v

# Run multiple bots in parallel (different categories/personalities)
python -m athena.discourse run -c discourse_finance.json &
python -m athena.discourse run -c discourse_geopolitics.json &
```

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--config` | `-c` | string | *(required)* | Path to the bot's JSON config file |
| `--dry-run` | | flag | off | Triage and generate replies but don't post to Discourse |
| `--once` | | flag | off | Run one poll cycle and exit (useful for cron/testing) |
| `--verbose` | `-v` | flag | off | Print triage scores, tool calls, and full replies to stdout |

### `post` — Create a New Topic

Generates a new forum topic using the LLM (with full tool access) and posts
it to the configured category. The bot uses the personality and tools from
the config file to research and write the post.

```bash
# Create a topic about Q4 earnings
python -m athena.discourse post -c discourse_user.json --prompt "Write about Q4 AAPL earnings"

# Dry-run — generate the topic but don't post (useful for reviewing output)
python -m athena.discourse post -c discourse_user.json --prompt "Analyze TSLA options" --dry-run -v

# Use a geopolitics-focused config for a current events topic
python -m athena.discourse post -c d.json --prompt "Review the recent Iran situation and discuss potential stocks to invest in on Monday morning."
```

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--config` | `-c` | string | *(required)* | Path to the bot's JSON config file |
| `--prompt` | `-p` | string | *(required)* | Prompt describing the topic to create |
| `--dry-run` | | flag | off | Generate the topic but don't post to Discourse |
| `--verbose` | `-v` | flag | off | Print tool calls and generated content to stdout |

### `chat` — Interactive CLI Chat

Opens an interactive chat session with the bot in your terminal. The bot
loads its full config (personality, memory, todo, tools) and you can talk
to it directly. Useful for testing personality tuning, tool usage, and
memory/todo management.

```bash
# Start a chat session
python -m athena.discourse chat -c discourse_finance.json

# Verbose mode — see tool calls and internal thinking
python -m athena.discourse chat -c discourse_finance.json -v
```

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--config` | `-c` | string | *(required)* | Path to the bot's JSON config file |
| `--verbose` | `-v` | flag | off | Print tool calls and thinking to stdout |

### Examples

**Monitor a finance category and reply to relevant threads:**
```bash
python -m athena.discourse run -c discourse_finance.json
```

**Test a single poll cycle without posting anything:**
```bash
python -m athena.discourse run -c discourse_finance.json --dry-run --once -v
```

**Write a new topic about earnings, review it before posting:**
```bash
python -m athena.discourse post -c discourse_user.json \
  --prompt "Write about Q4 AAPL earnings" \
  --dry-run -v
```

**Post a geopolitics-meets-markets analysis:**
```bash
python -m athena.discourse post -c d.json \
  --prompt "Review the recent Iran situation and discuss potential stocks to invest in on Monday morning."
```

**Post an options analysis directly:**
```bash
python -m athena.discourse post -c discourse_user.json \
  --prompt "Analyze TSLA options and what the implied distribution tells us about market expectations"
```

---

## 10. Implementation Phases

| Phase | Deliverable | Complexity |
|-------|-------------|------------|
| 1 | `api.py` — Discourse REST client (fetch topics, posts; create replies) | Low |
| 2 | `state.py` — Thread state model + JSON persistence | Low |
| 3 | `triage.py` — Stage 1 triage with configurable model | Low-Medium |
| 4 | `agent.py` — Stage 2 reply generation with tool loop | Medium |
| 5 | `worker.py` — Polling loop, rate limiting, orchestration, CLI | Medium |
| 6 | Templates + personality tuning | Low |

Phases 1-5 form the MVP. Phase 6 is iterative tuning once it's live.

---

## 11. Open Questions

1. ~~**Should the bot create new topics**, or only reply to existing ones?~~ — Resolved:
   the `post` subcommand creates new topics via `python -m athena.discourse post --prompt "..."`.
2. **Should multiple bots be able to converse with each other** in the same
   thread? If so, we need cross-bot coordination to avoid ping-pong loops.
3. **Quoting**: Should the bot quote specific parts of posts it's replying to
   (Discourse supports `[quote]` BBCode)? This improves readability in long
   threads.
4. **Image/chart support**: The AthenaOS tools can generate charts. Should the
   bot upload chart images to Discourse as part of its replies? Discourse
   supports image uploads via the API.
