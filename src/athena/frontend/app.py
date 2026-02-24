"""Flask application factory for the ATHENA frontend."""

import atexit
import json
import os
import threading
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, Response, render_template, request

import litellm

from ..portfolio import (
    load_portfolio_from_excel,
    save_portfolio_to_excel,
    Portfolio,
    calculate_portfolio_value_by_day,
    calculate_portfolio_value_on_date,
    TransactionType,
)
from ..currency import Currency
from ..agents.agents import format_positions_table, format_cash_balances, format_transaction_log
from ..metrics import (
    calculate_daily_returns,
    calculate_sharpe_ratio_cumulative,
    calculate_sharpe_ratio_by_day_cumulative,
)
from ..dashboard.generator import calculate_drawdown_periods
from ..tools import Tool, VisualTool
from ..tools.prices import GetOptionPrice, GetStockPrice, StockPriceHistory, StockPriceWidget
from ..pricingdata import PricingDataManager
from ..tools.dashboard import (
    PortfolioSummaryWidget,
    PortfolioValueChartWidget,
    DailyReturnsChartWidget,
    SharpeRatioChartWidget,
)
from ..tools.web import NewsSearch, CrawlPage
from ..tools.events import ETEventsSearch, ETEventsWidget
from ..tools.filings import SECFilings, SECFilingContent, SECFilingsWidget, InsiderTrades, InsiderTradesWidget
from ..tools.tasks import ManageTasks, ScheduledTasksWidget
from ..tasks import ScheduledTask, load_tasks_from_excel, save_tasks_to_excel, parse_schedule
from .scheduler import TaskScheduler
from .. import pricingdata

load_dotenv()

CHAT_MODEL = "anthropic/claude-opus-4-5"
MAX_TOKENS = 16000


def _load_config(config_path: Path) -> dict:
    """Load config.json, with backwards compatibility for widgets.json array format.

    Args:
        config_path: Path to config.json. Falls back to widgets.json in the
            same directory if the primary file does not exist.

    Returns:
        Parsed configuration dict, or empty dict on missing/invalid file.
    """
    path = config_path
    if not path.exists():
        # Fall back to widgets.json for backwards compatibility
        path = config_path.parent / "widgets.json"
    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}

    # Backwards compat: bare array = widgets list (old widgets.json format)
    if isinstance(data, list):
        return {"widgets": data}
    return data


def _normalize_widget_rows(widgets_config: list) -> list[list[dict]]:
    """Convert widget config to row-based format.

    Supports both the new row format (``[[{...}, {...}], [{...}]]``) and the
    old flat format (``[{...}, {...}]``) where each dict becomes its own row.

    Args:
        widgets_config: Widget configuration list from config.json.

    Returns:
        List of rows, where each row is a list of widget dicts (max 4 per row).

    Raises:
        ValueError: If any row contains more than 4 widgets.
    """
    if not widgets_config:
        return []
    if isinstance(widgets_config[0], dict):
        # Old flat format: each widget is its own row
        return [[w] for w in widgets_config]
    rows = []
    for row in widgets_config:
        if not isinstance(row, list):
            continue
        if len(row) > 4:
            raise ValueError(f"Widget row has {len(row)} items; maximum is 4")
        rows.append(row)
    return rows


def _flatten_widget_rows(rows: list[list[dict]]) -> list[dict]:
    """Flatten row-based widget config to a flat list.

    Args:
        rows: Row-based widget configuration (list of lists).

    Returns:
        Single flat list of widget dicts.
    """
    return [w for row in rows for w in row]


def _build_widget_context(
    widgets_list: list[dict],
    visual_tools_by_name: dict[str, VisualTool],
) -> str:
    """Execute configured widgets and collect ``to_context()`` from each.

    Args:
        widgets_list: Flat list of widget config dicts (each with ``tool``
            and optional ``kwargs`` keys).
        visual_tools_by_name: Mapping of widget name to ``VisualTool`` instance.

    Returns:
        Newline-joined context strings, one bullet per widget.
    """
    lines: list[str] = []
    for entry in widgets_list:
        tool_name = entry.get("tool")
        kwargs = entry.get("kwargs", {})
        tool = visual_tools_by_name.get(tool_name)
        if tool is None:
            continue
        try:
            tool.execute(**kwargs)
            lines.append(f"- {tool.to_context()}")
        except Exception:
            continue

    return "\n".join(lines)


_LITELLM_FIELDS = frozenset(("role", "content", "tool_calls", "tool_call_id"))


def _strip_extra_fields(msg: dict) -> dict:
    """Return a copy of *msg* keeping only fields recognized by litellm.

    Args:
        msg: A conversation message dict that may contain extra keys
            (e.g. ``source``).

    Returns:
        New dict containing only the keys in ``_LITELLM_FIELDS``.
    """
    return {k: v for k, v in msg.items() if k in _LITELLM_FIELDS}


def _build_system_prompt(
    portfolio: Portfolio,
    widget_context: str = "",
    portfolio_file: str = "",
    portfolio_description: str = "",
    tasks: list | None = None,
) -> str:
    """Build a system prompt containing portfolio context for the chat.

    Args:
        portfolio: The loaded portfolio to summarize.
        widget_context: Pre-rendered context strings from live widgets.
        portfolio_file: Path to the portfolio Excel file (for display).
        portfolio_description: Human-readable portfolio description.
        tasks: Optional list of ``ScheduledTask`` objects to include.

    Returns:
        Complete system prompt string for the LLM.
    """
    now = datetime.now(timezone.utc)

    positions_table = format_positions_table(portfolio, now)
    cash = format_cash_balances(portfolio, now)
    transactions = format_transaction_log(portfolio)
    total_value = calculate_portfolio_value_on_date(portfolio, now, Currency.USD)

    widget_section = ""
    if widget_context:
        widget_section = f"""

## Live Widget Data
The user has the following live data widgets visible on their dashboard:

{widget_context}
"""

    portfolio_source = ""
    if portfolio_file or portfolio_description:
        parts = []
        if portfolio_file:
            parts.append(f"**Source file:** {portfolio_file}")
        if portfolio_description:
            parts.append(f"**Description:** {portfolio_description}")
        portfolio_source = "\n## Portfolio Source\n" + "\n".join(parts) + "\n"

    tasks_section = ""
    if tasks:
        task_lines = []
        for i, t in enumerate(tasks):
            last = t.last_run.strftime("%Y-%m-%d %H:%M UTC") if t.last_run else "never"
            task_lines.append(f"  [{i}] {t.name} | Schedule: {t.schedule} | Last run: {last}")
        tasks_section = (
            "\n## Scheduled Tasks\n"
            "The following tasks are configured to run on schedule:\n"
            + "\n".join(task_lines)
            + "\nYou can manage these tasks using the manage_tasks tool.\n"
        )

    return f"""You are ATHENA, an expert financial portfolio analyst. You are helping the user \
understand and analyze their investment portfolio.
{portfolio_source}
You have access to the following portfolio data:

## Current Positions
{positions_table}

## Cash Balances
{cash}

## Total Portfolio Market Value
${total_value:,.2f} USD

## Transaction History
{transactions}
{widget_section}{tasks_section}
## Instructions
- Answer questions about the portfolio accurately based on the data above.
- You can calculate metrics, explain holdings, discuss diversification, and offer analytical insights.
- When the user asks about specific holdings, reference the actual data.
- Format financial numbers with commas and 2 decimal places.
- Use markdown formatting in your responses for readability.
- Be concise but thorough.
- You have access to tools that can fetch live stock prices. Use them when the user asks for \
current or recent prices of specific tickers, especially for tickers not shown in the portfolio data above.
- If the user has live widget data visible, you can reference those prices directly without needing to call tools.
- The current date/time is {now.strftime('%Y-%m-%d %H:%M UTC')}."""


def _build_dashboard_data(
    portfolio: Portfolio,
    risk_free_rate: float = 0.04,
) -> dict:
    """Pre-compute all dashboard metrics for a portfolio.

    Args:
        portfolio: Portfolio to analyze.
        risk_free_rate: Annual risk-free rate for Sharpe ratio calculation.

    Returns:
        Dict containing portfolio value chart data, daily returns chart data,
        Sharpe ratio chart data, drawdown periods, and summary statistics.
    """
    target_currency = portfolio.primary_currency

    print("  [1/4] Calculating portfolio values …", flush=True)
    portfolio_values = calculate_portfolio_value_by_day(portfolio, target_currency)

    if not portfolio_values:
        return {"error": "No portfolio values available"}

    sorted_dates = sorted(portfolio_values.keys())
    start_value = float(portfolio_values[sorted_dates[0]])
    end_value = float(portfolio_values[sorted_dates[-1]])
    total_return = ((end_value - start_value) / start_value * 100) if start_value != 0 else 0

    print("  [2/4] Computing Sharpe ratio …", flush=True)
    try:
        _daily_sharpe, annual_sharpe = calculate_sharpe_ratio_cumulative(
            portfolio, target_currency, risk_free_rate
        )
    except ValueError:
        annual_sharpe = None

    daily_returns = calculate_daily_returns(portfolio_values)

    print("  [3/4] Computing rolling Sharpe ratio …", flush=True)
    sharpe_by_day = calculate_sharpe_ratio_by_day_cumulative(
        portfolio, target_currency, risk_free_rate
    )

    print("  [4/4] Computing drawdown periods …", flush=True)
    drawdown_periods = calculate_drawdown_periods(portfolio_values)

    value_chart = {
        "labels": [d.strftime("%Y-%m-%d") for d in sorted_dates],
        "values": [float(portfolio_values[d]) for d in sorted_dates],
    }

    returns_sorted = sorted(daily_returns.keys())
    returns_chart = {
        "labels": [d.strftime("%Y-%m-%d") for d in returns_sorted],
        "values": [daily_returns[d] * 100 for d in returns_sorted],
    }

    sharpe_dates = sorted(sharpe_by_day.keys())
    sharpe_chart = {
        "labels": [d.strftime("%Y-%m-%d") for d in sharpe_dates],
        "values": [sharpe_by_day[d][1] for d in sharpe_dates],
    }

    return {
        "currency": target_currency.value,
        "start_date": sorted_dates[0].strftime("%Y-%m-%d"),
        "end_date": sorted_dates[-1].strftime("%Y-%m-%d"),
        "start_value": start_value,
        "end_value": end_value,
        "total_return": round(total_return, 2),
        "annual_sharpe": round(annual_sharpe, 2) if annual_sharpe is not None else None,
        "value_chart": value_chart,
        "returns_chart": returns_chart,
        "sharpe_chart": sharpe_chart,
        "drawdown_periods": drawdown_periods,
    }


def create_app(
    portfolio_file: str | None = None,
    pricing_manager: PricingDataManager | None = None,
    force_cache_refresh: bool = False,
) -> Flask:
    """Create and configure the Flask application.

    Args:
        portfolio_file: Path to the portfolio Excel file. If ``None``, the
            app starts without a loaded portfolio.
        pricing_manager: Optional shared ``PricingDataManager`` instance.
            When provided, it is attached to the portfolio for live pricing.
        force_cache_refresh: When ``True``, bypass disk cache for price
            history lookups.

    Returns:
        Configured Flask application instance with all routes registered.
    """
    app = Flask(__name__)

    # ── Configuration ────────────────────────────────────
    config_path = Path.cwd() / "config.json"
    config = _load_config(config_path)
    widget_rows: list[list[dict]] = _normalize_widget_rows(
        config.get("widgets", [])
    )

    portfolio_file_path: str | None = portfolio_file
    portfolio: Portfolio | None = None
    system_prompt: str = ""
    conversation_messages: list[dict] = []
    conversation_lock = threading.Lock()
    cached_dashboard: dict | None = None
    _gateway_stream: dict | None = None  # {"source": "...", "content": "..."}

    # ── Tasks ────────────────────────────────────────────
    tasks_file_path: str | None = config.get("tasks_file")
    scheduled_tasks: list[ScheduledTask] = []

    if tasks_file_path and Path(tasks_file_path).exists():
        scheduled_tasks = load_tasks_from_excel(tasks_file_path)
        print(f"Loaded {len(scheduled_tasks)} scheduled task(s).")
    elif tasks_file_path:
        print(f"Tasks file not found: {tasks_file_path} (starting with empty task list)")

    def _get_tasks() -> list[ScheduledTask]:
        return scheduled_tasks

    def _add_task(name: str, schedule: str, description: str,
                  added_by: str = "user") -> ScheduledTask:
        interval, daily_time = parse_schedule(schedule)
        task = ScheduledTask(
            name=name, schedule=schedule, description=description,
            last_run=None, added_by=added_by,
        )
        task._interval_seconds = interval
        task._daily_time_utc = daily_time
        scheduled_tasks.append(task)
        if tasks_file_path:
            save_tasks_to_excel(scheduled_tasks, tasks_file_path)
        return task

    def _add_task_from_llm(name: str, schedule: str, description: str) -> ScheduledTask:
        return _add_task(name, schedule, description, added_by="athena")

    def _update_task(idx: int, updates: dict) -> bool:
        if idx < 0 or idx >= len(scheduled_tasks):
            return False
        task = scheduled_tasks[idx]
        if "name" in updates:
            task.name = updates["name"]
        if "schedule" in updates:
            task.schedule = updates["schedule"]
            interval, daily_time = parse_schedule(task.schedule)
            task._interval_seconds = interval
            task._daily_time_utc = daily_time
        if "description" in updates:
            task.description = updates["description"]
        if tasks_file_path:
            save_tasks_to_excel(scheduled_tasks, tasks_file_path)
        return True

    def _delete_task(idx: int) -> bool:
        if idx < 0 or idx >= len(scheduled_tasks):
            return False
        scheduled_tasks.pop(idx)
        if tasks_file_path:
            save_tasks_to_excel(scheduled_tasks, tasks_file_path)
        return True

    # Tool registry (LLM-callable)
    available_tools: list[Tool] = [
        GetStockPrice(pricing_manager=pricing_manager),
        GetOptionPrice(),
        StockPriceHistory(force_cache_refresh=force_cache_refresh),
        NewsSearch(),
        CrawlPage(),
        ETEventsSearch(),
        SECFilings(),
        SECFilingContent(),
        InsiderTrades(),
        ManageTasks(
            get_tasks=_get_tasks,
            add_task=_add_task_from_llm,
            update_task=_update_task,
            delete_task=_delete_task,
        ),
    ]
    tools_json = [t.to_json() for t in available_tools]
    tools_by_name: dict[str, Tool] = {t.name: t for t in available_tools}
    tool_labels: dict[str, str] = {t.name: t.label for t in available_tools}

    def get_dashboard_data() -> dict | None:
        """Return cached dashboard data, computing lazily if needed."""
        nonlocal cached_dashboard
        if cached_dashboard is not None:
            return cached_dashboard
        if portfolio is None:
            return None
        try:
            cached_dashboard = _build_dashboard_data(portfolio)
        except Exception:
            return None
        return cached_dashboard

    # Visual tool registry (user-facing widgets)
    visual_tools: list[VisualTool] = [
        StockPriceWidget(),
        PortfolioSummaryWidget(get_dashboard_data),
        PortfolioValueChartWidget(get_dashboard_data),
        DailyReturnsChartWidget(get_dashboard_data),
        SharpeRatioChartWidget(get_dashboard_data),
        SECFilingsWidget(),
        InsiderTradesWidget(),
        ETEventsWidget(),
        ScheduledTasksWidget(_get_tasks),
    ]
    visual_tools_by_name: dict[str, VisualTool] = {t.name: t for t in visual_tools}

    portfolio_description: str = config.get("portfolio_description", "")

    if portfolio_file:
        if not Path(portfolio_file).exists():
            raise FileNotFoundError(
                f"Portfolio file not found: {portfolio_file}"
            )

        portfolio = load_portfolio_from_excel(
            portfolio_file,
            primary_currency=Currency.USD,
            error_out_negative_cash=False,
            error_out_negative_quantity=False,
        )

        if pricing_manager is not None:
            portfolio.pricing_manager = pricing_manager

        print("Pre-computing dashboard metrics…")
        pricingdata.verbose = True
        try:
            cached_dashboard = _build_dashboard_data(portfolio)
            print("Dashboard metrics ready.")
        except Exception as e:
            print(f"Warning: failed to pre-compute dashboard: {e}")
            cached_dashboard = None

        flat_widgets = _flatten_widget_rows(widget_rows)
        widget_ctx = _build_widget_context(flat_widgets, visual_tools_by_name)
        system_prompt = _build_system_prompt(
            portfolio, widget_ctx,
            portfolio_file=portfolio_file,
            portfolio_description=portfolio_description,
            tasks=scheduled_tasks,
        )

    # ── Page ──────────────────────────────────────────────

    show_investing_warning = config.get("show_investing_warning") is not False

    @app.route("/")
    def index():
        return render_template("index.html", show_investing_warning=show_investing_warning)

    # ── Chat API ──────────────────────────────────────────

    @app.route("/api/chat", methods=["POST"])
    def chat():
        nonlocal conversation_messages

        if portfolio is None:
            return {"error": "No portfolio loaded"}, 400

        data = request.get_json()
        if not data or "message" not in data:
            return {"error": "Missing 'message' field"}, 400

        user_message = data["message"].strip()
        if not user_message:
            return {"error": "Empty message"}, 400

        if not os.getenv("ANTHROPIC_API_KEY"):
            return {"error": "ANTHROPIC_API_KEY not set"}, 500

        conversation_messages.append({"role": "user", "content": user_message, "source": "web"})

        litellm_messages = (
            [{"role": "system", "content": system_prompt}]
            + [_strip_extra_fields(m) for m in conversation_messages]
        )

        def generate():
            full_response = ""
            messages = list(litellm_messages)

            try:
                while True:
                    response = litellm.completion(
                        model=CHAT_MODEL,
                        messages=messages,
                        max_tokens=MAX_TOKENS,
                        tools=tools_json,
                        stream=True,
                    )

                    # Accumulate streamed tool calls by index
                    tool_calls_acc: dict[int, dict] = {}
                    chunk_content = ""

                    for chunk in response:
                        delta = chunk.choices[0].delta

                        if delta.content:
                            chunk_content += delta.content
                            full_response += delta.content
                            sse_data = json.dumps({"type": "chunk", "content": delta.content})
                            yield f"data: {sse_data}\n\n"

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
                                        tool_calls_acc[idx]["function"]["name"] += tc.function.name
                                    if tc.function.arguments:
                                        tool_calls_acc[idx]["function"]["arguments"] += tc.function.arguments

                    # If no tool calls were made, we're done
                    if not tool_calls_acc:
                        break

                    # Build the assistant message that includes the tool calls
                    tc_list = [tool_calls_acc[i] for i in sorted(tool_calls_acc)]
                    assistant_msg: dict = {
                        "role": "assistant",
                        "content": chunk_content or None,
                        "tool_calls": tc_list,
                    }
                    messages.append(assistant_msg)
                    conversation_messages.append(assistant_msg)

                    # Notify frontend that tools are being executed
                    tool_display = [tool_labels.get(tc["function"]["name"], tc["function"]["name"]) for tc in tc_list]
                    yield f"data: {json.dumps({'type': 'tool_use', 'tools': tool_display})}\n\n"

                    # Execute each tool and append results
                    for tc in tc_list:
                        tool = tools_by_name.get(tc["function"]["name"])
                        if tool:
                            try:
                                result = tool.handle_tool_call(tc["function"]["arguments"])
                            except Exception as exc:
                                result = f"Error executing {tc['function']['name']}: {exc}"
                        else:
                            result = f"Error: unknown tool '{tc['function']['name']}'"

                        tool_msg = {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": result,
                        }
                        messages.append(tool_msg)
                        conversation_messages.append(tool_msg)

                    # Signal a new message bubble before the follow-up response
                    yield f"data: {json.dumps({'type': 'new_message'})}\n\n"

                    # Loop back to let the model produce a final response

                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                conversation_messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                error_msg = str(e)
                yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"
                if conversation_messages and conversation_messages[-1]["role"] == "user":
                    conversation_messages.pop()

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.route("/api/reset", methods=["POST"])
    def reset():
        nonlocal conversation_messages
        conversation_messages = []
        return {"status": "ok"}

    @app.route("/api/chatlog")
    def chatlog():
        return {
            "system_prompt": system_prompt,
            "messages": conversation_messages,
            "tools": tools_json,
        }

    # ── Widgets API ─────────────────────────────────────

    @app.route("/api/widgets")
    def widgets():
        if not widget_rows:
            return {"rows": []}

        result_rows = []
        for row in widget_rows:
            row_results = []
            for entry in row:
                tool_name = entry.get("tool")
                kwargs = entry.get("kwargs", {})
                tool = visual_tools_by_name.get(tool_name)
                if tool is None:
                    row_results.append({
                        "tool": tool_name,
                        "kwargs": kwargs,
                        "html": f'<div class="widget-card widget-error">Unknown widget: {tool_name}</div>',
                        "error": f"Unknown widget: {tool_name}",
                        "chart_config": None,
                    })
                    continue

                try:
                    tool.execute(**kwargs)
                    row_results.append({
                        "tool": tool_name,
                        "kwargs": kwargs,
                        "html": tool.to_html(),
                        "chart_config": getattr(tool, "chart_config", None),
                    })
                except Exception as e:
                    row_results.append({
                        "tool": tool_name,
                        "kwargs": kwargs,
                        "html": f'<div class="widget-card widget-error">{tool_name}: {e}</div>',
                        "error": str(e),
                        "chart_config": None,
                    })

            result_rows.append(row_results)

        return {"rows": result_rows}

    @app.route("/api/widgets/reload", methods=["POST"])
    def widgets_reload():
        """Re-read config.json and update widget_rows if changed."""
        nonlocal widget_rows, system_prompt
        new_config = _load_config(config_path)
        new_rows = _normalize_widget_rows(new_config.get("widgets", []))
        changed = new_rows != widget_rows
        widget_rows[:] = []
        widget_rows.extend(new_rows)
        if changed and portfolio is not None:
            flat_widgets = _flatten_widget_rows(widget_rows)
            widget_ctx = _build_widget_context(flat_widgets, visual_tools_by_name)
            system_prompt = _build_system_prompt(
                portfolio, widget_ctx,
                portfolio_file=portfolio_file_path,
                portfolio_description=portfolio_description,
                tasks=scheduled_tasks,
            )
        return {"status": "ok", "changed": changed}

    @app.route("/api/widgets/available")
    def widgets_available():
        """List all registered visual tools and their parameter schemas."""
        result = []
        for tool in visual_tools:
            params = tool.parameters
            props = params.get("properties", {})
            required = params.get("required", [])
            result.append({
                "name": tool.name,
                "description": tool.description,
                "has_params": len(props) > 0,
                "params": [
                    {
                        "name": k,
                        "type": v.get("type", "string"),
                        "description": v.get("description", ""),
                        "required": k in required,
                    }
                    for k, v in props.items()
                ],
            })
        return {"widgets": result}

    @app.route("/api/widgets/config", methods=["GET"])
    def widgets_config_get():
        """Return current widget layout config."""
        return {"widgets": widget_rows}

    @app.route("/api/widgets/config", methods=["PUT"])
    def widgets_config_put():
        """Save new widget layout to config.json and update state."""
        nonlocal widget_rows, system_prompt
        data = request.get_json()
        if not data or "widgets" not in data:
            return {"error": "Missing 'widgets' field"}, 400

        new_rows = _normalize_widget_rows(data["widgets"])
        widget_rows[:] = []
        widget_rows.extend(new_rows)

        # Persist to config.json
        full_config = _load_config(config_path)
        full_config["widgets"] = new_rows
        config_path.write_text(json.dumps(full_config, indent=4) + "\n")

        # Rebuild system prompt with new widget context
        if portfolio is not None:
            flat_widgets = _flatten_widget_rows(widget_rows)
            widget_ctx = _build_widget_context(flat_widgets, visual_tools_by_name)
            system_prompt = _build_system_prompt(
                portfolio, widget_ctx,
                portfolio_file=portfolio_file_path,
                portfolio_description=portfolio_description,
                tasks=scheduled_tasks,
            )

        return {"status": "ok"}

    # ── Dashboard API ─────────────────────────────────────

    def _invalidate_dashboard():
        nonlocal cached_dashboard
        cached_dashboard = None

    @app.route("/api/dashboard")
    def dashboard_data():
        nonlocal cached_dashboard

        if portfolio is None:
            return {"error": "No portfolio loaded"}, 400

        if cached_dashboard is not None:
            return cached_dashboard

        try:
            cached_dashboard = _build_dashboard_data(portfolio)
        except Exception as e:
            return {"error": str(e)}, 500

        return cached_dashboard

    # ── Transaction CRUD API ──────────────────────────────

    @app.route("/api/transactions")
    def list_transactions():
        if portfolio is None:
            return {"error": "No portfolio loaded"}, 400

        txns = []
        for i, txn in enumerate(portfolio.transactions):
            txns.append({
                "index": i,
                "symbol": txn.symbol,
                "datetime": txn.transaction_datetime.isoformat(),
                "transaction_type": txn.transaction_type.value,
                "quantity": str(txn.quantity),
                "price": str(txn.price),
                "currency": txn.currency.value,
            })

        return {
            "transactions": txns,
            "transaction_types": [t.value for t in TransactionType],
        }

    @app.route("/api/transactions", methods=["POST"])
    def add_transaction():
        if portfolio is None:
            return {"error": "No portfolio loaded"}, 400

        data = request.get_json()
        required = ["symbol", "datetime", "transaction_type", "quantity", "price", "currency"]
        for field in required:
            if field not in data:
                return {"error": f"Missing field: {field}"}, 400

        try:
            txn_dt = datetime.fromisoformat(data["datetime"])
            txn_type = TransactionType(data["transaction_type"])
            quantity = Decimal(str(data["quantity"]))
            price = Decimal(str(data["price"]))
            currency = Currency(data["currency"])
        except (ValueError, KeyError) as e:
            return {"error": f"Invalid value: {e}"}, 400

        portfolio.add_transaction_now(
            symbol=data["symbol"].upper().strip(),
            transaction_type=txn_type,
            quantity=quantity,
            price=price,
            currency=currency,
            transaction_datetime=txn_dt,
        )

        _invalidate_dashboard()
        return {"status": "ok", "index": len(portfolio.transactions) - 1}

    @app.route("/api/transactions/<int:idx>", methods=["DELETE"])
    def delete_transaction(idx: int):
        if portfolio is None:
            return {"error": "No portfolio loaded"}, 400

        if idx < 0 or idx >= len(portfolio.transactions):
            return {"error": f"Invalid index: {idx}"}, 400

        portfolio.transactions.pop(idx)
        _invalidate_dashboard()
        return {"status": "ok"}

    # ── Tasks CRUD API ───────────────────────────────────

    @app.route("/api/tasks")
    def list_tasks():
        return {
            "tasks": [
                {
                    "index": i,
                    "name": t.name,
                    "schedule": t.schedule,
                    "description": t.description,
                    "last_run": t.last_run.isoformat() if t.last_run else None,
                    "added_by": t.added_by,
                }
                for i, t in enumerate(scheduled_tasks)
            ]
        }

    @app.route("/api/tasks", methods=["POST"])
    def add_task_route():
        data = request.get_json()
        for fld in ("name", "schedule", "description"):
            if not data or fld not in data:
                return {"error": f"Missing field: {fld}"}, 400
        try:
            _add_task(data["name"], data["schedule"], data["description"])
        except ValueError as e:
            return {"error": str(e)}, 400
        return {"status": "ok", "index": len(scheduled_tasks) - 1}

    @app.route("/api/tasks/<int:idx>", methods=["PUT"])
    def update_task_route(idx: int):
        data = request.get_json()
        if not data:
            return {"error": "No data"}, 400
        try:
            ok = _update_task(idx, data)
        except ValueError as e:
            return {"error": str(e)}, 400
        if not ok:
            return {"error": f"Invalid index: {idx}"}, 400
        return {"status": "ok"}

    @app.route("/api/tasks/<int:idx>", methods=["DELETE"])
    def delete_task_route(idx: int):
        if not _delete_task(idx):
            return {"error": f"Invalid index: {idx}"}, 400
        return {"status": "ok"}

    @app.route("/api/tasks/save", methods=["POST"])
    def save_tasks_route():
        if not tasks_file_path:
            return {"error": "No tasks_file configured"}, 400
        try:
            save_tasks_to_excel(scheduled_tasks, tasks_file_path)
        except Exception as e:
            return {"error": str(e)}, 500
        return {"status": "ok"}

    @app.route("/api/tasks/<int:idx>/run", methods=["POST"])
    def run_task_route(idx: int):
        if idx < 0 or idx >= len(scheduled_tasks):
            return {"error": f"Invalid index: {idx}"}, 400
        task = scheduled_tasks[idx]
        prompt = (
            f"[Scheduled Task: {task.name}]\n"
            f"Schedule: {task.schedule}\n\n"
            f"{task.description}"
        )
        # Run via gateway handler (thread-safe, shows in chat)
        threading.Thread(
            target=_run_task_in_background,
            args=(task, prompt),
            daemon=True,
        ).start()
        return {"status": "ok", "message": f"Task '{task.name}' triggered."}

    def _run_task_in_background(task: ScheduledTask, prompt: str) -> None:
        try:
            response = process_gateway_message(prompt, "scheduler")
            task.last_run = datetime.now(timezone.utc)
            if tasks_file_path:
                save_tasks_to_excel(scheduled_tasks, tasks_file_path)
            # Forward response to all active gateways (e.g. Telegram)
            for gw in active_gateways:
                try:
                    gw.send_message(f"[{task.name}]\n{response}")
                except Exception as e:
                    print(f"[Run Now] Failed to forward to {gw.name}: {e}", flush=True)
        except Exception as e:
            print(f"[Run Now] Task '{task.name}' failed: {e}", flush=True)

    # ── Save API ──────────────────────────────────────────

    @app.route("/api/save", methods=["POST"])
    def save():
        nonlocal system_prompt

        if portfolio is None or portfolio_file_path is None:
            return {"error": "No portfolio loaded"}, 400

        try:
            save_portfolio_to_excel(portfolio, portfolio_file_path)
        except Exception as e:
            return {"error": str(e)}, 500

        flat_widgets = _flatten_widget_rows(widget_rows)
        widget_ctx = _build_widget_context(flat_widgets, visual_tools_by_name)
        system_prompt = _build_system_prompt(
            portfolio, widget_ctx,
            portfolio_file=portfolio_file_path,
            portfolio_description=portfolio_description,
        )
        return {"status": "ok"}

    # ── Messages API (for gateway sync) ──────────────────

    @app.route("/api/messages")
    def messages():
        return {
            "messages": conversation_messages,
            "count": len(conversation_messages),
            "gateway_stream": _gateway_stream,
            "tool_labels": tool_labels,
        }

    # ── Gateway Message Handler ──────────────────────────

    def process_gateway_message(text: str, source: str) -> str:
        """Process a message from an external gateway. Thread-safe.

        Uses streaming so the frontend can show progressive updates
        via the ``gateway_stream`` field in ``/api/messages``.
        """
        nonlocal conversation_messages, _gateway_stream

        if portfolio is None:
            return "No portfolio loaded."

        if not os.getenv("ANTHROPIC_API_KEY"):
            return "ANTHROPIC_API_KEY not set."

        with conversation_lock:
            conversation_messages.append({"role": "user", "content": text, "source": source})

            llm_messages = (
                [{"role": "system", "content": system_prompt}]
                + [_strip_extra_fields(m) for m in conversation_messages]
            )

            full_response = ""
            _gateway_stream = {"source": source, "content": ""}

            try:
                while True:
                    response = litellm.completion(
                        model=CHAT_MODEL,
                        messages=llm_messages,
                        max_tokens=MAX_TOKENS,
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
                            _gateway_stream["content"] = full_response

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
                                        tool_calls_acc[idx]["function"]["name"] += tc.function.name
                                    if tc.function.arguments:
                                        tool_calls_acc[idx]["function"]["arguments"] += tc.function.arguments

                    if not tool_calls_acc:
                        break

                    # Handle tool calls
                    tc_list = [tool_calls_acc[i] for i in sorted(tool_calls_acc)]
                    assistant_msg = {
                        "role": "assistant",
                        "content": chunk_content or None,
                        "tool_calls": tc_list,
                    }
                    llm_messages.append(assistant_msg)
                    conversation_messages.append(assistant_msg)

                    # Signal tool usage for frontend polling
                    tool_display = [tool_labels.get(tc["function"]["name"], tc["function"]["name"]) for tc in tc_list]
                    _gateway_stream["tools"] = tool_display

                    for tc in tc_list:
                        tool = tools_by_name.get(tc["function"]["name"])
                        if tool:
                            try:
                                result = tool.handle_tool_call(tc["function"]["arguments"])
                            except Exception as exc:
                                result = f"Error executing {tc['function']['name']}: {exc}"
                        else:
                            result = f"Error: unknown tool '{tc['function']['name']}'"
                        tool_msg = {"role": "tool", "tool_call_id": tc["id"], "content": result}
                        llm_messages.append(tool_msg)
                        conversation_messages.append(tool_msg)

                    # Reset stream content for next response chunk
                    _gateway_stream["tools"] = None
                    full_response = ""
                    _gateway_stream["content"] = ""

                conversation_messages.append({"role": "assistant", "content": full_response, "source": source})
            except Exception as e:
                full_response = f"Error: {e}"
                if conversation_messages and conversation_messages[-1].get("source") == source:
                    conversation_messages.pop()
            finally:
                _gateway_stream = None

        return full_response

    # ── Gateways ─────────────────────────────────────────

    from .gateways import create_gateways, TelegramGateway

    def _persist_telegram_chat_id(chat_id: int) -> None:
        """Save the Telegram chat_id to config.json so it survives restarts."""
        try:
            full_config = _load_config(config_path)
            full_config.setdefault("gateways", {}).setdefault("telegram", {})["chat_id"] = chat_id
            config_path.write_text(json.dumps(full_config, indent=4) + "\n")
            print(f"[Telegram] Persisted chat_id {chat_id} to config.json")
        except Exception as e:
            print(f"[Telegram] Failed to persist chat_id: {e}")

    active_gateways = create_gateways(config)
    for gw in active_gateways:
        gw.set_message_handler(process_gateway_message)
        if isinstance(gw, TelegramGateway):
            gw._on_chat_id_changed = _persist_telegram_chat_id
        try:
            gw.start()
            print(f"Gateway started: {gw.name}")
        except Exception as e:
            print(f"Warning: failed to start {gw.name} gateway: {e}")

    def _shutdown_gateways():
        for gw in active_gateways:
            try:
                gw.stop()
            except Exception:
                pass

    atexit.register(_shutdown_gateways)

    # ── Task Scheduler ────────────────────────────────────

    _scheduler: TaskScheduler | None = None
    if scheduled_tasks and tasks_file_path:
        _scheduler = TaskScheduler(
            get_tasks=_get_tasks,
            process_message=process_gateway_message,
            tasks_file=tasks_file_path,
            check_interval=60,
            gateways=active_gateways,
        )
        _scheduler.start()
        print("Task scheduler started.")

    def _shutdown_scheduler():
        if _scheduler:
            _scheduler.stop()

    atexit.register(_shutdown_scheduler)

    return app
