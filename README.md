# Agentic Toolkit for Holistic Economic Narratives and Analysis (ATHENA, v0.1.1)

Track your portfolio, individual positions, forex, and other financial instruments in one place. Pull data from the Federal Reserve, Yahoo! Finance, and custom sources. Track metrics to benchmark yourself against hedge funds. Use LLMs to manage your portfolio.

Athena is a toolkit designed to easily track investments and delegate decision-making to a large language model. Our vision is to have LLMs help with tracking major news developments, run scenario analyses, and ultimately open and close positions.

The data structure lives in Excel; it can be edited by you to input positions, override choices, or incorporate your existing portfolios.

**This is an experimental toolkit.** It should be used to inform your research and does not represent any sort of official investment advice.

## Installation

Install via `pip`:
```bash
pip install athenaos
```

Install via GitHub:

```bash
pip install git+https://github.com/wgryc/athena-os.git
```

If you are cloning the repo and installing from local source, run the following: `pip install -e .` or `pip install -e ".[dev]"` to run tests.

## Initial Runs and Reports

Track your portfolio positions, currencies, and values:

```bash
athenaos report sample_transactions.xlsx
```

Hedge fund metrics around your portfolio performance:

```bash
athenaos metrics sample_transactions.xlsx
```

## Data Structure

Transactions and portfolio information are stored in Excel files and are easily editable by users. A transactions file needs the following columns:

- SYMBOL: the ticker symbol. By default, use the Yahoo Finance! ticker format. However, you can use anything that is supported by the pricing APIs you are using.
- DATE AND TIME: ISO format for the date-time transaction is preferred, but you can also use standard Excel dates. If time zone information is missing, we assume NYC time. If times are missing (i.e., only dates are provided) we assume 12pm NYC time.
- TRANSACTION TYPE: we support BUY, SELL, CASH_IN, CASH_OUT, DIVIDEND, INTEREST, FEE, and CURRENCY_EXCHANGE.
- PRICE: the price paid per unit of SYMBOL.
- QUANTITY: the quantity being purchased.
- CURRENCY: the currency being used. We currently support USD, CAD, EUR, TWD, SGD, AUD, JPY, KRW, GBP, BRL, CNY, HKD, MXN, ZAR, CHF, and THB.

When logging currency exchanges, the SYMBOL is the target currency (e.g. "HKD") and the CURRENCY is the source currency (e.g., "USD"). The price is how much of the CURRENCY it costs to buy 1 unit of the target (SYMBOL) currency.

## Agentic Trading

Agents require connections to third-party APIs and API keys should be stored in a local `.env` file. If you are trading stocks, we recommend using the [Massive](https://massive.com/) API while for CBOE-traded commodities, we recommend [DataBento](https://databento.com/). Finally, the current agents use the Emerging Trajectories "events" API to get information and trade on it.

```bash
athenaos demo --commodities demo_commodities.xlsx
```

```bash
athenaos demo --meme-stocks demo_meme_stocks.xlsx
```

```bash
athenaos demo --us-stocks demo_us_stocks.xlsx
```

## Dashboards

Generate interactive HTML dashboards to visualize your portfolio performance, returns, and Sharpe ratio over time.

```bash
athenaos dashboard sample_transactions.xlsx
```

This creates a `portfolio_dashboard.html` file in the current directory. To specify a custom output filename:

```bash
athenaos dashboard sample_transactions.xlsx --output my_dashboard.html
```

If the output file already exists, Athena will automatically append "copy" to the filename to avoid overwriting.

## Frontend & Chat

Launch the interactive web frontend with AI-powered chat:

```bash
athenaos frontend sample_transactions.xlsx
```

Or, set `portfolio_file` in `config.json` and run without arguments:

```bash
athenaos frontend
```

### Configuration (`config.json`)

The frontend reads from a `config.json` file in the current directory. This replaces the old `widgets.json` format (which is still supported for backwards compatibility).

```json
{
    "portfolio_file": "sample_transactions.xlsx",
    "widgets": [
        {"tool": "stock_price_widget", "kwargs": {"symbol": "AAPL"}},
        {"tool": "stock_price_widget", "kwargs": {"symbol": "MSFT"}}
    ],
    "gateways": {
        "telegram": {
            "bot_token": "YOUR_BOT_TOKEN"
        }
    }
}
```

### Telegram Bot Gateway

You can connect a Telegram bot to Athena so that messages you send via Telegram are processed by the same AI chat. Responses appear both in Telegram and in the web frontend (with a "TELEGRAM" badge).

**Setup:**

1. Open Telegram and message [@BotFather](https://t.me/botfather).
2. Send `/newbot`, follow the prompts to name your bot, and copy the bot token.
3. Add the token to your `config.json`:
   ```json
   {
       "gateways": {
           "telegram": {
               "bot_token": "123456789:ABCdefGHIjklMNOpqrsTUVwxyz"
           }
       }
   }
   ```
4. Start the frontend (`athenaos frontend`). You should see `Gateway started: telegram` in the console.
5. Open your bot in Telegram and send a message. The AI will respond in Telegram, and both the question and answer will appear in the web chat panel in real time.

The gateway system is extensible -- additional platforms (Discord, Slack, etc.) can be added via the same `gateways` configuration pattern.

## Scheduled Tasks

The `athenaos frontend` command can also run scheduled tasks when it is running in server mode. To facilitate data entry and review, the tasks are defined via an Excel file. This file has the following columns:

- TASK NAME: a short and pithy name for the task, for easy referencing in chats.
- SCHEDULE: define how often you want to schedule a task. You can write this in plain English (or language of your choice). "Once per hour" or "1230pm ET daily" are both acceptable.
- DESCRIPTION: the actual you want the agents to complete. Agents will choose their own tools unless you specifically call them out.
- LAST RUN: we ask the LLM to update the file to mention when the tool was last run.
- ADDED BY: options hare are typically "user" and "athena". The idea is to track who added the task, as Athena can add tasks directly, too.

Tasks can be added by Athena via tool use. There is a tab at the top of the frontend for easy visual editing of tasks, too.

The default is that there is no tasks file, and one should be added to the `config.json` (or equivalent file) with the `tasks_file` parameter.