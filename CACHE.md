# Caching in ATHENA

## Philosophy

Historical market data doesn't change. Once we know AAPL closed at $150.23 on 2024-03-15, that fact is permanent. There's no reason to ask an API for that same number ever again.

Caching exists to avoid redundant API calls — both within a single session (in-memory) and across sessions (disk). Every pricing data manager follows the same layered approach.

## Cache Layers

### 1. Disk Cache (persistent across runs)

Each pricing data manager stores fetched data as CSV files under `.cache/`:

| Manager   | Path                                | Format       |
|-----------|-------------------------------------|--------------|
| YFinance  | `.cache/yfinance_prices/{SYMBOL}.csv` | Date, Open, High, Low, Close, Adj Close, Volume |
| Databento | `.cache/databento_prices/{SYMBOL}.csv` | Date, Close |
| Massive   | `.cache/massive_prices/{SYMBOL}.csv`  | Date, Close  |

On first access for a symbol, the manager reads the CSV into memory. If the requested date is already there, no API call is made. If not, only the missing range is fetched, merged, and written back.

The `.cache/` directory is gitignored and local to each working directory.

### 2. In-Memory Cache (per session)

Each manager instance holds a `_price_cache: dict[tuple[str, date], Decimal]` dictionary. Once a price is in memory — whether loaded from disk or fetched from an API — all subsequent lookups for that (symbol, date) pair are instant with no I/O.

The in-memory cache is populated from:
- Disk cache (on first access per symbol)
- API responses (on cache miss)

### 3. API Fetch (last resort)

When neither disk nor memory has the needed data, the manager fetches from the API. To minimize future misses, it fetches a broad window (up to 730 calendar days) in a single request and caches every returned date — both in memory and to disk.

## The `--no-cache` Flag

`--no-cache` means: **skip the disk cache, fetch fresh from the API**.

Specifically:
- **Disk reads are skipped** — the manager won't load from CSV files
- **In-memory cache still works** — within the session, the same symbol/date won't be fetched twice
- **Disk writes still happen** — fresh API data is saved to disk so future normal runs benefit

This is useful when you suspect the disk cache has bad data, or you want to ensure everything comes directly from the source.

## Current-Day Prices

Daily close prices (OHLCV bars) are only available after a trading day ends. For "what is the current price?" queries:

- **YFinance**: Uses `ticker.fast_info['lastPrice']` for live/intraday prices
- **Databento**: Uses `tbbo` (tick-by-tick best bid/offer) schema with ~20-minute delay, returning the mid-price of bid/ask. Falls back to the most recent OHLCV close if the live quote fails.
- **Massive**: Fetches the most recent daily aggregate available

## Weekend/Holiday Handling

All managers use a `days_back` loop (up to 8 days) when looking up a date. If you request a price for a Saturday, it returns Friday's close. This handles weekends, holidays, and any other non-trading days transparently.
