from abc import ABC, abstractmethod
from decimal import Decimal
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
import os
import sys
import warnings

import yfinance as yf  # type: ignore[import-untyped]
import pandas as pd
from massive import RESTClient  # type: ignore[import-untyped]
import databento as db  # type: ignore[import-untyped]

from .currency import Currency

# Track which symbols have been force-refreshed this session
_refreshed_symbols: set[str] = set()

# When True, print status messages during data fetching (e.g. "Fetching AAPL …").
# Defaults to False so CLI commands aren't polluted; the frontend sets this to True.
verbose: bool = False


def get_yfinance_cache_path(symbol: str) -> Path:
    """Get the cache file path for a given symbol.

    Args:
        symbol: The ticker symbol (e.g., "AAPL", "MSFT").

    Returns:
        Path to the CSV cache file under ``.cache/yfinance_prices/``.
    """
    return Path.cwd() / ".cache" / "yfinance_prices" / f"{symbol}.csv"


def fetch_yfinance_data(symbol: str, min_date: date, max_date: date, force_cache_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch yfinance data for a symbol, using cache intelligently.

    If cached data exists, checks if it covers the requested date range.
    If not, expands the request to include all dates from cache + requested range,
    then updates the cache with the merged data.

    Args:
        symbol: The ticker symbol (e.g., "AAPL", "MSFT").
        min_date: The minimum date needed.
        max_date: The maximum date needed.
        force_cache_refresh: If True, force a fresh fetch from Yahoo Finance
            (only once per symbol per session).

    Returns:
        DataFrame with columns: Date, Open, High, Low, Close, Adj Close, Volume.
    """
    cache_path = get_yfinance_cache_path(symbol)

    cached_df: pd.DataFrame | None = None
    cached_min: date | None = None
    cached_max: date | None = None

    # Always try to read cache first (needed for range calculation and fallback)
    if cache_path.exists():
        try:
            cached_df = pd.read_csv(cache_path, parse_dates=['Date'])  # type: ignore[call-overload]
            if not cached_df.empty:
                cached_df['Date'] = pd.to_datetime(cached_df['Date']).dt.date
                cached_min = cached_df['Date'].min()
                cached_max = cached_df['Date'].max()
        except Exception:
            # If cache is corrupted, we'll just refetch
            cached_df = None

    # Determine if we need to fetch new data
    fetch_start = min_date
    fetch_end = max_date

    # Check if we should force refresh (only once per symbol per session)
    should_force_refresh = force_cache_refresh and symbol not in _refreshed_symbols

    if should_force_refresh:
        # Force refresh: fetch fresh data but use cache for range expansion
        need_fetch = True
        if cached_df is not None and not cached_df.empty and cached_min is not None and cached_max is not None:
            # Expand fetch range to cover existing cache too, so we refresh everything
            fetch_start = min(min_date, cached_min)
            fetch_end = max(max_date, cached_max)
    elif cached_df is None or cached_df.empty or cached_min is None or cached_max is None:
        # No valid cache, fetch the full range
        need_fetch = True
    elif min_date < cached_min or max_date > cached_max:
        # We have cached data but it doesn't cover our range
        need_fetch = True
        fetch_start = min(min_date, cached_min)
        fetch_end = max(max_date, cached_max)
    else:
        # Cache covers our range
        need_fetch = False

    if need_fetch:
        # Track that we've refreshed this symbol
        _refreshed_symbols.add(symbol)

        # yfinance end date is exclusive, so add 1 day
        fetch_end_exclusive = fetch_end + timedelta(days=1)

        if verbose:
            print(f"  Fetching {symbol} ({fetch_start} to {fetch_end}) …", flush=True)
        try:
            ticker = yf.Ticker(symbol)
            new_df: pd.DataFrame = ticker.history(  # type: ignore[call-arg]
                start=fetch_start.isoformat(),
                end=fetch_end_exclusive.isoformat(),
                auto_adjust=False
            )
        except Exception as e:
            # yfinance can fail if Yahoo Finance returns None (e.g., rate limiting,
            # requesting data for a date with no trading yet, network issues)
            print(f"Warning: yfinance request failed for {symbol}: {e}", file=sys.stderr)
            if cached_df is not None and not cached_df.empty:
                return cached_df
            return pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])

        if new_df.empty:
            # Empty response often indicates rate limiting from Yahoo Finance
            print(f"Warning: yfinance returned no data for {symbol} (possible rate limiting)", file=sys.stderr)
            # If we got no data but have cache, return cache
            if cached_df is not None and not cached_df.empty:
                return cached_df
            # Otherwise return empty DataFrame with expected columns
            return pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])

        # Reset index to make Date a column
        new_df = new_df.reset_index()
        new_df['Date'] = pd.to_datetime(new_df['Date']).dt.date

        # Keep only the columns we need
        columns_to_keep = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        available_columns = [c for c in columns_to_keep if c in new_df.columns]
        new_df = new_df[available_columns]

        # Merge with cached data if we had any
        if cached_df is not None and not cached_df.empty:
            # Combine and remove duplicates, keeping newer data
            combined_df = pd.concat([cached_df, new_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['Date'], keep='last')
            combined_df = combined_df.sort_values('Date').reset_index(drop=True)
        else:
            combined_df = new_df.sort_values('Date').reset_index(drop=True)

        # Save to cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(cache_path, index=False)

        return combined_df

    # Cache covers our range, return it (we know it's not None here)
    assert cached_df is not None
    return cached_df

class PricePoint:
    """A single price observation for a financial instrument."""

    def __init__(self, symbol: str, price_datetime: datetime, price: Decimal, base_currency:Currency = Currency.USD):
        """Initialize a PricePoint.

        Args:
            symbol: Ticker or contract symbol (e.g., "AAPL", "NGH26").
            price_datetime: The datetime the price was observed.
            price: The observed price as a Decimal.
            base_currency: Currency the price is denominated in.
        """
        self.symbol: str = symbol
        self.price_datetime: datetime = price_datetime
        self.price: Decimal = price
        self.base_currency:Currency = base_currency

class PricingDataManager(ABC):
    """Abstract base class for all pricing data providers."""

    @abstractmethod
    def get_price_point(self, symbol: str, price_datetime: datetime) -> PricePoint:
        raise NotImplementedError("This method should be overridden by subclasses.")

class FixedPricingDataManager(PricingDataManager):
    """Pricing manager that returns a fixed price for any symbol/datetime."""

    def __init__(self, price_for_everything:Decimal = Decimal("1.0")):
        """Initialize with a fixed price.

        Args:
            price_for_everything: The constant price returned for every query.
        """
        self.price = price_for_everything

    def get_price_point(self, symbol: str, price_datetime: datetime) -> PricePoint:
        """Return a PricePoint with the fixed price.

        Args:
            symbol: The ticker symbol (ignored; always returns the fixed price).
            price_datetime: The requested datetime (passed through to the result).

        Returns:
            PricePoint with the fixed price in USD.
        """
        price_point = PricePoint(
            symbol=symbol,
            price_datetime=price_datetime,
            price=self.price,
            base_currency=Currency.USD
        )

        return price_point

class YFinancePricingDataManager(PricingDataManager):

    def __init__(self, force_cache_refresh: bool = False):
        """Initialize the YFinance pricing manager.

        Args:
            force_cache_refresh: If True, bypass the disk cache and fetch
                fresh data from Yahoo Finance (once per symbol per session).
        """
        self.force_cache_refresh = force_cache_refresh

    def get_price_point(self, symbol: str, price_datetime: datetime) -> PricePoint:
        """Get price for a symbol on a specific date.

        If no data is available for the requested date (weekend, holiday),
        looks back up to 7 days to find the most recent trading day.

        For today's date during market hours, uses the current market price
        since the daily close isn't available yet.
        """
        target_date = price_datetime.date()
        today = date.today()

        # If requesting today's price, try to get the current/live market price first
        # since the daily close isn't available until market closes
        if target_date == today:
            try:
                ticker = yf.Ticker(symbol)
                # Use fast_info['lastPrice'] which provides the most recent price
                # including pre-market and after-hours trading, unlike regularMarketPrice
                # which only returns the last regular session close
                fast_info = ticker.fast_info
                current_price = fast_info.get('lastPrice')
                if current_price is not None:
                    price = Decimal(str(current_price)).quantize(Decimal("0.01"))
                    return PricePoint(
                        symbol=symbol,
                        price_datetime=datetime.now(),
                        price=price,
                        base_currency=Currency.USD
                    )
            except Exception:
                pass  # Fall through to historical data lookup

        # Fetch data - request a window to handle weekends/holidays
        # We ask for 10 days before to ensure we have data even if target is a Monday after a long weekend
        min_date = target_date - timedelta(days=10)
        max_date = target_date

        df = fetch_yfinance_data(symbol, min_date, max_date, self.force_cache_refresh)

        if df.empty:
            raise ValueError(f"No price data available for {symbol}")

        # Look for the target date or the most recent date before it
        for days_back in range(8):
            lookup_date = target_date - timedelta(days=days_back)
            matching_rows = df[df['Date'] == lookup_date]
            if not matching_rows.empty:
                row = matching_rows.iloc[0]
                price = Decimal(str(row['Close'])).quantize(Decimal("0.01"))
                return PricePoint(
                    symbol=symbol,
                    price_datetime=datetime.combine(lookup_date, datetime.min.time()),
                    price=price,
                    base_currency=Currency.USD
                )

        raise ValueError(
            f"No price data available for {symbol} on {target_date} or the previous 7 days."
        )


class MassivePricingDataManager(PricingDataManager):
    """Pricing data manager using the Massive API."""

    _PREFETCH_DAYS = 730

    def __init__(self, api_key: str, force_cache_refresh: bool = False):
        """Initialize the Massive client.

        Args:
            api_key: Massive API key.
            force_cache_refresh: If True, ignore disk cache and fetch fresh
                from API (but still use in-memory cache within the session).
        """
        self._client = RESTClient(api_key)
        self._force_cache_refresh = force_cache_refresh
        # In-memory cache: (symbol, date) → Decimal close price.
        self._price_cache: dict[tuple[str, date], Decimal] = {}
        # Track which symbols have been loaded from disk this session.
        self._disk_loaded: set[str] = set()

    @staticmethod
    def _get_cache_path(symbol: str) -> Path:
        """Get the disk-cache file path for a symbol.

        Args:
            symbol: The ticker symbol.

        Returns:
            Path to the CSV cache file under ``.cache/massive_prices/``.
        """
        return Path.cwd() / ".cache" / "massive_prices" / f"{symbol}.csv"

    def _load_disk_cache(self, symbol: str) -> None:
        """Load cached prices for a symbol from disk into memory.

        Args:
            symbol: The ticker symbol to load from disk cache.
        """
        if symbol in self._disk_loaded:
            return
        self._disk_loaded.add(symbol)

        cache_path = self._get_cache_path(symbol)
        if not cache_path.exists():
            return
        try:
            df = pd.read_csv(cache_path)
            count = 0
            for _, row in df.iterrows():
                trade_date = date.fromisoformat(str(row["Date"]))
                price = Decimal(str(row["Close"])).quantize(Decimal("0.01"))
                self._price_cache[(symbol, trade_date)] = price
                count += 1
            if verbose:
                print(f"  Massive: loaded {count} days from disk cache for {symbol}", flush=True)
        except Exception:
            pass  # Corrupted cache, will refetch from API

    def _save_disk_cache(self, symbol: str) -> None:
        """Persist all cached prices for a symbol to disk.

        Args:
            symbol: The ticker symbol whose in-memory prices are written to CSV.
        """
        cache_path = self._get_cache_path(symbol)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for (sym, trade_date), price in self._price_cache.items():
            if sym == symbol:
                rows.append({"Date": trade_date.isoformat(), "Close": str(price)})

        if rows:
            df = pd.DataFrame(rows).sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
            df.to_csv(cache_path, index=False)

    def _populate_cache(self, symbol: str, min_date: date, max_date: date) -> None:
        """Fetch daily aggregates from Massive and cache all returned dates.

        Args:
            symbol: The ticker symbol to fetch.
            min_date: Start of the date range (inclusive).
            max_date: End of the date range (inclusive).
        """
        try:
            aggs = list(self._client.list_aggs(
                ticker=symbol,
                multiplier=1,
                timespan="day",
                from_=min_date.isoformat(),
                to=max_date.isoformat(),
                limit=5000,
            ))
        except Exception as e:
            raise ValueError(f"Error fetching price data for {symbol}: {e}") from e

        for agg in aggs:
            agg_date = datetime.fromtimestamp(agg.timestamp / 1000).date()  # type: ignore[union-attr]
            self._price_cache[(symbol, agg_date)] = Decimal(str(agg.close)).quantize(Decimal("0.01"))  # type: ignore[union-attr]

        if verbose:
            print(f"  Massive: fetched {len(aggs)} days for {symbol}", flush=True)

        self._save_disk_cache(symbol)

    def get_price_point(self, symbol: str, price_datetime: datetime) -> PricePoint:
        """Get price for a symbol on a specific date.

        Loads from disk cache on first access (unless --no-cache). Falls back
        to the API when data is missing. Recent dates (today/yesterday) always
        trigger a fresh API fetch.
        """
        target_date = price_datetime.date()
        today = date.today()
        is_recent = (today - target_date).days <= 1

        # Load from disk cache on first access (unless --no-cache).
        if not self._force_cache_refresh:
            self._load_disk_cache(symbol)

        # For recent dates, always fetch fresh data first.
        if is_recent:
            fresh_start = target_date - timedelta(days=10)
            self._populate_cache(symbol, fresh_start, today)

        # Check cache (including nearby dates for weekends/holidays).
        for days_back in range(8):
            lookup_date = target_date - timedelta(days=days_back)
            cached = self._price_cache.get((symbol, lookup_date))
            if cached is not None:
                return PricePoint(
                    symbol=symbol,
                    price_datetime=datetime.combine(lookup_date, datetime.min.time()),
                    price=cached,
                    base_currency=Currency.USD,
                )

        # Full cache miss — fetch a broad range.
        min_date = target_date - timedelta(days=self._PREFETCH_DAYS)
        self._populate_cache(symbol, min_date, target_date)

        # Retry from cache after fetch.
        for days_back in range(8):
            lookup_date = target_date - timedelta(days=days_back)
            cached = self._price_cache.get((symbol, lookup_date))
            if cached is not None:
                return PricePoint(
                    symbol=symbol,
                    price_datetime=datetime.combine(lookup_date, datetime.min.time()),
                    price=cached,
                    base_currency=Currency.USD,
                )

        raise ValueError(
            f"No price data available for {symbol} on {target_date} or the previous 7 days."
        )


class DatabentoPricingDataManager(PricingDataManager):
    """Pricing data manager using the Databento API for futures and options."""

    # Dataset constants
    DATASET_CME = "GLBX.MDP3"  # CME Globex - futures & commodities (includes NYMEX)
    DATASET_OPRA = "OPRA.PILLAR"  # OPRA - US equity options

    # How far back to fetch on the first API call for a symbol (calendar days).
    _PREFETCH_DAYS = 730

    def __init__(self, api_key: str | None = None, dataset: str = DATASET_CME,
                 force_cache_refresh: bool = False):
        """Initialize the Databento client.

        Args:
            api_key: Databento API key. If None, reads from DATABENTO_API_KEY env var.
            dataset: Databento dataset ID (default: GLBX.MDP3 for CME/NYMEX futures)
            force_cache_refresh: If True, ignore disk cache and fetch fresh
                from API (but still use in-memory cache within the session).
        """
        self._api_key = api_key or os.getenv("DATABENTO_API_KEY")
        if not self._api_key:
            raise ValueError("Databento API key required. Set DATABENTO_API_KEY env var or pass api_key.")
        self._dataset = dataset
        self._force_cache_refresh = force_cache_refresh
        # In-memory cache: (symbol, date) → Decimal close price.
        self._price_cache: dict[tuple[str, date], Decimal] = {}
        # Remember which symbol variant resolved for each base symbol so we
        # skip variants that already failed.
        self._resolved_variant: dict[str, str] = {}
        # Track which symbols have been loaded from disk this session.
        self._disk_loaded: set[str] = set()

    @staticmethod
    def _get_cache_path(symbol: str) -> Path:
        """Get the disk-cache file path for a symbol.

        Args:
            symbol: The ticker or contract symbol.

        Returns:
            Path to the CSV cache file under ``.cache/databento_prices/``.
        """
        return Path.cwd() / ".cache" / "databento_prices" / f"{symbol}.csv"

    def _load_disk_cache(self, symbol: str) -> None:
        """Load cached prices for a symbol from disk into memory.

        Args:
            symbol: The ticker or contract symbol to load from disk cache.
        """
        if symbol in self._disk_loaded:
            return
        self._disk_loaded.add(symbol)

        cache_path = self._get_cache_path(symbol)
        if not cache_path.exists():
            return
        try:
            df = pd.read_csv(cache_path)
            count = 0
            for _, row in df.iterrows():
                trade_date = date.fromisoformat(str(row["Date"]))
                price = Decimal(str(row["Close"])).quantize(Decimal("0.0001"))
                self._price_cache[(symbol, trade_date)] = price
                count += 1
            if verbose:
                print(f"  Databento: loaded {count} days from disk cache for {symbol}", flush=True)
        except Exception:
            pass  # Corrupted cache, will refetch from API

    def _save_disk_cache(self, symbol: str) -> None:
        """Persist all cached prices for a symbol to disk.

        Args:
            symbol: The ticker or contract symbol whose prices are written.
        """
        cache_path = self._get_cache_path(symbol)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for (sym, trade_date), price in self._price_cache.items():
            if sym == symbol:
                rows.append({"Date": trade_date.isoformat(), "Close": str(price)})

        if rows:
            df = pd.DataFrame(rows).sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
            df.to_csv(cache_path, index=False)

    @staticmethod
    def _is_opra_symbol(symbol: str) -> bool:
        """Check if symbol is an OPRA options contract.

        Args:
            symbol: The symbol string to test.

        Returns:
            True if the symbol matches OSI options format, False otherwise.
        """
        from .metrics.options import parse_osi_symbol
        return parse_osi_symbol(symbol) is not None

    def _populate_cache(self, symbol: str, start_dt: datetime, end_dt: datetime) -> None:
        """Fetch price data from Databento and cache close/mid prices.

        Uses ohlcv-1d for futures (GLBX) and cbbo-1m for options (OPRA).

        Args:
            symbol: The ticker or contract symbol to fetch.
            start_dt: Start of the datetime range (inclusive).
            end_dt: End of the datetime range (exclusive).
        """
        is_opra = self._is_opra_symbol(symbol)

        if is_opra:
            self._populate_cache_opra(symbol, start_dt, end_dt)
        else:
            self._populate_cache_futures(symbol, start_dt, end_dt)

    def _populate_cache_futures(self, symbol: str, start_dt: datetime, end_dt: datetime) -> None:
        """Fetch daily OHLCV bars from GLBX for futures symbols.

        Args:
            symbol: Futures symbol (e.g., "NGH26").
            start_dt: Start of the datetime range (inclusive).
            end_dt: End of the datetime range (exclusive).
        """
        resolved = self._resolved_variant.get(symbol)
        variants = [resolved] if resolved else [symbol, f"{symbol}.FUT", symbol.upper()]

        for sym in variants:
            try:
                client = db.Historical(self._api_key)
                data = client.timeseries.get_range(
                    dataset=self._dataset,
                    symbols=sym,
                    schema="ohlcv-1d",
                    start=start_dt.isoformat(),
                    end=end_dt.isoformat(),
                )
                df = data.to_df()

                if df.empty:
                    continue

                self._resolved_variant[symbol] = sym

                df['trade_date'] = pd.to_datetime(df.index).date
                for _, row in df.iterrows():
                    trade_date = row['trade_date']
                    price = Decimal(str(float(row["close"]))).quantize(Decimal("0.0001"))
                    self._price_cache[(symbol, trade_date)] = price

                if verbose:
                    print(
                        f"  Databento: fetched {len(df)} days for {sym}",
                        flush=True,
                    )

                self._save_disk_cache(symbol)
                return  # success

            except Exception as e:
                if verbose:
                    print(f"  Databento: ohlcv-1d fetch failed for {sym}: {e}", flush=True)
                continue

    def _populate_cache_opra(self, symbol: str, start_dt: datetime, end_dt: datetime) -> None:
        """Fetch cbbo-1m data from OPRA for an individual options contract.

        Makes a single bulk query for the full date range (one API call),
        then groups by day and takes the last bar as the close mid-price.

        Tries the direct contract symbol first. Falls back to a parent
        query filtered to the contract if that fails.

        Args:
            symbol: OPRA options symbol (e.g., "SBUX  260417P00095000").
            start_dt: Start of the datetime range (inclusive).
            end_dt: End of the datetime range (exclusive).
        """
        from .metrics.options import parse_osi_symbol

        parsed = parse_osi_symbol(symbol)
        if parsed is None:
            return

        # OPRA historical data lags ~1 day. Clamp end to yesterday midnight
        # UTC to avoid data_end_after_available_end errors.
        yesterday_end = datetime.combine(
            date.today() - timedelta(days=1),
            datetime.min.time(),
            tzinfo=timezone.utc,
        )
        if end_dt > yesterday_end:
            end_dt = yesterday_end
        if start_dt >= end_dt:
            return

        client = db.Historical(self._api_key)
        df = None

        # Attempt 1: direct contract symbol — one contract's bars only.
        # The Databento client emits a misleading >5GB size warning based on
        # a pre-flight estimate that doesn't account for symbol filtering;
        # actual data for a single contract is typically <1MB.
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*size.*streaming.*5 GB.*")
                data = client.timeseries.get_range(
                    dataset=self.DATASET_OPRA,
                    symbols=symbol,
                    schema="cbbo-1m",
                    start=start_dt.isoformat(),
                    end=end_dt.isoformat(),
                )
            df = data.to_df()
            if df.empty:
                df = None
        except Exception:
            df = None

        # Attempt 2: parent query, filtered to the contract.
        if df is None:
            try:
                parent = f"{parsed['underlying']}.OPT"
                data = client.timeseries.get_range(
                    dataset=self.DATASET_OPRA,
                    symbols=parent,
                    schema="cbbo-1m",
                    stype_in="parent",
                    start=start_dt.isoformat(),
                    end=end_dt.isoformat(),
                )
                df = data.to_df()
                if not df.empty:
                    df = df.reset_index()
                    df = df[df["symbol"] == symbol]
                if df.empty:
                    df = None
            except Exception as e:
                if verbose:
                    print(f"  Databento OPRA: cbbo-1m fetch failed for {symbol}: {e}", flush=True)
                return

        if df is None:
            return

        df = df.reset_index(drop=True) if "ts_event" in df.columns else df.reset_index()
        df["trade_date"] = pd.to_datetime(df["ts_event"]).dt.date

        # Last bar of each trading day (closest to market close).
        last_bars = df.groupby("trade_date").last()

        count = 0
        for trade_date, row in last_bars.iterrows():
            bid = float(row["bid_px_00"])
            ask = float(row["ask_px_00"])
            if bid > 0 and ask > 0:
                mid = Decimal(str((bid + ask) / 2)).quantize(Decimal("0.0001"))
                self._price_cache[(symbol, trade_date)] = mid
                count += 1

        if count > 0:
            if verbose:
                print(f"  Databento OPRA: fetched {count} days for {symbol}", flush=True)
            self._save_disk_cache(symbol)

    def get_price_point(self, symbol: str, price_datetime: datetime) -> PricePoint:
        """Get price for a symbol on a specific date.

        Supports futures (GLBX.MDP3) and OPRA options (OPRA.PILLAR).
        OPRA symbols are detected automatically via OSI format parsing.

        Loads from disk cache on first access (unless --no-cache). Falls back
        to the Databento API when data is missing. Recent dates
        (today/yesterday) use live tbbo quotes since ohlcv-1d bars are only
        available after the trading day ends.

        Args:
            symbol: Futures symbol (e.g., "NGH26") or OPRA options symbol
                    (e.g., "SBUX  260417P00095000")
            price_datetime: The datetime to get the price for

        Returns:
            PricePoint with the daily close price (or live mid-price for today)
        """
        target_date = price_datetime.date()
        today = date.today()
        is_recent = (today - target_date).days <= 1

        # Load from disk cache on first access (unless --no-cache).
        if not self._force_cache_refresh:
            self._load_disk_cache(symbol)

        # 1. Check in-memory cache first (covers disk-loaded and previously
        #    fetched data). Handles weekends/holidays by looking back 7 days.
        for days_back in range(8):
            lookup_date = target_date - timedelta(days=days_back)
            cached = self._price_cache.get((symbol, lookup_date))
            if cached is not None:
                return PricePoint(
                    symbol=symbol,
                    price_datetime=datetime.combine(lookup_date, datetime.min.time()),
                    price=cached,
                    base_currency=Currency.USD,
                )

        # 2. For recent dates, try a live bid/ask quote (tbbo for futures,
        #    cbbo-1m for OPRA). This covers today's price before the daily
        #    close bar is available.
        if is_recent:
            try:
                ask, bid = self.get_quote(symbol)
                if ask > 0 and bid > 0:
                    mid = Decimal(str((ask + bid) / 2)).quantize(Decimal("0.0001"))
                    return PricePoint(
                        symbol=symbol,
                        price_datetime=datetime.now(),
                        price=mid,
                        base_currency=Currency.USD,
                    )
            except Exception as e:
                if verbose:
                    print(f"  Databento: live quote failed for {symbol}: {e}", flush=True)

            # Live quote unavailable — try refreshing cache for recent days.
            end_dt = datetime.now(timezone.utc) - timedelta(minutes=30)
            start_dt = datetime.combine(
                target_date - timedelta(days=10),
                datetime.min.time(),
                tzinfo=timezone.utc,
            )
            self._populate_cache(symbol, start_dt, end_dt)

            # Check cache again after refresh.
            for days_back in range(8):
                lookup_date = target_date - timedelta(days=days_back)
                cached = self._price_cache.get((symbol, lookup_date))
                if cached is not None:
                    return PricePoint(
                        symbol=symbol,
                        price_datetime=datetime.combine(lookup_date, datetime.min.time()),
                        price=cached,
                        base_currency=Currency.USD,
                    )

        # 3. Full cache miss — fetch a broad range and populate the cache.
        # OPRA cbbo-1m is much denser than futures ohlcv-1d, so use a
        # shorter prefetch window to avoid excessive API cost.
        prefetch = 30 if self._is_opra_symbol(symbol) else self._PREFETCH_DAYS
        end_dt = datetime.now(timezone.utc) - timedelta(minutes=30)
        if target_date < today:
            end_dt = datetime.combine(
                target_date + timedelta(days=1),
                datetime.min.time(),
                tzinfo=timezone.utc,
            )
        start_dt = datetime.combine(
            target_date - timedelta(days=prefetch),
            datetime.min.time(),
            tzinfo=timezone.utc,
        )

        self._populate_cache(symbol, start_dt, end_dt)

        # Retry from cache after fetch.
        for days_back in range(8):
            lookup_date = target_date - timedelta(days=days_back)
            cached = self._price_cache.get((symbol, lookup_date))
            if cached is not None:
                return PricePoint(
                    symbol=symbol,
                    price_datetime=datetime.combine(lookup_date, datetime.min.time()),
                    price=cached,
                    base_currency=Currency.USD,
                )

        raise ValueError(
            f"No price data available for {symbol} on {target_date} or the previous 7 days."
        )

    def get_quote(self, symbol: str) -> tuple[float, float]:
        """
        Get the latest bid/ask prices for a symbol.

        Uses tbbo for futures (GLBX) and cbbo-1m via parent query for
        options (OPRA).

        Args:
            symbol: Futures symbol (e.g., "NGH26") or OPRA options symbol
                    (e.g., "SBUX  260417P00095000")

        Returns:
            Tuple of (ask_price, bid_price). Returns (-1.0, -1.0) on error.
        """
        if self._is_opra_symbol(symbol):
            return self._get_quote_opra(symbol)
        return self._get_quote_futures(symbol)

    def _get_quote_futures(self, symbol: str) -> tuple[float, float]:
        """Live quote for futures via tbbo on GLBX.

        Args:
            symbol: Futures symbol (e.g., "NGH26").

        Returns:
            Tuple of (ask_price, bid_price). Returns (-1.0, -1.0) on error.
        """
        symbol_variants = [symbol, f"{symbol}.FUT", symbol.upper()]

        for sym in symbol_variants:
            try:
                client = db.Historical(self._api_key)

                end = datetime.now(timezone.utc) - timedelta(minutes=30)
                start = end - timedelta(days=3)

                data = client.timeseries.get_range(
                    dataset=self._dataset,
                    symbols=sym,
                    schema="tbbo",
                    start=start.isoformat(),
                    end=end.isoformat(),
                )

                df = data.to_df()

                if df.empty:
                    continue

                latest = df.iloc[-1]
                bid_price = float(latest["bid_px_00"])
                ask_price = float(latest["ask_px_00"])
                return ask_price, bid_price

            except Exception:
                continue

        return -1.0, -1.0

    def _get_quote_opra(self, symbol: str) -> tuple[float, float]:
        """Live quote for options via cbbo-1m on OPRA.

        Fetches recent days in one bulk call via _populate_cache_opra,
        then reads the most recent cached mid-price as bid=ask=mid.

        Args:
            symbol: OPRA options symbol (e.g., "SBUX  260417P00095000").

        Returns:
            Tuple of (ask_price, bid_price) where both equal the mid-price.
            Returns (-1.0, -1.0) if no recent data is available.
        """
        end = datetime.now(timezone.utc) - timedelta(minutes=30)
        start = end - timedelta(days=5)
        self._populate_cache_opra(symbol, start, end)

        # Return the most recent cached price as a "quote".
        today = date.today()
        for days_back in range(8):
            lookup = today - timedelta(days=days_back)
            cached = self._price_cache.get((symbol, lookup))
            if cached is not None:
                mid = float(cached)
                return mid, mid  # (ask, bid) — mid-price for both

        return -1.0, -1.0