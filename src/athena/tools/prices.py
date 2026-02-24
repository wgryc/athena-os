import re
from datetime import datetime, date, timedelta
from math import isnan
from typing import Any

import yfinance as yf

from athena.tools import Tool, VisualTool
from athena.pricingdata import PricePoint, PricingDataManager, YFinancePricingDataManager, fetch_yfinance_data


class GetStockPrice(Tool):
    """Tool that fetches the latest price for a stock ticker."""

    def __init__(self, pricing_manager: PricingDataManager | None = None):
        """Initialize the stock price lookup tool.

        Args:
            pricing_manager: Backend used to fetch price data. Defaults to
                ``YFinancePricingDataManager`` when not provided.
        """
        self._manager = pricing_manager or YFinancePricingDataManager()

    @property
    def name(self) -> str:
        return "get_stock_price"

    @property
    def label(self) -> str:
        return "Stock Price Lookup"

    @property
    def description(self) -> str:
        return (
            "Get the latest price for a stock ticker symbol. "
            "Returns the most recent closing price (or live price if markets are open)."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker symbol (e.g. 'AAPL', 'MSFT', 'GOOG').",
                },
            },
            "required": ["symbol"],
        }

    def execute(self, **kwargs: Any) -> str:
        symbol: str = kwargs["symbol"]
        price_point = self._manager.get_price_point(symbol, datetime.now())
        return (
            f"{price_point.symbol} last price: "
            f"${price_point.price} ({price_point.base_currency.value}) "
            f"as of {price_point.price_datetime:%Y-%m-%d %H:%M}"
        )


class GetOptionPrice(Tool):
    """Tool that fetches the latest price for a stock option via Yahoo Finance."""

    @property
    def name(self) -> str:
        return "get_option_price"

    @property
    def label(self) -> str:
        return "Option Price Lookup"

    @property
    def description(self) -> str:
        return (
            "Get the latest price for a stock option contract using Yahoo Finance. "
            "Either provide an OCC/OPRA symbol (e.g. 'SBUX  260417P00095000') OR "
            "the individual fields: underlying ticker, expiration date, option type, "
            "and strike price. Returns bid, ask, last price, volume, open interest, "
            "and implied volatility. Data is delayed ~15 minutes."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "occ_symbol": {
                    "type": "string",
                    "description": (
                        "OCC/OPRA option symbol (e.g. 'SBUX  260417P00095000'). "
                        "Format: TICKER (padded to 6 chars) + YYMMDD + C/P + 8-digit strike "
                        "(strike * 1000). If provided, the other fields are ignored."
                    ),
                },
                "symbol": {
                    "type": "string",
                    "description": "Underlying stock ticker symbol (e.g. 'AAPL', 'SBUX').",
                },
                "expiration": {
                    "type": "string",
                    "description": "Option expiration date in YYYY-MM-DD format (e.g. '2026-04-17').",
                },
                "option_type": {
                    "type": "string",
                    "enum": ["call", "put"],
                    "description": "Option type: 'call' or 'put'.",
                },
                "strike": {
                    "type": "number",
                    "description": "Strike price in dollars (e.g. 95.0).",
                },
            },
            "required": [],
        }

    @staticmethod
    def _parse_occ(occ_symbol: str) -> tuple[str, str, str, float]:
        """Parse an OCC/OPRA symbol into its component fields.

        Args:
            occ_symbol: OCC/OPRA option symbol string, e.g.
                ``"SBUX  260417P00095000"``.

        Returns:
            A tuple of ``(ticker, expiration, option_type, strike)`` where
            *expiration* is ``"YYYY-MM-DD"``, *option_type* is ``"call"`` or
            ``"put"``, and *strike* is a float in dollars.

        Raises:
            ValueError: If *occ_symbol* does not match the expected format.
        """
        clean = occ_symbol.strip()
        m = re.match(r'^([A-Z]+)\s*(\d{6})([CP])(\d{8})$', clean)
        if not m:
            raise ValueError(
                f"Invalid OCC symbol: '{occ_symbol}'. "
                f"Expected format: TICKER YYMMDDCSSSSSSSS (e.g. 'SBUX  260417P00095000')"
            )
        ticker = m.group(1)
        date_str = m.group(2)
        cp = m.group(3)
        strike_raw = m.group(4)
        expiration = datetime.strptime(date_str, "%y%m%d").strftime("%Y-%m-%d")
        option_type = "call" if cp == "C" else "put"
        strike = int(strike_raw) / 1000.0
        return ticker, expiration, option_type, strike

    def execute(self, **kwargs: Any) -> str:
        occ_symbol: str | None = kwargs.get("occ_symbol")
        if occ_symbol:
            try:
                symbol, expiration, option_type, strike = self._parse_occ(occ_symbol)
            except ValueError as e:
                return str(e)
        else:
            if not all(k in kwargs for k in ("symbol", "expiration", "option_type", "strike")):
                return (
                    "Please provide either an occ_symbol OR all of: "
                    "symbol, expiration, option_type, and strike."
                )
            symbol = kwargs["symbol"].upper().strip()
            expiration = kwargs["expiration"]
            option_type = kwargs["option_type"].lower()
            strike = float(kwargs["strike"])

        try:
            ticker = yf.Ticker(symbol)
            available = ticker.options
            if not available:
                return f"No options data available for {symbol}."

            if expiration not in available:
                closest = min(
                    available,
                    key=lambda d: abs(
                        datetime.strptime(d, "%Y-%m-%d")
                        - datetime.strptime(expiration, "%Y-%m-%d")
                    ),
                )
                return (
                    f"Expiration {expiration} not available for {symbol}. "
                    f"Closest available: {closest}. "
                    f"Available expirations: {', '.join(available[:10])}"
                )

            chain = ticker.option_chain(expiration)
            df = chain.calls if option_type == "call" else chain.puts

            match = df[df["strike"] == strike]
            if match.empty:
                nearby = sorted(df["strike"].tolist(), key=lambda s: abs(s - strike))[:5]
                return (
                    f"No {option_type} option found for {symbol} at ${strike:.2f} strike "
                    f"expiring {expiration}. "
                    f"Nearby strikes: {', '.join(f'${s:.2f}' for s in nearby)}"
                )

            row = match.iloc[0]

            def _safe(val: Any, default: float = 0.0) -> float:
                if val is None:
                    return default
                try:
                    f = float(val)
                    return default if isnan(f) else f
                except (TypeError, ValueError):
                    return default

            bid = _safe(row.get("bid"))
            ask = _safe(row.get("ask"))
            last = _safe(row.get("lastPrice"))
            volume = int(_safe(row.get("volume")))
            oi = int(_safe(row.get("openInterest")))
            iv = _safe(row.get("impliedVolatility"))
            itm = bool(row.get("inTheMoney", False))

            mid = (bid + ask) / 2 if bid and ask else last
            type_label = "Call" if option_type == "call" else "Put"

            return (
                f"{symbol} {expiration} ${strike:g} {type_label}\n"
                f"  Last: ${last:.2f}  Bid: ${bid:.2f}  Ask: ${ask:.2f}  Mid: ${mid:.2f}\n"
                f"  Volume: {volume:,}  Open Interest: {oi:,}\n"
                f"  Implied Volatility: {iv:.1%}  In The Money: {'Yes' if itm else 'No'}"
            )

        except Exception as e:
            return f"Error fetching option price for {symbol}: {e}"


class StockPriceHistory(Tool):
    """Tool that fetches historical price data for a stock ticker."""

    def __init__(self, force_cache_refresh: bool = False):
        """Initialize the stock price history tool.

        Args:
            force_cache_refresh: When ``True``, bypass the on-disk price cache
                and fetch fresh data from YFinance.
        """
        self._force_cache_refresh = force_cache_refresh

    @property
    def name(self) -> str:
        return "stock_price_history"

    @property
    def label(self) -> str:
        return "Stock Price History"

    @property
    def description(self) -> str:
        return (
            "Get historical daily price data for a stock over a specified number of "
            "trading days. Returns OHLCV data with summary statistics including price "
            "change, high, low, and average. Useful for analyzing price trends."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker symbol (e.g. 'AAPL', 'MSFT', 'GOOG').",
                },
                "days": {
                    "type": "integer",
                    "description": (
                        "Number of calendar days of history to retrieve (default 30, max 730). "
                        "Note: only trading days will have data."
                    ),
                },
            },
            "required": ["symbol"],
        }

    def execute(self, **kwargs: Any) -> str:
        symbol: str = kwargs["symbol"].upper().strip()
        days: int = min(int(kwargs.get("days", 30)), 730)

        today = date.today()
        start = today - timedelta(days=days)

        try:
            df = fetch_yfinance_data(symbol, start, today, self._force_cache_refresh)
        except Exception as e:
            return f"Error fetching price history for {symbol}: {e}"

        if df.empty:
            return f"No price data available for {symbol} over the past {days} days."

        # Filter to requested window (cache may contain older data).
        df = df[df["Date"] >= start].copy()
        if df.empty:
            return f"No trading data for {symbol} in the past {days} days."

        df = df.sort_values("Date").reset_index(drop=True)

        first_close = float(df.iloc[0]["Close"])
        last_close = float(df.iloc[-1]["Close"])
        change = last_close - first_close
        change_pct = (change / first_close) * 100 if first_close else 0
        high = float(df["High"].max())
        low = float(df["Low"].min())
        avg_close = float(df["Close"].mean())

        parts = [
            f"Price history for {symbol} — {len(df)} trading days "
            f"({df.iloc[0]['Date']} to {df.iloc[-1]['Date']}):\n",
            f"  Start: ${first_close:,.2f}  End: ${last_close:,.2f}  "
            f"Change: ${change:+,.2f} ({change_pct:+.1f}%)",
            f"  Period High: ${high:,.2f}  Low: ${low:,.2f}  Avg Close: ${avg_close:,.2f}\n",
            f"{'Date':<12} {'Open':>10} {'High':>10} {'Low':>10} {'Close':>10} {'Volume':>14}",
            "-" * 68,
        ]

        for _, row in df.iterrows():
            parts.append(
                f"{row['Date']}  "
                f"${float(row['Open']):>9,.2f} "
                f"${float(row['High']):>9,.2f} "
                f"${float(row['Low']):>9,.2f} "
                f"${float(row['Close']):>9,.2f} "
                f"{int(row['Volume']):>14,}"
            )

        return "\n".join(parts)


class StockPriceWidget(VisualTool):
    """Dashboard widget that displays a stock ticker's current price."""

    def __init__(self, force_cache_refresh: bool = False):
        """Initialize the stock price widget.

        Args:
            force_cache_refresh: When ``True``, bypass the on-disk price cache
                and fetch fresh data from YFinance.
        """
        self._manager = YFinancePricingDataManager(force_cache_refresh=force_cache_refresh)
        self._last_result: PricePoint | None = None

    @property
    def name(self) -> str:
        return "stock_price_widget"

    @property
    def description(self) -> str:
        return "Display the current price of a stock ticker."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker symbol (e.g. 'AAPL', 'MSFT', 'GOOG').",
                },
            },
            "required": ["symbol"],
        }

    def execute(self, **kwargs: Any) -> str:
        symbol: str = kwargs["symbol"]
        self._last_result = self._manager.get_price_point(symbol, datetime.now())
        return self.to_context()

    def to_context(self) -> str:
        if self._last_result is None:
            return "(no data)"
        p = self._last_result
        return (
            f"{p.symbol} last price: "
            f"${p.price} ({p.base_currency.value}) "
            f"as of {p.price_datetime:%Y-%m-%d %H:%M}"
        )

    def to_html(self) -> str:
        if self._last_result is None:
            return '<div class="widget-card widget-error">No data</div>'
        p = self._last_result
        return (
            f'<div class="widget-card">'
            f'  <div class="widget-symbol">{p.symbol}</div>'
            f'  <div class="widget-price">{p.price:,.2f} {p.base_currency.value}</div>'
            f'  <div class="widget-meta">'
            f'    {p.price_datetime:%Y-%m-%d %H:%M}'
            f'  </div>'
            f'</div>'
        )
