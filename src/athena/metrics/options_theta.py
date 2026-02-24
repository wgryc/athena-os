"""
Options theta decay analysis.

Reads options positions from a portfolio xlsx file (OSI/OPRA symbol format),
computes theta decay curves using Black-Scholes, and generates a standalone
HTML report with 2D decay charts, 3D pricing surfaces, IV vs realized vol
comparison, and a portfolio expected value timeline.

Usage:
    python -m athena.metrics.options_theta portfolio.xlsx
"""

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader

from .options import (
    parse_osi_symbol,
    bs_call_price,
    bs_put_price,
    implied_volatility,
    DEFAULT_RISK_FREE_RATE,
)

load_dotenv()

TEMPLATES_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class OptionPosition:
    """An options position parsed from a portfolio xlsx."""

    symbol: str
    underlying: str
    expiry: date
    option_type: str  # 'C' or 'P'
    strike: float
    purchase_date: date
    purchase_price: float
    quantity: float
    currency: str

    # Computed after spot/IV resolution
    spot_at_purchase: float = 0.0
    current_spot: float = 0.0
    implied_vol: float = 0.30


# ---------------------------------------------------------------------------
# Portfolio reading
# ---------------------------------------------------------------------------

def read_options_from_portfolio(path: str | Path) -> list[OptionPosition]:
    """Read options positions from a portfolio xlsx file.

    Options are identified by successfully parsing the SYMBOL column as
    an OSI symbol. Non-option rows are silently skipped. Sell
    transactions produce negative quantities.

    Args:
        path: Path to the portfolio ``.xlsx`` file.

    Returns:
        List of ``OptionPosition`` objects parsed from the file.
    """
    df = pd.read_excel(path)
    positions = []

    for _, row in df.iterrows():
        symbol = str(row.get("SYMBOL", "")).strip()
        parsed = parse_osi_symbol(symbol)
        if parsed is None:
            continue

        tx_type = str(row.get("TRANSACTION TYPE", "")).strip().upper()
        if tx_type not in ("BUY", "SELL"):
            continue

        purchase_date = pd.to_datetime(row["DATE AND TIME"]).date()
        price = float(row["PRICE"])
        quantity = float(row["QUANTITY"])
        if tx_type == "SELL":
            quantity = -quantity

        positions.append(OptionPosition(
            symbol=symbol,
            underlying=parsed["underlying"],
            expiry=parsed["expiry"],
            option_type=parsed["option_type"],
            strike=parsed["strike"],
            purchase_date=purchase_date,
            purchase_price=price,
            quantity=quantity,
            currency=str(row.get("CURRENCY", "USD")).strip(),
        ))

    return positions


def read_cash_events(path: str | Path) -> list[tuple[date, float]]:
    """Read CASH_IN and CASH_OUT events from a portfolio xlsx file.

    CASH_OUT amounts are negated so outflows appear as negative values.

    Args:
        path: Path to the portfolio ``.xlsx`` file.

    Returns:
        List of ``(date, amount)`` tuples.
    """
    df = pd.read_excel(path)
    events = []
    for _, row in df.iterrows():
        tx_type = str(row.get("TRANSACTION TYPE", "")).strip().upper()
        if tx_type in ("CASH_IN", "CASH_OUT"):
            d = pd.to_datetime(row["DATE AND TIME"]).date()
            amount = float(row["PRICE"]) * float(row["QUANTITY"])
            if tx_type == "CASH_OUT":
                amount = -amount
            events.append((d, amount))
    return events


# ---------------------------------------------------------------------------
# Spot price fetching
# ---------------------------------------------------------------------------

def get_spot_price(ticker: str) -> float:
    """Get the current spot price via YFinance ``fast_info``.

    Args:
        ticker: Stock ticker symbol (e.g. ``"AAPL"``).

    Returns:
        Last traded price as a float.
    """
    t = yf.Ticker(ticker)
    return float(t.fast_info["lastPrice"])


def get_historical_prices(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Get daily close prices from YFinance for a date range.

    Args:
        ticker: Stock ticker symbol (e.g. ``"AAPL"``).
        start: Start date (inclusive).
        end: End date (inclusive).

    Returns:
        DataFrame with a ``Close`` column indexed by date.
    """
    t = yf.Ticker(ticker)
    hist = t.history(
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
    )
    if hist.empty:
        return pd.DataFrame(columns=["Close"])
    return hist[["Close"]].copy()


# ---------------------------------------------------------------------------
# Realized volatility
# ---------------------------------------------------------------------------

def compute_realized_vol(
    prices: pd.DataFrame,
    start_date: date,
    window: int = 20,
) -> tuple[list[str], list[float]]:
    """Compute rolling annualized realized volatility.

    Uses log-returns and a rolling standard deviation annualised with
    a 252-day factor.

    Args:
        prices: DataFrame with a ``Close`` column indexed by date.
        start_date: Only include results on or after this date.
        window: Rolling window size in trading days.

    Returns:
        Tuple of ``(dates, vol_percentages)`` where dates are
        ISO-formatted strings and volatilities are in percent.
    """
    close = prices["Close"]
    log_ret = np.log(close / close.shift(1))
    rolling_vol = log_ret.rolling(window=window).std() * np.sqrt(252)

    valid = rolling_vol.dropna()
    dates = []
    values = []
    for idx, v in valid.items():
        d = idx.date() if hasattr(idx, "date") else idx
        if d < start_date:
            continue
        dates.append(d.isoformat())
        values.append(round(float(v) * 100, 2))
    return dates, values


# ---------------------------------------------------------------------------
# Historical option values (mark-to-model using actual spot prices)
# ---------------------------------------------------------------------------

def compute_historical_option_values(
    option_type: str,
    strike: float,
    iv: float,
    expiry: date,
    purchase_date: date,
    r: float,
    spot_history: pd.DataFrame,
) -> tuple[list[str], list[float]]:
    """Compute Black-Scholes value at each historical date.

    Uses actual spot prices from ``spot_history`` with a fixed implied
    volatility to mark-to-model the option from purchase to today.

    Args:
        option_type: ``"C"`` for call, ``"P"`` for put.
        strike: Option strike price.
        iv: Implied volatility (annualised, as a decimal).
        expiry: Option expiration date.
        purchase_date: Date the option was purchased.
        r: Annualized risk-free rate.
        spot_history: DataFrame with a ``Close`` column indexed by date.

    Returns:
        Tuple of ``(dates, values)`` where dates are ISO-formatted
        strings from ``purchase_date`` to today.
    """
    bs_fn = bs_call_price if option_type == "C" else bs_put_price
    today = date.today()
    dates = []
    values = []

    for idx, row in spot_history.iterrows():
        d = idx.date() if hasattr(idx, "date") else idx
        if d < purchase_date:
            continue
        if d > today:
            break
        T = (expiry - d).days / 365.25
        if T <= 0:
            break
        v = bs_fn(float(row["Close"]), strike, T, r, iv)
        dates.append(d.isoformat())
        values.append(round(v, 4))

    return dates, values


# ---------------------------------------------------------------------------
# Theta decay curve (2D)
# ---------------------------------------------------------------------------

def compute_theta_curve(
    option_type: str,
    strike: float,
    spot: float,
    iv: float,
    purchase_date: date,
    expiry: date,
    r: float,
    max_points: int = 365,
) -> tuple[list[str], list[float], list[float]]:
    """Compute theoretical option value from purchase date to expiry.

    Holds spot constant and varies only time to show pure theta decay.
    Today's date and the expiry date are always included in the output.

    Args:
        option_type: ``"C"`` for call, ``"P"`` for put.
        strike: Option strike price.
        spot: Underlying price (held constant across the curve).
        iv: Implied volatility (annualised, as a decimal).
        purchase_date: Start date for the curve.
        expiry: Option expiration date.
        r: Annualized risk-free rate.
        max_points: Maximum number of evaluation points.

    Returns:
        Tuple of ``(dates, theoretical_values, intrinsic_values)``
        where dates are ISO-formatted strings.
    """
    total_days = (expiry - purchase_date).days
    if total_days <= 0:
        return [], [], []

    today = date.today()
    step = max(1, total_days // max_points)
    bs_fn = bs_call_price if option_type == "C" else bs_put_price
    intrinsic_val = max(spot - strike, 0) if option_type == "C" else max(strike - spot, 0)

    # Build date set ensuring today and expiry are included
    date_set = set()
    for day_offset in range(0, total_days + 1, step):
        date_set.add(purchase_date + timedelta(days=day_offset))
    date_set.add(expiry)
    if purchase_date <= today <= expiry:
        date_set.add(today)

    sorted_dates = sorted(date_set)

    dates = []
    values = []
    intrinsics = []
    for d in sorted_dates:
        T = max((expiry - d).days / 365.25, 1e-6)
        v = bs_fn(spot, strike, T, r, iv)
        dates.append(d.isoformat())
        values.append(round(v, 4))
        intrinsics.append(round(intrinsic_val, 4))

    return dates, values, intrinsics


# ---------------------------------------------------------------------------
# 3D pricing surface
# ---------------------------------------------------------------------------

def compute_3d_surface(
    option_type: str,
    strike: float,
    spot: float,
    iv: float,
    purchase_date: date,
    expiry: date,
    r: float,
    n_time: int = 40,
    n_price: int = 40,
    price_range: float = 0.30,
) -> dict:
    """Compute a Black-Scholes pricing surface over time and price.

    Evaluates the option value on a grid of ``(days_since_purchase,
    underlying_price)`` for use with Plotly 3D surface charts.

    Args:
        option_type: ``"C"`` for call, ``"P"`` for put.
        strike: Option strike price.
        spot: Current underlying price (centre of the price axis).
        iv: Implied volatility (annualised, as a decimal).
        purchase_date: Start date (day 0 on the time axis).
        expiry: Option expiration date.
        r: Annualized risk-free rate.
        n_time: Number of points along the time axis.
        n_price: Number of points along the price axis.
        price_range: Fraction of spot to extend above/below for the
            price axis (e.g. 0.30 for +/- 30%).

    Returns:
        Dictionary with ``days`` (list[int]), ``prices`` (list[float]),
        and ``values`` (2D list oriented as
        ``values[price_idx][day_idx]``).
    """
    total_days = (expiry - purchase_date).days
    if total_days <= 0:
        return {"days": [], "prices": [], "values": []}

    bs_fn = bs_call_price if option_type == "C" else bs_put_price
    days = np.linspace(0, total_days, n_time, dtype=int)
    prices = np.linspace(spot * (1 - price_range), spot * (1 + price_range), n_price)

    # Compute values[price_idx][day_idx] — Plotly expects z[y_idx][x_idx]
    values = []
    for S in prices:
        row = []
        for day in days:
            T = max((total_days - int(day)) / 365.25, 1e-6)
            row.append(round(bs_fn(float(S), strike, T, r, iv), 4))
        values.append(row)

    return {
        "days": [int(d) for d in days],
        "prices": [round(float(p), 2) for p in prices],
        "values": values,
    }


# ---------------------------------------------------------------------------
# Portfolio timeline
# ---------------------------------------------------------------------------

def compute_portfolio_timeline(
    positions: list[OptionPosition],
    r: float,
    cash_events: list[tuple[date, float]] | None = None,
) -> dict:
    """Compute aggregate portfolio value from first purchase to last expiry.

    Tracks cash balance alongside Black-Scholes option values to
    produce a total NAV timeline.  Today's date is always included
    so the chart can draw a vertical marker.

    Args:
        positions: List of ``OptionPosition`` objects with resolved
            ``current_spot`` and ``implied_vol``.
        r: Annualized risk-free rate.
        cash_events: Optional list of ``(date, amount)`` tuples for
            cash inflows/outflows.

    Returns:
        Dictionary with keys ``dates``, ``options_value``, ``cash``,
        ``total_nav``, ``initial_capital``, ``today_idx``, and
        ``today_date``.
    """
    if not positions:
        return {
            "dates": [], "options_value": [], "cash": [], "total_nav": [],
            "initial_capital": 0, "today_idx": 0, "today_date": "",
        }

    cash_events = cash_events or []

    # Date range: include cash events in the range
    all_event_dates = (
        [p.purchase_date for p in positions]
        + [e[0] for e in cash_events]
    )
    first = min(all_event_dates)
    last = max(p.expiry for p in positions)
    today = date.today()
    total_days = (last - first).days
    if total_days <= 0:
        return {
            "dates": [], "options_value": [], "cash": [], "total_nav": [],
            "initial_capital": 0, "today_idx": 0, "today_date": "",
        }

    step = max(1, total_days // 500)
    initial_capital = sum(amt for _, amt in cash_events)

    # Precompute events sorted by date
    purchase_events = sorted(
        [(p.purchase_date, p.purchase_price * p.quantity) for p in positions],
        key=lambda x: x[0],
    )
    sorted_cash = sorted(cash_events, key=lambda x: x[0])

    # Build date set ensuring today is included
    date_set = set()
    for day_offset in range(0, total_days + 1, step):
        date_set.add(first + timedelta(days=day_offset))
    date_set.add(last)
    if first <= today <= last:
        date_set.add(today)

    sorted_dates = sorted(date_set)

    dates = []
    options_value_list = []
    cash_list = []
    total_nav_list = []
    today_idx = 0
    cash_balance = 0.0
    cash_ptr = 0
    purchase_ptr = 0

    for d in sorted_dates:
        dates.append(d.isoformat())
        if d <= today:
            today_idx = len(dates) - 1

        # Process cash inflows
        while cash_ptr < len(sorted_cash) and sorted_cash[cash_ptr][0] <= d:
            cash_balance += sorted_cash[cash_ptr][1]
            cash_ptr += 1

        # Process option purchases (cash outflows)
        while purchase_ptr < len(purchase_events) and purchase_events[purchase_ptr][0] <= d:
            cash_balance -= purchase_events[purchase_ptr][1]
            purchase_ptr += 1

        # Sum options values
        options_total = 0.0
        for pos in positions:
            if d < pos.purchase_date:
                continue
            if d > pos.expiry:
                intr = (max(pos.current_spot - pos.strike, 0) if pos.option_type == "C"
                        else max(pos.strike - pos.current_spot, 0))
                options_total += intr * pos.quantity
                continue
            T = max((pos.expiry - d).days / 365.25, 1e-6)
            bs_fn = bs_call_price if pos.option_type == "C" else bs_put_price
            options_total += bs_fn(pos.current_spot, pos.strike, T, r, pos.implied_vol) * pos.quantity

        options_value_list.append(round(options_total, 2))
        cash_list.append(round(cash_balance, 2))
        total_nav_list.append(round(cash_balance + options_total, 2))

    return {
        "dates": dates,
        "options_value": options_value_list,
        "cash": cash_list,
        "total_nav": total_nav_list,
        "initial_capital": round(initial_capital, 2),
        "today_idx": today_idx,
        "today_date": today.isoformat(),
    }


# ---------------------------------------------------------------------------
# Report data builder
# ---------------------------------------------------------------------------

def build_report_data(
    positions: list[OptionPosition],
    r: float,
    historical_prices: dict[str, pd.DataFrame] | None = None,
    cash_events: list[tuple[date, float]] | None = None,
) -> dict:
    """Build the complete data structure for the HTML template.

    Computes theta decay curves, 3D pricing surfaces, historical
    option values, realised-vs-implied volatility comparisons, and a
    portfolio-level NAV timeline for each position.

    Args:
        positions: List of ``OptionPosition`` objects with resolved
            ``current_spot`` and ``implied_vol``.
        r: Annualized risk-free rate.
        historical_prices: Mapping of ticker to historical price
            DataFrame (``Close`` column). Used for mark-to-model
            and realised volatility.
        cash_events: Optional list of ``(date, amount)`` tuples for
            cash inflows/outflows.

    Returns:
        Dictionary with keys ``generated_at``, ``options`` (per-position
        detail), and ``portfolio`` (aggregate timeline).
    """
    historical_prices = historical_prices or {}
    options_data = []

    for pos in positions:
        # Theta decay curve
        theta_dates, theta_vals, theta_intr = compute_theta_curve(
            pos.option_type, pos.strike, pos.current_spot,
            pos.implied_vol, pos.purchase_date, pos.expiry, r,
        )

        # 3D surface
        surface = compute_3d_surface(
            pos.option_type, pos.strike, pos.current_spot,
            pos.implied_vol, pos.purchase_date, pos.expiry, r,
        )

        # Historical option values (BS with actual spot prices)
        hist_prices = historical_prices.get(pos.underlying)
        if hist_prices is not None and not hist_prices.empty:
            hist_dates, hist_vals = compute_historical_option_values(
                pos.option_type, pos.strike, pos.implied_vol,
                pos.expiry, pos.purchase_date, r, hist_prices,
            )
            vol_dates, vol_vals = compute_realized_vol(
                hist_prices, pos.purchase_date,
            )
        else:
            hist_dates, hist_vals = [], []
            vol_dates, vol_vals = [], []

        # Current theoretical value
        dte = max((pos.expiry - date.today()).days, 0)
        T_now = max(dte / 365.25, 1e-6)
        bs_fn = bs_call_price if pos.option_type == "C" else bs_put_price
        current_val = bs_fn(pos.current_spot, pos.strike, T_now, r, pos.implied_vol)

        # Daily theta (numerical)
        if T_now > 2 / 365.25:
            theta_1d = (bs_fn(pos.current_spot, pos.strike, T_now - 1 / 365.25, r, pos.implied_vol)
                        - current_val)
        else:
            theta_1d = 0.0

        intrinsic = (max(pos.current_spot - pos.strike, 0) if pos.option_type == "C"
                     else max(pos.strike - pos.current_spot, 0))

        options_data.append({
            "symbol": pos.symbol.strip(),
            "underlying": pos.underlying,
            "type_label": "Call" if pos.option_type == "C" else "Put",
            "strike": pos.strike,
            "expiry": pos.expiry.isoformat(),
            "purchase_date": pos.purchase_date.isoformat(),
            "purchase_price": round(pos.purchase_price, 2),
            "quantity": pos.quantity,
            "current_spot": round(pos.current_spot, 2),
            "iv_pct": round(pos.implied_vol * 100, 1),
            "current_val": round(current_val, 2),
            "intrinsic": round(intrinsic, 2),
            "time_value": round(max(current_val - intrinsic, 0), 2),
            "theta_1d": round(theta_1d, 4),
            "dte": dte,
            "today_date": date.today().isoformat(),
            "theta_curve": {
                "dates": theta_dates,
                "values": theta_vals,
                "intrinsic": theta_intr,
            },
            "surface": surface,
            "historical": {
                "dates": hist_dates,
                "values": hist_vals,
            },
            "vol_comparison": {
                "dates": vol_dates,
                "realized_vol": vol_vals,
                "implied_vol_pct": round(pos.implied_vol * 100, 2),
            },
        })

    portfolio = compute_portfolio_timeline(positions, r, cash_events)

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "options": options_data,
        "portfolio": portfolio,
    }


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------

def generate_report(
    positions: list[OptionPosition],
    r: float,
    output_path: str | Path,
    historical_prices: dict[str, pd.DataFrame] | None = None,
    cash_events: list[tuple[date, float]] | None = None,
) -> Path:
    """Generate the options theta HTML report.

    Args:
        positions: List of ``OptionPosition`` objects with resolved
            ``current_spot`` and ``implied_vol``.
        r: Annualized risk-free rate.
        output_path: File path for the generated HTML report.
        historical_prices: Mapping of ticker to historical price
            DataFrame. Used for mark-to-model and vol charts.
        cash_events: Optional list of ``(date, amount)`` tuples for
            cash inflows/outflows.

    Returns:
        Resolved ``Path`` to the written report file.
    """
    data = build_report_data(positions, r, historical_prices, cash_events)

    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))
    template = env.get_template("options_theta_template.html")
    html = template.render(data=json.dumps(data))

    output = Path(output_path)
    output.write_text(html)
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate options theta decay analysis report."
    )
    parser.add_argument(
        "portfolio",
        help="Path to portfolio xlsx file with options positions",
    )
    parser.add_argument(
        "--output", "-o", default="options_theta_report.html",
        help="Output HTML file path",
    )
    parser.add_argument(
        "--risk-free-rate", "-r", type=float, default=DEFAULT_RISK_FREE_RATE,
        help=f"Annualized risk-free rate (default: {DEFAULT_RISK_FREE_RATE})",
    )
    args = parser.parse_args()

    # Read options positions and cash events from portfolio
    positions = read_options_from_portfolio(args.portfolio)
    cash_events = read_cash_events(args.portfolio)

    if not positions:
        print("ERROR: No options positions found in portfolio.", file=sys.stderr)
        print("Options must use OSI symbols (e.g. 'AAPL  260320C00220000').", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(positions)} options position(s)")
    if cash_events:
        total_cash = sum(amt for _, amt in cash_events)
        print(f"  Cash events: {len(cash_events)}, total: ${total_cash:,.2f}")

    # Fetch current spot prices for each underlying
    underlyings = set(p.underlying for p in positions)
    current_spots = {}
    for ticker in underlyings:
        try:
            current_spots[ticker] = get_spot_price(ticker)
            print(f"  {ticker} spot: ${current_spots[ticker]:.2f}")
        except Exception as e:
            print(f"ERROR: Could not fetch spot for {ticker}: {e}", file=sys.stderr)
            sys.exit(1)

    # Fetch historical prices for all underlyings (for vol and historical values)
    earliest = min(p.purchase_date for p in positions)
    historical_prices: dict[str, pd.DataFrame] = {}
    for ticker in underlyings:
        try:
            historical_prices[ticker] = get_historical_prices(
                ticker,
                earliest - timedelta(days=40),  # warmup for 20-day rolling vol
                date.today(),
            )
            print(f"  {ticker} history: {len(historical_prices[ticker])} days")
        except Exception as e:
            print(f"  Warning: no history for {ticker}: {e}", file=sys.stderr)

    # Resolve IV for each position
    for pos in positions:
        pos.current_spot = current_spots[pos.underlying]

        # Historical spot for IV derivation (use bulk historical data)
        hist = historical_prices.get(pos.underlying)
        if hist is not None and not hist.empty:
            hist_dates_idx = hist.index.map(
                lambda x: x.date() if hasattr(x, "date") else x
            )
            mask = hist_dates_idx <= pos.purchase_date
            if mask.any():
                pos.spot_at_purchase = float(hist.loc[mask, "Close"].iloc[-1])
            else:
                pos.spot_at_purchase = pos.current_spot
        else:
            pos.spot_at_purchase = pos.current_spot

        # Back out IV from purchase price
        T_at_purchase = (pos.expiry - pos.purchase_date).days / 365.25
        iv = implied_volatility(
            pos.purchase_price, pos.spot_at_purchase,
            pos.strike, T_at_purchase,
            args.risk_free_rate, pos.option_type,
        )
        pos.implied_vol = iv if iv is not None else 0.30  # fallback

        print(
            f"  {pos.symbol.strip()}: IV={pos.implied_vol * 100:.1f}%, "
            f"spot@purchase=${pos.spot_at_purchase:.2f}, "
            f"spot@now=${pos.current_spot:.2f}"
        )

    # Generate report
    path = generate_report(
        positions, args.risk_free_rate, args.output,
        historical_prices, cash_events,
    )
    print(f"Report written to {path}")


if __name__ == "__main__":
    main()
