"""
Alpha and Beta calculations for portfolio performance analysis.

This module provides functions to calculate Jensen's Alpha and Beta,
which measure a portfolio's risk-adjusted performance relative to a
market benchmark (e.g., S&P 500, NASDAQ).

Key concepts:
- Beta: Measures systematic risk (sensitivity to market movements).
  Calculated via regression: strategy_returns = alpha + beta * benchmark_returns
- Alpha (Jensen's Alpha): Excess return over CAPM-predicted return.
  alpha = strategy_return - [rf + beta * (benchmark_return - rf)]
"""

from dataclasses import dataclass
from datetime import datetime, date
from decimal import Decimal
from typing import Tuple

import numpy as np
from scipy import stats  # type: ignore[import-untyped]

from ..currency import Currency
from ..portfolio import Portfolio, calculate_portfolio_value_by_day
from ..pricingdata import fetch_yfinance_data, PricingDataManager, YFinancePricingDataManager
from ..riskfreerates import (
    RiskFreeRateManager,
    FREDRiskFreeRateManager,
    RiskFreeRateSeries,
)
from .sharpe import calculate_daily_returns


# Common benchmark tickers
BENCHMARK_SP500 = "^GSPC"
BENCHMARK_NASDAQ = "^IXIC"
BENCHMARK_DOW = "^DJI"
BENCHMARK_RUSSELL2000 = "^RUT"


@dataclass
class AlphaBetaResult:
    """Result container for alpha/beta calculations."""
    alpha: float            # Jensen's alpha (daily)
    alpha_annualized: float # Annualized alpha
    beta: float             # Market beta
    r_squared: float        # Coefficient of determination (0-1)
    correlation: float      # Correlation with benchmark


def _get_default_risk_free_manager(
    min_date: date,
    max_date: date
) -> RiskFreeRateManager:
    """Create the default risk-free rate manager (FRED DTB3)."""
    return FREDRiskFreeRateManager(
        series=RiskFreeRateSeries.DTB3,
        min_date=min_date,
        max_date=max_date,
        use_cache=True
    )


def fetch_benchmark_returns(
    benchmark_ticker: str,
    min_date: date,
    max_date: date,
    force_cache_refresh: bool = False
) -> dict[datetime, float]:
    """
    Fetch benchmark price data and calculate daily returns.

    Args:
        benchmark_ticker: Yahoo Finance ticker (e.g., "^GSPC" for S&P 500).
        min_date: Start date for data.
        max_date: End date for data.
        force_cache_refresh: Whether to force refresh cached data.

    Returns:
        Dictionary mapping dates to daily returns.

    Raises:
        ValueError: If no benchmark data is available.
    """
    df = fetch_yfinance_data(benchmark_ticker, min_date, max_date, force_cache_refresh)

    if df.empty:
        raise ValueError(f"No benchmark data available for {benchmark_ticker}")

    # Calculate daily returns from Close prices
    returns: dict[datetime, float] = {}
    sorted_df = df.sort_values('Date')

    prev_close = None
    for _, row in sorted_df.iterrows():
        current_close = float(row['Close'])
        current_date = row['Date']

        if prev_close is not None and prev_close != 0:
            daily_return = (current_close - prev_close) / prev_close
            # Convert date to datetime for consistency with portfolio returns
            dt = datetime.combine(current_date, datetime.min.time())
            returns[dt] = daily_return

        prev_close = current_close

    return returns


def align_strategy_with_benchmark(
    strategy_returns: dict[datetime, float],
    benchmark_returns: dict[datetime, float],
    rf_manager: RiskFreeRateManager | None = None,
    trading_days_per_year: int = 252
) -> Tuple[list[float], list[float], list[float], list[datetime]]:
    """
    Align strategy returns with benchmark returns on trading days.

    Only includes dates where both strategy and benchmark have data.
    Optionally includes risk-free rates for each date.

    Args:
        strategy_returns: Dictionary mapping dates to strategy returns.
        benchmark_returns: Dictionary mapping dates to benchmark returns.
        rf_manager: Optional risk-free rate manager. If None, uses 0.0 for all dates.
        trading_days_per_year: Number of trading days per year for daily rate calc.

    Returns:
        Tuple of (aligned_strategy_returns, aligned_benchmark_returns,
                  aligned_rf_rates, aligned_dates).

    Raises:
        ValueError: If no overlapping dates are found.
    """
    # Find overlapping dates
    strategy_dates = set(strategy_returns.keys())
    benchmark_dates = set(benchmark_returns.keys())
    common_dates = strategy_dates & benchmark_dates

    if not common_dates:
        raise ValueError(
            "No overlapping dates between strategy and benchmark returns."
        )

    aligned_strategy: list[float] = []
    aligned_benchmark: list[float] = []
    aligned_rf: list[float] = []
    aligned_dates: list[datetime] = []

    for dt in sorted(common_dates):
        dt_date = dt.date()

        # Skip weekends
        if dt_date.weekday() >= 5:
            continue

        # Get risk-free rate for this date
        if rf_manager is not None:
            try:
                annual_rate = rf_manager.get_rate(dt_date)
                daily_rf = float(annual_rate) / trading_days_per_year
            except ValueError:
                # No risk-free rate available, skip this date
                continue
        else:
            daily_rf = 0.0

        aligned_strategy.append(strategy_returns[dt])
        aligned_benchmark.append(benchmark_returns[dt])
        aligned_rf.append(daily_rf)
        aligned_dates.append(dt)

    if len(aligned_strategy) == 0:
        raise ValueError(
            "No overlapping dates after filtering weekends and risk-free rate availability."
        )

    return aligned_strategy, aligned_benchmark, aligned_rf, aligned_dates


def calculate_alpha_beta(
    strategy_returns: list[float],
    benchmark_returns: list[float],
    risk_free_rates: list[float] | None = None,
    trading_days_per_year: int = 252
) -> AlphaBetaResult:
    """
    Calculate alpha and beta using OLS regression.

    Uses the CAPM model: excess_strategy = alpha + beta * excess_benchmark

    Args:
        strategy_returns: List of daily strategy returns.
        benchmark_returns: List of daily benchmark returns (same length).
        risk_free_rates: Optional list of daily risk-free rates.
                         If None, uses 0.0 (raw returns instead of excess).
        trading_days_per_year: Trading days per year for annualization (default 252).

    Returns:
        AlphaBetaResult with alpha, beta, r_squared, and correlation.

    Raises:
        ValueError: If inputs have different lengths or fewer than 2 observations.
    """
    if len(strategy_returns) != len(benchmark_returns):
        raise ValueError(
            f"Strategy and benchmark returns must have same length. "
            f"Got {len(strategy_returns)} and {len(benchmark_returns)}."
        )

    if len(strategy_returns) < 2:
        raise ValueError("Need at least two return observations.")

    strategy_arr = np.array(strategy_returns, dtype=float)
    benchmark_arr = np.array(benchmark_returns, dtype=float)

    # Calculate excess returns if risk-free rates provided
    if risk_free_rates is not None:
        if len(risk_free_rates) != len(strategy_returns):
            raise ValueError(
                f"Risk-free rates must have same length as returns. "
                f"Got {len(risk_free_rates)} rates and {len(strategy_returns)} returns."
            )
        rf_arr = np.array(risk_free_rates, dtype=float)
        excess_strategy = strategy_arr - rf_arr
        excess_benchmark = benchmark_arr - rf_arr
    else:
        excess_strategy = strategy_arr
        excess_benchmark = benchmark_arr

    # Perform linear regression: excess_strategy = alpha + beta * excess_benchmark
    slope, intercept, r_value, _, _ = stats.linregress(excess_benchmark, excess_strategy)

    beta = float(slope)
    alpha_daily = float(intercept)
    r_squared = float(r_value ** 2)
    correlation = float(r_value)

    # Annualize alpha (compound daily alpha over trading days)
    # For small daily returns, this approximates: alpha_annual ≈ alpha_daily * trading_days
    alpha_annualized = alpha_daily * trading_days_per_year

    return AlphaBetaResult(
        alpha=alpha_daily,
        alpha_annualized=alpha_annualized,
        beta=beta,
        r_squared=r_squared,
        correlation=correlation
    )


def calculate_alpha_beta_cumulative(
    portfolio: Portfolio,
    target_currency: Currency,
    benchmark_ticker: str = BENCHMARK_SP500,
    rf_manager: RiskFreeRateManager | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    trading_days_per_year: int = 252,
    price_manager: PricingDataManager | None = None
) -> AlphaBetaResult:
    """
    Calculate cumulative alpha and beta for a portfolio vs benchmark.

    Args:
        portfolio: Portfolio object containing transactions and settings.
        target_currency: The currency to convert all values into.
        benchmark_ticker: Yahoo Finance ticker for benchmark (default: ^GSPC).
        rf_manager: Risk-free rate manager. If None, uses FRED DTB3.
        start_date: The start date for the calculation (inclusive).
                    If None, uses the earliest transaction date.
        end_date: The end date for the calculation (inclusive).
                  If None, uses the latest transaction date.
        trading_days_per_year: Trading days per year for annualization (default 252).
        price_manager: PricingDataManager for portfolio valuation.
                       If None, uses YFinancePricingDataManager.

    Returns:
        AlphaBetaResult with alpha, beta, r_squared, and correlation.

    Raises:
        ValueError: If insufficient data or calculation error.
    """
    # Get daily portfolio values
    portfolio_values = calculate_portfolio_value_by_day(
        portfolio,
        target_currency,
        start_date,
        end_date
    )

    if len(portfolio_values) < 2:
        raise ValueError("Need at least two portfolio value observations.")

    # Calculate strategy daily returns
    strategy_returns = calculate_daily_returns(portfolio_values)

    if len(strategy_returns) < 2:
        raise ValueError("Need at least two return observations.")

    # Determine date range
    return_dates = sorted(strategy_returns.keys())
    min_date = return_dates[0].date()
    max_date = return_dates[-1].date()

    # Fetch benchmark returns
    benchmark_returns = fetch_benchmark_returns(
        benchmark_ticker,
        min_date,
        max_date
    )

    # Create default risk-free manager if not provided
    if rf_manager is None:
        rf_manager = _get_default_risk_free_manager(min_date, max_date)

    # Align returns
    aligned_strategy, aligned_benchmark, aligned_rf, _ = align_strategy_with_benchmark(
        strategy_returns,
        benchmark_returns,
        rf_manager,
        trading_days_per_year
    )

    if len(aligned_strategy) < 2:
        raise ValueError(
            "Need at least two aligned observations after filtering to trading days."
        )

    return calculate_alpha_beta(
        aligned_strategy,
        aligned_benchmark,
        aligned_rf,
        trading_days_per_year
    )


def calculate_alpha_beta_by_day_cumulative(
    portfolio: Portfolio,
    target_currency: Currency,
    benchmark_ticker: str = BENCHMARK_SP500,
    rf_manager: RiskFreeRateManager | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    trading_days_per_year: int = 252,
    min_observations: int = 20,
    price_manager: PricingDataManager | None = None
) -> dict[datetime, AlphaBetaResult]:
    """
    Calculate cumulative alpha/beta for each trading day in a date range.

    For each trading day, calculates alpha/beta using all returns from
    start_date up to that day (cumulative/expanding window).

    Args:
        portfolio: Portfolio object containing transactions and settings.
        target_currency: The currency to convert all values into.
        benchmark_ticker: Yahoo Finance ticker for benchmark (default: ^GSPC).
        rf_manager: Risk-free rate manager. If None, uses FRED DTB3.
        start_date: The start date for the calculation (inclusive).
        end_date: The end date for the calculation (inclusive).
        trading_days_per_year: Trading days per year for annualization (default 252).
        min_observations: Minimum observations before calculating (default 20).
        price_manager: PricingDataManager for portfolio valuation.

    Returns:
        Dictionary mapping each trading day to AlphaBetaResult.
        Days with insufficient observations are excluded.
    """
    # Get daily portfolio values
    portfolio_values = calculate_portfolio_value_by_day(
        portfolio,
        target_currency,
        start_date,
        end_date
    )

    if len(portfolio_values) < 2:
        return {}

    # Calculate strategy daily returns
    strategy_returns = calculate_daily_returns(portfolio_values)

    if len(strategy_returns) < min_observations:
        return {}

    # Determine date range
    return_dates = sorted(strategy_returns.keys())
    min_dt = return_dates[0].date()
    max_dt = return_dates[-1].date()

    # Fetch benchmark returns
    benchmark_returns = fetch_benchmark_returns(benchmark_ticker, min_dt, max_dt)

    # Create default risk-free manager if not provided
    if rf_manager is None:
        rf_manager = _get_default_risk_free_manager(min_dt, max_dt)

    # Align returns
    aligned_strategy, aligned_benchmark, aligned_rf, aligned_dates = align_strategy_with_benchmark(
        strategy_returns,
        benchmark_returns,
        rf_manager,
        trading_days_per_year
    )

    if len(aligned_strategy) < min_observations:
        return {}

    # Calculate cumulative alpha/beta for each trading day
    result: dict[datetime, AlphaBetaResult] = {}
    cumulative_strategy: list[float] = []
    cumulative_benchmark: list[float] = []
    cumulative_rf: list[float] = []

    for i, dt in enumerate(aligned_dates):
        cumulative_strategy.append(aligned_strategy[i])
        cumulative_benchmark.append(aligned_benchmark[i])
        cumulative_rf.append(aligned_rf[i])

        if len(cumulative_strategy) < min_observations:
            continue

        try:
            ab_result = calculate_alpha_beta(
                cumulative_strategy,
                cumulative_benchmark,
                cumulative_rf,
                trading_days_per_year
            )
            result[dt] = ab_result
        except ValueError:
            # Skip if calculation fails (e.g., zero variance)
            continue

    return result


def calculate_alpha_beta_by_day_rolling_window(
    portfolio: Portfolio,
    target_currency: Currency,
    window_size: int,
    benchmark_ticker: str = BENCHMARK_SP500,
    rf_manager: RiskFreeRateManager | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    trading_days_per_year: int = 252,
    price_manager: PricingDataManager | None = None
) -> dict[datetime, AlphaBetaResult]:
    """
    Calculate rolling window alpha/beta for each trading day.

    For each trading day, calculates alpha/beta using the trailing
    window_size trading days of returns.

    Args:
        portfolio: Portfolio object containing transactions and settings.
        target_currency: The currency to convert all values into.
        window_size: Number of trailing trading days to use for each calculation.
                     Common values: 21 (month), 63 (quarter), 252 (year).
        benchmark_ticker: Yahoo Finance ticker for benchmark (default: ^GSPC).
        rf_manager: Risk-free rate manager. If None, uses FRED DTB3.
        start_date: The start date for the calculation (inclusive).
        end_date: The end date for the calculation (inclusive).
        trading_days_per_year: Trading days per year for annualization (default 252).
        price_manager: PricingDataManager for portfolio valuation.

    Returns:
        Dictionary mapping each trading day to AlphaBetaResult.
        Days with insufficient observations are excluded.
    """
    if window_size < 2:
        raise ValueError("window_size must be at least 2.")

    # Get daily portfolio values
    portfolio_values = calculate_portfolio_value_by_day(
        portfolio,
        target_currency,
        start_date,
        end_date
    )

    if len(portfolio_values) < 2:
        return {}

    # Calculate strategy daily returns
    strategy_returns = calculate_daily_returns(portfolio_values)

    if len(strategy_returns) < window_size:
        return {}

    # Determine date range
    return_dates = sorted(strategy_returns.keys())
    min_dt = return_dates[0].date()
    max_dt = return_dates[-1].date()

    # Fetch benchmark returns
    benchmark_returns = fetch_benchmark_returns(benchmark_ticker, min_dt, max_dt)

    # Create default risk-free manager if not provided
    if rf_manager is None:
        rf_manager = _get_default_risk_free_manager(min_dt, max_dt)

    # Align returns
    aligned_strategy, aligned_benchmark, aligned_rf, aligned_dates = align_strategy_with_benchmark(
        strategy_returns,
        benchmark_returns,
        rf_manager,
        trading_days_per_year
    )

    if len(aligned_strategy) < window_size:
        return {}

    # Calculate rolling alpha/beta for each trading day
    result: dict[datetime, AlphaBetaResult] = {}

    for i in range(window_size - 1, len(aligned_dates)):
        window_strategy = aligned_strategy[i - window_size + 1:i + 1]
        window_benchmark = aligned_benchmark[i - window_size + 1:i + 1]
        window_rf = aligned_rf[i - window_size + 1:i + 1]
        dt = aligned_dates[i]

        try:
            ab_result = calculate_alpha_beta(
                window_strategy,
                window_benchmark,
                window_rf,
                trading_days_per_year
            )
            result[dt] = ab_result
        except ValueError:
            # Skip if calculation fails
            continue

    return result


def calculate_alpha_beta_from_values(
    portfolio_values: dict[datetime, Decimal],
    benchmark_ticker: str = BENCHMARK_SP500,
    rf_manager: RiskFreeRateManager | None = None,
    trading_days_per_year: int = 252
) -> AlphaBetaResult:
    """
    Calculate alpha/beta directly from portfolio values dictionary.

    Convenience function when you already have portfolio values and don't
    need to calculate them from a Portfolio object.

    Args:
        portfolio_values: Dictionary mapping dates to portfolio values.
        benchmark_ticker: Yahoo Finance ticker for benchmark (default: ^GSPC).
        rf_manager: Risk-free rate manager. If None, uses FRED DTB3.
        trading_days_per_year: Trading days per year for annualization (default 252).

    Returns:
        AlphaBetaResult with alpha, beta, r_squared, and correlation.

    Raises:
        ValueError: If insufficient data or calculation error.
    """
    # Calculate strategy daily returns
    strategy_returns = calculate_daily_returns(portfolio_values)

    if len(strategy_returns) < 2:
        raise ValueError("Need at least two return observations.")

    # Determine date range
    return_dates = sorted(strategy_returns.keys())
    min_date = return_dates[0].date()
    max_date = return_dates[-1].date()

    # Fetch benchmark returns
    benchmark_returns = fetch_benchmark_returns(benchmark_ticker, min_date, max_date)

    # Create default risk-free manager if not provided
    if rf_manager is None:
        rf_manager = _get_default_risk_free_manager(min_date, max_date)

    # Align returns
    aligned_strategy, aligned_benchmark, aligned_rf, _ = align_strategy_with_benchmark(
        strategy_returns,
        benchmark_returns,
        rf_manager,
        trading_days_per_year
    )

    if len(aligned_strategy) < 2:
        raise ValueError(
            "Need at least two aligned observations after filtering to trading days."
        )

    return calculate_alpha_beta(
        aligned_strategy,
        aligned_benchmark,
        aligned_rf,
        trading_days_per_year
    )
