"""
Advanced Sharpe Ratio calculations using dynamic risk-free rates.

This module provides Sharpe ratio calculations that use real Treasury Bill rates
from FRED (Federal Reserve Economic Data) instead of a constant risk-free rate.

Key differences from sharpe.py:
- Uses actual daily risk-free rates instead of a constant annual rate
- Only calculates on trading days (when risk-free rate data is available)
- Default annualization uses 252 trading days instead of 365 calendar days
"""

from datetime import datetime, date
from decimal import Decimal
from typing import Tuple

import numpy as np

from ..currency import Currency
from ..portfolio import Portfolio, calculate_portfolio_value_by_day
from ..riskfreerates import (
    RiskFreeRateManager,
    FREDRiskFreeRateManager,
    RiskFreeRateSeries,
)
from .sharpe import calculate_daily_returns


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


def align_returns_with_risk_free_rates(
    portfolio_returns: dict[datetime, float],
    rf_manager: RiskFreeRateManager,
    trading_days_per_year: int = 252
) -> Tuple[list[float], list[float], list[datetime]]:
    """
    Align portfolio returns with risk-free rates on trading days only.

    Includes dates where portfolio returns exist and either:
    - Actual risk-free rate data is available for that date, OR
    - It's a recent weekday where FRED data may not be published yet
      (uses most recent available rate as fallback)

    Weekends are always excluded.

    Args:
        portfolio_returns: Dictionary mapping dates to portfolio returns.
        rf_manager: Risk-free rate manager to get daily rates from.
        trading_days_per_year: Number of trading days per year for daily rate calc.

    Returns:
        Tuple of (aligned_portfolio_returns, aligned_rf_rates, aligned_dates).
        All three lists have the same length and correspond to each other.

    Raises:
        ValueError: If no overlapping dates are found.
    """
    aligned_returns: list[float] = []
    aligned_rf_rates: list[float] = []
    aligned_dates: list[datetime] = []

    sorted_dates = sorted(portfolio_returns.keys())

    # Get the set of actual trading days from the risk-free rate manager
    if hasattr(rf_manager, 'get_rates_for_range') and sorted_dates:
        min_date = sorted_dates[0].date()
        max_date = sorted_dates[-1].date()
        available_rates = rf_manager.get_rates_for_range(min_date, max_date)
        trading_days = set(available_rates.keys())
        # Track the most recent available rate for fallback on recent days
        last_known_rate: float | None = None
        if available_rates:
            most_recent_date = max(available_rates.keys())
            last_known_rate = float(available_rates[most_recent_date])
    else:
        # Fallback for managers that don't support get_rates_for_range
        trading_days = None
        last_known_rate = None

    for dt in sorted_dates:
        dt_date = dt.date()

        # Always skip weekends (Saturday=5, Sunday=6)
        if dt_date.weekday() >= 5:
            continue

        # Check if we have actual FRED data for this date
        has_fred_data = trading_days is None or dt_date in trading_days

        if has_fred_data:
            try:
                annual_rate = rf_manager.get_rate(dt_date)
                daily_rf = float(annual_rate) / trading_days_per_year
                # Update last known rate for potential future fallback
                last_known_rate = float(annual_rate)

                aligned_returns.append(portfolio_returns[dt])
                aligned_rf_rates.append(daily_rf)
                aligned_dates.append(dt)
            except ValueError:
                # No risk-free rate available for this date, skip it
                continue
        elif last_known_rate is not None:
            # Recent weekday without FRED data - use last known rate as fallback
            # This handles cases where FRED data lags by a day or two
            daily_rf = last_known_rate / trading_days_per_year

            aligned_returns.append(portfolio_returns[dt])
            aligned_rf_rates.append(daily_rf)
            aligned_dates.append(dt)
        # else: skip - no fallback rate available

    if len(aligned_returns) == 0:
        raise ValueError(
            "No overlapping dates between portfolio returns and risk-free rate data."
        )

    return aligned_returns, aligned_rf_rates, aligned_dates


def calculate_sharpe_ratio_advanced(
    returns: list[float],
    daily_risk_free_rates: list[float],
    trading_days_per_year: int = 252
) -> Tuple[float, float]:
    """
    Calculate Sharpe ratio using variable daily risk-free rates.

    Uses the standard (non-geometric) approach:
    excess_return = portfolio_return - risk_free_rate

    Args:
        returns: List of daily portfolio returns (aligned with rf rates).
        daily_risk_free_rates: List of daily risk-free rates (same length as returns).
        trading_days_per_year: Trading days per year for annualization (default 252).

    Returns:
        Tuple of (daily_sharpe, annual_sharpe).

    Raises:
        ValueError: If inputs have different lengths, fewer than 2 observations,
                    or zero standard deviation.
    """
    if len(returns) != len(daily_risk_free_rates):
        raise ValueError(
            f"Returns and risk-free rates must have same length. "
            f"Got {len(returns)} returns and {len(daily_risk_free_rates)} rates."
        )

    if len(returns) < 2:
        raise ValueError("Need at least two return observations.")

    # Calculate excess returns (standard approach: simple subtraction)
    portfolio_returns = np.array(returns, dtype=float)
    rf_rates = np.array(daily_risk_free_rates, dtype=float)
    excess_returns = portfolio_returns - rf_rates

    mean_excess = excess_returns.mean()
    std_excess = excess_returns.std(ddof=1)

    if np.isclose(std_excess, 0.0):
        raise ValueError("Sample standard deviation is zero; Sharpe is undefined.")

    daily_sharpe = float(mean_excess / std_excess)
    annual_sharpe = daily_sharpe * np.sqrt(trading_days_per_year)

    return daily_sharpe, annual_sharpe


def calculate_sharpe_ratio_cumulative_advanced(
    portfolio: Portfolio,
    target_currency: Currency,
    rf_manager: RiskFreeRateManager | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    trading_days_per_year: int = 252
) -> Tuple[float, float]:
    """
    Calculate the cumulative Sharpe ratio using dynamic risk-free rates.

    Args:
        portfolio: Portfolio object containing transactions and settings.
        target_currency: The currency to convert all values into.
        rf_manager: Risk-free rate manager. If None, uses FRED DTB3.
        start_date: The start date for the calculation (inclusive).
                    If None, uses the earliest transaction date.
        end_date: The end date for the calculation (inclusive).
                  If None, uses the latest transaction date.
        trading_days_per_year: Trading days per year for annualization (default 252).

    Returns:
        Tuple of (daily_sharpe, annual_sharpe).

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

    # Calculate daily returns
    daily_returns = calculate_daily_returns(portfolio_values)

    if len(daily_returns) < 2:
        raise ValueError("Need at least two return observations.")

    # Determine date range for risk-free rate data
    return_dates = sorted(daily_returns.keys())
    min_date = return_dates[0].date()
    max_date = return_dates[-1].date()

    # Create default manager if not provided
    if rf_manager is None:
        rf_manager = _get_default_risk_free_manager(min_date, max_date)

    # Align returns with risk-free rates (trading days only)
    aligned_returns, aligned_rf_rates, _ = align_returns_with_risk_free_rates(
        daily_returns,
        rf_manager,
        trading_days_per_year
    )

    if len(aligned_returns) < 2:
        raise ValueError(
            "Need at least two aligned observations after filtering to trading days."
        )

    return calculate_sharpe_ratio_advanced(
        aligned_returns,
        aligned_rf_rates,
        trading_days_per_year
    )


def calculate_sharpe_ratio_by_day_cumulative_advanced(
    portfolio: Portfolio,
    target_currency: Currency,
    rf_manager: RiskFreeRateManager | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    trading_days_per_year: int = 252,
    min_observations: int = 2
) -> dict[datetime, Tuple[float, float]]:
    """
    Calculate cumulative Sharpe ratios for each trading day in a date range.

    For each trading day, calculates the Sharpe ratio using all returns from
    start_date up to that day (cumulative/expanding window).

    Only includes dates where risk-free rate data is available.

    Args:
        portfolio: Portfolio object containing transactions and settings.
        target_currency: The currency to convert all values into.
        rf_manager: Risk-free rate manager. If None, uses FRED DTB3.
        start_date: The start date for the calculation (inclusive).
                    If None, uses the earliest transaction date.
        end_date: The end date for the calculation (inclusive).
                  If None, uses the latest transaction date.
        trading_days_per_year: Trading days per year for annualization (default 252).
        min_observations: Minimum number of observations required before
                          calculating Sharpe. Defaults to 2.

    Returns:
        Dictionary mapping each trading day to (daily_sharpe, annual_sharpe).
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

    # Calculate daily returns
    daily_returns = calculate_daily_returns(portfolio_values)

    if len(daily_returns) < min_observations:
        return {}

    # Determine date range for risk-free rate data
    return_dates = sorted(daily_returns.keys())
    min_dt = return_dates[0].date()
    max_dt = return_dates[-1].date()

    # Create default manager if not provided
    if rf_manager is None:
        rf_manager = _get_default_risk_free_manager(min_dt, max_dt)

    # Align returns with risk-free rates (trading days only)
    aligned_returns, aligned_rf_rates, aligned_dates = align_returns_with_risk_free_rates(
        daily_returns,
        rf_manager,
        trading_days_per_year
    )

    if len(aligned_returns) < min_observations:
        return {}

    # Calculate cumulative Sharpe for each trading day
    result: dict[datetime, Tuple[float, float]] = {}
    cumulative_returns: list[float] = []
    cumulative_rf_rates: list[float] = []

    for i, dt in enumerate(aligned_dates):
        cumulative_returns.append(aligned_returns[i])
        cumulative_rf_rates.append(aligned_rf_rates[i])

        if len(cumulative_returns) < min_observations:
            continue

        excess_returns = np.array(cumulative_returns) - np.array(cumulative_rf_rates)
        mean_excess = excess_returns.mean()
        std_excess = excess_returns.std(ddof=1)

        if np.isclose(std_excess, 0.0):
            continue

        daily_sharpe = float(mean_excess / std_excess)
        annual_sharpe = daily_sharpe * np.sqrt(trading_days_per_year)

        result[dt] = (daily_sharpe, annual_sharpe)

    return result


def calculate_sharpe_ratio_by_day_rolling_window_advanced(
    portfolio: Portfolio,
    target_currency: Currency,
    window_size: int,
    rf_manager: RiskFreeRateManager | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    trading_days_per_year: int = 252
) -> dict[datetime, Tuple[float, float]]:
    """
    Calculate rolling window Sharpe ratios for each trading day.

    For each trading day, calculates the Sharpe ratio using the trailing
    window_size trading days of returns.

    Only includes dates where risk-free rate data is available.

    Args:
        portfolio: Portfolio object containing transactions and settings.
        target_currency: The currency to convert all values into.
        window_size: Number of trailing trading days to use for each calculation.
                     Common values: 21 (month), 63 (quarter), 252 (year).
        rf_manager: Risk-free rate manager. If None, uses FRED DTB3.
        start_date: The start date for the calculation (inclusive).
                    If None, uses the earliest transaction date.
        end_date: The end date for the calculation (inclusive).
                  If None, uses the latest transaction date.
        trading_days_per_year: Trading days per year for annualization (default 252).

    Returns:
        Dictionary mapping each trading day to (daily_sharpe, annual_sharpe).
        Days with insufficient observations (fewer than window_size) are excluded.
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

    # Calculate daily returns
    daily_returns = calculate_daily_returns(portfolio_values)

    if len(daily_returns) < window_size:
        return {}

    # Determine date range for risk-free rate data
    return_dates = sorted(daily_returns.keys())
    min_dt = return_dates[0].date()
    max_dt = return_dates[-1].date()

    # Create default manager if not provided
    if rf_manager is None:
        rf_manager = _get_default_risk_free_manager(min_dt, max_dt)

    # Align returns with risk-free rates (trading days only)
    aligned_returns, aligned_rf_rates, aligned_dates = align_returns_with_risk_free_rates(
        daily_returns,
        rf_manager,
        trading_days_per_year
    )

    if len(aligned_returns) < window_size:
        return {}

    # Calculate rolling Sharpe for each trading day
    result: dict[datetime, Tuple[float, float]] = {}

    for i in range(window_size - 1, len(aligned_dates)):
        window_returns = aligned_returns[i - window_size + 1:i + 1]
        window_rf_rates = aligned_rf_rates[i - window_size + 1:i + 1]
        dt = aligned_dates[i]

        excess_returns = np.array(window_returns) - np.array(window_rf_rates)
        mean_excess = excess_returns.mean()
        std_excess = excess_returns.std(ddof=1)

        if np.isclose(std_excess, 0.0):
            continue

        daily_sharpe = float(mean_excess / std_excess)
        annual_sharpe = daily_sharpe * np.sqrt(trading_days_per_year)

        result[dt] = (daily_sharpe, annual_sharpe)

    return result


def calculate_sharpe_ratio_from_values_advanced(
    portfolio_values: dict[datetime, Decimal],
    rf_manager: RiskFreeRateManager | None = None,
    trading_days_per_year: int = 252
) -> Tuple[float, float]:
    """
    Calculate Sharpe ratios directly from portfolio values using dynamic RF rates.

    Convenience function when you already have portfolio values and don't
    need to calculate them from a Portfolio object.

    Args:
        portfolio_values: Dictionary mapping dates to portfolio values.
        rf_manager: Risk-free rate manager. If None, uses FRED DTB3.
        trading_days_per_year: Trading days per year for annualization (default 252).

    Returns:
        Tuple of (daily_sharpe, annual_sharpe).

    Raises:
        ValueError: If insufficient data or calculation error.
    """
    daily_returns = calculate_daily_returns(portfolio_values)

    if len(daily_returns) < 2:
        raise ValueError("Need at least two return observations.")

    # Determine date range for risk-free rate data
    return_dates = sorted(daily_returns.keys())
    min_date = return_dates[0].date()
    max_date = return_dates[-1].date()

    # Create default manager if not provided
    if rf_manager is None:
        rf_manager = _get_default_risk_free_manager(min_date, max_date)

    # Align returns with risk-free rates
    aligned_returns, aligned_rf_rates, _ = align_returns_with_risk_free_rates(
        daily_returns,
        rf_manager,
        trading_days_per_year
    )

    if len(aligned_returns) < 2:
        raise ValueError(
            "Need at least two aligned observations after filtering to trading days."
        )

    return calculate_sharpe_ratio_advanced(
        aligned_returns,
        aligned_rf_rates,
        trading_days_per_year
    )
