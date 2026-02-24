from datetime import datetime
from decimal import Decimal
from typing import Tuple

import numpy as np

from ..currency import Currency
from ..portfolio import Portfolio, calculate_portfolio_value_by_day


def calculate_daily_returns(
    portfolio_values: dict[datetime, Decimal]
) -> dict[datetime, float]:
    """
    Calculate daily returns from a time series of portfolio values.

    Args:
        portfolio_values: Dictionary mapping dates to portfolio values.

    Returns:
        Dictionary mapping dates to daily returns (as floats).
        The first date will not have a return (needs previous day).
    """
    if len(portfolio_values) < 2:
        return {}

    sorted_dates = sorted(portfolio_values.keys())
    returns: dict[datetime, float] = {}

    for i in range(1, len(sorted_dates)):
        prev_date = sorted_dates[i - 1]
        curr_date = sorted_dates[i]

        prev_value = float(portfolio_values[prev_date])
        curr_value = float(portfolio_values[curr_date])

        if prev_value != 0:
            daily_return = (curr_value - prev_value) / prev_value
            returns[curr_date] = daily_return

    return returns


def calculate_sharpe_ratio(
    returns: list[float],
    annual_risk_free_rate: float,
    periods_in_year: int = 365
) -> Tuple[float, float]:
    """
    Calculate daily and annualized Sharpe ratios from a list of returns.

    Args:
        returns: List of daily returns as decimals (e.g., 0.002 = 0.2%).
        annual_risk_free_rate: Annual nominal risk-free rate (e.g., 0.03 for 3%).
        periods_in_year: Trading periods in a year (365 for daily).

    Returns:
        Tuple of (daily_sharpe, annual_sharpe).

    Raises:
        ValueError: If fewer than two observations or zero standard deviation.
    """
    if len(returns) < 2:
        raise ValueError("Need at least two return observations.")

    daily_rf = annual_risk_free_rate / periods_in_year

    excess_returns = np.array(returns, dtype=float) - daily_rf
    mean_excess = excess_returns.mean()
    std_excess = excess_returns.std(ddof=1)

    if np.isclose(std_excess, 0.0):
        raise ValueError("Sample standard deviation is zero; Sharpe is undefined.")

    daily_sharpe = float(mean_excess / std_excess)
    annual_sharpe = daily_sharpe * np.sqrt(periods_in_year)

    return daily_sharpe, annual_sharpe


def calculate_sharpe_ratio_by_day_cumulative(
    portfolio: Portfolio,
    target_currency: Currency,
    annual_risk_free_rate: float,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    periods_in_year: int = 365,
    min_observations: int = 2
) -> dict[datetime, Tuple[float, float]]:
    """
    Calculate cumulative Sharpe ratios for each day in a date range.

    For each day, calculates the Sharpe ratio using all returns from
    start_date up to that day (cumulative/expanding window).

    Args:
        portfolio: Portfolio object containing transactions and settings.
        target_currency: The currency to convert all values into.
        annual_risk_free_rate: Annual nominal risk-free rate (e.g., 0.03 for 3%).
        start_date: The start date for the calculation (inclusive).
                    If None, uses the earliest transaction date.
        end_date: The end date for the calculation (inclusive).
                  If None, uses the latest transaction date.
        periods_in_year: Trading periods in a year (365 for daily, 252 for
                         US equities trading days).
        min_observations: Minimum number of return observations required
                          before calculating Sharpe. Defaults to 2.

    Returns:
        Dictionary mapping each date to a tuple of (daily_sharpe, annual_sharpe).
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

    # Calculate cumulative Sharpe for each day
    sorted_dates = sorted(daily_returns.keys())
    daily_rf = annual_risk_free_rate / periods_in_year

    result: dict[datetime, Tuple[float, float]] = {}
    cumulative_returns: list[float] = []

    for date in sorted_dates:
        cumulative_returns.append(daily_returns[date])

        if len(cumulative_returns) < min_observations:
            continue

        excess_returns = np.array(cumulative_returns, dtype=float) - daily_rf
        mean_excess = excess_returns.mean()
        std_excess = excess_returns.std(ddof=1)

        if np.isclose(std_excess, 0.0):
            # Skip days where std is zero
            continue

        daily_sharpe = float(mean_excess / std_excess)
        annual_sharpe = daily_sharpe * np.sqrt(periods_in_year)

        result[date] = (daily_sharpe, annual_sharpe)

    return result


def calculate_sharpe_ratio_cumulative(
    portfolio: Portfolio,
    target_currency: Currency,
    annual_risk_free_rate: float,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    periods_in_year: int = 365
) -> Tuple[float, float]:
    """
    Calculate the cumulative Sharpe ratio for a portfolio over a date range.

    Args:
        portfolio: Portfolio object containing transactions and settings.
        target_currency: The currency to convert all values into.
        annual_risk_free_rate: Annual nominal risk-free rate (e.g., 0.03 for 3%).
        start_date: The start date for the calculation (inclusive).
                    If None, uses the earliest transaction date.
        end_date: The end date for the calculation (inclusive).
                  If None, uses the latest transaction date.
        periods_in_year: Trading periods in a year (365 for daily, 252 for
                         US equities trading days).

    Returns:
        Tuple of (daily_sharpe, annual_sharpe).

    Raises:
        ValueError: If fewer than two observations or zero standard deviation.
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

    # Calculate Sharpe ratio
    returns_list = [daily_returns[d] for d in sorted(daily_returns.keys())]
    return calculate_sharpe_ratio(returns_list, annual_risk_free_rate, periods_in_year)


def calculate_sharpe_ratio_by_day_rolling_window(
    portfolio: Portfolio,
    target_currency: Currency,
    annual_risk_free_rate: float,
    window_size: int,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    periods_in_year: int = 365
) -> dict[datetime, Tuple[float, float]]:
    """
    Calculate rolling window Sharpe ratios for each day in a date range.

    For each day, calculates the Sharpe ratio using the trailing
    window_size days of returns.

    Args:
        portfolio: Portfolio object containing transactions and settings.
        target_currency: The currency to convert all values into.
        annual_risk_free_rate: Annual nominal risk-free rate (e.g., 0.03 for 3%).
        window_size: Number of trailing days to use for each calculation.
                     Common values: 30, 60, 90, 252 (trading year).
        start_date: The start date for the calculation (inclusive).
                    If None, uses the earliest transaction date.
        end_date: The end date for the calculation (inclusive).
                  If None, uses the latest transaction date.
        periods_in_year: Trading periods in a year (365 for daily, 252 for
                         US equities trading days).

    Returns:
        Dictionary mapping each date to a tuple of (daily_sharpe, annual_sharpe).
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

    # Calculate rolling Sharpe for each day
    sorted_dates = sorted(daily_returns.keys())
    returns_list = [daily_returns[d] for d in sorted_dates]
    daily_rf = annual_risk_free_rate / periods_in_year

    result: dict[datetime, Tuple[float, float]] = {}

    for i in range(window_size - 1, len(sorted_dates)):
        window_returns = returns_list[i - window_size + 1:i + 1]
        date = sorted_dates[i]

        excess_returns = np.array(window_returns, dtype=float) - daily_rf
        mean_excess = excess_returns.mean()
        std_excess = excess_returns.std(ddof=1)

        if np.isclose(std_excess, 0.0):
            # Skip days where std is zero
            continue

        daily_sharpe = float(mean_excess / std_excess)
        annual_sharpe = daily_sharpe * np.sqrt(periods_in_year)

        result[date] = (daily_sharpe, annual_sharpe)

    return result


def calculate_sharpe_ratio_from_values(
    portfolio_values: dict[datetime, Decimal],
    annual_risk_free_rate: float,
    periods_in_year: int = 365
) -> Tuple[float, float]:
    """
    Calculate Sharpe ratios directly from a dictionary of portfolio values.

    This is a convenience function when you already have portfolio values
    and don't need to calculate them from a Portfolio object.

    Args:
        portfolio_values: Dictionary mapping dates to portfolio values.
        annual_risk_free_rate: Annual nominal risk-free rate (e.g., 0.03 for 3%).
        periods_in_year: Trading periods in a year (365 for daily).

    Returns:
        Tuple of (daily_sharpe, annual_sharpe).

    Raises:
        ValueError: If fewer than two observations or zero standard deviation.
    """
    daily_returns = calculate_daily_returns(portfolio_values)

    if len(daily_returns) < 2:
        raise ValueError("Need at least two return observations.")

    returns_list = [daily_returns[d] for d in sorted(daily_returns.keys())]

    return calculate_sharpe_ratio(returns_list, annual_risk_free_rate, periods_in_year)
