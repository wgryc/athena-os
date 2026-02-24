from datetime import datetime
from decimal import Decimal
from typing import Tuple

import numpy as np

from ..currency import Currency
from ..portfolio import Portfolio, calculate_portfolio_value_by_day
from .sharpe import calculate_daily_returns


def calculate_downside_deviation(
    returns: list[float],
    target_return: float = 0.0
) -> float:
    """
    Calculate downside deviation from a list of returns.

    Downside deviation measures the volatility of returns that fall below
    a target return (typically the risk-free rate or zero).

    Args:
        returns: List of returns as decimals (e.g., 0.002 = 0.2%).
        target_return: The minimum acceptable return (default 0.0).

    Returns:
        The downside deviation as a float.

    Raises:
        ValueError: If fewer than two observations.
    """
    if len(returns) < 2:
        raise ValueError("Need at least two return observations.")

    returns_array = np.array(returns, dtype=float)
    downside_returns = np.minimum(returns_array - target_return, 0.0)
    downside_deviation = np.sqrt(np.mean(downside_returns ** 2))

    return float(downside_deviation)


def calculate_sortino_ratio(
    returns: list[float],
    annual_risk_free_rate: float,
    periods_in_year: int = 365
) -> Tuple[float, float]:
    """
    Calculate daily and annualized Sortino ratios from a list of returns.

    The Sortino ratio is similar to the Sharpe ratio but only considers
    downside volatility, making it a better measure for investors who
    are primarily concerned with downside risk.

    Args:
        returns: List of daily returns as decimals (e.g., 0.002 = 0.2%).
        annual_risk_free_rate: Annual nominal risk-free rate (e.g., 0.03 for 3%).
        periods_in_year: Trading periods in a year (365 for daily).

    Returns:
        Tuple of (daily_sortino, annual_sortino).

    Raises:
        ValueError: If fewer than two observations or zero downside deviation.
    """
    if len(returns) < 2:
        raise ValueError("Need at least two return observations.")

    daily_rf = annual_risk_free_rate / periods_in_year

    returns_array = np.array(returns, dtype=float)
    mean_return = returns_array.mean()
    excess_return = mean_return - daily_rf

    downside_deviation = calculate_downside_deviation(returns, daily_rf)

    if np.isclose(downside_deviation, 0.0):
        raise ValueError("Downside deviation is zero; Sortino is undefined.")

    daily_sortino = float(excess_return / downside_deviation)
    annual_sortino = daily_sortino * np.sqrt(periods_in_year)

    return daily_sortino, annual_sortino


def calculate_sortino_ratio_by_day_cumulative(
    portfolio: Portfolio,
    target_currency: Currency,
    annual_risk_free_rate: float,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    periods_in_year: int = 365,
    min_observations: int = 2
) -> dict[datetime, Tuple[float, float]]:
    """
    Calculate cumulative Sortino ratios for each day in a date range.

    For each day, calculates the Sortino ratio using all returns from
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
                          before calculating Sortino. Defaults to 2.

    Returns:
        Dictionary mapping each date to a tuple of (daily_sortino, annual_sortino).
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

    # Calculate cumulative Sortino for each day
    sorted_dates = sorted(daily_returns.keys())
    daily_rf = annual_risk_free_rate / periods_in_year

    result: dict[datetime, Tuple[float, float]] = {}
    cumulative_returns: list[float] = []

    for date in sorted_dates:
        cumulative_returns.append(daily_returns[date])

        if len(cumulative_returns) < min_observations:
            continue

        returns_array = np.array(cumulative_returns, dtype=float)
        mean_return = returns_array.mean()
        excess_return = mean_return - daily_rf

        downside_returns = np.minimum(returns_array - daily_rf, 0.0)
        downside_deviation = np.sqrt(np.mean(downside_returns ** 2))

        if np.isclose(downside_deviation, 0.0):
            # Skip days where downside deviation is zero
            continue

        daily_sortino = float(excess_return / downside_deviation)
        annual_sortino = daily_sortino * np.sqrt(periods_in_year)

        result[date] = (daily_sortino, annual_sortino)

    return result


def calculate_sortino_ratio_cumulative(
    portfolio: Portfolio,
    target_currency: Currency,
    annual_risk_free_rate: float,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    periods_in_year: int = 365
) -> Tuple[float, float]:
    """
    Calculate the cumulative Sortino ratio for a portfolio over a date range.

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
        Tuple of (daily_sortino, annual_sortino).

    Raises:
        ValueError: If fewer than two observations or zero downside deviation.
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

    # Calculate Sortino ratio
    returns_list = [daily_returns[d] for d in sorted(daily_returns.keys())]
    return calculate_sortino_ratio(returns_list, annual_risk_free_rate, periods_in_year)


def calculate_sortino_ratio_by_day_rolling_window(
    portfolio: Portfolio,
    target_currency: Currency,
    annual_risk_free_rate: float,
    window_size: int,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    periods_in_year: int = 365
) -> dict[datetime, Tuple[float, float]]:
    """
    Calculate rolling window Sortino ratios for each day in a date range.

    For each day, calculates the Sortino ratio using the trailing
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
        Dictionary mapping each date to a tuple of (daily_sortino, annual_sortino).
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

    # Calculate rolling Sortino for each day
    sorted_dates = sorted(daily_returns.keys())
    returns_list = [daily_returns[d] for d in sorted_dates]
    daily_rf = annual_risk_free_rate / periods_in_year

    result: dict[datetime, Tuple[float, float]] = {}

    for i in range(window_size - 1, len(sorted_dates)):
        window_returns = returns_list[i - window_size + 1:i + 1]
        date = sorted_dates[i]

        returns_array = np.array(window_returns, dtype=float)
        mean_return = returns_array.mean()
        excess_return = mean_return - daily_rf

        downside_returns = np.minimum(returns_array - daily_rf, 0.0)
        downside_deviation = np.sqrt(np.mean(downside_returns ** 2))

        if np.isclose(downside_deviation, 0.0):
            # Skip days where downside deviation is zero
            continue

        daily_sortino = float(excess_return / downside_deviation)
        annual_sortino = daily_sortino * np.sqrt(periods_in_year)

        result[date] = (daily_sortino, annual_sortino)

    return result


def calculate_sortino_ratio_from_values(
    portfolio_values: dict[datetime, Decimal],
    annual_risk_free_rate: float,
    periods_in_year: int = 365
) -> Tuple[float, float]:
    """
    Calculate Sortino ratios directly from a dictionary of portfolio values.

    This is a convenience function when you already have portfolio values
    and don't need to calculate them from a Portfolio object.

    Args:
        portfolio_values: Dictionary mapping dates to portfolio values.
        annual_risk_free_rate: Annual nominal risk-free rate (e.g., 0.03 for 3%).
        periods_in_year: Trading periods in a year (365 for daily).

    Returns:
        Tuple of (daily_sortino, annual_sortino).

    Raises:
        ValueError: If fewer than two observations or zero downside deviation.
    """
    daily_returns = calculate_daily_returns(portfolio_values)

    if len(daily_returns) < 2:
        raise ValueError("Need at least two return observations.")

    returns_list = [daily_returns[d] for d in sorted(daily_returns.keys())]

    return calculate_sortino_ratio(returns_list, annual_risk_free_rate, periods_in_year)
