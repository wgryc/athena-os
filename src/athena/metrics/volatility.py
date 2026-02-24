from datetime import datetime
from decimal import Decimal
from typing import Tuple

import numpy as np

from ..currency import Currency
from ..portfolio import Portfolio, calculate_portfolio_value_by_day
from .sharpe import calculate_daily_returns


def calculate_volatility(
    returns: list[float],
    periods_in_year: int = 365
) -> Tuple[float, float]:
    """
    Calculate daily and annualized volatility from a list of returns.

    Volatility is measured as the standard deviation of returns, which
    quantifies the dispersion of returns around their mean.

    Args:
        returns: List of daily returns as decimals (e.g., 0.002 = 0.2%).
        periods_in_year: Trading periods in a year (365 for daily).

    Returns:
        Tuple of (daily_volatility, annual_volatility).

    Raises:
        ValueError: If fewer than two observations.
    """
    if len(returns) < 2:
        raise ValueError("Need at least two return observations.")

    returns_array = np.array(returns, dtype=float)
    daily_volatility = float(returns_array.std(ddof=1))
    annual_volatility = daily_volatility * np.sqrt(periods_in_year)

    return daily_volatility, annual_volatility


def calculate_volatility_cumulative(
    portfolio: Portfolio,
    target_currency: Currency,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    periods_in_year: int = 365
) -> Tuple[float, float]:
    """
    Calculate the volatility for a portfolio over a date range.

    Args:
        portfolio: Portfolio object containing transactions and settings.
        target_currency: The currency to convert all values into.
        start_date: The start date for the calculation (inclusive).
                    If None, uses the earliest transaction date.
        end_date: The end date for the calculation (inclusive).
                  If None, uses the latest transaction date.
        periods_in_year: Trading periods in a year (365 for daily, 252 for
                         US equities trading days).

    Returns:
        Tuple of (daily_volatility, annual_volatility).

    Raises:
        ValueError: If fewer than two observations.
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

    returns_list = [daily_returns[d] for d in sorted(daily_returns.keys())]
    return calculate_volatility(returns_list, periods_in_year)


def calculate_volatility_by_day_cumulative(
    portfolio: Portfolio,
    target_currency: Currency,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    periods_in_year: int = 365,
    min_observations: int = 2
) -> dict[datetime, Tuple[float, float]]:
    """
    Calculate cumulative volatility for each day in a date range.

    For each day, calculates the volatility using all returns from
    start_date up to that day (cumulative/expanding window).

    Args:
        portfolio: Portfolio object containing transactions and settings.
        target_currency: The currency to convert all values into.
        start_date: The start date for the calculation (inclusive).
                    If None, uses the earliest transaction date.
        end_date: The end date for the calculation (inclusive).
                  If None, uses the latest transaction date.
        periods_in_year: Trading periods in a year (365 for daily, 252 for
                         US equities trading days).
        min_observations: Minimum number of return observations required
                          before calculating volatility. Defaults to 2.

    Returns:
        Dictionary mapping each date to a tuple of (daily_volatility, annual_volatility).
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

    sorted_dates = sorted(daily_returns.keys())
    result: dict[datetime, Tuple[float, float]] = {}
    cumulative_returns: list[float] = []

    for date in sorted_dates:
        cumulative_returns.append(daily_returns[date])

        if len(cumulative_returns) < min_observations:
            continue

        returns_array = np.array(cumulative_returns, dtype=float)
        daily_volatility = float(returns_array.std(ddof=1))
        annual_volatility = daily_volatility * np.sqrt(periods_in_year)

        result[date] = (daily_volatility, annual_volatility)

    return result


def calculate_volatility_by_day_rolling_window(
    portfolio: Portfolio,
    target_currency: Currency,
    window_size: int,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    periods_in_year: int = 365
) -> dict[datetime, Tuple[float, float]]:
    """
    Calculate rolling window volatility for each day in a date range.

    For each day, calculates the volatility using the trailing
    window_size days of returns.

    Args:
        portfolio: Portfolio object containing transactions and settings.
        target_currency: The currency to convert all values into.
        window_size: Number of trailing days to use for each calculation.
                     Common values: 20, 30, 60, 90.
        start_date: The start date for the calculation (inclusive).
                    If None, uses the earliest transaction date.
        end_date: The end date for the calculation (inclusive).
                  If None, uses the latest transaction date.
        periods_in_year: Trading periods in a year (365 for daily, 252 for
                         US equities trading days).

    Returns:
        Dictionary mapping each date to a tuple of (daily_volatility, annual_volatility).
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

    sorted_dates = sorted(daily_returns.keys())
    returns_list = [daily_returns[d] for d in sorted_dates]

    result: dict[datetime, Tuple[float, float]] = {}

    for i in range(window_size - 1, len(sorted_dates)):
        window_returns = returns_list[i - window_size + 1:i + 1]
        date = sorted_dates[i]

        returns_array = np.array(window_returns, dtype=float)
        daily_volatility = float(returns_array.std(ddof=1))
        annual_volatility = daily_volatility * np.sqrt(periods_in_year)

        result[date] = (daily_volatility, annual_volatility)

    return result


def calculate_volatility_from_values(
    portfolio_values: dict[datetime, Decimal],
    periods_in_year: int = 365
) -> Tuple[float, float]:
    """
    Calculate volatility directly from a dictionary of portfolio values.

    This is a convenience function when you already have portfolio values
    and don't need to calculate them from a Portfolio object.

    Args:
        portfolio_values: Dictionary mapping dates to portfolio values.
        periods_in_year: Trading periods in a year (365 for daily).

    Returns:
        Tuple of (daily_volatility, annual_volatility).

    Raises:
        ValueError: If fewer than two observations.
    """
    daily_returns = calculate_daily_returns(portfolio_values)

    if len(daily_returns) < 2:
        raise ValueError("Need at least two return observations.")

    returns_list = [daily_returns[d] for d in sorted(daily_returns.keys())]
    return calculate_volatility(returns_list, periods_in_year)


def calculate_upside_volatility(
    returns: list[float],
    periods_in_year: int = 365
) -> Tuple[float, float]:
    """
    Calculate upside volatility (volatility of positive returns only).

    Upside volatility measures the dispersion of returns above zero,
    capturing the variability of gains.

    Args:
        returns: List of daily returns as decimals (e.g., 0.002 = 0.2%).
        periods_in_year: Trading periods in a year (365 for daily).

    Returns:
        Tuple of (daily_upside_volatility, annual_upside_volatility).

    Raises:
        ValueError: If fewer than two positive return observations.
    """
    positive_returns = [r for r in returns if r > 0]

    if len(positive_returns) < 2:
        raise ValueError("Need at least two positive return observations.")

    returns_array = np.array(positive_returns, dtype=float)
    daily_volatility = float(returns_array.std(ddof=1))
    annual_volatility = daily_volatility * np.sqrt(periods_in_year)

    return daily_volatility, annual_volatility


def calculate_downside_volatility(
    returns: list[float],
    periods_in_year: int = 365
) -> Tuple[float, float]:
    """
    Calculate downside volatility (volatility of negative returns only).

    Downside volatility measures the dispersion of returns below zero,
    capturing the variability of losses.

    Args:
        returns: List of daily returns as decimals (e.g., 0.002 = 0.2%).
        periods_in_year: Trading periods in a year (365 for daily).

    Returns:
        Tuple of (daily_downside_volatility, annual_downside_volatility).

    Raises:
        ValueError: If fewer than two negative return observations.
    """
    negative_returns = [r for r in returns if r < 0]

    if len(negative_returns) < 2:
        raise ValueError("Need at least two negative return observations.")

    returns_array = np.array(negative_returns, dtype=float)
    daily_volatility = float(returns_array.std(ddof=1))
    annual_volatility = daily_volatility * np.sqrt(periods_in_year)

    return daily_volatility, annual_volatility


def calculate_volatility_ratio(
    returns: list[float]
) -> float | None:
    """
    Calculate the ratio of downside to upside volatility.

    A ratio greater than 1 indicates more variability in losses than gains.
    A ratio less than 1 indicates more variability in gains than losses.

    Args:
        returns: List of daily returns as decimals (e.g., 0.002 = 0.2%).

    Returns:
        The volatility ratio (downside/upside), or None if insufficient data.
    """
    positive_returns = [r for r in returns if r > 0]
    negative_returns = [r for r in returns if r < 0]

    if len(positive_returns) < 2 or len(negative_returns) < 2:
        return None

    upside_std = np.array(positive_returns, dtype=float).std(ddof=1)
    downside_std = np.array(negative_returns, dtype=float).std(ddof=1)

    if np.isclose(upside_std, 0.0):
        return None

    return float(downside_std / upside_std)


def calculate_volatility_statistics(
    portfolio: Portfolio,
    target_currency: Currency,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    periods_in_year: int = 365
) -> dict[str, float | None]:
    """
    Calculate comprehensive volatility statistics for a portfolio.

    Args:
        portfolio: Portfolio object containing transactions and settings.
        target_currency: The currency to convert all values into.
        start_date: The start date for the calculation (inclusive).
                    If None, uses the earliest transaction date.
        end_date: The end date for the calculation (inclusive).
                  If None, uses the latest transaction date.
        periods_in_year: Trading periods in a year (365 for daily).

    Returns:
        Dictionary containing:
        - daily_volatility: Daily standard deviation of returns
        - annual_volatility: Annualized standard deviation
        - daily_upside_volatility: Daily std of positive returns (or None)
        - annual_upside_volatility: Annualized std of positive returns (or None)
        - daily_downside_volatility: Daily std of negative returns (or None)
        - annual_downside_volatility: Annualized std of negative returns (or None)
        - volatility_ratio: Downside/upside volatility ratio (or None)
    """
    # Get daily portfolio values
    portfolio_values = calculate_portfolio_value_by_day(
        portfolio,
        target_currency,
        start_date,
        end_date
    )

    if len(portfolio_values) < 2:
        return {
            "daily_volatility": None,
            "annual_volatility": None,
            "daily_upside_volatility": None,
            "annual_upside_volatility": None,
            "daily_downside_volatility": None,
            "annual_downside_volatility": None,
            "volatility_ratio": None,
        }

    daily_returns = calculate_daily_returns(portfolio_values)
    returns_list = [daily_returns[d] for d in sorted(daily_returns.keys())]

    # Calculate overall volatility
    try:
        daily_vol, annual_vol = calculate_volatility(returns_list, periods_in_year)
    except ValueError:
        daily_vol, annual_vol = None, None

    # Calculate upside volatility
    try:
        daily_up_vol, annual_up_vol = calculate_upside_volatility(returns_list, periods_in_year)
    except ValueError:
        daily_up_vol, annual_up_vol = None, None

    # Calculate downside volatility
    try:
        daily_down_vol, annual_down_vol = calculate_downside_volatility(returns_list, periods_in_year)
    except ValueError:
        daily_down_vol, annual_down_vol = None, None

    # Calculate volatility ratio
    vol_ratio = calculate_volatility_ratio(returns_list)

    return {
        "daily_volatility": daily_vol,
        "annual_volatility": annual_vol,
        "daily_upside_volatility": daily_up_vol,
        "annual_upside_volatility": annual_up_vol,
        "daily_downside_volatility": daily_down_vol,
        "annual_downside_volatility": annual_down_vol,
        "volatility_ratio": vol_ratio,
    }
