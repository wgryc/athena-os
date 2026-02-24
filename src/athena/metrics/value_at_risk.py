from datetime import datetime
from decimal import Decimal
from typing import Tuple

import numpy as np
from scipy import stats

from ..currency import Currency
from ..portfolio import Portfolio, calculate_portfolio_value_by_day
from .sharpe import calculate_daily_returns


def calculate_var_historical(
    returns: list[float],
    confidence_level: float = 0.95
) -> float:
    """
    Calculate Value at Risk using the historical simulation method.

    Historical VaR uses the actual distribution of past returns to estimate
    potential losses at a given confidence level.

    Args:
        returns: List of daily returns as decimals (e.g., 0.002 = 0.2%).
        confidence_level: Confidence level for VaR (e.g., 0.95 for 95%).

    Returns:
        VaR as a negative float representing the potential loss
        (e.g., -0.05 means 5% potential loss at the given confidence level).

    Raises:
        ValueError: If fewer than two observations or invalid confidence level.
    """
    if len(returns) < 2:
        raise ValueError("Need at least two return observations.")

    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between 0 and 1.")

    returns_array = np.array(returns, dtype=float)
    var = float(np.percentile(returns_array, (1 - confidence_level) * 100))

    return var


def calculate_var_parametric(
    returns: list[float],
    confidence_level: float = 0.95
) -> float:
    """
    Calculate Value at Risk using the parametric (variance-covariance) method.

    Parametric VaR assumes returns are normally distributed and uses the
    mean and standard deviation to estimate potential losses.

    Args:
        returns: List of daily returns as decimals (e.g., 0.002 = 0.2%).
        confidence_level: Confidence level for VaR (e.g., 0.95 for 95%).

    Returns:
        VaR as a negative float representing the potential loss
        (e.g., -0.05 means 5% potential loss at the given confidence level).

    Raises:
        ValueError: If fewer than two observations or invalid confidence level.
    """
    if len(returns) < 2:
        raise ValueError("Need at least two return observations.")

    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between 0 and 1.")

    returns_array = np.array(returns, dtype=float)
    mean_return = returns_array.mean()
    std_return = returns_array.std(ddof=1)

    # Get the z-score for the confidence level
    z_score = stats.norm.ppf(1 - confidence_level)
    var = float(mean_return + z_score * std_return)

    return var


def calculate_cvar(
    returns: list[float],
    confidence_level: float = 0.95
) -> float:
    """
    Calculate Conditional Value at Risk (Expected Shortfall).

    CVaR represents the expected loss given that the loss exceeds the VaR
    threshold. It provides a more complete picture of tail risk than VaR alone.

    Args:
        returns: List of daily returns as decimals (e.g., 0.002 = 0.2%).
        confidence_level: Confidence level for CVaR (e.g., 0.95 for 95%).

    Returns:
        CVaR as a negative float representing the expected loss beyond VaR
        (e.g., -0.08 means 8% expected loss in worst cases).

    Raises:
        ValueError: If fewer than two observations or invalid confidence level.
    """
    if len(returns) < 2:
        raise ValueError("Need at least two return observations.")

    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between 0 and 1.")

    returns_array = np.array(returns, dtype=float)
    var = calculate_var_historical(returns, confidence_level)

    # CVaR is the mean of returns that are worse than VaR
    tail_returns = returns_array[returns_array <= var]

    if len(tail_returns) == 0:
        return var

    return float(tail_returns.mean())


def calculate_var_by_day_cumulative(
    portfolio: Portfolio,
    target_currency: Currency,
    confidence_level: float = 0.95,
    method: str = "historical",
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    min_observations: int = 30
) -> dict[datetime, float]:
    """
    Calculate cumulative Value at Risk for each day in a date range.

    For each day, calculates VaR using all returns from start_date up to
    that day (cumulative/expanding window).

    Args:
        portfolio: Portfolio object containing transactions and settings.
        target_currency: The currency to convert all values into.
        confidence_level: Confidence level for VaR (e.g., 0.95 for 95%).
        method: VaR calculation method ("historical" or "parametric").
        start_date: The start date for the calculation (inclusive).
                    If None, uses the earliest transaction date.
        end_date: The end date for the calculation (inclusive).
                  If None, uses the latest transaction date.
        min_observations: Minimum number of return observations required
                          before calculating VaR. Defaults to 30.

    Returns:
        Dictionary mapping each date to the cumulative VaR up to that date.
        Days with insufficient observations are excluded.
    """
    if method not in ("historical", "parametric"):
        raise ValueError("Method must be 'historical' or 'parametric'.")

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
    result: dict[datetime, float] = {}
    cumulative_returns: list[float] = []

    var_func = calculate_var_historical if method == "historical" else calculate_var_parametric

    for date in sorted_dates:
        cumulative_returns.append(daily_returns[date])

        if len(cumulative_returns) < min_observations:
            continue

        try:
            var = var_func(cumulative_returns, confidence_level)
            result[date] = var
        except ValueError:
            continue

    return result


def calculate_var_cumulative(
    portfolio: Portfolio,
    target_currency: Currency,
    confidence_level: float = 0.95,
    method: str = "historical",
    start_date: datetime | None = None,
    end_date: datetime | None = None
) -> Tuple[float, float]:
    """
    Calculate Value at Risk and CVaR for a portfolio over a date range.

    Args:
        portfolio: Portfolio object containing transactions and settings.
        target_currency: The currency to convert all values into.
        confidence_level: Confidence level for VaR (e.g., 0.95 for 95%).
        method: VaR calculation method ("historical" or "parametric").
        start_date: The start date for the calculation (inclusive).
                    If None, uses the earliest transaction date.
        end_date: The end date for the calculation (inclusive).
                  If None, uses the latest transaction date.

    Returns:
        Tuple of (var, cvar) as negative floats.

    Raises:
        ValueError: If fewer than two observations or invalid parameters.
    """
    if method not in ("historical", "parametric"):
        raise ValueError("Method must be 'historical' or 'parametric'.")

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

    var_func = calculate_var_historical if method == "historical" else calculate_var_parametric
    var = var_func(returns_list, confidence_level)
    cvar = calculate_cvar(returns_list, confidence_level)

    return var, cvar


def calculate_var_by_day_rolling_window(
    portfolio: Portfolio,
    target_currency: Currency,
    window_size: int,
    confidence_level: float = 0.95,
    method: str = "historical",
    start_date: datetime | None = None,
    end_date: datetime | None = None
) -> dict[datetime, float]:
    """
    Calculate rolling window Value at Risk for each day in a date range.

    For each day, calculates VaR using the trailing window_size days of returns.

    Args:
        portfolio: Portfolio object containing transactions and settings.
        target_currency: The currency to convert all values into.
        window_size: Number of trailing days to use for each calculation.
                     Common values: 30, 60, 90, 252 (trading year).
        confidence_level: Confidence level for VaR (e.g., 0.95 for 95%).
        method: VaR calculation method ("historical" or "parametric").
        start_date: The start date for the calculation (inclusive).
                    If None, uses the earliest transaction date.
        end_date: The end date for the calculation (inclusive).
                  If None, uses the latest transaction date.

    Returns:
        Dictionary mapping each date to the rolling VaR.
        Days with insufficient observations (fewer than window_size) are excluded.
    """
    if window_size < 2:
        raise ValueError("window_size must be at least 2.")

    if method not in ("historical", "parametric"):
        raise ValueError("Method must be 'historical' or 'parametric'.")

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

    var_func = calculate_var_historical if method == "historical" else calculate_var_parametric

    result: dict[datetime, float] = {}

    for i in range(window_size - 1, len(sorted_dates)):
        window_returns = returns_list[i - window_size + 1:i + 1]
        date = sorted_dates[i]

        try:
            var = var_func(window_returns, confidence_level)
            result[date] = var
        except ValueError:
            continue

    return result


def calculate_var_from_values(
    portfolio_values: dict[datetime, Decimal],
    confidence_level: float = 0.95,
    method: str = "historical"
) -> Tuple[float, float]:
    """
    Calculate VaR and CVaR directly from a dictionary of portfolio values.

    This is a convenience function when you already have portfolio values
    and don't need to calculate them from a Portfolio object.

    Args:
        portfolio_values: Dictionary mapping dates to portfolio values.
        confidence_level: Confidence level for VaR (e.g., 0.95 for 95%).
        method: VaR calculation method ("historical" or "parametric").

    Returns:
        Tuple of (var, cvar) as negative floats.

    Raises:
        ValueError: If fewer than two observations or invalid parameters.
    """
    if method not in ("historical", "parametric"):
        raise ValueError("Method must be 'historical' or 'parametric'.")

    daily_returns = calculate_daily_returns(portfolio_values)

    if len(daily_returns) < 2:
        raise ValueError("Need at least two return observations.")

    returns_list = [daily_returns[d] for d in sorted(daily_returns.keys())]

    var_func = calculate_var_historical if method == "historical" else calculate_var_parametric
    var = var_func(returns_list, confidence_level)
    cvar = calculate_cvar(returns_list, confidence_level)

    return var, cvar
