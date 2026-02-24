from datetime import datetime
from decimal import Decimal
from typing import Tuple

import numpy as np

from ..currency import Currency
from ..portfolio import Portfolio, calculate_portfolio_value_by_day


def calculate_max_drawdown(
    portfolio_values: dict[datetime, Decimal]
) -> Tuple[float, datetime | None, datetime | None]:
    """
    Calculate the maximum drawdown from a time series of portfolio values.

    Maximum drawdown measures the largest peak-to-trough decline in portfolio
    value, expressed as a percentage of the peak value.

    Args:
        portfolio_values: Dictionary mapping dates to portfolio values.

    Returns:
        Tuple of (max_drawdown, peak_date, trough_date).
        max_drawdown is a negative float (e.g., -0.20 = 20% drawdown).
        peak_date and trough_date are None if insufficient data.

    Raises:
        ValueError: If fewer than two observations.
    """
    if len(portfolio_values) < 2:
        raise ValueError("Need at least two portfolio value observations.")

    sorted_dates = sorted(portfolio_values.keys())
    values = [float(portfolio_values[d]) for d in sorted_dates]

    max_drawdown = 0.0
    peak_value = values[0]
    peak_date: datetime | None = sorted_dates[0]
    trough_date: datetime | None = None
    current_peak_date = sorted_dates[0]

    for i, (date, value) in enumerate(zip(sorted_dates, values)):
        if value > peak_value:
            peak_value = value
            current_peak_date = date

        if peak_value > 0:
            drawdown = (value - peak_value) / peak_value
            if drawdown < max_drawdown:
                max_drawdown = drawdown
                peak_date = current_peak_date
                trough_date = date

    return max_drawdown, peak_date, trough_date


def calculate_drawdown_series(
    portfolio_values: dict[datetime, Decimal]
) -> dict[datetime, float]:
    """
    Calculate the drawdown at each point in time from a time series of portfolio values.

    Args:
        portfolio_values: Dictionary mapping dates to portfolio values.

    Returns:
        Dictionary mapping dates to drawdown values (negative floats).
        A drawdown of -0.10 means the portfolio is 10% below its peak.
    """
    if len(portfolio_values) < 1:
        return {}

    sorted_dates = sorted(portfolio_values.keys())
    values = [float(portfolio_values[d]) for d in sorted_dates]

    drawdowns: dict[datetime, float] = {}
    peak_value = values[0]

    for date, value in zip(sorted_dates, values):
        if value > peak_value:
            peak_value = value

        if peak_value > 0:
            drawdowns[date] = (value - peak_value) / peak_value
        else:
            drawdowns[date] = 0.0

    return drawdowns


def calculate_max_drawdown_by_day_cumulative(
    portfolio: Portfolio,
    target_currency: Currency,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    min_observations: int = 2
) -> dict[datetime, float]:
    """
    Calculate cumulative maximum drawdown for each day in a date range.

    For each day, calculates the maximum drawdown using all portfolio values
    from start_date up to that day (cumulative/expanding window).

    Args:
        portfolio: Portfolio object containing transactions and settings.
        target_currency: The currency to convert all values into.
        start_date: The start date for the calculation (inclusive).
                    If None, uses the earliest transaction date.
        end_date: The end date for the calculation (inclusive).
                  If None, uses the latest transaction date.
        min_observations: Minimum number of observations required
                          before calculating max drawdown. Defaults to 2.

    Returns:
        Dictionary mapping each date to the cumulative max drawdown up to that date.
        Days with insufficient observations are excluded.
    """
    # Get daily portfolio values
    portfolio_values = calculate_portfolio_value_by_day(
        portfolio,
        target_currency,
        start_date,
        end_date
    )

    if len(portfolio_values) < min_observations:
        return {}

    sorted_dates = sorted(portfolio_values.keys())
    values = [float(portfolio_values[d]) for d in sorted_dates]

    result: dict[datetime, float] = {}
    max_drawdown = 0.0
    peak_value = values[0]

    for i, (date, value) in enumerate(zip(sorted_dates, values)):
        if value > peak_value:
            peak_value = value

        if peak_value > 0:
            drawdown = (value - peak_value) / peak_value
            if drawdown < max_drawdown:
                max_drawdown = drawdown

        if i + 1 >= min_observations:
            result[date] = max_drawdown

    return result


def calculate_max_drawdown_cumulative(
    portfolio: Portfolio,
    target_currency: Currency,
    start_date: datetime | None = None,
    end_date: datetime | None = None
) -> Tuple[float, datetime | None, datetime | None]:
    """
    Calculate the maximum drawdown for a portfolio over a date range.

    Args:
        portfolio: Portfolio object containing transactions and settings.
        target_currency: The currency to convert all values into.
        start_date: The start date for the calculation (inclusive).
                    If None, uses the earliest transaction date.
        end_date: The end date for the calculation (inclusive).
                  If None, uses the latest transaction date.

    Returns:
        Tuple of (max_drawdown, peak_date, trough_date).
        max_drawdown is a negative float (e.g., -0.20 = 20% drawdown).

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

    return calculate_max_drawdown(portfolio_values)


def calculate_max_drawdown_by_day_rolling_window(
    portfolio: Portfolio,
    target_currency: Currency,
    window_size: int,
    start_date: datetime | None = None,
    end_date: datetime | None = None
) -> dict[datetime, float]:
    """
    Calculate rolling window maximum drawdown for each day in a date range.

    For each day, calculates the maximum drawdown using the trailing
    window_size days of portfolio values.

    Args:
        portfolio: Portfolio object containing transactions and settings.
        target_currency: The currency to convert all values into.
        window_size: Number of trailing days to use for each calculation.
                     Common values: 30, 60, 90, 252 (trading year).
        start_date: The start date for the calculation (inclusive).
                    If None, uses the earliest transaction date.
        end_date: The end date for the calculation (inclusive).
                  If None, uses the latest transaction date.

    Returns:
        Dictionary mapping each date to the rolling max drawdown.
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

    if len(portfolio_values) < window_size:
        return {}

    sorted_dates = sorted(portfolio_values.keys())
    values = [float(portfolio_values[d]) for d in sorted_dates]

    result: dict[datetime, float] = {}

    for i in range(window_size - 1, len(sorted_dates)):
        window_values = values[i - window_size + 1:i + 1]
        date = sorted_dates[i]

        # Calculate max drawdown for this window
        max_drawdown = 0.0
        peak_value = window_values[0]

        for value in window_values:
            if value > peak_value:
                peak_value = value

            if peak_value > 0:
                drawdown = (value - peak_value) / peak_value
                if drawdown < max_drawdown:
                    max_drawdown = drawdown

        result[date] = max_drawdown

    return result


def calculate_max_drawdown_from_values(
    portfolio_values: dict[datetime, Decimal]
) -> Tuple[float, datetime | None, datetime | None]:
    """
    Calculate maximum drawdown directly from a dictionary of portfolio values.

    This is a convenience function when you already have portfolio values
    and don't need to calculate them from a Portfolio object.

    Args:
        portfolio_values: Dictionary mapping dates to portfolio values.

    Returns:
        Tuple of (max_drawdown, peak_date, trough_date).
        max_drawdown is a negative float (e.g., -0.20 = 20% drawdown).

    Raises:
        ValueError: If fewer than two observations.
    """
    return calculate_max_drawdown(portfolio_values)
