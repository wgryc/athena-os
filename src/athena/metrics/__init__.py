"""Portfolio risk and performance metrics.

Provides functions for calculating Sharpe ratio, Sortino ratio, maximum
drawdown, Value at Risk, volatility, win rate, and alpha/beta against
market benchmarks.  Sub-modules for options-implied analytics
(forward curves, theta decay) are available but not re-exported here.
"""

from .alpha_beta import (
    BENCHMARK_DOW,
    BENCHMARK_NASDAQ,
    BENCHMARK_RUSSELL2000,
    BENCHMARK_SP500,
    AlphaBetaResult,
    align_strategy_with_benchmark,
    calculate_alpha_beta,
    calculate_alpha_beta_by_day_cumulative,
    calculate_alpha_beta_by_day_rolling_window,
    calculate_alpha_beta_cumulative,
    calculate_alpha_beta_from_values,
    fetch_benchmark_returns,
)
from .max_drawdown import (
    calculate_drawdown_series,
    calculate_max_drawdown,
    calculate_max_drawdown_by_day_cumulative,
    calculate_max_drawdown_by_day_rolling_window,
    calculate_max_drawdown_cumulative,
    calculate_max_drawdown_from_values,
)
from .sharpe import (
    calculate_daily_returns,
    calculate_sharpe_ratio,
    calculate_sharpe_ratio_by_day_cumulative,
    calculate_sharpe_ratio_by_day_rolling_window,
    calculate_sharpe_ratio_cumulative,
    calculate_sharpe_ratio_from_values,
)
from .sharpe_advanced import (
    align_returns_with_risk_free_rates,
    calculate_sharpe_ratio_advanced,
    calculate_sharpe_ratio_by_day_cumulative_advanced,
    calculate_sharpe_ratio_by_day_rolling_window_advanced,
    calculate_sharpe_ratio_cumulative_advanced,
    calculate_sharpe_ratio_from_values_advanced,
)
from .sortino import (
    calculate_downside_deviation,
    calculate_sortino_ratio,
    calculate_sortino_ratio_by_day_cumulative,
    calculate_sortino_ratio_by_day_rolling_window,
    calculate_sortino_ratio_cumulative,
    calculate_sortino_ratio_from_values,
)
from .value_at_risk import (
    calculate_cvar,
    calculate_var_by_day_cumulative,
    calculate_var_by_day_rolling_window,
    calculate_var_cumulative,
    calculate_var_from_values,
    calculate_var_historical,
    calculate_var_parametric,
)
from .volatility import (
    calculate_downside_volatility,
    calculate_upside_volatility,
    calculate_volatility,
    calculate_volatility_by_day_cumulative,
    calculate_volatility_by_day_rolling_window,
    calculate_volatility_cumulative,
    calculate_volatility_from_values,
    calculate_volatility_ratio,
    calculate_volatility_statistics,
)
from .win_rate import (
    ClosedPosition,
    WinRateResult,
    calculate_win_rate_all_positions,
    calculate_win_rate_by_symbol,
    calculate_win_rate_closed,
    get_closed_positions,
)

__all__ = [
    # Alpha/Beta functions
    "BENCHMARK_DOW",
    "BENCHMARK_NASDAQ",
    "BENCHMARK_RUSSELL2000",
    "BENCHMARK_SP500",
    "AlphaBetaResult",
    "align_strategy_with_benchmark",
    "calculate_alpha_beta",
    "calculate_alpha_beta_by_day_cumulative",
    "calculate_alpha_beta_by_day_rolling_window",
    "calculate_alpha_beta_cumulative",
    "calculate_alpha_beta_from_values",
    "fetch_benchmark_returns",
    # Max drawdown functions
    "calculate_drawdown_series",
    "calculate_max_drawdown",
    "calculate_max_drawdown_by_day_cumulative",
    "calculate_max_drawdown_by_day_rolling_window",
    "calculate_max_drawdown_cumulative",
    "calculate_max_drawdown_from_values",
    # Sharpe ratio functions
    "calculate_daily_returns",
    "calculate_sharpe_ratio",
    "calculate_sharpe_ratio_by_day_cumulative",
    "calculate_sharpe_ratio_by_day_rolling_window",
    "calculate_sharpe_ratio_cumulative",
    "calculate_sharpe_ratio_from_values",
    # Sharpe ratio advanced functions (with FRED risk-free rates)
    "align_returns_with_risk_free_rates",
    "calculate_sharpe_ratio_advanced",
    "calculate_sharpe_ratio_by_day_cumulative_advanced",
    "calculate_sharpe_ratio_by_day_rolling_window_advanced",
    "calculate_sharpe_ratio_cumulative_advanced",
    "calculate_sharpe_ratio_from_values_advanced",
    # Sortino ratio functions
    "calculate_downside_deviation",
    "calculate_sortino_ratio",
    "calculate_sortino_ratio_by_day_cumulative",
    "calculate_sortino_ratio_by_day_rolling_window",
    "calculate_sortino_ratio_cumulative",
    "calculate_sortino_ratio_from_values",
    # Value at Risk functions
    "calculate_cvar",
    "calculate_var_by_day_cumulative",
    "calculate_var_by_day_rolling_window",
    "calculate_var_cumulative",
    "calculate_var_from_values",
    "calculate_var_historical",
    "calculate_var_parametric",
    # Volatility functions
    "calculate_downside_volatility",
    "calculate_upside_volatility",
    "calculate_volatility",
    "calculate_volatility_by_day_cumulative",
    "calculate_volatility_by_day_rolling_window",
    "calculate_volatility_cumulative",
    "calculate_volatility_from_values",
    "calculate_volatility_ratio",
    "calculate_volatility_statistics",
    # Win rate functions
    "ClosedPosition",
    "WinRateResult",
    "calculate_win_rate_all_positions",
    "calculate_win_rate_by_symbol",
    "calculate_win_rate_closed",
    "get_closed_positions",
]
