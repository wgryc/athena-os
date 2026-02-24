"""Tests for alpha_beta module."""

import pytest
import numpy as np
from datetime import datetime

from athena.metrics.alpha_beta import (
    AlphaBetaResult,
    calculate_alpha_beta,
    align_strategy_with_benchmark,
)


class TestCalculateAlphaBeta:
    """Tests for the core calculate_alpha_beta function."""

    def test_perfect_correlation_beta_one(self):
        """Strategy that exactly matches benchmark should have beta=1, alpha=0."""
        # Same returns for strategy and benchmark
        returns = [0.01, -0.02, 0.015, 0.005, -0.01, 0.02, -0.005, 0.01, 0.003, -0.008]

        result = calculate_alpha_beta(
            strategy_returns=returns,
            benchmark_returns=returns,
            risk_free_rates=None
        )

        assert isinstance(result, AlphaBetaResult)
        assert abs(result.beta - 1.0) < 0.0001
        assert abs(result.alpha) < 0.0001
        assert abs(result.r_squared - 1.0) < 0.0001
        assert abs(result.correlation - 1.0) < 0.0001

    def test_double_beta(self):
        """Strategy with 2x leverage should have beta=2."""
        benchmark = [0.01, -0.02, 0.015, 0.005, -0.01, 0.02, -0.005, 0.01, 0.003, -0.008]
        strategy = [r * 2 for r in benchmark]

        result = calculate_alpha_beta(
            strategy_returns=strategy,
            benchmark_returns=benchmark,
            risk_free_rates=None
        )

        assert abs(result.beta - 2.0) < 0.0001
        assert abs(result.alpha) < 0.0001
        assert abs(result.r_squared - 1.0) < 0.0001

    def test_negative_beta(self):
        """Inverse strategy should have beta=-1."""
        benchmark = [0.01, -0.02, 0.015, 0.005, -0.01, 0.02, -0.005, 0.01, 0.003, -0.008]
        strategy = [-r for r in benchmark]

        result = calculate_alpha_beta(
            strategy_returns=strategy,
            benchmark_returns=benchmark,
            risk_free_rates=None
        )

        assert abs(result.beta - (-1.0)) < 0.0001
        assert abs(result.correlation - (-1.0)) < 0.0001

    def test_positive_alpha(self):
        """Strategy with consistent outperformance should have positive alpha."""
        benchmark = [0.01, -0.02, 0.015, 0.005, -0.01, 0.02, -0.005, 0.01, 0.003, -0.008]
        # Add 0.5% daily alpha
        daily_alpha = 0.005
        strategy = [r + daily_alpha for r in benchmark]

        result = calculate_alpha_beta(
            strategy_returns=strategy,
            benchmark_returns=benchmark,
            risk_free_rates=None
        )

        assert abs(result.alpha - daily_alpha) < 0.0001
        assert abs(result.beta - 1.0) < 0.0001

    def test_with_risk_free_rates(self):
        """Test that risk-free rates are properly accounted for."""
        benchmark = [0.01, -0.02, 0.015, 0.005, -0.01, 0.02, -0.005, 0.01, 0.003, -0.008]
        strategy = benchmark.copy()
        rf_rates = [0.0001] * len(benchmark)  # Small constant risk-free rate

        result = calculate_alpha_beta(
            strategy_returns=strategy,
            benchmark_returns=benchmark,
            risk_free_rates=rf_rates
        )

        # With identical strategy and benchmark, beta should still be 1
        assert abs(result.beta - 1.0) < 0.0001

    def test_zero_beta_uncorrelated(self):
        """Uncorrelated strategy should have beta near 0."""
        np.random.seed(42)
        benchmark = list(np.random.normal(0.001, 0.02, 100))
        strategy = list(np.random.normal(0.001, 0.02, 100))

        result = calculate_alpha_beta(
            strategy_returns=strategy,
            benchmark_returns=benchmark,
            risk_free_rates=None
        )

        # Beta should be near 0 for uncorrelated returns
        assert abs(result.beta) < 0.5
        # R-squared should be low
        assert result.r_squared < 0.3

    def test_annualized_alpha(self):
        """Test that alpha is annualized correctly."""
        benchmark = [0.01, -0.02, 0.015, 0.005, -0.01, 0.02, -0.005, 0.01, 0.003, -0.008]
        daily_alpha = 0.001
        strategy = [r + daily_alpha for r in benchmark]

        trading_days = 252
        result = calculate_alpha_beta(
            strategy_returns=strategy,
            benchmark_returns=benchmark,
            risk_free_rates=None,
            trading_days_per_year=trading_days
        )

        expected_annual = daily_alpha * trading_days
        assert abs(result.alpha_annualized - expected_annual) < 0.0001

    def test_insufficient_data_raises(self):
        """Should raise ValueError with fewer than 2 observations."""
        with pytest.raises(ValueError, match="at least two"):
            calculate_alpha_beta([0.01], [0.01], None)

    def test_mismatched_lengths_raises(self):
        """Should raise ValueError when input lengths differ."""
        with pytest.raises(ValueError, match="same length"):
            calculate_alpha_beta([0.01, 0.02], [0.01], None)

    def test_mismatched_rf_rates_raises(self):
        """Should raise ValueError when risk-free rates length differs."""
        with pytest.raises(ValueError, match="same length"):
            calculate_alpha_beta([0.01, 0.02], [0.01, 0.02], [0.001])


class TestAlignStrategyWithBenchmark:
    """Tests for aligning strategy and benchmark returns."""

    def test_align_overlapping_dates(self):
        """Should only include dates present in both dictionaries."""
        dt1 = datetime(2024, 1, 2)  # Tuesday
        dt2 = datetime(2024, 1, 3)  # Wednesday
        dt3 = datetime(2024, 1, 4)  # Thursday
        dt4 = datetime(2024, 1, 5)  # Friday

        strategy = {dt1: 0.01, dt2: 0.02, dt3: 0.015}
        benchmark = {dt2: 0.005, dt3: 0.01, dt4: 0.008}

        aligned_s, aligned_b, aligned_rf, aligned_dates = align_strategy_with_benchmark(
            strategy, benchmark, rf_manager=None
        )

        # Only dt2 and dt3 should be included
        assert len(aligned_s) == 2
        assert len(aligned_b) == 2
        assert aligned_dates == [dt2, dt3]

    def test_align_skips_weekends(self):
        """Should skip weekend dates."""
        dt_friday = datetime(2024, 1, 5)    # Friday
        dt_saturday = datetime(2024, 1, 6)  # Saturday
        dt_sunday = datetime(2024, 1, 7)    # Sunday
        dt_monday = datetime(2024, 1, 8)    # Monday

        strategy = {dt_friday: 0.01, dt_saturday: 0.02, dt_sunday: 0.015, dt_monday: 0.01}
        benchmark = {dt_friday: 0.005, dt_saturday: 0.01, dt_sunday: 0.008, dt_monday: 0.005}

        aligned_s, aligned_b, aligned_rf, aligned_dates = align_strategy_with_benchmark(
            strategy, benchmark, rf_manager=None
        )

        # Saturday and Sunday should be excluded
        assert len(aligned_s) == 2
        assert dt_friday in aligned_dates
        assert dt_monday in aligned_dates
        assert dt_saturday not in aligned_dates
        assert dt_sunday not in aligned_dates

    def test_no_overlap_raises(self):
        """Should raise ValueError when no overlapping dates."""
        dt1 = datetime(2024, 1, 2)
        dt2 = datetime(2024, 1, 3)

        strategy = {dt1: 0.01}
        benchmark = {dt2: 0.02}

        with pytest.raises(ValueError, match="No overlapping dates"):
            align_strategy_with_benchmark(strategy, benchmark, rf_manager=None)
