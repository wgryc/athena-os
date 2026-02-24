#!/usr/bin/env python3
"""Metrics subcommand - Display portfolio metrics report."""

import os
import warnings
from datetime import datetime, timezone
from decimal import Decimal

from dotenv import load_dotenv

load_dotenv()

from ..portfolio import load_portfolio_from_excel, get_positions
from ..pricingdata import (
    MassivePricingDataManager,
    YFinancePricingDataManager,
    DatabentoPricingDataManager,
)
from ..currency import Currency
from ..metrics import (
    # Alpha/Beta
    calculate_alpha_beta_cumulative,
    BENCHMARK_SP500,
    BENCHMARK_NASDAQ,
    # Sharpe ratio
    calculate_sharpe_ratio_cumulative,
    # Sharpe ratio (advanced with FRED data)
    calculate_sharpe_ratio_cumulative_advanced,
    # Sortino ratio
    calculate_sortino_ratio_cumulative,
    # Max drawdown
    calculate_max_drawdown_cumulative,
    # Value at Risk
    calculate_var_cumulative,
    # Volatility
    calculate_volatility_statistics,
    # Win rate
    calculate_win_rate_closed,
    calculate_win_rate_all_positions,
)
from rich.console import Console
from rich.table import Table
from rich.panel import Panel


def format_percentage(value: float | None, precision: int = 2) -> str:
    """Format a decimal value as a percentage string.

    Args:
        value: Decimal value to format (e.g. 0.05 becomes "5.00%").
        precision: Number of decimal places in the output.

    Returns:
        Formatted percentage string, or "N/A" if value is None.
    """
    if value is None:
        return "N/A"
    return f"{value * 100:.{precision}f}%"


def format_ratio(value: float | None, precision: int = 4) -> str:
    """Format a ratio value as a fixed-point string.

    Args:
        value: Ratio value to format.
        precision: Number of decimal places in the output.

    Returns:
        Formatted ratio string, or "N/A" if value is None.
    """
    if value is None:
        return "N/A"
    return f"{value:.{precision}f}"


def format_currency(
    value: Decimal | float | None, symbol: str = "$", precision: int = 2
) -> str:
    """Format a currency value with symbol and thousands separators.

    Args:
        value: Monetary amount to format.
        symbol: Currency symbol to prepend.
        precision: Number of decimal places in the output.

    Returns:
        Formatted currency string, or "N/A" if value is None.
    """
    if value is None:
        return "N/A"
    return f"{symbol}{float(value):,.{precision}f}"


def register_subcommand(subparsers):
    """Register the metrics subcommand with the argument parser.

    Args:
        subparsers: The argparse subparsers action to add the command to.
    """
    parser = subparsers.add_parser(
        "metrics",
        help="Display portfolio metrics report",
        description="Display comprehensive portfolio metrics including risk-adjusted returns, alpha/beta, volatility, and more.",
    )
    parser.add_argument("filename", help="Path to the Excel portfolio file")
    parser.add_argument(
        "--currency",
        "-c",
        default="USD",
        help="Primary currency for portfolio valuation (default: USD)",
    )
    parser.add_argument(
        "--risk-free-rate",
        "-r",
        type=float,
        default=0.05,
        help="Annual risk-free rate for Sharpe/Sortino calculations (default: 0.05 = 5%%)",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Confidence level for VaR calculations (default: 0.95 = 95%%)",
    )
    parser.add_argument(
        "--periods-in-year",
        type=int,
        default=365,
        help="Trading periods in a year (default: 365, use 252 for trading days only)",
    )
    parser.add_argument(
        "--ignore-errors",
        action="store_true",
        help="Ignore negative cash and quantity errors",
    )
    parser.add_argument(
        "--massive",
        action="store_true",
        help="Use Massive API for pricing data (requires MASSIVE_API_KEY env var)",
    )
    parser.add_argument(
        "--databento",
        action="store_true",
        help="Use Databento API for pricing data (requires DATABENTO_API_KEY env var)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass cache and fetch fresh pricing data",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show debug info including current prices for open positions",
    )
    parser.add_argument(
        "--date",
        "-d",
        type=str,
        default=None,
        help="Calculate metrics as of end of business day on this date (format: DD-MM-YYYY). If not specified, uses current time.",
    )
    parser.set_defaults(func=run)


def run(args):
    """Display comprehensive portfolio metrics to the console.

    Args:
        args: Parsed argparse namespace with filename, currency,
            risk_free_rate, confidence_level, periods_in_year,
            ignore_errors, massive, databento, no_cache, debug,
            and date attributes.

    Returns:
        int: Exit code (0 for success, 1 for errors).
    """
    # Suppress warnings if --ignore-errors is set
    if args.ignore_errors:
        warnings.filterwarnings("ignore", category=UserWarning)

    # Parse currency
    try:
        primary_currency = Currency(args.currency.upper())
    except ValueError:
        print(f"Error: Unknown currency '{args.currency}'")
        return 1

    # Load the portfolio from Excel
    portfolio = load_portfolio_from_excel(
        args.filename,
        primary_currency=primary_currency,
        error_out_negative_cash=not args.ignore_errors,
        error_out_negative_quantity=not args.ignore_errors,
        create_if_missing=False,
    )

    # Set up pricing manager
    if args.massive:
        api_key = os.getenv("MASSIVE_API_KEY")
        if not api_key:
            print("Error: MASSIVE_API_KEY environment variable not set")
            return 1
        portfolio.pricing_manager = MassivePricingDataManager(api_key)
    elif args.databento:
        api_key = os.getenv("DATABENTO_API_KEY")
        if not api_key:
            print("Error: DATABENTO_API_KEY environment variable not set")
            return 1
        portfolio.pricing_manager = DatabentoPricingDataManager(api_key)
    elif args.no_cache:
        portfolio.pricing_manager = YFinancePricingDataManager(force_cache_refresh=True)

    console = Console()

    # Parse --date argument or use current time
    if args.date:
        try:
            # Parse DD-MM-YYYY format and set to end of business day (23:59:59 UTC)
            parsed_date = datetime.strptime(args.date, "%d-%m-%Y")
            valuation_datetime = parsed_date.replace(
                hour=23, minute=59, second=59, tzinfo=timezone.utc
            )
            date_display = f"As of {parsed_date.strftime('%d %b %Y')} (EOD)"
        except ValueError:
            print(
                f"Error: Invalid date format '{args.date}'. Expected DD-MM-YYYY (e.g., 15-01-2025)"
            )
            return 1
    else:
        valuation_datetime = datetime.now(timezone.utc)
        local_now = datetime.now().astimezone()
        date_display = local_now.strftime("%Y-%m-%d %H:%M %Z")

    console.print(f"\n[bold]Portfolio Metrics Report[/bold] - {date_display}")
    console.print(
        f"[dim]Currency: {primary_currency.value} | Risk-Free Rate: {args.risk_free_rate:.2%} | Confidence: {args.confidence_level:.0%}[/dim]\n"
    )

    # ==================== Risk-Adjusted Return Metrics ====================
    # Use valuation_datetime for all metric calculations
    now = valuation_datetime

    risk_table = Table(title="Risk-Adjusted Return Metrics")
    risk_table.add_column("Metric", style="cyan", justify="left")
    risk_table.add_column("Daily", justify="right")
    risk_table.add_column("Annualized", justify="right")

    # Sharpe Ratio
    try:
        daily_sharpe, annual_sharpe = calculate_sharpe_ratio_cumulative(
            portfolio,
            primary_currency,
            args.risk_free_rate,
            periods_in_year=args.periods_in_year,
            end_date=now,
        )
        risk_table.add_row(
            "Sharpe Ratio", format_ratio(daily_sharpe), format_ratio(annual_sharpe)
        )
    except ValueError as e:
        risk_table.add_row(
            "Sharpe Ratio", "[dim]Insufficient data[/dim]", f"[dim]{e}[/dim]"
        )

    # Sharpe Ratio (DTB3) - using real Treasury Bill rates from FRED
    try:
        daily_sharpe_dtb3, annual_sharpe_dtb3 = calculate_sharpe_ratio_cumulative_advanced(
            portfolio,
            primary_currency,
            rf_manager=None,  # Uses default FRED DTB3
            trading_days_per_year=252,
            end_date=now,
        )
        risk_table.add_row(
            "Sharpe Ratio (DTB3)",
            format_ratio(daily_sharpe_dtb3),
            format_ratio(annual_sharpe_dtb3),
        )
    except ValueError as e:
        risk_table.add_row(
            "Sharpe Ratio (DTB3)", "[dim]Insufficient data[/dim]", f"[dim]{e}[/dim]"
        )

    # Sortino Ratio
    try:
        daily_sortino, annual_sortino = calculate_sortino_ratio_cumulative(
            portfolio,
            primary_currency,
            args.risk_free_rate,
            periods_in_year=args.periods_in_year,
            end_date=now,
        )
        risk_table.add_row(
            "Sortino Ratio", format_ratio(daily_sortino), format_ratio(annual_sortino)
        )
    except ValueError as e:
        risk_table.add_row(
            "Sortino Ratio", "[dim]Insufficient data[/dim]", f"[dim]{e}[/dim]"
        )

    console.print(risk_table)
    console.print()

    # ==================== Alpha & Beta Metrics ====================
    ab_table = Table(title="Alpha & Beta (vs Market Benchmarks)")
    ab_table.add_column("Benchmark", style="cyan", justify="left")
    ab_table.add_column("Alpha (Annual)", justify="right")
    ab_table.add_column("Beta", justify="right")
    ab_table.add_column("R²", justify="right")

    # S&P 500
    try:
        sp500_result = calculate_alpha_beta_cumulative(
            portfolio, primary_currency, benchmark_ticker=BENCHMARK_SP500, end_date=now
        )
        alpha_color = "green" if sp500_result.alpha_annualized > 0 else "red"
        ab_table.add_row(
            "S&P 500 (^GSPC)",
            f"[{alpha_color}]{format_percentage(sp500_result.alpha_annualized)}[/{alpha_color}]",
            format_ratio(sp500_result.beta, 2),
            format_percentage(sp500_result.r_squared),
        )
    except ValueError:
        ab_table.add_row(
            "S&P 500 (^GSPC)",
            "[dim]Insufficient data[/dim]",
            "[dim]—[/dim]",
            "[dim]—[/dim]",
        )

    # NASDAQ
    try:
        nasdaq_result = calculate_alpha_beta_cumulative(
            portfolio, primary_currency, benchmark_ticker=BENCHMARK_NASDAQ, end_date=now
        )
        alpha_color = "green" if nasdaq_result.alpha_annualized > 0 else "red"
        ab_table.add_row(
            "NASDAQ (^IXIC)",
            f"[{alpha_color}]{format_percentage(nasdaq_result.alpha_annualized)}[/{alpha_color}]",
            format_ratio(nasdaq_result.beta, 2),
            format_percentage(nasdaq_result.r_squared),
        )
    except ValueError:
        ab_table.add_row(
            "NASDAQ (^IXIC)",
            "[dim]Insufficient data[/dim]",
            "[dim]—[/dim]",
            "[dim]—[/dim]",
        )

    console.print(ab_table)
    console.print()

    # ==================== Volatility Metrics ====================
    vol_table = Table(title="Volatility Metrics")
    vol_table.add_column("Metric", style="cyan", justify="left")
    vol_table.add_column("Daily", justify="right")
    vol_table.add_column("Annualized", justify="right")

    vol_stats = calculate_volatility_statistics(
        portfolio, primary_currency, periods_in_year=args.periods_in_year, end_date=now
    )

    vol_table.add_row(
        "Total Volatility",
        format_percentage(vol_stats["daily_volatility"]),
        format_percentage(vol_stats["annual_volatility"]),
    )
    vol_table.add_row(
        "Upside Volatility",
        format_percentage(vol_stats["daily_upside_volatility"]),
        format_percentage(vol_stats["annual_upside_volatility"]),
    )
    vol_table.add_row(
        "Downside Volatility",
        format_percentage(vol_stats["daily_downside_volatility"]),
        format_percentage(vol_stats["annual_downside_volatility"]),
    )

    console.print(vol_table)

    # Volatility ratio
    if vol_stats["volatility_ratio"] is not None:
        ratio = vol_stats["volatility_ratio"]
        if ratio > 1:
            ratio_desc = "[red]More downside variability[/red]"
        elif ratio < 1:
            ratio_desc = "[green]More upside variability[/green]"
        else:
            ratio_desc = "Balanced"
        console.print(
            f"  Volatility Ratio (Down/Up): {format_ratio(ratio)} - {ratio_desc}"
        )
    console.print()

    # ==================== Drawdown Metrics ====================
    dd_table = Table(title="Drawdown Metrics")
    dd_table.add_column("Metric", style="cyan", justify="left")
    dd_table.add_column("Value", justify="right")
    dd_table.add_column("Details", justify="left")

    try:
        max_dd, peak_date, trough_date = calculate_max_drawdown_cumulative(
            portfolio, primary_currency, end_date=now
        )
        dd_pct = format_percentage(max_dd)
        if peak_date and trough_date:
            details = f"Peak: {peak_date.strftime('%Y-%m-%d')} → Trough: {trough_date.strftime('%Y-%m-%d')}"
        else:
            details = ""
        dd_table.add_row("Maximum Drawdown", f"[red]{dd_pct}[/red]", details)
    except ValueError as e:
        dd_table.add_row("Maximum Drawdown", "[dim]Insufficient data[/dim]", str(e))

    console.print(dd_table)
    console.print()

    # ==================== Value at Risk Metrics ====================
    var_table = Table(title=f"Value at Risk ({args.confidence_level:.0%} Confidence)")
    var_table.add_column("Metric", style="cyan", justify="left")
    var_table.add_column("Historical", justify="right")
    var_table.add_column("Parametric", justify="right")

    # Historical VaR
    try:
        var_hist, cvar_hist = calculate_var_cumulative(
            portfolio,
            primary_currency,
            confidence_level=args.confidence_level,
            method="historical",
            end_date=now,
        )
        var_hist_str = f"[red]{format_percentage(var_hist)}[/red]"
        cvar_hist_str = f"[red]{format_percentage(cvar_hist)}[/red]"
    except ValueError:
        var_hist_str = "[dim]Insufficient data[/dim]"
        cvar_hist_str = "[dim]Insufficient data[/dim]"

    # Parametric VaR
    try:
        var_param, cvar_param = calculate_var_cumulative(
            portfolio,
            primary_currency,
            confidence_level=args.confidence_level,
            method="parametric",
            end_date=now,
        )
        var_param_str = f"[red]{format_percentage(var_param)}[/red]"
        cvar_param_str = f"[red]{format_percentage(cvar_param)}[/red]"
    except ValueError:
        var_param_str = "[dim]Insufficient data[/dim]"
        cvar_param_str = "[dim]Insufficient data[/dim]"

    var_table.add_row("Value at Risk (VaR)", var_hist_str, var_param_str)
    var_table.add_row("Conditional VaR (CVaR)", cvar_hist_str, cvar_param_str)

    console.print(var_table)
    console.print()

    # ==================== Win Rate Metrics ====================
    win_table = Table(title="Win Rate Metrics")
    win_table.add_column("Metric", style="cyan", justify="left")
    win_table.add_column("Closed Positions", justify="right")
    win_table.add_column("All Positions", justify="right")

    closed_wr = calculate_win_rate_closed(portfolio)
    all_wr = calculate_win_rate_all_positions(portfolio, as_of=now)

    # Win Rate
    if closed_wr.total_positions > 0:
        closed_wr_str = f"[green]{closed_wr.win_rate:.1%}[/green]"
    else:
        closed_wr_str = "[dim]No closed positions[/dim]"

    if all_wr.total_positions > 0:
        all_wr_str = f"[green]{all_wr.win_rate:.1%}[/green]"
    else:
        all_wr_str = "[dim]No positions[/dim]"

    win_table.add_row("Win Rate", closed_wr_str, all_wr_str)

    # Position counts
    win_table.add_row(
        "Total Positions", str(closed_wr.total_positions), str(all_wr.total_positions)
    )
    win_table.add_row(
        "Winning",
        f"[green]{closed_wr.winning_positions}[/green]",
        f"[green]{all_wr.winning_positions}[/green]",
    )
    win_table.add_row(
        "Losing",
        f"[red]{closed_wr.losing_positions}[/red]",
        f"[red]{all_wr.losing_positions}[/red]",
    )
    win_table.add_row(
        "Breakeven",
        str(closed_wr.breakeven_positions),
        str(all_wr.breakeven_positions),
    )

    # Average win/loss
    win_table.add_row(
        "Average Win",
        format_currency(closed_wr.average_win),
        format_currency(all_wr.average_win),
    )
    win_table.add_row(
        "Average Loss",
        format_currency(closed_wr.average_loss),
        format_currency(all_wr.average_loss),
    )

    # Win/Loss Ratio
    win_table.add_row(
        "Win/Loss Ratio",
        format_ratio(closed_wr.win_loss_ratio, 2),
        format_ratio(all_wr.win_loss_ratio, 2),
    )

    # Total P&L
    win_table.add_row(
        "Total Gain/Loss",
        format_currency(closed_wr.total_gain_loss),
        format_currency(all_wr.total_gain_loss),
    )

    console.print(win_table)
    console.print()

    # ==================== Summary Panel ====================
    summary_lines = []

    # Best metric highlight
    if vol_stats["annual_volatility"] is not None:
        annual_vol = vol_stats["annual_volatility"]
        if annual_vol < 0.15:
            summary_lines.append(
                "[green]✓ Low volatility portfolio (<15% annual)[/green]"
            )
        elif annual_vol > 0.30:
            summary_lines.append(
                "[red]⚠ High volatility portfolio (>30% annual)[/red]"
            )

    try:
        _, annual_sharpe = calculate_sharpe_ratio_cumulative(
            portfolio,
            primary_currency,
            args.risk_free_rate,
            periods_in_year=args.periods_in_year,
            end_date=now,
        )
        if annual_sharpe > 1.0:
            summary_lines.append(
                "[green]✓ Good risk-adjusted returns (Sharpe > 1.0)[/green]"
            )
        elif annual_sharpe < 0:
            summary_lines.append(
                "[red]⚠ Negative risk-adjusted returns (Sharpe < 0)[/red]"
            )
    except ValueError:
        pass

    try:
        max_dd, _, _ = calculate_max_drawdown_cumulative(
            portfolio, primary_currency, end_date=now
        )
        if max_dd > -0.10:
            summary_lines.append("[green]✓ Low maximum drawdown (<10%)[/green]")
        elif max_dd < -0.30:
            summary_lines.append(
                "[red]⚠ Significant maximum drawdown (>30%)[/red]"
            )
    except ValueError:
        pass

    if closed_wr.total_positions > 0 and closed_wr.win_rate > 0.5:
        summary_lines.append(
            f"[green]✓ Positive win rate on closed positions ({closed_wr.win_rate:.1%})[/green]"
        )

    if summary_lines:
        console.print(Panel("\n".join(summary_lines), title="Summary Insights"))

    # Debug output: show prices for open positions at valuation date
    if args.debug:
        console.print()
        debug_title = (
            "Debug: Open Position Prices"
            + (f" (as of {args.date})" if args.date else " (Current)")
        )
        debug_table = Table(title=debug_title)
        debug_table.add_column("Symbol", style="cyan", justify="left")
        debug_table.add_column("Quantity", justify="right")
        debug_table.add_column("Price", justify="right")
        debug_table.add_column("Value", justify="right")
        debug_table.add_column("Price Date", justify="left")

        positions = get_positions(now, portfolio)
        for pos in sorted(positions, key=lambda p: p.symbol):
            if pos.quantity != 0:
                # Get the price point to show the date
                price_point = portfolio.pricing_manager.get_price_point(pos.symbol, now)
                debug_table.add_row(
                    pos.symbol,
                    f"{pos.quantity:,.4f}",
                    f"${price_point.price:,.2f}",
                    f"${pos.total_value:,.2f}",
                    price_point.price_datetime.strftime("%Y-%m-%d %H:%M"),
                )

        console.print(debug_table)

    return 0
