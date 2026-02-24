#!/usr/bin/env python3
"""Report subcommand - Display portfolio holdings report."""

import os
import warnings
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

from ..portfolio import (
    load_portfolio_from_excel,
    get_positions,
    get_cash_balances,
    calculate_portfolio_value_on_date,
)
from ..pricingdata import (
    MassivePricingDataManager,
    YFinancePricingDataManager,
    DatabentoPricingDataManager,
)
from ..currency import Currency
from rich.console import Console
from rich.table import Table
from rich.panel import Panel


def register_subcommand(subparsers):
    """Register the report subcommand with the argument parser.

    Args:
        subparsers: The argparse subparsers action to add the command to.
    """
    parser = subparsers.add_parser(
        "report",
        help="Display portfolio holdings report",
        description="Display current portfolio holdings, positions, and cash balances from an Excel file.",
    )
    parser.add_argument("filename", help="Path to the Excel portfolio file")
    parser.add_argument(
        "--currency",
        "-c",
        default="USD",
        help="Primary currency for portfolio valuation (default: USD)",
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
    parser.set_defaults(func=run)


def run(args):
    """Display portfolio holdings, positions, and cash balances.

    Args:
        args: Parsed argparse namespace with filename, currency,
            ignore_errors, massive, databento, and no_cache attributes.

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
    now = datetime.now(timezone.utc)
    local_now = datetime.now().astimezone()

    # Get current positions
    positions = get_positions(now, portfolio)

    # Get cash balances by currency
    cash_balances = get_cash_balances(portfolio, as_of=now)

    # Get total portfolio value
    total_value = calculate_portfolio_value_on_date(portfolio, now, primary_currency)

    # Create holdings table
    holdings_table = Table(
        title=f"Open Positions on {local_now.strftime('%Y-%m-%d %H:%M %Z')}"
    )
    holdings_table.add_column("Symbol", style="cyan", justify="left")
    holdings_table.add_column("Quantity", style="magenta", justify="right")
    holdings_table.add_column("Unit Price\n(Book → Market)", justify="right")
    holdings_table.add_column(
        f"Book Value ({primary_currency.value})", style="yellow", justify="right"
    )
    holdings_table.add_column(
        f"Market Value ({primary_currency.value})", style="green", justify="right"
    )
    holdings_table.add_column("Gain/Loss %", justify="right")

    for position in positions:
        # Format gain/loss percentage with color
        if position.gain_loss_percent is not None:
            gain_pct = position.gain_loss_percent
            if gain_pct >= 0:
                gain_str = f"[green]+{gain_pct:.2f}%[/green]"
            else:
                gain_str = f"[red]{gain_pct:.2f}%[/red]"
        else:
            gain_str = "N/A"

        # Format unit price: book unit price → market unit price
        if position.book_value is not None and position.quantity != 0:
            book_unit_price = position.book_value / position.quantity
            if position.unit_price is not None:
                unit_price_str = f"[yellow]${book_unit_price:,.2f}[/yellow] → [green]${position.unit_price:,.2f}[/green]"
            else:
                unit_price_str = f"[yellow]${book_unit_price:,.2f}[/yellow] → N/A"
        elif position.unit_price is not None:
            unit_price_str = f"N/A → [green]${position.unit_price:,.2f}[/green]"
        else:
            unit_price_str = "N/A"

        holdings_table.add_row(
            position.symbol,
            f"{position.quantity:,.0f}",
            unit_price_str,
            f"${position.book_value:,.2f}" if position.book_value else "N/A",
            f"${position.total_value:,.2f}",
            gain_str,
        )

    console.print(holdings_table)

    # Create cash balances table
    cash_table = Table(title="Cash Balances")
    cash_table.add_column("Currency", style="cyan", justify="left")
    cash_table.add_column("Balance", style="yellow", justify="right")

    for currency, balance in sorted(cash_balances.items(), key=lambda x: x[0].value):
        cash_table.add_row(currency.value, f"{balance:,.2f}")

    console.print(cash_table)

    # Print total portfolio value
    console.print(
        Panel(
            f"[bold green]Total Portfolio Value: ${total_value:,.2f}[/bold green]",
            title="Summary",
        )
    )

    return 0
