#!/usr/bin/env python3
"""Dashboard command for generating HTML portfolio dashboards."""

import os
import warnings
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

from ..currency import Currency
from ..dashboard import generate_dashboard
from ..portfolio import load_portfolio_from_excel
from ..pricingdata import (
    MassivePricingDataManager,
    YFinancePricingDataManager,
    DatabentoPricingDataManager,
)

load_dotenv()


def get_unique_filename(filepath: Path) -> Path:
    """
    Get a unique filename by appending '_copy' if the file exists.

    Args:
        filepath: The desired file path.

    Returns:
        A unique file path that doesn't exist.
    """
    if not filepath.exists():
        return filepath

    stem = filepath.stem
    suffix = filepath.suffix
    parent = filepath.parent

    # Keep appending '_copy' until we find a unique name
    while filepath.exists():
        stem = f"{stem}_copy"
        filepath = parent / f"{stem}{suffix}"

    return filepath


def register_subcommand(subparsers):
    """Register the dashboard subcommand.

    Args:
        subparsers: The argparse subparsers action to add the command to.
    """
    parser = subparsers.add_parser(
        "dashboard",
        help="Generate HTML portfolio dashboard",
        description="Generate an interactive HTML dashboard from a portfolio Excel file.",
    )

    parser.add_argument("filename", help="Path to the Excel portfolio file")
    parser.add_argument(
        "--output",
        "-o",
        metavar="FILE",
        help="Output HTML filename (default: dashboard_<input_filename>.html)",
    )
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
        help="Annual risk-free rate for Sharpe calculations (default: 0.05 = 5%%)",
    )
    parser.add_argument(
        "--title",
        "-t",
        default="Portfolio Dashboard",
        help="Dashboard title (default: 'Portfolio Dashboard')",
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

    parser.set_defaults(func=run_dashboard)


def run_dashboard(args):
    """Generate an HTML portfolio dashboard from an Excel file.

    Args:
        args: Parsed argparse namespace with filename, output, currency,
            risk_free_rate, title, ignore_errors, massive, databento,
            and no_cache attributes.

    Returns:
        int: Exit code (0 for success, 1 for errors).
    """
    # Suppress warnings if --ignore-errors is set
    if args.ignore_errors:
        warnings.filterwarnings("ignore", category=UserWarning)

    console = Console()

    # Parse currency
    try:
        primary_currency = Currency(args.currency.upper())
    except ValueError:
        console.print(f"[red]Error: Unknown currency '{args.currency}'[/red]")
        return 1

    # Load the portfolio
    console.print(f"Loading portfolio from [cyan]{args.filename}[/cyan]...")
    try:
        portfolio = load_portfolio_from_excel(
            args.filename,
            primary_currency=primary_currency,
            error_out_negative_cash=not args.ignore_errors,
            error_out_negative_quantity=not args.ignore_errors,
            create_if_missing=False,
        )
    except FileNotFoundError:
        console.print(f"[red]Error: File '{args.filename}' not found[/red]")
        return 1
    except Exception as e:
        console.print(f"[red]Error loading portfolio: {e}[/red]")
        return 1

    # Set up pricing manager
    if args.massive:
        api_key = os.getenv("MASSIVE_API_KEY")
        if not api_key:
            console.print("[red]Error: MASSIVE_API_KEY environment variable not set[/red]")
            return 1
        portfolio.pricing_manager = MassivePricingDataManager(api_key)
    elif args.databento:
        api_key = os.getenv("DATABENTO_API_KEY")
        if not api_key:
            console.print("[red]Error: DATABENTO_API_KEY environment variable not set[/red]")
            return 1
        portfolio.pricing_manager = DatabentoPricingDataManager(api_key)
    elif args.no_cache:
        portfolio.pricing_manager = YFinancePricingDataManager(force_cache_refresh=True)

    # Generate dashboard HTML
    console.print("Generating dashboard...")
    try:
        html_content = generate_dashboard(
            portfolio=portfolio,
            target_currency=primary_currency,
            annual_risk_free_rate=args.risk_free_rate,
            title=args.title,
        )
    except ValueError as e:
        console.print(f"[red]Error generating dashboard: {e}[/red]")
        return 1

    # Determine output filename
    if args.output:
        output_path = Path(args.output)
    else:
        input_stem = Path(args.filename).stem
        output_path = Path(f"dashboard_{input_stem}.html")

    # Ensure it has .html extension
    if output_path.suffix.lower() != ".html":
        output_path = output_path.with_suffix(".html")

    # Get unique filename if file exists
    original_path = output_path
    output_path = get_unique_filename(output_path)

    if output_path != original_path:
        console.print(f"[yellow]File '{original_path}' exists, saving as '{output_path}'[/yellow]")

    # Write HTML file
    output_path.write_text(html_content, encoding="utf-8")
    console.print(f"[green]Dashboard saved to: {output_path}[/green]")

    return 0
