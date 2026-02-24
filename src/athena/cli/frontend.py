#!/usr/bin/env python3
"""Frontend command for launching the ATHENA local web server."""

import json
import os
from pathlib import Path

from rich.console import Console

# Default port: 4770 — a nod to 447 BC, when construction of
# Athena's Parthenon began on the Acropolis in Athens.
DEFAULT_PORT = 4770


def register_subcommand(subparsers):
    """Register the frontend subcommand.

    Args:
        subparsers: The argparse subparsers action to add the command to.
    """
    parser = subparsers.add_parser(
        "frontend",
        help="Launch the ATHENA local web frontend",
        description="Start a local Flask web server for the ATHENA frontend.",
    )

    parser.add_argument(
        "filename",
        nargs="?",
        default=None,
        help="Path to the Excel portfolio file (or set portfolio_file in config.json)",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port to run the server on (default: {DEFAULT_PORT})",
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

    parser.set_defaults(func=run_frontend)


def _resolve_portfolio_file(cli_filename: str | None) -> str | None:
    """Resolve portfolio file from CLI arg or config.json fallback.

    Args:
        cli_filename: Portfolio file path from CLI argument, or None.

    Returns:
        The resolved portfolio file path, or None if not found.
    """
    if cli_filename:
        return cli_filename

    config_path = Path.cwd() / "config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
            if isinstance(config, dict):
                return config.get("portfolio_file")
        except (json.JSONDecodeError, OSError):
            pass

    return None


def _build_pricing_manager(args, console):
    """Build a PricingDataManager from CLI flags.

    Args:
        args: Parsed argparse namespace with massive, databento,
            and no_cache attributes.
        console: Rich Console instance for status output.

    Returns:
        A PricingDataManager instance, None for default (YFinance),
        or False if a required API key is missing.
    """
    no_cache = args.no_cache

    if args.massive:
        from ..pricingdata import MassivePricingDataManager
        api_key = os.getenv("MASSIVE_API_KEY")
        if not api_key:
            console.print("[red]Error: MASSIVE_API_KEY environment variable not set[/red]")
            return False
        label = "Massive"
        if no_cache:
            label += " (cache bypass)"
        console.print(f"Using [cyan]{label}[/cyan] pricing data manager")
        return MassivePricingDataManager(api_key, force_cache_refresh=no_cache)
    elif args.databento:
        from ..pricingdata import DatabentoPricingDataManager
        api_key = os.getenv("DATABENTO_API_KEY")
        if not api_key:
            console.print("[red]Error: DATABENTO_API_KEY environment variable not set[/red]")
            return False
        label = "Databento"
        if no_cache:
            label += " (cache bypass)"
        console.print(f"Using [cyan]{label}[/cyan] pricing data manager")
        return DatabentoPricingDataManager(api_key, force_cache_refresh=no_cache)
    elif no_cache:
        from ..pricingdata import YFinancePricingDataManager
        console.print("Using YFinance with [cyan]cache bypass[/cyan]")
        return YFinancePricingDataManager(force_cache_refresh=True)
    return None


def run_frontend(args):
    """Launch the Flask development server for the ATHENA frontend.

    Args:
        args: Parsed argparse namespace with filename, port, massive,
            databento, and no_cache attributes.

    Returns:
        int: Exit code (0 for success, 1 for errors).
    """
    from ..frontend import create_app

    console = Console()

    portfolio_file = _resolve_portfolio_file(args.filename)
    if not portfolio_file:
        console.print(
            "[red]Error: No portfolio file specified. "
            "Provide a filename argument or set 'portfolio_file' in config.json.[/red]"
        )
        return 1

    if not os.path.exists(portfolio_file):
        console.print(
            f"[red]Error: Portfolio file '{portfolio_file}' not found.\n"
            "Check the path or update 'portfolio_file' in config.json.[/red]"
        )
        return 1

    pricing_manager = _build_pricing_manager(args, console)
    if pricing_manager is False:
        return 1

    console.print(f"Loading portfolio from [cyan]{portfolio_file}[/cyan]...")
    try:
        app = create_app(
            portfolio_file=portfolio_file,
            pricing_manager=pricing_manager,
            force_cache_refresh=args.no_cache,
        )
    except Exception as e:
        console.print(f"[red]Error: Failed to load portfolio — {e}[/red]")
        return 1

    console.print(
        f"[bold]Starting ATHENA frontend on [cyan]http://127.0.0.1:{args.port}[/cyan][/bold]"
    )
    console.print("[dim]Press Ctrl+C to stop the server.[/dim]\n")

    app.run(host="127.0.0.1", port=args.port)
    return 0
