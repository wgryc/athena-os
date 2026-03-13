#!/usr/bin/env python3
"""Main entry point for the ATHENA CLI."""

import argparse
import sys

ATHENA_LOGO = """
  █████╗ ████████╗██╗  ██╗███████╗███╗   ██╗ █████╗
 ██╔══██╗╚══██╔══╝██║  ██║██╔════╝████╗  ██║██╔══██╗
 ███████║   ██║   ███████║█████╗  ██╔██╗ ██║███████║
 ██╔══██║   ██║   ██╔══██║██╔══╝  ██║╚██╗██║██╔══██║
 ██║  ██║   ██║   ██║  ██║███████╗██║ ╚████║██║  ██║
 ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═══╝╚═╝  ╚═╝
 ATHENA — Agentic Toolkit for Holistic Economic Narratives and Analysis
"""

INVESTING_WARNING = (
    " \033[33m⚠  This is an experimental investing toolkit. All ideas, research,\n"
    "    and analysis should be verified by a human. Nothing here should be\n"
    "    construed as investment advice.\033[0m"
)


def main():
    """Parse CLI arguments and dispatch to the appropriate subcommand.

    Returns:
        int: Exit code (0 for success, non-zero for errors).
    """
    parser = argparse.ArgumentParser(
        prog="athenaos",
        description="ATHENA - Agentic Toolkit for Holistic Economic Narratives and Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  athenaos report portfolio.xlsx              Display portfolio holdings report
  athenaos metrics portfolio.xlsx             Display portfolio metrics report
  athenaos report portfolio.xlsx -c CAD       Use CAD as primary currency
  athenaos metrics portfolio.xlsx --debug     Show debug info with prices
        """,
    )

    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        help="Available commands",
    )

    # Import subcommand modules and register them
    from .report import register_subcommand as register_report
    from .metrics import register_subcommand as register_metrics
    from .demo import register_subcommand as register_demo
    from .dashboard import register_subcommand as register_dashboard
    from .frontend import register_subcommand as register_frontend
    from .version import register_subcommand as register_version

    register_report(subparsers)
    register_metrics(subparsers)
    register_demo(subparsers)
    register_dashboard(subparsers)
    register_frontend(subparsers)
    register_version(subparsers)

    # Parse arguments
    args = parser.parse_args()

    # If no command specified, show help
    if args.command is None:
        print(ATHENA_LOGO)
        print(INVESTING_WARNING)
        print()
        parser.print_help()
        return 0

    # Print logo before running command
    print(ATHENA_LOGO)
    print(INVESTING_WARNING)
    print()

    # Run the appropriate command
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
