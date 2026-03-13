"""Version subcommand for the ATHENA CLI."""

from importlib.metadata import version


def register_subcommand(subparsers):
    """Register the version subcommand with the argument parser.

    Args:
        subparsers: The argparse subparsers object to register with.
    """
    parser = subparsers.add_parser(
        "version",
        help="Display ATHENA version information",
        description="Display the current ATHENA version and project links.",
    )
    parser.set_defaults(func=run)


def run(args):
    """Display version information and project URL.

    Args:
        args: Parsed CLI arguments.

    Returns:
        int: Exit code (0 for success).
    """
    try:
        ver = version("athenaos")
    except Exception:
        ver = "unknown"

    print(f" Version: {ver}")
    print(f" Visit us: https://athena-os.ai/")
    return 0
