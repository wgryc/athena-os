#!/usr/bin/env python3
"""Demo command for running agentic trading."""

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from ..agents import run_investing_agent
from ..pricingdata import DatabentoPricingDataManager

# Agent configurations mapping to A, B, C
AGENT_CONFIGS = {
    "meme": {
        "project_code": "reddit_investing_comments",
        "events_section_title": "REDDIT COMMENTS",
        "events_description": (
            "The comments below are a list of comments with their specific unique IDs, "
            "the time stamp when they were posted, and then the comment itself. "
            "Note that comments can sometimes be false, lies, etc."
        ),
        "data_source_description": "trends and memes on Reddit",
        "data_summary_description": "A summary of recent meme stock discussions and notes.",
        "system_prompt_file": "system_prompt.j2",
        "use_commodities": False,
    },
    "us": {
        "project_code": "tracker_usa",
        "events_section_title": "US STOCK EVENTS AND NEWS",
        "events_description": (
            "The news below shows what's been crawled or found recently. "
            "We have a unique event ID for the news, the time stamp when they were posted, "
            "and then the comment itself. Note that news can also be false or incorrect; "
            "the more often that events show up in the list, the more likely they are correct."
        ),
        "data_source_description": "major investments in the US stock market",
        "data_summary_description": (
            "A summary of recent events and news articles outlining things taking place "
            "with specific companies or economic situations in the US."
        ),
        "system_prompt_file": "system_prompt.j2",
        "use_commodities": False,
    },
    "commodities": {
        "project_code": "commodities_tracker",
        "events_section_title": "COMMODITIES EVENTS AND NEWS",
        "events_description": (
            "The news below shows what's been crawled or found recently. "
            "We have a unique event ID for the news, the time stamp when they were posted, "
            "and then the comment itself. Note that news can also be false or incorrect; "
            "the more often that events show up in the list, the more likely they are correct."
        ),
        "data_source_description": "major investments in the commodities market",
        "data_summary_description": (
            "A summary of recent events and news articles outlining things taking place "
            "with specific commodities and related economic situations around the globe."
        ),
        "system_prompt_file": "system_prompt_commodities.j2",
        "use_commodities": True,
    },
}


def register_subcommand(subparsers):
    """Register the demo subcommand.

    Args:
        subparsers: The argparse subparsers action to add the command to.
    """
    parser = subparsers.add_parser(
        "demo",
        help="Run agentic trading demo",
        description="Run an AI-powered trading agent with different market focus areas.",
    )

    # Create mutually exclusive group for the agent type
    agent_group = parser.add_mutually_exclusive_group(required=True)
    agent_group.add_argument(
        "--meme-stocks",
        metavar="FILE",
        dest="meme_stocks_file",
        help="Run meme stocks agent tracking Reddit trends (output to FILE)",
    )
    agent_group.add_argument(
        "--us-stocks",
        metavar="FILE",
        dest="us_stocks_file",
        help="Run US stocks agent tracking market news (output to FILE)",
    )
    agent_group.add_argument(
        "--commodities",
        metavar="FILE",
        dest="commodities_file",
        help="Run commodities agent tracking energy futures (output to FILE)",
    )

    parser.set_defaults(func=run_demo)


def run_demo(args):
    """Run an AI-powered trading agent based on the selected market focus.

    Args:
        args: Parsed argparse namespace with meme_stocks_file,
            us_stocks_file, or commodities_file attribute.

    Returns:
        int: Exit code (0 for success, 1 for errors).
    """
    # Determine which agent to run based on provided argument
    if args.meme_stocks_file:
        config_key = "meme"
        portfolio_file = args.meme_stocks_file
    elif args.us_stocks_file:
        config_key = "us"
        portfolio_file = args.us_stocks_file
    elif args.commodities_file:
        config_key = "commodities"
        portfolio_file = args.commodities_file
    else:
        print("Error: No agent type specified.")
        return 1

    config = AGENT_CONFIGS[config_key]

    print(f"Running {config_key} agent...")
    print(f"Portfolio file: {portfolio_file}")
    print(f"Project code: {config['project_code']}")
    print()

    # Set up pricing manager for commodities
    pricing_manager = None
    if config["use_commodities"]:
        pricing_manager = DatabentoPricingDataManager()

    summary = run_investing_agent(
        portfolio_file_name=portfolio_file,
        project_code=config["project_code"],
        events_section_title=config["events_section_title"],
        events_description=config["events_description"],
        data_source_description=config["data_source_description"],
        data_summary_description=config["data_summary_description"],
        system_prompt_file=config["system_prompt_file"],
        pricing_manager=pricing_manager,
        use_commodities=config["use_commodities"],
        generate_summary=True,
    )

    console = Console()
    console.print(f"\n[green]Portfolio saved to: {portfolio_file}[/green]\n")

    if summary:
        console.print(Panel(Markdown(summary), title="Trading Summary", border_style="cyan"))

    return 0
