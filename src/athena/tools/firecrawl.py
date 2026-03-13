"""Firecrawl-powered tools: Google search and page scraping with JS rendering."""

from __future__ import annotations

import os
from typing import Any

from firecrawl import FirecrawlApp

from athena.tools import Tool


class FirecrawlSearch(Tool):
    """Search Google via Firecrawl and return results with page content."""

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.environ.get("FIRECRAWL_API_KEY", "")

    @property
    def name(self) -> str:
        return "firecrawl_search"

    @property
    def label(self) -> str:
        return "Web Search (Firecrawl)"

    @property
    def description(self) -> str:
        return (
            "Search Google and return results with their full page content "
            "in LLM-friendly markdown. Each result includes the URL, title, "
            "and scraped markdown content. Use this for general web research."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max number of results to return (default 5, max 10).",
                },
            },
            "required": ["query"],
        }

    def execute(self, **kwargs: Any) -> str:
        query: str = kwargs["query"]
        limit: int = min(kwargs.get("limit", 5), 10)

        app = FirecrawlApp(api_key=self._api_key)

        try:
            results = app.search(
                query,
                limit=limit,
                scrape_options={"formats": ["markdown"]},
            )
        except Exception as e:
            return f"Firecrawl search failed: {e}"

        # Handle both list and dict response formats
        if isinstance(results, dict):
            items = results.get("data", [])
        else:
            items = results

        if not items:
            return f"No results found for '{query}'."

        parts: list[str] = [f"Search results for: {query}\n"]

        for i, item in enumerate(items, 1):
            url = item.get("url", "N/A")
            title = item.get("metadata", {}).get("title", "N/A") if isinstance(item.get("metadata"), dict) else "N/A"
            markdown = item.get("markdown", "(no content)")

            # Truncate very long pages to keep context manageable
            if len(markdown) > 5000:
                markdown = markdown[:5000] + "\n\n... (truncated)"

            parts.append(
                f"--- Result {i} ---\n"
                f"URL: {url}\n"
                f"Title: {title}\n"
                f"Content:\n{markdown}\n"
            )

        return "\n".join(parts)


class FirecrawlScrape(Tool):
    """Scrape a single web page via Firecrawl with full JS rendering."""

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.environ.get("FIRECRAWL_API_KEY", "")

    @property
    def name(self) -> str:
        return "firecrawl_scrape"

    @property
    def label(self) -> str:
        return "Page Scraper (Firecrawl)"

    @property
    def description(self) -> str:
        return (
            "Scrape a web page and return its content as clean, LLM-friendly "
            "markdown. Handles JavaScript-rendered pages. Use this when you "
            "have a specific URL to read."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full URL of the web page to scrape.",
                },
            },
            "required": ["url"],
        }

    def execute(self, **kwargs: Any) -> str:
        url: str = kwargs["url"]

        app = FirecrawlApp(api_key=self._api_key)

        try:
            result = app.scrape_url(url, params={"formats": ["markdown"]})
        except Exception as e:
            return f"Firecrawl scrape failed for {url}: {e}"

        if not result:
            return f"No content returned for {url}."

        metadata = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
        title = metadata.get("title", "N/A")
        description = metadata.get("description", "N/A")
        markdown = result.get("markdown", "(no content)")

        # Truncate very long pages
        if len(markdown) > 15000:
            markdown = markdown[:15000] + "\n\n... (truncated)"

        return (
            f"URL: {url}\n"
            f"Title: {title}\n"
            f"Description: {description}\n"
            f"Content:\n{markdown}"
        )
