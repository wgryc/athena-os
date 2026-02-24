"""Web-based tools: news search and page crawling."""

from __future__ import annotations

import os
import concurrent.futures
from typing import Any

import trafilatura
from scrapingbee import ScrapingBeeClient
from serpapi import GoogleSearch

from athena.tools import Tool


def _scrape_page(url: str, api_key: str) -> dict[str, str | None]:
    """Fetch a page via ScrapingBee and extract text with trafilatura.

    Args:
        url: The URL to scrape.
        api_key: ScrapingBee API key.

    Returns:
        A dict with keys ``url``, ``title``, ``description``, and ``content``.
        Values are ``None`` when extraction fails.
    """
    try:
        client = ScrapingBeeClient(api_key=api_key)
        response = client.get(url, params={"render_js": False})
        if not response.ok:
            return {"url": url, "title": None, "description": None, "content": None}
        html = response.text
    except Exception:
        return {"url": url, "title": None, "description": None, "content": None}

    metadata = trafilatura.extract_metadata(html)
    content = trafilatura.extract(html)

    return {
        "url": url,
        "title": metadata.title if metadata else None,
        "description": metadata.description if metadata else None,
        "content": content,
    }


class NewsSearch(Tool):
    """Search Google News and Bing News for a keyword, then scrape article content."""

    def __init__(
        self,
        serpapi_key: str | None = None,
        scrapingbee_key: str | None = None,
    ):
        """Initialize the news search tool.

        Args:
            serpapi_key: SerpAPI key. Falls back to the ``SERPAPI_KEY``
                environment variable when not provided.
            scrapingbee_key: ScrapingBee key. Falls back to the
                ``SCRAPING_BEE_API_KEY`` environment variable when not provided.
        """
        self._serpapi_key = serpapi_key or os.environ.get("SERPAPI_KEY", "")
        self._scrapingbee_key = scrapingbee_key or os.environ.get("SCRAPING_BEE_API_KEY", "")

    @property
    def name(self) -> str:
        return "news_search"

    @property
    def label(self) -> str:
        return "News Search"

    @property
    def description(self) -> str:
        return (
            "Search for recent English-language news articles about a keyword. "
            "Returns up to 10 results each from Google News and Bing News, "
            "including the URL, page title, meta description, and full article text. "
            "Optionally filter to articles from the past day or past week."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "The search keyword or phrase for the news query.",
                },
                "recency": {
                    "type": "string",
                    "enum": ["day", "week"],
                    "description": (
                        "Optional time filter. 'day' limits results to the past 24 hours, "
                        "'week' limits to the past 7 days. Omit for no time restriction."
                    ),
                },
            },
            "required": ["keyword"],
        }

    def _search_google_news(self, keyword: str, recency: str | None = None) -> list[dict[str, str]]:
        """Query Google News via SerpAPI. Returns up to 10 results."""
        q = keyword
        if recency == "day":
            q += " when:1d"
        elif recency == "week":
            q += " when:7d"
        params = {
            "engine": "google_news",
            "q": q,
            "gl": "us",
            "hl": "en",
            "api_key": self._serpapi_key,
        }
        search = GoogleSearch(params)
        data = search.get_dict()
        results = []
        for item in data.get("news_results", [])[:10]:
            link = item.get("link", "")
            if link:
                results.append({
                    "source": "google_news",
                    "title": item.get("title", ""),
                    "url": link,
                    "snippet": item.get("snippet", ""),
                })
        return results

    _BING_RECENCY = {"day": "interval=7", "week": "interval=8"}

    def _search_bing_news(self, keyword: str, recency: str | None = None) -> list[dict[str, str]]:
        """Query Bing News via SerpAPI. Returns up to 10 results."""
        params: dict[str, str] = {
            "engine": "bing_news",
            "q": keyword,
            "cc": "us",
            "api_key": self._serpapi_key,
        }
        if recency and recency in self._BING_RECENCY:
            params["qft"] = self._BING_RECENCY[recency]
        search = GoogleSearch(params)
        data = search.get_dict()
        results = []
        for item in data.get("organic_results", [])[:10]:
            link = item.get("link", "")
            if link:
                results.append({
                    "source": "bing_news",
                    "title": item.get("title", ""),
                    "url": link,
                    "snippet": item.get("snippet", ""),
                })
        return results

    def execute(self, **kwargs: Any) -> str:
        keyword: str = kwargs["keyword"]
        recency: str | None = kwargs.get("recency")

        # Fetch search results from both engines in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            google_future = pool.submit(self._search_google_news, keyword, recency)
            bing_future = pool.submit(self._search_bing_news, keyword, recency)
            google_results = google_future.result()
            bing_results = bing_future.result()

        all_results = google_results + bing_results

        if not all_results:
            return f"No news results found for '{keyword}'."

        # Deduplicate by URL
        seen_urls: set[str] = set()
        unique_results: list[dict[str, str]] = []
        for r in all_results:
            if r["url"] not in seen_urls:
                seen_urls.add(r["url"])
                unique_results.append(r)

        # Scrape all article pages in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
            futures = {
                pool.submit(_scrape_page, r["url"], self._scrapingbee_key): r
                for r in unique_results
            }
            scraped: dict[str, dict[str, str | None]] = {}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                scraped[result["url"]] = result

        # Build output
        parts: list[str] = []
        parts.append(f"News results for: {keyword}\n")

        for i, r in enumerate(unique_results, 1):
            page = scraped.get(r["url"], {})
            title = page.get("title") or r.get("title", "N/A")
            description = page.get("description") or r.get("snippet", "N/A")
            content = page.get("content") or "(could not extract content)"

            parts.append(
                f"--- Article {i} [{r['source']}] ---\n"
                f"URL: {r['url']}\n"
                f"Title: {title}\n"
                f"Description: {description}\n"
                f"Content:\n{content}\n"
            )

        return "\n".join(parts)


class CrawlPage(Tool):
    """Crawl a single web page and extract its main text content."""

    def __init__(self, scrapingbee_key: str | None = None):
        """Initialize the web page crawler tool.

        Args:
            scrapingbee_key: ScrapingBee key. Falls back to the
                ``SCRAPING_BEE_API_KEY`` environment variable when not provided.
        """
        self._scrapingbee_key = scrapingbee_key or os.environ.get("SCRAPING_BEE_API_KEY", "")

    @property
    def name(self) -> str:
        return "crawl_page"

    @property
    def label(self) -> str:
        return "Web Page Crawler"

    @property
    def description(self) -> str:
        return (
            "Fetch a web page and extract its main text content. "
            "Returns the page title, meta description, and the main body text."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full URL of the web page to crawl.",
                },
            },
            "required": ["url"],
        }

    def execute(self, **kwargs: Any) -> str:
        url: str = kwargs["url"]

        client = ScrapingBeeClient(api_key=self._scrapingbee_key)
        response = client.get(url, params={"render_js": False})

        if not response.ok:
            return f"Failed to fetch {url} (HTTP {response.status_code})."

        html = response.text
        metadata = trafilatura.extract_metadata(html)
        content = trafilatura.extract(html)

        title = metadata.title if metadata else "N/A"
        description = metadata.description if metadata else "N/A"
        content = content or "(could not extract content)"

        return (
            f"URL: {url}\n"
            f"Title: {title}\n"
            f"Description: {description}\n"
            f"Content:\n{content}"
        )
