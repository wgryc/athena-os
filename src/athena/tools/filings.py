"""SEC EDGAR filing tools powered by edgartools."""

from __future__ import annotations

import os
from typing import Any

from edgar import Company, set_identity

from athena.tools import Tool, VisualTool

# SEC requires a user-agent identity for EDGAR access.
_EDGAR_IDENTITY = os.environ.get("EDGAR_IDENTITY", "ATHENA app@athena.local")
set_identity(_EDGAR_IDENTITY)

_DEFAULT_FORMS = ["10-K", "10-Q", "8-K"]

_TRANSACTION_LABELS = {
    "P": "Buy",
    "S": "Sale",
    "M": "Exercise",
    "A": "Grant",
    "G": "Gift",
    "F": "Tax",
    "C": "Conversion",
    "D": "Dispose",
}


class SECFilings(Tool):
    """Fetch recent SEC filings for a company from EDGAR."""

    @property
    def name(self) -> str:
        return "sec_filings"

    @property
    def label(self) -> str:
        return "SEC Filings Lookup"

    @property
    def description(self) -> str:
        return (
            "Look up recent SEC filings (10-K, 10-Q, 8-K, etc.) for a company. "
            "Returns filing form type, date, description, and a link to the filing."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol (e.g. 'AAPL', 'MSFT').",
                },
                "form_type": {
                    "type": "string",
                    "description": (
                        "Optional SEC form type to filter by (e.g. '10-K', '10-Q', '8-K'). "
                        "Omit to return all major filing types."
                    ),
                },
                "count": {
                    "type": "integer",
                    "description": "Number of filings to return (default 10, max 25).",
                },
            },
            "required": ["ticker"],
        }

    def execute(self, **kwargs: Any) -> str:
        ticker: str = kwargs["ticker"].upper().strip()
        form_type: str | None = kwargs.get("form_type")
        count: int = min(int(kwargs.get("count", 10)), 25)

        try:
            company = Company(ticker)
        except Exception as e:
            return f"Could not find company for ticker '{ticker}': {e}"

        forms = [form_type] if form_type else _DEFAULT_FORMS
        try:
            filings = company.get_filings(form=forms).head(count)
        except Exception as e:
            return f"Error fetching filings for {ticker}: {e}"

        results = list(filings)
        if not results:
            form_label = form_type or "10-K/10-Q/8-K"
            return f"No {form_label} filings found for {ticker}."

        parts = [f"SEC Filings for {company.name} ({ticker}) — {len(results)} results:\n"]
        for i, f in enumerate(results, 1):
            parts.append(
                f"  {i}. [{f.form}] {f.filing_date} — {f.primary_doc_description}\n"
                f"     {f.homepage_url}"
            )

        return "\n".join(parts)


_MAX_CONTENT_LENGTH = 20_000


class SECFilingContent(Tool):
    """Retrieve the text content of a specific SEC filing from EDGAR."""

    @property
    def name(self) -> str:
        return "sec_filing_content"

    @property
    def label(self) -> str:
        return "SEC Filing Content"

    @property
    def description(self) -> str:
        return (
            "Retrieve the text content of a specific SEC filing. Use this after "
            "sec_filings to read the actual content of a filing. Returns the full "
            "filing as markdown, or a specific section for 10-K/10-Q filings."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol (e.g. 'AAPL', 'MSFT').",
                },
                "form_type": {
                    "type": "string",
                    "description": (
                        "SEC form type (e.g. '10-K', '10-Q', '8-K'). Default '10-K'."
                    ),
                },
                "filing_date": {
                    "type": "string",
                    "description": (
                        "Filing date in YYYY-MM-DD format to select a specific filing. "
                        "If omitted, returns the most recent filing of the given form type."
                    ),
                },
                "section": {
                    "type": "string",
                    "description": (
                        "Optional section to extract from 10-K or 10-Q filings. "
                        "For 10-K: 'Item 1' (Business), 'Item 1A' (Risk Factors), "
                        "'Item 7' (MD&A), 'Item 8' (Financial Statements), etc. "
                        "Also accepts aliases like 'business', 'risk_factors', 'mda'. "
                        "For 10-Q: use 'Part I, Item 2' format. "
                        "If omitted, returns the full filing."
                    ),
                },
            },
            "required": ["ticker"],
        }

    def execute(self, **kwargs: Any) -> str:
        ticker: str = kwargs["ticker"].upper().strip()
        form_type: str = kwargs.get("form_type", "10-K").upper().strip()
        filing_date: str | None = kwargs.get("filing_date")
        section: str | None = kwargs.get("section")

        try:
            company = Company(ticker)
        except Exception as e:
            return f"Could not find company for ticker '{ticker}': {e}"

        try:
            filings = company.get_filings(form=form_type)
        except Exception as e:
            return f"Error fetching filings for {ticker}: {e}"

        # Select the target filing.
        filing = None
        if filing_date:
            for f in filings.head(50):
                if str(f.filing_date) == filing_date:
                    filing = f
                    break
            if filing is None:
                return (
                    f"No {form_type} filing found for {ticker} on {filing_date}. "
                    f"Use sec_filings to list available filings."
                )
        else:
            results = list(filings.head(1))
            if not results:
                return f"No {form_type} filings found for {ticker}."
            filing = results[0]

        header = (
            f"SEC {filing.form} filing for {company.name} ({ticker})\n"
            f"Filed: {filing.filing_date}\n"
            f"URL: {filing.homepage_url}\n"
        )

        # Extract a specific section if requested (10-K / 10-Q only).
        if section:
            try:
                obj = filing.obj()
            except Exception:
                return header + f"\nSection extraction not supported for {filing.form} filings."

            try:
                content = str(obj[section])
            except (KeyError, IndexError, TypeError):
                return header + f"\nSection '{section}' not found in this {filing.form} filing."

            return header + f"\n--- {section} ---\n\n{content[:_MAX_CONTENT_LENGTH]}"

        # Full filing as markdown.
        try:
            content = filing.markdown()
        except Exception:
            try:
                content = filing.text()
            except Exception as e:
                return header + f"\nCould not retrieve filing content: {e}"

        if len(content) > _MAX_CONTENT_LENGTH:
            content = (
                content[:_MAX_CONTENT_LENGTH]
                + f"\n\n... [truncated — full filing is {len(content):,} characters. "
                f"Use the 'section' parameter to read specific sections.]"
            )

        return header + "\n" + content


class SECFilingsWidget(VisualTool):
    """Dashboard widget displaying recent SEC filings for a company."""

    def __init__(self):
        self._filings: list[dict] = []
        self._company_name: str = ""
        self._ticker: str = ""

    @property
    def name(self) -> str:
        return "sec_filings_widget"

    @property
    def description(self) -> str:
        return "Display a table of recent SEC filings for a company."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol (e.g. 'AAPL', 'MSFT').",
                },
                "count": {
                    "type": "integer",
                    "description": "Number of filings to show (default 10).",
                },
            },
            "required": ["ticker"],
        }

    def execute(self, **kwargs: Any) -> str:
        self._ticker = kwargs["ticker"].upper().strip()
        count: int = min(int(kwargs.get("count", 10)), 25)

        company = Company(self._ticker)
        self._company_name = company.name

        filings = company.get_filings(form=_DEFAULT_FORMS).head(count)
        self._filings = []
        for f in filings:
            self._filings.append({
                "form": f.form,
                "date": str(f.filing_date),
                "description": f.primary_doc_description,
                "url": f.homepage_url,
            })

        return self.to_context()

    def to_context(self) -> str:
        if not self._filings:
            return f"(no SEC filings for {self._ticker})"
        lines = [f"Recent SEC filings for {self._company_name} ({self._ticker}):"]
        for f in self._filings:
            lines.append(f"  [{f['form']}] {f['date']} — {f['description']}")
        return " ".join(lines)

    def to_html(self) -> str:
        if not self._filings:
            return '<div class="widget-card widget-error">No SEC filings found</div>'

        rows = ""
        for f in self._filings:
            form_class = f["form"].lower().replace("-", "")
            rows += (
                "<tr>"
                f'<td><span class="filing-badge filing-{form_class}">{f["form"]}</span></td>'
                f'<td>{f["date"]}</td>'
                f'<td><a href="{f["url"]}" target="_blank" rel="noopener">{f["description"]}</a></td>'
                "</tr>"
            )

        return (
            '<div class="widget-card">'
            f'<h3 class="widget-title">SEC Filings — {self._company_name}</h3>'
            '<table class="filing-table">'
            "<thead><tr><th>Form</th><th>Date</th><th>Description</th></tr></thead>"
            f"<tbody>{rows}</tbody>"
            "</table>"
            "</div>"
        )


def _parse_form4_trades(filing) -> list[dict]:
    """Extract transactions from a Form 4 filing object.

    Checks both non-derivative (common stock) and derivative (RSUs, options)
    tables.  The edgartools ``NonDerivativeTransactions`` iterator has a bug
    where ``len()`` returns 0 but iteration yields infinite ``None`` values,
    so we guard with a ``len()`` check before iterating.

    Args:
        filing: An edgartools ``Filing`` object for a Form 4 filing.

    Returns:
        A list of dicts, each containing keys: ``insider``, ``position``,
        ``date``, ``security``, ``code``, ``type``, ``shares``, ``price``,
        ``acquired``, ``remaining``, and ``url``. Returns a single
        placeholder entry when no transactions can be extracted.
    """
    trades: list[dict] = []
    try:
        obj = filing.obj()
    except Exception:
        return trades

    insider = getattr(obj, "insider_name", None) or "Unknown"
    position = getattr(obj, "position", None) or ""
    report_date = getattr(obj, "reporting_period", None) or str(filing.filing_date)
    url = getattr(filing, "homepage_url", "")

    # Non-derivative transactions (common stock buys/sells/exercises).
    nd_table = getattr(obj, "non_derivative_table", None)
    if nd_table and hasattr(nd_table, "transactions") and len(nd_table.transactions):
        for txn in nd_table.transactions:
            if txn is None:
                continue
            code = getattr(txn, "transaction_code", "") or ""
            trades.append({
                "insider": insider,
                "position": position,
                "date": str(getattr(txn, "date", report_date) or report_date),
                "security": getattr(txn, "security", "Common Stock") or "Common Stock",
                "code": code,
                "type": _TRANSACTION_LABELS.get(code, code),
                "shares": int(getattr(txn, "shares", 0) or 0),
                "price": float(getattr(txn, "price", 0) or 0),
                "acquired": getattr(txn, "acquired_disposed", "") == "A",
                "remaining": int(getattr(txn, "remaining", 0) or 0),
                "url": url,
            })

    # Derivative transactions (RSUs, options, etc.).
    d_table = getattr(obj, "derivative_table", None)
    if not trades and d_table and hasattr(d_table, "transactions") and len(d_table.transactions):
        for txn in d_table.transactions:
            if txn is None:
                continue
            code = getattr(txn, "transaction_code", "") or ""
            security = getattr(txn, "security", "") or ""
            shares_raw = getattr(txn, "shares", 0) or getattr(txn, "underlying_shares", 0) or 0
            trades.append({
                "insider": insider,
                "position": position,
                "date": str(getattr(txn, "date", report_date) or report_date),
                "security": security,
                "code": code,
                "type": _TRANSACTION_LABELS.get(code, code),
                "shares": int(shares_raw),
                "price": float(getattr(txn, "price", 0) or 0),
                "acquired": getattr(txn, "acquired_disposed", "") == "A",
                "remaining": int(getattr(txn, "remaining", 0) or 0),
                "url": url,
            })

    if not trades:
        trades.append({
            "insider": insider,
            "position": position,
            "date": str(report_date),
            "security": "",
            "code": "",
            "type": "Filing",
            "shares": 0,
            "price": 0,
            "acquired": False,
            "remaining": 0,
            "url": url,
        })

    return trades


class InsiderTrades(Tool):
    """Fetch recent Form 4 insider trades for a company from EDGAR."""

    @property
    def name(self) -> str:
        return "insider_trades"

    @property
    def label(self) -> str:
        return "Insider Trades"

    @property
    def description(self) -> str:
        return (
            "Look up recent insider trades (Form 4 filings) for a company. "
            "Returns insider name, position, transaction type, shares, and price."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol (e.g. 'AAPL', 'MSFT').",
                },
                "count": {
                    "type": "integer",
                    "description": "Number of Form 4 filings to fetch (default 10, max 25).",
                },
            },
            "required": ["ticker"],
        }

    def execute(self, **kwargs: Any) -> str:
        ticker: str = kwargs["ticker"].upper().strip()
        count: int = min(int(kwargs.get("count", 10)), 25)

        try:
            company = Company(ticker)
        except Exception as e:
            return f"Could not find company for ticker '{ticker}': {e}"

        try:
            filings = company.get_filings(form="4").head(count)
        except Exception as e:
            return f"Error fetching Form 4 filings for {ticker}: {e}"

        all_trades: list[dict] = []
        for f in filings:
            all_trades.extend(_parse_form4_trades(f))

        if not all_trades:
            return f"No insider trades found for {ticker}."

        parts = [f"Insider Trades for {company.name} ({ticker}) — {len(all_trades)} transactions:\n"]
        for i, t in enumerate(all_trades, 1):
            direction = "+" if t["acquired"] else "-"
            price_str = f"${t['price']:.2f}" if t["price"] else "N/A"
            parts.append(
                f"  {i}. {t['date']} | {t['insider']} ({t['position']}) | "
                f"{t['type']} {direction}{t['shares']:,} shares @ {price_str}"
            )

        return "\n".join(parts)


class InsiderTradesWidget(VisualTool):
    """Dashboard widget displaying recent Form 4 insider trades."""

    def __init__(self):
        self._trades: list[dict] = []
        self._company_name: str = ""
        self._ticker: str = ""

    @property
    def name(self) -> str:
        return "insider_trades_widget"

    @property
    def description(self) -> str:
        return "Display a table of recent insider trades (Form 4) for a company."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol (e.g. 'AAPL', 'MSFT').",
                },
                "count": {
                    "type": "integer",
                    "description": "Number of Form 4 filings to fetch (default 10).",
                },
            },
            "required": ["ticker"],
        }

    def execute(self, **kwargs: Any) -> str:
        self._ticker = kwargs["ticker"].upper().strip()
        count: int = min(int(kwargs.get("count", 10)), 25)

        company = Company(self._ticker)
        self._company_name = company.name

        filings = company.get_filings(form="4").head(count)
        self._trades = []
        for f in filings:
            self._trades.extend(_parse_form4_trades(f))

        return self.to_context()

    def to_context(self) -> str:
        if not self._trades:
            return f"(no insider trades for {self._ticker})"
        lines = [f"Recent insider trades for {self._company_name} ({self._ticker}):"]
        for t in self._trades:
            direction = "Buy" if t["acquired"] else "Sell"
            price_str = f"${t['price']:.2f}" if t["price"] else "N/A"
            lines.append(
                f"  {t['date']} {t['insider']} ({t['position']}) "
                f"{direction} {t['shares']:,} @ {price_str}"
            )
        return " ".join(lines)

    def to_html(self) -> str:
        if not self._trades:
            return '<div class="widget-card widget-error">No insider trades found</div>'

        rows = ""
        for t in self._trades:
            if t["acquired"]:
                direction = "buy"
                badge = '<span class="trade-badge trade-buy">BUY</span>'
            else:
                direction = "sell"
                badge = '<span class="trade-badge trade-sell">SELL</span>'
            price_str = f"${t['price']:,.2f}" if t["price"] else "—"
            shares_str = f"{t['shares']:,}" if t["shares"] else "—"
            rows += (
                "<tr>"
                f'<td>{t["date"]}</td>'
                f'<td><a href="{t["url"]}" target="_blank" rel="noopener">{t["insider"]}</a></td>'
                f'<td>{t["position"]}</td>'
                f"<td>{badge}</td>"
                f'<td class="num">{shares_str}</td>'
                f'<td class="num">{price_str}</td>'
                "</tr>"
            )

        return (
            '<div class="widget-card">'
            f'<h3 class="widget-title">Insider Trades — {self._company_name}</h3>'
            '<table class="filing-table">'
            "<thead><tr>"
            "<th>Date</th><th>Insider</th><th>Position</th>"
            "<th>Type</th><th>Shares</th><th>Price</th>"
            "</tr></thead>"
            f"<tbody>{rows}</tbody>"
            "</table>"
            "</div>"
        )
