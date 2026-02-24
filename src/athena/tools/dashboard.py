"""Dashboard widgets for the ATHENA frontend.

Each widget wraps a slice of pre-computed dashboard data (portfolio values,
returns, Sharpe ratio, drawdown periods) and exposes it as a VisualTool with
HTML rendering, Chart.js config, and plain-text context for the LLM.
"""

from typing import Any, Callable

from . import VisualTool


class DashboardWidget(VisualTool):
    """Base for widgets that depend on pre-computed dashboard data."""

    def __init__(self, get_dashboard_data: Callable[[], dict | None]):
        """Initialize the dashboard widget.

        Args:
            get_dashboard_data: Callable that returns a dict of pre-computed
                portfolio data (values, returns, Sharpe, drawdowns) or ``None``
                when data is unavailable.
        """
        self._get_dashboard_data = get_dashboard_data
        self._data: dict | None = None
        self.chart_config: dict | None = None

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}


# ---------------------------------------------------------------------------
# Widget 1: Portfolio Summary (metrics row)
# ---------------------------------------------------------------------------


class PortfolioSummaryWidget(DashboardWidget):
    """Four key portfolio metrics as a single full-width widget."""

    @property
    def name(self) -> str:
        return "portfolio_summary_widget"

    @property
    def description(self) -> str:
        return "Display portfolio summary metrics: start/end value, total return, and annual Sharpe ratio."

    def execute(self, **kwargs: Any) -> str:
        self._data = self._get_dashboard_data()
        self.chart_config = None
        return self.to_context()

    def to_context(self) -> str:
        if not self._data or "error" in self._data:
            return "(no portfolio data)"
        d = self._data
        ret_sign = "+" if d["total_return"] >= 0 else ""
        sharpe = f'{d["annual_sharpe"]:.2f}' if d["annual_sharpe"] is not None else "N/A"
        return (
            f'Portfolio Summary ({d["currency"]}): '
            f'Starting Value ${d["start_value"]:,.2f} ({d["start_date"]}), '
            f'Ending Value ${d["end_value"]:,.2f} ({d["end_date"]}), '
            f'Total Return {ret_sign}{d["total_return"]:.2f}%, '
            f'Annual Sharpe Ratio {sharpe}'
        )

    def to_html(self) -> str:
        if not self._data or "error" in self._data:
            return '<div class="widget-card widget-error">No portfolio data</div>'
        d = self._data
        fmt = f'{d["start_value"]:,.2f}'
        ret_class = "positive" if d["total_return"] >= 0 else "negative"
        ret_sign = "+" if d["total_return"] >= 0 else ""
        sharpe_text = f'{d["annual_sharpe"]:.2f}' if d["annual_sharpe"] is not None else "N/A"
        return (
            '<div class="metrics-grid">'
            '<div class="metric-card">'
            '<div class="metric-label">Starting Value</div>'
            f'<div class="metric-value">{d["start_value"]:,.2f} {d["currency"]}</div>'
            '</div>'
            '<div class="metric-card">'
            '<div class="metric-label">Ending Value</div>'
            f'<div class="metric-value">{d["end_value"]:,.2f} {d["currency"]}</div>'
            '</div>'
            '<div class="metric-card">'
            '<div class="metric-label">Total Return</div>'
            f'<div class="metric-value {ret_class}">{ret_sign}{d["total_return"]:.2f}%</div>'
            '</div>'
            '<div class="metric-card">'
            '<div class="metric-label">Annual Sharpe Ratio</div>'
            f'<div class="metric-value">{sharpe_text}</div>'
            '</div>'
            '</div>'
        )


# ---------------------------------------------------------------------------
# Widget 2: Portfolio Value chart (with drawdown annotations)
# ---------------------------------------------------------------------------


class PortfolioValueChartWidget(DashboardWidget):
    """Portfolio value line chart with drawdown period annotations."""

    @property
    def name(self) -> str:
        return "portfolio_value_chart_widget"

    @property
    def description(self) -> str:
        return "Display portfolio value over time with drawdown period annotations."

    def execute(self, **kwargs: Any) -> str:
        self._data = self._get_dashboard_data()
        if not self._data or "error" in self._data:
            self.chart_config = None
            return self.to_context()

        d = self._data
        annotations = {}
        for i, p in enumerate(d.get("drawdown_periods", [])):
            annotations[f"dd{i}"] = {
                "type": "box",
                "xMin": p["start"],
                "xMax": p["end"],
                "backgroundColor": "rgba(209,77,65,0.15)",
                "borderWidth": 0,
            }

        self.chart_config = {
            "type": "line",
            "data": {
                "labels": d["value_chart"]["labels"],
                "datasets": [
                    {
                        "label": "Portfolio Value",
                        "data": d["value_chart"]["values"],
                        "borderColor": "#4385BE",
                        "backgroundColor": "rgba(67,133,190,0.1)",
                        "fill": True,
                        "tension": 0.1,
                        "pointRadius": 0,
                        "pointHitRadius": 10,
                        "pointStyle": "line",
                    },
                    {
                        "label": "Drawdown Period",
                        "data": [],
                        "borderColor": "rgba(209,77,65,0.5)",
                        "backgroundColor": "rgba(209,77,65,0.15)",
                        "pointRadius": 0,
                        "pointStyle": "line",
                        "hidden": len(d.get("drawdown_periods", [])) == 0,
                    },
                ],
            },
            "options": {
                "responsive": True,
                "maintainAspectRatio": False,
                "plugins": {
                    "legend": {
                        "display": True,
                        "position": "top",
                        "labels": {"usePointStyle": True, "boxWidth": 12},
                    },
                    "annotation": {"annotations": annotations},
                },
                "scales": {
                    "x": {
                        "grid": {"display": False},
                        "ticks": {"maxTicksLimit": 10},
                    },
                    "y": {
                        "ticks": {"_currencyPrefix": d["currency"]},
                    },
                },
            },
        }
        return self.to_context()

    def to_context(self) -> str:
        if not self._data or "error" in self._data:
            return "(no portfolio value data)"
        d = self._data
        values = d["value_chart"]["values"]
        labels = d["value_chart"]["labels"]
        if not values:
            return "(no portfolio value data)"
        peak_val = max(values)
        peak_idx = values.index(peak_val)
        current_val = values[-1]
        dd_periods = d.get("drawdown_periods", [])
        dd_count = len(dd_periods)
        lines = [
            f"Portfolio Value Chart ({d['currency']}): "
            f"${values[0]:,.0f} ({labels[0]}) to ${current_val:,.0f} ({labels[-1]}).",
            f"Peak: ${peak_val:,.0f} on {labels[peak_idx]}.",
            f"{dd_count} drawdown period(s) detected.",
        ]
        if current_val < peak_val:
            pct = (current_val - peak_val) / peak_val * 100
            lines.append(f"Currently in drawdown: {pct:.2f}% from peak.")
        for i, p in enumerate(dd_periods, 1):
            ongoing = " (ongoing)" if p["end"] == labels[-1] else ""
            lines.append(f"  Drawdown {i}: {p['start']} to {p['end']}{ongoing}.")
        return " ".join(lines)

    def to_html(self) -> str:
        if not self._data or "error" in self._data:
            return '<div class="widget-card widget-error">No portfolio value data</div>'
        return (
            '<div class="chart-card">'
            '<h3 class="chart-title">Portfolio Value Over Time</h3>'
            '<div class="chart-container"><canvas class="widget-chart"></canvas></div>'
            '</div>'
        )


# ---------------------------------------------------------------------------
# Widget 3: Daily Returns bar chart
# ---------------------------------------------------------------------------


class DailyReturnsChartWidget(DashboardWidget):
    """Bar chart of daily portfolio returns."""

    @property
    def name(self) -> str:
        return "daily_returns_chart_widget"

    @property
    def description(self) -> str:
        return "Display a bar chart of daily portfolio returns."

    def execute(self, **kwargs: Any) -> str:
        self._data = self._get_dashboard_data()
        if not self._data or "error" in self._data:
            self.chart_config = None
            return self.to_context()

        d = self._data
        values = d["returns_chart"]["values"]
        self.chart_config = {
            "type": "bar",
            "data": {
                "labels": d["returns_chart"]["labels"],
                "datasets": [
                    {
                        "label": "Daily Return",
                        "data": values,
                        "backgroundColor": [
                            "rgba(92,185,122,0.7)" if v >= 0 else "rgba(209,77,65,0.7)"
                            for v in values
                        ],
                        "borderRadius": 2,
                    }
                ],
            },
            "options": {
                "responsive": True,
                "maintainAspectRatio": False,
                "plugins": {"legend": {"display": False}},
                "scales": {
                    "x": {
                        "grid": {"display": False},
                        "ticks": {"maxTicksLimit": 10},
                    },
                    "y": {
                        "ticks": {"_percentSuffix": True},
                    },
                },
            },
        }
        return self.to_context()

    def to_context(self) -> str:
        if not self._data or "error" in self._data:
            return "(no daily returns data)"
        values = self._data["returns_chart"]["values"]
        if not values:
            return "(no daily returns data)"
        pos = sum(1 for v in values if v >= 0)
        neg = len(values) - pos
        avg = sum(values) / len(values)
        best = max(values)
        worst = min(values)
        labels = self._data["returns_chart"]["labels"]
        best_date = labels[values.index(best)]
        worst_date = labels[values.index(worst)]
        return (
            f"Daily Returns: {len(values)} trading days. "
            f"{pos} positive ({pos / len(values) * 100:.0f}%), "
            f"{neg} negative ({neg / len(values) * 100:.0f}%). "
            f"Average: {avg:.2f}%. "
            f"Best: +{best:.2f}% ({best_date}). "
            f"Worst: {worst:.2f}% ({worst_date})."
        )

    def to_html(self) -> str:
        if not self._data or "error" in self._data:
            return '<div class="widget-card widget-error">No daily returns data</div>'
        return (
            '<div class="chart-card">'
            '<h3 class="chart-title">Daily Returns (%)</h3>'
            '<div class="chart-container"><canvas class="widget-chart"></canvas></div>'
            '</div>'
        )


# ---------------------------------------------------------------------------
# Widget 4: Cumulative Sharpe Ratio chart
# ---------------------------------------------------------------------------


class SharpeRatioChartWidget(DashboardWidget):
    """Line chart of cumulative annual Sharpe ratio over time."""

    @property
    def name(self) -> str:
        return "sharpe_ratio_chart_widget"

    @property
    def description(self) -> str:
        return "Display cumulative annual Sharpe ratio over time."

    def execute(self, **kwargs: Any) -> str:
        self._data = self._get_dashboard_data()
        if not self._data or "error" in self._data:
            self.chart_config = None
            return self.to_context()

        d = self._data
        self.chart_config = {
            "type": "line",
            "data": {
                "labels": d["sharpe_chart"]["labels"],
                "datasets": [
                    {
                        "label": "Annual Sharpe",
                        "data": d["sharpe_chart"]["values"],
                        "borderColor": "#8B7EC8",
                        "backgroundColor": "rgba(139,126,200,0.1)",
                        "fill": True,
                        "tension": 0.1,
                        "pointRadius": 0,
                        "pointHitRadius": 10,
                    }
                ],
            },
            "options": {
                "responsive": True,
                "maintainAspectRatio": False,
                "plugins": {"legend": {"display": False}},
                "scales": {
                    "x": {
                        "grid": {"display": False},
                        "ticks": {"maxTicksLimit": 10},
                    },
                    "y": {},
                },
            },
        }
        return self.to_context()

    def to_context(self) -> str:
        if not self._data or "error" in self._data:
            return "(no Sharpe ratio data)"
        values = self._data["sharpe_chart"]["values"]
        if not values:
            return "(no Sharpe ratio data)"
        current = values[-1]
        peak = max(values)
        trough = min(values)
        if len(values) > 1:
            trend = "improving" if values[-1] > values[-2] else "declining"
        else:
            trend = "stable"
        return (
            f"Cumulative Sharpe Ratio: Current {current:.2f}, "
            f"Peak {peak:.2f}, Trough {trough:.2f}. "
            f"Trend: {trend}."
        )

    def to_html(self) -> str:
        if not self._data or "error" in self._data:
            return '<div class="widget-card widget-error">No Sharpe ratio data</div>'
        return (
            '<div class="chart-card">'
            '<h3 class="chart-title">Cumulative Annual Sharpe Ratio</h3>'
            '<div class="chart-container"><canvas class="widget-chart"></canvas></div>'
            '</div>'
        )
