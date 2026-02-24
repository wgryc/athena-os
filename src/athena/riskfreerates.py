from abc import ABC, abstractmethod
from datetime import datetime, date, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
import csv
import hashlib
import io
import urllib.request


class RiskFreeRateSeries(Enum):
    """FRED series identifiers for risk-free rates."""
    DTB3 = "DTB3"      # 3-Month Treasury Bill Secondary Market Rate
    DTB6 = "DTB6"      # 6-Month Treasury Bill Secondary Market Rate
    DTB1YR = "DTB1YR"  # 1-Year Treasury Bill Secondary Market Rate
    DGS1 = "DGS1"      # 1-Year Treasury Constant Maturity Rate
    DGS2 = "DGS2"      # 2-Year Treasury Constant Maturity Rate
    DGS5 = "DGS5"      # 5-Year Treasury Constant Maturity Rate
    DGS10 = "DGS10"    # 10-Year Treasury Constant Maturity Rate


class RiskFreeRateManager(ABC):
    """Abstract base class for risk-free rate data providers."""

    def __init__(
        self,
        min_date: date | None = None,
        max_date: date | None = None
    ):
        self.min_date = min_date
        self.max_date = max_date

    @abstractmethod
    def get_rate(self, rate_date: date) -> Decimal:
        """
        Get the annualized risk-free rate for a specific date.

        Args:
            rate_date: The date to get the rate for.

        Returns:
            The annualized rate as a decimal (e.g., Decimal("0.045") for 4.5%).

        Raises:
            ValueError: If no rate is available for the date.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def get_daily_rate(self, rate_date: date, trading_days_per_year: int = 252) -> Decimal:
        """
        Get the daily risk-free rate for a specific date.

        Args:
            rate_date: The date to get the rate for.
            trading_days_per_year: Number of trading days per year (default 252).

        Returns:
            The daily rate as a decimal.

        Raises:
            ValueError: If no rate is available for the date.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def get_rates_for_range(
        self,
        start_date: date,
        end_date: date
    ) -> dict[date, Decimal]:
        """
        Get all available annualized rates for a date range.

        Args:
            start_date: Start of the date range (inclusive).
            end_date: End of the date range (inclusive).

        Returns:
            Dictionary mapping dates to annualized rates.
            Only includes dates where data is available (trading days).
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def get_trading_days_per_year(self) -> int:
        """Return the typical number of trading days per year for US markets."""
        return 252


class FixedRiskFreeRateManager(RiskFreeRateManager):
    """Returns a fixed risk-free rate for all dates."""

    def __init__(self, annual_rate: Decimal = Decimal("0.05")):
        """
        Initialize with a fixed annual rate.

        Args:
            annual_rate: The fixed annual rate as a decimal (e.g., 0.05 for 5%).
        """
        super().__init__()
        self.annual_rate = annual_rate

    def get_rate(self, rate_date: date) -> Decimal:
        return self.annual_rate

    def get_daily_rate(self, rate_date: date, trading_days_per_year: int = 252) -> Decimal:
        return self.annual_rate / trading_days_per_year

    def get_rates_for_range(
        self,
        start_date: date,
        end_date: date
    ) -> dict[date, Decimal]:
        result: dict[date, Decimal] = {}
        current = start_date
        while current <= end_date:
            result[current] = self.annual_rate
            current += timedelta(days=1)
        return result


class FREDRiskFreeRateManager(RiskFreeRateManager):
    """
    Risk-free rate manager using FRED (Federal Reserve Economic Data).

    Fetches Treasury Bill rates directly from FRED's CSV endpoint.
    Data is cached locally to avoid repeated downloads.
    """

    def __init__(
        self,
        series: RiskFreeRateSeries = RiskFreeRateSeries.DTB3,
        min_date: date | None = None,
        max_date: date | None = None,
        use_cache: bool = True
    ):
        """
        Initialize the FRED risk-free rate manager.

        Args:
            series: The FRED series to use (default: DTB3 - 3-Month T-Bill).
            min_date: Earliest date to fetch data for (default: 5 years ago).
            max_date: Latest date to fetch data for (default: today).
            use_cache: Whether to use local caching (default: True).
        """
        if min_date is None:
            min_date = date.today() - timedelta(days=5 * 365)
        if max_date is None:
            max_date = date.today()

        super().__init__(min_date, max_date)

        self.series = series
        self.use_cache = use_cache

        # Dictionary to store rates: {date: Decimal}
        # Rates are stored as decimals (e.g., 0.045 for 4.5%)
        self.rates: dict[date, Decimal] = {}

        # Build the FRED CSV URL
        url = self._build_url(min_date, max_date)

        # Generate cache file path
        cache_key = hashlib.md5(
            f"{series.value}_{min_date}_{max_date}".encode()
        ).hexdigest()[:12]
        self._cache_path = (
            Path.cwd() / ".cache" / "fred_risk_free_rates" / f"{series.value}_{cache_key}.csv"
        )

        self._fetch_and_parse_data(url)

    def _build_url(self, start_date: date, end_date: date) -> str:
        """Build the FRED CSV download URL."""
        # Format dates as YYYY-MM-DD
        cosd = start_date.strftime("%Y-%m-%d")
        coed = end_date.strftime("%Y-%m-%d")
        today = date.today().strftime("%Y-%m-%d")

        # Simplified FRED URL - only the essential parameters
        url = (
            f"https://fred.stlouisfed.org/graph/fredgraph.csv?"
            f"id={self.series.value}&"
            f"cosd={cosd}&"
            f"coed={coed}&"
            f"fq=Daily&"
            f"fam=avg&"
            f"vintage_date={today}&"
            f"revision_date={today}"
        )
        return url

    def _is_cache_valid(self) -> bool:
        """Check if cache file exists and was modified today."""
        if not self._cache_path.exists():
            return False
        mtime = datetime.fromtimestamp(self._cache_path.stat().st_mtime)
        return mtime.date() == date.today()

    def _fetch_and_parse_data(self, url: str) -> None:
        """Fetch CSV data from FRED and parse it into rates."""
        data: str

        if self.use_cache and self._is_cache_valid():
            data = self._cache_path.read_text(encoding='utf-8')
        else:
            headers = {
                'User-Agent': (
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                    'AppleWebKit/537.36 (KHTML, like Gecko) '
                    'Chrome/120.0.0.0 Safari/537.36'
                ),
                'Accept': 'text/csv,text/plain,*/*',
                'Accept-Language': 'en-US,en;q=0.5',
            }
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req) as response:
                data = response.read().decode('utf-8')

            if self.use_cache:
                self._cache_path.parent.mkdir(parents=True, exist_ok=True)
                self._cache_path.write_text(data, encoding='utf-8')

        # Parse CSV
        # FRED CSV format:
        # DATE,DTB3
        # 2021-01-15,0.09
        # 2021-01-19,0.08
        # ...
        # Note: Missing data is marked as "."
        reader = csv.reader(io.StringIO(data))
        rows = list(reader)

        if len(rows) < 2:
            return

        # First row is header, data starts at row 1
        for row in rows[1:]:
            if len(row) < 2:
                continue

            date_str = row[0].strip()
            value_str = row[1].strip()

            # Skip missing data (FRED uses "." for missing values)
            if value_str == "." or not value_str:
                continue

            try:
                # Parse date (YYYY-MM-DD format)
                rate_date = datetime.strptime(date_str, "%Y-%m-%d").date()

                # FRED returns rates as percentage points (e.g., 4.5 means 4.5%)
                # Convert to decimal (e.g., 0.045)
                rate_pct = Decimal(value_str)
                rate_decimal = rate_pct / 100

                self.rates[rate_date] = rate_decimal

            except (ValueError, ArithmeticError):
                continue

    def _get_rate_with_lookback(self, rate_date: date, max_lookback_days: int = 14) -> Decimal | None:
        """
        Get rate for a date, looking back if not available.

        Args:
            rate_date: The target date.
            max_lookback_days: Maximum days to look back for data.

        Returns:
            The rate if found, None otherwise.
        """
        for days_back in range(max_lookback_days + 1):
            lookup_date = rate_date - timedelta(days=days_back)
            if lookup_date in self.rates:
                return self.rates[lookup_date]
        return None

    def get_rate(self, rate_date: date) -> Decimal:
        """
        Get the annualized risk-free rate for a specific date.

        If no data is available for the exact date (weekend, holiday),
        looks back up to 14 days to find the most recent trading day.

        Args:
            rate_date: The date to get the rate for.

        Returns:
            The annualized rate as a decimal (e.g., Decimal("0.045") for 4.5%).

        Raises:
            ValueError: If no rate is available for the date or recent history.
        """
        rate = self._get_rate_with_lookback(rate_date)
        if rate is None:
            raise ValueError(
                f"No risk-free rate available for {rate_date} "
                f"or the previous 14 days."
            )
        return rate

    def get_daily_rate(self, rate_date: date, trading_days_per_year: int = 252) -> Decimal:
        """
        Get the daily risk-free rate for a specific date.

        Converts the annualized rate to a daily rate by dividing by
        the number of trading days per year.

        Args:
            rate_date: The date to get the rate for.
            trading_days_per_year: Number of trading days per year (default 252).

        Returns:
            The daily rate as a decimal.

        Raises:
            ValueError: If no rate is available for the date.
        """
        annual_rate = self.get_rate(rate_date)
        return annual_rate / trading_days_per_year

    def get_rates_for_range(
        self,
        start_date: date,
        end_date: date
    ) -> dict[date, Decimal]:
        """
        Get all available annualized rates for a date range.

        Only returns dates where actual FRED data exists (trading days).
        Does not interpolate or carry forward rates.

        Args:
            start_date: Start of the date range (inclusive).
            end_date: End of the date range (inclusive).

        Returns:
            Dictionary mapping dates to annualized rates (as decimals).
        """
        result: dict[date, Decimal] = {}
        for rate_date, rate in self.rates.items():
            if start_date <= rate_date <= end_date:
                result[rate_date] = rate
        return result

    def get_available_dates(self) -> list[date]:
        """
        Get all dates for which rate data is available.

        Returns:
            Sorted list of dates with available data.
        """
        return sorted(self.rates.keys())
