from enum import Enum
from abc import ABC, abstractmethod
from decimal import Decimal
import datetime as dt_module
from datetime import datetime
import csv
import io
import urllib.request
import hashlib
from pathlib import Path 

class Currency(Enum):
    """Supported currencies for exchange rate conversions."""

    USD = "USD"
    CAD = "CAD"
    EUR = "EUR"
    TWD = "TWD"
    SGD = "SGD"
    AUD = "AUD"
    JPY = "JPY"
    KRW = "KRW"
    GBP = "GBP"
    BRL = "BRL"
    CNY = "CNY"
    HKD = "HKD"
    MXN = "MXN"
    ZAR = "ZAR"
    CHF = "CHF"
    THB = "THB"

class ExchangeRateManager(ABC):
    """Abstract base class for currency exchange rate providers."""

    def __init__(self, min_datetime:datetime|None = None, max_datetime:datetime|None = None):
        """Initialize the exchange rate manager.

        Args:
            min_datetime: Earliest date for which rates are available.
            max_datetime: Latest date for which rates are available.
        """
        self.min_datetime = min_datetime
        self.max_datetime = max_datetime

    @abstractmethod
    def get_exchange_rate(self, from_currency: Currency, to_currency: Currency, datetime:datetime|None = None) -> Decimal:
        """Get the exchange rate between two currencies.

        Args:
            from_currency: The source currency.
            to_currency: The target currency.
            datetime: The date for the rate lookup. If None, uses the current date.

        Returns:
            The exchange rate as a Decimal.

        Raises:
            NotImplementedError: Always, must be overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
class FixedExchangeRateManager(ExchangeRateManager):
    """Exchange rate manager using fixed, hardcoded rates.

    Provides static exchange rates that do not vary by date. Useful for
    testing or when live rates are not needed.
    """

    global_exchange_rates = {
        (Currency.USD, Currency.CAD): Decimal("1.25"),
        (Currency.CAD, Currency.USD): Decimal("0.80"),
        (Currency.USD, Currency.EUR): Decimal("0.85"),
        (Currency.EUR, Currency.USD): Decimal("1.18"),
        (Currency.USD, Currency.TWD): Decimal("27.5"),
        (Currency.TWD, Currency.USD): Decimal("0.036"),
        (Currency.USD, Currency.SGD): Decimal("1.35"),
        (Currency.SGD, Currency.USD): Decimal("0.74"),
        (Currency.USD, Currency.AUD): Decimal("1.55"),
        (Currency.AUD, Currency.USD): Decimal("0.65"),
        (Currency.USD, Currency.JPY): Decimal("150.0"),
        (Currency.JPY, Currency.USD): Decimal("0.0067"),
        (Currency.USD, Currency.KRW): Decimal("1300.0"),
        (Currency.KRW, Currency.USD): Decimal("0.00077"),
        (Currency.USD, Currency.GBP): Decimal("0.79"),
        (Currency.GBP, Currency.USD): Decimal("1.27"),
        (Currency.USD, Currency.BRL): Decimal("5.0"),
        (Currency.BRL, Currency.USD): Decimal("0.20"),
        (Currency.USD, Currency.CNY): Decimal("7.25"),
        (Currency.CNY, Currency.USD): Decimal("0.14"),
        (Currency.USD, Currency.HKD): Decimal("7.80"),
        (Currency.HKD, Currency.USD): Decimal("0.128"),
        (Currency.USD, Currency.MXN): Decimal("17.0"),
        (Currency.MXN, Currency.USD): Decimal("0.059"),
        (Currency.USD, Currency.ZAR): Decimal("18.5"),
        (Currency.ZAR, Currency.USD): Decimal("0.054"),
        (Currency.USD, Currency.CHF): Decimal("0.88"),
        (Currency.CHF, Currency.USD): Decimal("1.14"),
        (Currency.USD, Currency.THB): Decimal("35.0"),
        (Currency.THB, Currency.USD): Decimal("0.029"),
    }

    def __init__(self, exchange_rates:dict[tuple[Currency, Currency], Decimal]={}):
        """Initialize with optional custom exchange rates.

        Args:
            exchange_rates: Custom rates to use. Missing pairs are filled
                from global_exchange_rates defaults.
        """
        self.exchange_rates = exchange_rates
        for (from_currency, to_currency), rate in self.global_exchange_rates.items():
            if (from_currency, to_currency) not in self.exchange_rates:
                self.exchange_rates[(from_currency, to_currency)] = rate

    def set_exchange_rate(self, from_currency: Currency, to_currency: Currency, rate: Decimal):
        """Set or override the exchange rate for a currency pair.

        Args:
            from_currency: The source currency.
            to_currency: The target currency.
            rate: The exchange rate to set.
        """
        self.exchange_rates[(from_currency, to_currency)] = rate

    def get_exchange_rate(self, from_currency: Currency, to_currency: Currency, datetime:datetime|None=None) -> Decimal:
        """Get the fixed exchange rate between two currencies.

        Falls back to USD-triangulated conversion if no direct rate exists.

        Args:
            from_currency: The source currency.
            to_currency: The target currency.
            datetime: Ignored; included for interface compatibility.

        Returns:
            The exchange rate as a Decimal.

        Raises:
            ValueError: If no rate is available for the currency pair.
        """
        if from_currency == to_currency:
            return Decimal("1.0")

        # Try direct lookup
        if (from_currency, to_currency) in self.exchange_rates:
            return self.exchange_rates[(from_currency, to_currency)]

        # If neither currency is USD, try converting via USD
        if from_currency != Currency.USD and to_currency != Currency.USD:
            from_to_usd = (from_currency, Currency.USD)
            usd_to_target = (Currency.USD, to_currency)

            if from_to_usd in self.exchange_rates and usd_to_target in self.exchange_rates:
                rate_to_usd = self.exchange_rates[from_to_usd]
                rate_from_usd = self.exchange_rates[usd_to_target]
                return rate_to_usd * rate_from_usd

        raise ValueError(f"Exchange rate from {from_currency} to {to_currency} not available.")
        
class FederalReserveExchangeRateManager(ExchangeRateManager):
    """Exchange rate manager that fetches historical rates from the Federal Reserve H.10 data.

    Downloads daily exchange rate CSV data from the Fed's website and caches
    it locally. Supports lookback of up to 14 days for missing dates.
    """

    # Mapping from Fed's currency codes to our Currency enum
    FED_CURRENCY_MAP = {
        "EU": Currency.EUR,
        "CA": Currency.CAD,
        "SI": Currency.SGD,
        "AL": Currency.AUD,
        "JA": Currency.JPY,
        "KO": Currency.KRW,
        "TA": Currency.TWD,
        "UK": Currency.GBP,
        "BZ": Currency.BRL,
        "CH": Currency.CNY,
        "HK": Currency.HKD,
        "MX": Currency.MXN,
        "SF": Currency.ZAR,
        "SZ": Currency.CHF,
        "TH": Currency.THB,
    }

    def __init__(self, min_datetime:datetime|None = None, max_datetime:datetime|None = None, use_cache: bool = False):
        """Initialize by fetching exchange rate data from the Federal Reserve.

        Args:
            min_datetime: Start of the date range. Defaults to 2020-01-01.
            max_datetime: End of the date range. Defaults to now.
            use_cache: If True, use disk-cached CSV data when available and fresh.
        """
        if min_datetime is None:
            min_datetime = datetime(2020, 1, 1)

        if max_datetime is None:
            max_datetime = datetime.now()

        super().__init__(min_datetime, max_datetime)

        # Format dates as mm/dd/yyyy for the Fed API
        from_date = min_datetime.strftime("%m/%d/%Y")
        to_date = max_datetime.strftime("%m/%d/%Y")

        base_url = (
            f"https://www.federalreserve.gov/datadownload/Output.aspx?"
            f"rel=H10&series=1f5e3a7e4b72dcddfd7ca4c7c6a8cd55&lastobs=&"
            f"from={from_date}&to={to_date}&filetype=csv&label=include&layout=seriescolumn"
        )

        # Dictionary to store exchange rates: {(date_str, currency_pair_str): Decimal}
        # e.g., {("2023-01-03", "USD->CAD"): Decimal("1.3664")}
        self.exchange_rates: dict[tuple[str, str], Decimal] = {}

        # Generate cache file path based on date range (in project's .cache folder)
        cache_key = hashlib.md5(f"{from_date}_{to_date}".encode()).hexdigest()[:12]
        self._cache_path = Path.cwd() / ".cache" / "fed_exchange_rates" / f"fed_rates_{cache_key}.csv"

        self._fetch_and_parse_data(base_url, use_cache)

    def _is_cache_valid(self) -> bool:
        """Check if cache file exists and was modified today.

        Returns:
            True if the cache file exists and was last modified today.
        """
        if not self._cache_path.exists():
            return False
        mtime = datetime.fromtimestamp(self._cache_path.stat().st_mtime)
        return mtime.date() == datetime.now().date()

    def _fetch_and_parse_data(self, url: str, use_cache: bool) -> None:
        """Fetch CSV data from the Federal Reserve and parse it into exchange rates.

        Args:
            url: The Federal Reserve H.10 CSV download URL.
            use_cache: If True, read from disk cache when valid and write
                fetched data back to disk.
        """
        data: str

        if use_cache and self._is_cache_valid():
            data = self._cache_path.read_text(encoding='utf-8')
        else:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Referer': 'https://www.federalreserve.gov/releases/h10/hist/default.htm',
            }
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req) as response:
                data = response.read().decode('utf-8')
            if use_cache:
                self._cache_path.parent.mkdir(parents=True, exist_ok=True)
                self._cache_path.write_text(data, encoding='utf-8')

        reader = csv.reader(io.StringIO(data))
        rows = list(reader)

        # Row 4 (index 4) contains the unique identifiers that tell us the direction
        # e.g., "H10/H10/RXI$US_N.B.EU" or "H10/H10/RXI_N.B.CA"
        identifier_row = rows[4]

        # Row 5 (index 5) contains the column headers for the time series
        # e.g., "Time Period", "RXI$US_N.B.EU", "RXI_N.B.CA", "RXI_N.B.SI"

        # Parse column metadata to determine currency pairs
        # Column 0 is "Time Period", columns 1+ are currencies
        column_pairs: list[str | None] = []  # Will hold currency pair strings like "USD->CAD"

        for col_idx in range(1, len(identifier_row)):
            identifier = identifier_row[col_idx].strip().strip('"')
            # Extract the series code, e.g., "RXI$US_N.B.EU" from "H10/H10/RXI$US_N.B.EU"
            series_code = identifier.split("/")[-1] if "/" in identifier else identifier

            # Determine currency and direction from series code
            # RXI$US_N.B.XX means USD per 1 XX (so XX -> USD)
            # RXI_N.B.XX means XX per 1 USD (so USD -> XX)
            if "$US" in series_code:
                # USD per foreign currency: foreign -> USD
                # Extract currency code (last 2 chars before any trailing chars)
                currency_code = series_code.split(".")[-1]
                foreign_currency = self.FED_CURRENCY_MAP.get(currency_code)
                if foreign_currency:
                    column_pairs.append(f"{foreign_currency.value}->USD")
                else:
                    column_pairs.append(None)
            else:
                # Foreign currency per USD: USD -> foreign
                currency_code = series_code.split(".")[-1]
                foreign_currency = self.FED_CURRENCY_MAP.get(currency_code)
                if foreign_currency:
                    column_pairs.append(f"USD->{foreign_currency.value}")
                else:
                    column_pairs.append(None)

        # Data rows start at index 6
        for row in rows[6:]:
            if len(row) < 2:
                continue

            date_str = row[0].strip()
            # Validate it looks like a date (yyyy-mm-dd)
            if not date_str or len(date_str) != 10 or date_str[4] != '-':
                continue

            for col_idx in range(1, len(row)):
                if col_idx - 1 >= len(column_pairs):
                    continue

                pair = column_pairs[col_idx - 1]
                if pair is None:
                    continue

                value_str = row[col_idx].strip()
                # Skip "ND" (No Data) entries
                if value_str == "ND" or not value_str:
                    continue

                try:
                    rate = Decimal(value_str)
                    self.exchange_rates[(date_str, pair)] = rate
                except (ValueError, ArithmeticError):
                    continue

    def _get_rate_for_pair(self, pair: str, datetime: datetime) -> Decimal | None:
        """Try to get exchange rate for a pair, looking back up to 14 days.

        Args:
            pair: Currency pair string in "XXX->YYY" format.
            datetime: The target date to start the lookback from.

        Returns:
            The exchange rate as a Decimal, or None if no rate is found
            within the 14-day lookback window.
        """
        for days_back in range(15):
            lookup_date = datetime - dt_module.timedelta(days=days_back)
            date_str = lookup_date.strftime("%Y-%m-%d")
            if (date_str, pair) in self.exchange_rates:
                return self.exchange_rates[(date_str, pair)]
        return None

    def get_exchange_rate(self, from_currency: Currency, to_currency: Currency, datetime: datetime | None = None) -> Decimal:
        """Get exchange rate for a currency pair on a specific date.

        If no data is available for the requested date, looks back up to 14 days
        to handle weekends, bank holidays, and publication delays.
        If neither currency is USD, converts via USD (e.g., HKD -> USD -> CAD).

        Args:
            from_currency: The source currency.
            to_currency: The target currency.
            datetime: The date for the rate lookup. Defaults to now.

        Returns:
            The exchange rate as a Decimal.

        Raises:
            ValueError: If no rate is available for the pair within the
                14-day lookback window.
        """
        if from_currency == to_currency:
            return Decimal("1.0")

        if datetime is None:
            datetime = dt_module.datetime.now()

        pair = f"{from_currency.value}->{to_currency.value}"
        inverse_pair = f"{to_currency.value}->{from_currency.value}"

        # Try direct lookup
        rate = self._get_rate_for_pair(pair, datetime)
        if rate is not None:
            return rate

        # Try inverse lookup
        inverse_rate = self._get_rate_for_pair(inverse_pair, datetime)
        if inverse_rate is not None:
            return Decimal("1") / inverse_rate

        # If neither currency is USD, try converting via USD
        if from_currency != Currency.USD and to_currency != Currency.USD:
            from_to_usd_pair = f"{from_currency.value}->USD"
            usd_to_target_pair = f"USD->{to_currency.value}"

            rate_to_usd = self._get_rate_for_pair(from_to_usd_pair, datetime)
            if rate_to_usd is None:
                # Try inverse: USD -> from_currency
                inverse_from_usd = self._get_rate_for_pair(f"USD->{from_currency.value}", datetime)
                if inverse_from_usd is not None:
                    rate_to_usd = Decimal("1") / inverse_from_usd

            rate_from_usd = self._get_rate_for_pair(usd_to_target_pair, datetime)
            if rate_from_usd is None:
                # Try inverse: to_currency -> USD
                inverse_to_usd = self._get_rate_for_pair(f"{to_currency.value}->USD", datetime)
                if inverse_to_usd is not None:
                    rate_from_usd = Decimal("1") / inverse_to_usd

            if rate_to_usd is not None and rate_from_usd is not None:
                return rate_to_usd * rate_from_usd

        raise ValueError(
            f"Exchange rate from {from_currency.value} to {to_currency.value} "
            f"not available for date {datetime.strftime('%Y-%m-%d')} or the previous 14 days."
        )
    