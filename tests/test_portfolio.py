"""Tests for portfolio operations including loading, valuation, and short selling."""

from datetime import datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from athena.portfolio import Transaction, TransactionType, Portfolio, Position, load_portfolio_from_excel, calculate_portfolio_final_cash_available, calculate_cash_available_by_day, calculate_portfolio_value_on_date, calculate_portfolio_value_by_day, calculate_portfolio_book_value_on_date, get_positions
from athena.currency import Currency, FixedExchangeRateManager
from athena.pricingdata import PricingDataManager, PricePoint


def test_portfolio_with_cad_transaction():
    """Verify a CAD-denominated transaction is stored correctly in a portfolio."""
    exchange_rate_manager = FixedExchangeRateManager({
        (Currency.CAD, Currency.USD): Decimal("0.75"),
        (Currency.USD, Currency.CAD): Decimal("1.33"),
    })

    transaction = Transaction(
        symbol="TD.TO",
        transaction_datetime=datetime(2025, 1, 15, 10, 0, 0, tzinfo=ZoneInfo("America/New_York")),
        transaction_type=TransactionType.BUY,
        quantity=Decimal("2"),
        price=Decimal("100.00"),
        currency=Currency.CAD,
    )

    portfolio = Portfolio(transactions=[transaction], primary_currency=Currency.CAD)

    assert len(portfolio.transactions) == 1
    assert portfolio.transactions[0].symbol == "TD.TO"
    assert portfolio.transactions[0].quantity == Decimal("2")
    assert portfolio.transactions[0].price == Decimal("100.00")
    assert portfolio.transactions[0].currency == Currency.CAD
    assert exchange_rate_manager.get_exchange_rate(Currency.CAD, Currency.USD) == Decimal("0.75")


def test_load_portfolio_from_excel():
    """Verify that loading a portfolio from an Excel file parses all transactions."""
    test_file = Path(__file__).parent / "sample_transactions.xlsx"
    portfolio = load_portfolio_from_excel(str(test_file))
    assert len(portfolio.transactions) == 2


def test_calculate_portfolio_final_value_cad():
    """Verify final cash available is zero after a buy-then-sell round trip in CAD."""
    test_file = Path(__file__).parent / "sample_transactions.xlsx"
    portfolio = load_portfolio_from_excel(
        str(test_file),
        error_out_negative_cash=False
    )
    final_value = calculate_portfolio_final_cash_available(portfolio, Currency.CAD)
    assert final_value == Decimal("0")


def test_calculate_portfolio_final_value_cad_negative_cash_error():
    """Verify ValueError is raised when negative cash is detected with error flag enabled."""
    test_file = Path(__file__).parent / "sample_transactions.xlsx"
    portfolio = load_portfolio_from_excel(
        str(test_file),
        error_out_negative_cash=True
    )
    with pytest.raises(ValueError, match="Negative cash balance detected"):
        calculate_portfolio_final_cash_available(portfolio, Currency.CAD)


def test_calculate_value_by_day():
    """
    Test calculate_value_by_day with sample_transactions.xlsx.

    Transactions:
    - 2025-12-08: BUY TD.TO at $100 x 2 = -$200 CAD
    - 2025-12-12: SELL TD.TO at $100 x 2 = +$200 CAD

    Expected values by day (in CAD):
    - Dec 8: -200 (after buy)
    - Dec 9: -200 (no change)
    - Dec 10: -200 (no change)
    - Dec 11: -200 (no change)
    - Dec 12: 0 (after sell)
    """
    test_file = Path(__file__).parent / "sample_transactions.xlsx"

    exchange_rate_manager = FixedExchangeRateManager()

    portfolio = load_portfolio_from_excel(
        str(test_file),
        error_out_negative_cash=False
    )
    portfolio.exchange_rate_manager = exchange_rate_manager

    result = calculate_cash_available_by_day(
        portfolio,
        Currency.CAD,
        end_date=datetime(2025, 12, 12),
    )

    expected = {
        datetime(2025, 12, 8): Decimal("-200"),
        datetime(2025, 12, 9): Decimal("-200"),
        datetime(2025, 12, 10): Decimal("-200"),
        datetime(2025, 12, 11): Decimal("-200"),
        datetime(2025, 12, 12): Decimal("0"),
    }

    assert result == expected


class FixedCADPricingDataManager(PricingDataManager):
    """A pricing manager that returns a fixed CAD price for all symbols.

    Used in tests to provide deterministic pricing without external data sources.

    Args:
        price: The fixed price to return for every symbol lookup.
    """

    def __init__(self, price: Decimal):
        self.price = price

    def get_price_point(self, symbol: str, price_datetime: datetime) -> PricePoint:
        """Return a fixed-price PricePoint in CAD.

        Args:
            symbol: The ticker symbol (ignored; same price returned for all).
            price_datetime: The datetime for the price point.

        Returns:
            A PricePoint with the fixed price in CAD.
        """
        return PricePoint(
            symbol=symbol,
            price_datetime=price_datetime,
            price=self.price,
            base_currency=Currency.CAD
        )


def test_portfolio_value_is_zero_when_holdings_offset_negative_cash():
    """
    Test that portfolio value is 0 when holdings value exactly offsets negative cash.

    Transactions from sample_transactions.xlsx:
    - 2025-12-08: BUY TD.TO at $100 x 2 = -$200 CAD cash, +2 shares
    - 2025-12-12: SELL TD.TO at $100 x 2 = +$200 CAD cash, -2 shares

    With holdings priced at $100 CAD each:
    - Dec 8-11: -$200 CAD cash + 2 shares @ $100 = $0 total value
    - Dec 12: $0 CAD cash + 0 shares = $0 total value

    This test verifies both:
    1. calculate_portfolio_value_on_date returns 0 for each day (called 5 times)
    2. calculate_portfolio_value_by_day returns 0 for all days in the range
    """
    test_file = Path(__file__).parent / "sample_transactions.xlsx"

    exchange_rate_manager = FixedExchangeRateManager()
    pricing_manager = FixedCADPricingDataManager(Decimal("100"))

    portfolio = load_portfolio_from_excel(
        str(test_file),
        error_out_negative_cash=False
    )
    portfolio.exchange_rate_manager = exchange_rate_manager
    portfolio.pricing_manager = pricing_manager

    tz = ZoneInfo("America/New_York")

    # Test 1: Call calculate_portfolio_value_on_date 5 times (once per day)
    assert calculate_portfolio_value_on_date(portfolio, datetime(2025, 12, 8, 23, 59, 59, tzinfo=tz), Currency.CAD) == Decimal("0")
    assert calculate_portfolio_value_on_date(portfolio, datetime(2025, 12, 9, 23, 59, 59, tzinfo=tz), Currency.CAD) == Decimal("0")
    assert calculate_portfolio_value_on_date(portfolio, datetime(2025, 12, 10, 23, 59, 59, tzinfo=tz), Currency.CAD) == Decimal("0")
    assert calculate_portfolio_value_on_date(portfolio, datetime(2025, 12, 11, 23, 59, 59, tzinfo=tz), Currency.CAD) == Decimal("0")
    assert calculate_portfolio_value_on_date(portfolio, datetime(2025, 12, 12, 23, 59, 59, tzinfo=tz), Currency.CAD) == Decimal("0")

    # Test 2: Call calculate_portfolio_value_by_day and verify all days are 0
    result = calculate_portfolio_value_by_day(
        portfolio,
        Currency.CAD,
        start_date=datetime(2025, 12, 8, tzinfo=tz),
        end_date=datetime(2025, 12, 12, tzinfo=tz)
    )

    expected = {
        datetime(2025, 12, 8): Decimal("0"),
        datetime(2025, 12, 9): Decimal("0"),
        datetime(2025, 12, 10): Decimal("0"),
        datetime(2025, 12, 11): Decimal("0"),
        datetime(2025, 12, 12): Decimal("0"),
    }

    assert result == expected


# ---------------------------------------------------------------------------
# Short Selling Tests
# ---------------------------------------------------------------------------

class FixedUSDPricingDataManager(PricingDataManager):
    """A pricing manager that returns a fixed USD price for all symbols.

    Used in tests to provide deterministic pricing without external data sources.

    Args:
        price: The fixed price to return for every symbol lookup.
    """

    def __init__(self, price: Decimal):
        self.price = price

    def get_price_point(self, symbol: str, price_datetime: datetime) -> PricePoint:
        """Return a fixed-price PricePoint in USD.

        Args:
            symbol: The ticker symbol (ignored; same price returned for all).
            price_datetime: The datetime for the price point.

        Returns:
            A PricePoint with the fixed price in USD.
        """
        return PricePoint(
            symbol=symbol,
            price_datetime=price_datetime,
            price=self.price,
            base_currency=Currency.USD,
        )


def _make_short_portfolio(
    transactions: list[Transaction],
    pricing_price: Decimal = Decimal("100"),
) -> Portfolio:
    """Create a portfolio with short selling enabled for testing.

    Args:
        transactions: List of transactions to include in the portfolio.
        pricing_price: The fixed price used by the test pricing manager.

    Returns:
        A Portfolio configured with fixed exchange rates and fixed pricing,
        with negative cash and negative quantity checks disabled.
    """
    portfolio = Portfolio(
        transactions=transactions,
        primary_currency=Currency.USD,
        error_out_negative_cash=False,
        error_out_negative_quantity=False,
    )
    portfolio.exchange_rate_manager = FixedExchangeRateManager()
    portfolio.pricing_manager = FixedUSDPricingDataManager(pricing_price)
    return portfolio


def test_short_position_creation():
    """SELL without holdings creates a short (negative quantity) position."""
    tz = ZoneInfo("America/New_York")
    txns = [
        Transaction("AAPL", datetime(2025, 1, 1, 10, 0, 0, tzinfo=tz), TransactionType.CASH_IN, Decimal("10000"), Decimal("1"), Currency.USD),
        Transaction("AAPL", datetime(2025, 1, 2, 10, 0, 0, tzinfo=tz), TransactionType.SELL, Decimal("10"), Decimal("100"), Currency.USD),
    ]
    portfolio = _make_short_portfolio(txns)

    positions = get_positions(datetime(2025, 1, 3, tzinfo=tz), portfolio)
    assert len(positions) == 1
    assert positions[0].symbol == "AAPL"
    assert positions[0].quantity == Decimal("-10")


def test_short_position_market_value():
    """
    Short 10 at $100, price drops to $90.
    Cash = 10000 + 1000 (from short) = 11000
    Position value = -10 * 90 = -900
    Total portfolio = 11000 + (-900) = 10100
    """
    tz = ZoneInfo("America/New_York")
    txns = [
        Transaction("AAPL", datetime(2025, 1, 1, 10, 0, 0, tzinfo=tz), TransactionType.CASH_IN, Decimal("10000"), Decimal("1"), Currency.USD),
        Transaction("AAPL", datetime(2025, 1, 2, 10, 0, 0, tzinfo=tz), TransactionType.SELL, Decimal("10"), Decimal("100"), Currency.USD),
    ]
    portfolio = _make_short_portfolio(txns, pricing_price=Decimal("90"))

    value = calculate_portfolio_value_on_date(
        portfolio, datetime(2025, 1, 3, 23, 59, 59, tzinfo=tz), Currency.USD
    )
    assert value == Decimal("10100")


def test_cover_short():
    """
    Short 10 at $100, then buy 10 at $90 to cover.
    Cash: +10000 (initial) + 1000 (short sale) - 900 (cover) = 10100
    Holdings: 0
    """
    tz = ZoneInfo("America/New_York")
    txns = [
        Transaction("AAPL", datetime(2025, 1, 1, 10, 0, 0, tzinfo=tz), TransactionType.CASH_IN, Decimal("10000"), Decimal("1"), Currency.USD),
        Transaction("AAPL", datetime(2025, 1, 2, 10, 0, 0, tzinfo=tz), TransactionType.SELL, Decimal("10"), Decimal("100"), Currency.USD),
        Transaction("AAPL", datetime(2025, 1, 3, 10, 0, 0, tzinfo=tz), TransactionType.BUY, Decimal("10"), Decimal("90"), Currency.USD),
    ]
    portfolio = _make_short_portfolio(txns)

    cash = calculate_portfolio_final_cash_available(portfolio, Currency.USD)
    assert cash == Decimal("10100")

    positions = get_positions(datetime(2025, 1, 4, tzinfo=tz), portfolio)
    # No positions remain (quantity == 0 is filtered out)
    assert len(positions) == 0


def test_short_gain_loss_profitable():
    """
    Short 10 at $100, price drops to $80.
    book_value = -1000 (negative = short obligation)
    total_value = -10 * 80 = -800
    gain_loss = -800 - (-1000) = +200
    gain_loss_percent = 200 / abs(-1000) * 100 = 20%
    """
    tz = ZoneInfo("America/New_York")
    txns = [
        Transaction("AAPL", datetime(2025, 1, 1, 10, 0, 0, tzinfo=tz), TransactionType.CASH_IN, Decimal("10000"), Decimal("1"), Currency.USD),
        Transaction("AAPL", datetime(2025, 1, 2, 10, 0, 0, tzinfo=tz), TransactionType.SELL, Decimal("10"), Decimal("100"), Currency.USD),
    ]
    portfolio = _make_short_portfolio(txns, pricing_price=Decimal("80"))

    positions = get_positions(datetime(2025, 1, 3, tzinfo=tz), portfolio)
    assert len(positions) == 1
    pos = positions[0]
    assert pos.gain_loss == Decimal("200")
    assert pos.gain_loss_percent == Decimal("20")


def test_short_gain_loss_unprofitable():
    """
    Short 10 at $100, price rises to $120.
    book_value = -1000
    total_value = -10 * 120 = -1200
    gain_loss = -1200 - (-1000) = -200
    gain_loss_percent = -200 / abs(-1000) * 100 = -20%
    """
    tz = ZoneInfo("America/New_York")
    txns = [
        Transaction("AAPL", datetime(2025, 1, 1, 10, 0, 0, tzinfo=tz), TransactionType.CASH_IN, Decimal("10000"), Decimal("1"), Currency.USD),
        Transaction("AAPL", datetime(2025, 1, 2, 10, 0, 0, tzinfo=tz), TransactionType.SELL, Decimal("10"), Decimal("100"), Currency.USD),
    ]
    portfolio = _make_short_portfolio(txns, pricing_price=Decimal("120"))

    positions = get_positions(datetime(2025, 1, 3, tzinfo=tz), portfolio)
    assert len(positions) == 1
    pos = positions[0]
    assert pos.gain_loss == Decimal("-200")
    assert pos.gain_loss_percent == Decimal("-20")


def test_partial_cover_short():
    """
    Short 10 at $100, buy 5 at $90.
    Remaining: -5 shares
    Cost basis: started at -1000, cover 5 removes half -> -500 remaining
    Cash: +10000 + 1000 - 450 = 10550
    """
    tz = ZoneInfo("America/New_York")
    txns = [
        Transaction("AAPL", datetime(2025, 1, 1, 10, 0, 0, tzinfo=tz), TransactionType.CASH_IN, Decimal("10000"), Decimal("1"), Currency.USD),
        Transaction("AAPL", datetime(2025, 1, 2, 10, 0, 0, tzinfo=tz), TransactionType.SELL, Decimal("10"), Decimal("100"), Currency.USD),
        Transaction("AAPL", datetime(2025, 1, 3, 10, 0, 0, tzinfo=tz), TransactionType.BUY, Decimal("5"), Decimal("90"), Currency.USD),
    ]
    portfolio = _make_short_portfolio(txns, pricing_price=Decimal("90"))

    positions = get_positions(datetime(2025, 1, 4, tzinfo=tz), portfolio)
    assert len(positions) == 1
    pos = positions[0]
    assert pos.quantity == Decimal("-5")
    # Book value should be -500 (half of -1000 remaining)
    assert pos.book_value == Decimal("-500")
    # Market value: -5 * 90 = -450
    assert pos.total_value == Decimal("-450")

    cash = calculate_portfolio_final_cash_available(portfolio, Currency.USD)
    assert cash == Decimal("10550")


def test_sell_through_zero_long_to_short():
    """
    BUY 5 at $100, then SELL 10 at $110.
    First 5 sold close the long (cost basis $500 removed).
    Remaining 5 open a short at $110 (cost basis = -550).
    Holdings: -5
    Cash: +10000 - 500 (buy) + 1100 (sell) = 10600
    """
    tz = ZoneInfo("America/New_York")
    txns = [
        Transaction("AAPL", datetime(2025, 1, 1, 10, 0, 0, tzinfo=tz), TransactionType.CASH_IN, Decimal("10000"), Decimal("1"), Currency.USD),
        Transaction("AAPL", datetime(2025, 1, 2, 10, 0, 0, tzinfo=tz), TransactionType.BUY, Decimal("5"), Decimal("100"), Currency.USD),
        Transaction("AAPL", datetime(2025, 1, 3, 10, 0, 0, tzinfo=tz), TransactionType.SELL, Decimal("10"), Decimal("110"), Currency.USD),
    ]
    portfolio = _make_short_portfolio(txns, pricing_price=Decimal("110"))

    positions = get_positions(datetime(2025, 1, 4, tzinfo=tz), portfolio)
    assert len(positions) == 1
    pos = positions[0]
    assert pos.quantity == Decimal("-5")
    # Short cost basis: -5 * 110 = -550
    assert pos.book_value == Decimal("-550")

    cash = calculate_portfolio_final_cash_available(portfolio, Currency.USD)
    assert cash == Decimal("10600")


def test_short_selling_blocked_by_default():
    """error_out_negative_quantity=True (default) still raises ValueError."""
    tz = ZoneInfo("America/New_York")
    txns = [
        Transaction("AAPL", datetime(2025, 1, 1, 10, 0, 0, tzinfo=tz), TransactionType.CASH_IN, Decimal("10000"), Decimal("1"), Currency.USD),
        Transaction("AAPL", datetime(2025, 1, 2, 10, 0, 0, tzinfo=tz), TransactionType.SELL, Decimal("10"), Decimal("100"), Currency.USD),
    ]
    portfolio = Portfolio(
        transactions=txns,
        primary_currency=Currency.USD,
        error_out_negative_cash=False,
        error_out_negative_quantity=True,
    )
    portfolio.exchange_rate_manager = FixedExchangeRateManager()
    portfolio.pricing_manager = FixedUSDPricingDataManager(Decimal("100"))

    with pytest.raises(ValueError, match="Negative quantity detected"):
        get_positions(datetime(2025, 1, 3, tzinfo=tz), portfolio)
