"""Tests for currency exchange rate managers including Federal Reserve data."""

from athena.currency import Currency, FederalReserveExchangeRateManager
from datetime import datetime
from decimal import Decimal


def test_fedreserve_currency_data():
    """Verify Federal Reserve exchange rates return correct CAD/USD rates for known dates."""
    fr = FederalReserveExchangeRateManager(use_cache=True)
    rate = fr.get_exchange_rate(Currency.CAD, Currency.USD, datetime(2025, 12, 12))
    assert rate == Decimal("0.7255314517884350286584923456")
    rate = fr.get_exchange_rate(Currency.CAD, Currency.USD, datetime(2025, 11, 27))
    assert rate == Decimal("0.7116931179275496405949754466")
