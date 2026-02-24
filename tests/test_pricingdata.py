"""Tests for pricing data managers including YFinance historical price lookups."""

from athena.pricingdata import YFinancePricingDataManager
from datetime import datetime
from decimal import Decimal


def test_yfinance_pricing_data_manager():
    """Verify YFinance returns correct historical prices for known symbol/date pairs."""
    pm = YFinancePricingDataManager()

    # TD.TO on Dec 31, 2025
    price_point = pm.get_price_point("TD.TO", datetime(2025, 12, 31))
    assert price_point.price == Decimal("129.36")

    # TSLA on Oct 6, 2025
    price_point = pm.get_price_point("TSLA", datetime(2025, 10, 6))
    assert price_point.price == Decimal("453.25")

    # 3037.TW on July 14, 2025
    price_point = pm.get_price_point("3037.TW", datetime(2025, 7, 14))
    assert price_point.price == Decimal("120")
