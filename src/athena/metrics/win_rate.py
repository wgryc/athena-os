from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Tuple

from ..currency import Currency
from ..portfolio import Portfolio, TransactionType, get_positions


@dataclass
class ClosedPosition:
    """Represents a closed position with its realized gain/loss."""
    symbol: str
    quantity: Decimal
    buy_price: Decimal
    sell_price: Decimal
    buy_date: datetime
    sell_date: datetime
    currency: Currency
    realized_gain_loss: Decimal
    realized_gain_loss_percent: Decimal


@dataclass
class WinRateResult:
    """Result of a win rate calculation."""
    win_rate: float
    total_positions: int
    winning_positions: int
    losing_positions: int
    breakeven_positions: int
    total_gain_loss: Decimal
    average_win: Decimal | None
    average_loss: Decimal | None
    win_loss_ratio: float | None


def get_closed_positions(
    portfolio: Portfolio,
    as_of: datetime | None = None
) -> list[ClosedPosition]:
    """
    Get all closed positions from a portfolio.

    A closed position is created when shares are sold. This uses FIFO
    (First In, First Out) matching to pair buys with sells.

    Args:
        portfolio: Portfolio object containing transactions.
        as_of: Only consider transactions up to this datetime.
               If None, considers all transactions.

    Returns:
        List of ClosedPosition objects representing realized trades.
    """
    # Track open lots for each symbol using FIFO
    # Each lot is (quantity, price, date, currency)
    open_lots: dict[str, list[tuple[Decimal, Decimal, datetime, Currency]]] = {}
    closed_positions: list[ClosedPosition] = []

    sorted_transactions = sorted(
        portfolio.transactions,
        key=lambda t: t.transaction_datetime
    )

    for txn in sorted_transactions:
        if as_of is not None and txn.transaction_datetime > as_of:
            break

        symbol = txn.symbol
        quantity = txn.quantity
        price = Decimal(str(txn.price))
        currency = txn.currency

        if txn.transaction_type == TransactionType.BUY:
            if symbol not in open_lots:
                open_lots[symbol] = []
            open_lots[symbol].append((quantity, price, txn.transaction_datetime, currency))

        elif txn.transaction_type == TransactionType.SELL:
            if symbol not in open_lots or not open_lots[symbol]:
                continue

            remaining_to_sell = quantity
            sell_price = price
            sell_date = txn.transaction_datetime

            while remaining_to_sell > 0 and open_lots[symbol]:
                lot = open_lots[symbol][0]
                lot_qty, lot_price, lot_date, lot_currency = lot

                if lot_qty <= remaining_to_sell:
                    # Close entire lot
                    open_lots[symbol].pop(0)
                    closed_qty = lot_qty
                    remaining_to_sell -= lot_qty
                else:
                    # Partially close lot
                    open_lots[symbol][0] = (
                        lot_qty - remaining_to_sell,
                        lot_price,
                        lot_date,
                        lot_currency
                    )
                    closed_qty = remaining_to_sell
                    remaining_to_sell = Decimal("0")

                # Calculate realized gain/loss
                buy_value = closed_qty * lot_price
                sell_value = closed_qty * sell_price
                realized_gain_loss = sell_value - buy_value

                if buy_value != 0:
                    realized_gain_loss_percent = (realized_gain_loss / buy_value) * 100
                else:
                    realized_gain_loss_percent = Decimal("0")

                closed_positions.append(ClosedPosition(
                    symbol=symbol,
                    quantity=closed_qty,
                    buy_price=lot_price,
                    sell_price=sell_price,
                    buy_date=lot_date,
                    sell_date=sell_date,
                    currency=lot_currency,
                    realized_gain_loss=realized_gain_loss,
                    realized_gain_loss_percent=realized_gain_loss_percent
                ))

    return closed_positions


def calculate_win_rate_closed(
    portfolio: Portfolio,
    as_of: datetime | None = None
) -> WinRateResult:
    """
    Calculate win rate based on closed positions only.

    A winning position is one where the sell price is higher than the buy price.
    Uses FIFO matching to pair buys with sells.

    Args:
        portfolio: Portfolio object containing transactions.
        as_of: Only consider transactions up to this datetime.
               If None, considers all transactions.

    Returns:
        WinRateResult containing win rate statistics for closed positions.
    """
    closed_positions = get_closed_positions(portfolio, as_of)

    if not closed_positions:
        return WinRateResult(
            win_rate=0.0,
            total_positions=0,
            winning_positions=0,
            losing_positions=0,
            breakeven_positions=0,
            total_gain_loss=Decimal("0"),
            average_win=None,
            average_loss=None,
            win_loss_ratio=None
        )

    winning = [p for p in closed_positions if p.realized_gain_loss > 0]
    losing = [p for p in closed_positions if p.realized_gain_loss < 0]
    breakeven = [p for p in closed_positions if p.realized_gain_loss == 0]

    total_positions = len(closed_positions)
    winning_positions = len(winning)
    losing_positions = len(losing)
    breakeven_positions = len(breakeven)

    win_rate = winning_positions / total_positions if total_positions > 0 else 0.0

    total_gain_loss = sum(p.realized_gain_loss for p in closed_positions)

    average_win = (
        sum(p.realized_gain_loss for p in winning) / winning_positions
        if winning_positions > 0 else None
    )

    average_loss = (
        sum(p.realized_gain_loss for p in losing) / losing_positions
        if losing_positions > 0 else None
    )

    win_loss_ratio = (
        float(abs(average_win / average_loss))
        if average_win is not None and average_loss is not None and average_loss != 0
        else None
    )

    return WinRateResult(
        win_rate=win_rate,
        total_positions=total_positions,
        winning_positions=winning_positions,
        losing_positions=losing_positions,
        breakeven_positions=breakeven_positions,
        total_gain_loss=total_gain_loss,
        average_win=average_win,
        average_loss=average_loss,
        win_loss_ratio=win_loss_ratio
    )


def calculate_win_rate_all_positions(
    portfolio: Portfolio,
    as_of: datetime | None = None
) -> WinRateResult:
    """
    Calculate win rate based on all positions (open and closed).

    For closed positions, uses realized gain/loss.
    For open positions, uses unrealized gain/loss based on current market value.

    Args:
        portfolio: Portfolio object containing transactions.
        as_of: The datetime to evaluate positions at.
               If None, uses the latest transaction date.

    Returns:
        WinRateResult containing win rate statistics for all positions.
    """
    if as_of is None:
        if not portfolio.transactions:
            return WinRateResult(
                win_rate=0.0,
                total_positions=0,
                winning_positions=0,
                losing_positions=0,
                breakeven_positions=0,
                total_gain_loss=Decimal("0"),
                average_win=None,
                average_loss=None,
                win_loss_ratio=None
            )
        sorted_txns = sorted(portfolio.transactions, key=lambda t: t.transaction_datetime)
        as_of = sorted_txns[-1].transaction_datetime

    # Get closed positions
    closed_positions = get_closed_positions(portfolio, as_of)

    # Get open positions
    open_positions = get_positions(as_of, portfolio)

    # Combine gains/losses
    all_gains_losses: list[Decimal] = []

    # Add closed position gains/losses
    for pos in closed_positions:
        all_gains_losses.append(pos.realized_gain_loss)

    # Add open position gains/losses (unrealized)
    for pos in open_positions:
        if pos.gain_loss is not None:
            all_gains_losses.append(pos.gain_loss)

    if not all_gains_losses:
        return WinRateResult(
            win_rate=0.0,
            total_positions=0,
            winning_positions=0,
            losing_positions=0,
            breakeven_positions=0,
            total_gain_loss=Decimal("0"),
            average_win=None,
            average_loss=None,
            win_loss_ratio=None
        )

    winning = [gl for gl in all_gains_losses if gl > 0]
    losing = [gl for gl in all_gains_losses if gl < 0]
    breakeven = [gl for gl in all_gains_losses if gl == 0]

    total_positions = len(all_gains_losses)
    winning_positions = len(winning)
    losing_positions = len(losing)
    breakeven_positions = len(breakeven)

    win_rate = winning_positions / total_positions if total_positions > 0 else 0.0

    total_gain_loss = sum(all_gains_losses)

    average_win = (
        sum(winning) / winning_positions
        if winning_positions > 0 else None
    )

    average_loss = (
        sum(losing) / losing_positions
        if losing_positions > 0 else None
    )

    win_loss_ratio = (
        float(abs(average_win / average_loss))
        if average_win is not None and average_loss is not None and average_loss != 0
        else None
    )

    return WinRateResult(
        win_rate=win_rate,
        total_positions=total_positions,
        winning_positions=winning_positions,
        losing_positions=losing_positions,
        breakeven_positions=breakeven_positions,
        total_gain_loss=total_gain_loss,
        average_win=average_win,
        average_loss=average_loss,
        win_loss_ratio=win_loss_ratio
    )


def calculate_win_rate_by_symbol(
    portfolio: Portfolio,
    as_of: datetime | None = None,
    include_open: bool = False
) -> dict[str, WinRateResult]:
    """
    Calculate win rate statistics grouped by symbol.

    Args:
        portfolio: Portfolio object containing transactions.
        as_of: Only consider transactions up to this datetime.
               If None, considers all transactions.
        include_open: If True, includes unrealized gains from open positions.

    Returns:
        Dictionary mapping symbol to WinRateResult.
    """
    if as_of is None and portfolio.transactions:
        sorted_txns = sorted(portfolio.transactions, key=lambda t: t.transaction_datetime)
        as_of = sorted_txns[-1].transaction_datetime

    closed_positions = get_closed_positions(portfolio, as_of)

    # Group closed positions by symbol
    by_symbol: dict[str, list[Decimal]] = {}
    for pos in closed_positions:
        if pos.symbol not in by_symbol:
            by_symbol[pos.symbol] = []
        by_symbol[pos.symbol].append(pos.realized_gain_loss)

    # Add open positions if requested
    if include_open and as_of is not None:
        open_positions = get_positions(as_of, portfolio)
        for pos in open_positions:
            if pos.gain_loss is not None:
                if pos.symbol not in by_symbol:
                    by_symbol[pos.symbol] = []
                by_symbol[pos.symbol].append(pos.gain_loss)

    # Calculate win rate for each symbol
    results: dict[str, WinRateResult] = {}
    for symbol, gains_losses in by_symbol.items():
        if not gains_losses:
            continue

        winning = [gl for gl in gains_losses if gl > 0]
        losing = [gl for gl in gains_losses if gl < 0]
        breakeven = [gl for gl in gains_losses if gl == 0]

        total_positions = len(gains_losses)
        winning_positions = len(winning)
        losing_positions = len(losing)
        breakeven_positions = len(breakeven)

        win_rate = winning_positions / total_positions if total_positions > 0 else 0.0

        total_gain_loss = sum(gains_losses)

        average_win = (
            sum(winning) / winning_positions
            if winning_positions > 0 else None
        )

        average_loss = (
            sum(losing) / losing_positions
            if losing_positions > 0 else None
        )

        win_loss_ratio = (
            float(abs(average_win / average_loss))
            if average_win is not None and average_loss is not None and average_loss != 0
            else None
        )

        results[symbol] = WinRateResult(
            win_rate=win_rate,
            total_positions=total_positions,
            winning_positions=winning_positions,
            losing_positions=losing_positions,
            breakeven_positions=breakeven_positions,
            total_gain_loss=total_gain_loss,
            average_win=average_win,
            average_loss=average_loss,
            win_loss_ratio=win_loss_ratio
        )

    return results
