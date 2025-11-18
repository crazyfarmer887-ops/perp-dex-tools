from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Dict, Optional, Sequence, Union

NumberLike = Union[str, float, int, Decimal]


def _to_decimal(value: NumberLike) -> Optional[Decimal]:
    try:
        decimal_value = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return decimal_value


def _normalize_side(side: str) -> Optional[str]:
    if not side:
        return None
    normalized = side.strip().lower()
    mapping = {
        'buy': 'buy',
        'sell': 'sell',
        'long': 'buy',
        'short': 'sell'
    }
    return mapping.get(normalized)


@dataclass
class PnlUpdate:
    exchange: str
    side: str
    price: Decimal
    quantity: Decimal
    realized_delta: Decimal
    exchange_realized: Decimal
    aggregated_realized: Decimal


class _ExchangePnlTracker:
    def __init__(self, name: str):
        self.name = name
        self.position = Decimal('0')
        self.avg_entry_price: Optional[Decimal] = None
        self.realized_pnl = Decimal('0')

    def apply_fill(self, side: str, price: Decimal, quantity: Decimal) -> Decimal:
        if quantity <= 0 or price <= 0:
            return Decimal('0')

        normalized_side = _normalize_side(side)
        if normalized_side is None:
            return Decimal('0')

        signed_qty = quantity if normalized_side == 'buy' else -quantity
        previous_position = self.position
        realized = Decimal('0')

        if previous_position == 0 or previous_position * signed_qty > 0:
            new_position = previous_position + signed_qty
            if previous_position == 0:
                self.avg_entry_price = price
            else:
                total = (self.avg_entry_price or Decimal('0')) * abs(previous_position) + price * abs(signed_qty)
                if new_position != 0:
                    self.avg_entry_price = total / abs(new_position)
                else:
                    self.avg_entry_price = Decimal('0')
            self.position = new_position
            return realized

        closing_qty = min(abs(signed_qty), abs(previous_position))
        reference_price = self.avg_entry_price or Decimal('0')
        if previous_position > 0:
            realized = (price - reference_price) * closing_qty
        else:
            realized = (reference_price - price) * closing_qty

        new_position = previous_position + signed_qty
        if abs(signed_qty) < abs(previous_position):
            self.position = new_position
        elif new_position == 0:
            self.position = Decimal('0')
            self.avg_entry_price = Decimal('0')
        else:
            self.position = new_position
            self.avg_entry_price = price

        return realized


class HedgedPnlTracker:
    """
    Tracks per-exchange realized PnL and their aggregated sum for hedge mode pairs.
    """

    def __init__(self, exchanges: Sequence[str], quote_symbol: str = 'USDT'):
        if not exchanges or len(exchanges) < 2:
            raise ValueError("HedgedPnlTracker requires at least two exchange names")

        cleaned_names = []
        for name in exchanges:
            if not name:
                continue
            stripped = name.strip()
            if stripped:
                cleaned_names.append(stripped)

        if len(cleaned_names) < 2:
            raise ValueError("HedgedPnlTracker requires at least two non-empty exchange names")

        self.quote_symbol = quote_symbol
        self._trackers: Dict[str, _ExchangePnlTracker] = {
            name: _ExchangePnlTracker(name) for name in cleaned_names
        }

    @property
    def aggregated_realized_pnl(self) -> Decimal:
        total = Decimal('0')
        for tracker in self._trackers.values():
            total += tracker.realized_pnl
        return total

    def get_exchange_realized(self, exchange: str) -> Optional[Decimal]:
        tracker = self._trackers.get(exchange)
        if tracker is None:
            return None
        return tracker.realized_pnl

    def record_fill(
        self,
        exchange: str,
        side: str,
        price: NumberLike,
        quantity: NumberLike
    ) -> Optional[PnlUpdate]:
        tracker = self._trackers.get(exchange)
        if tracker is None:
            return None

        normalized_side = _normalize_side(side)
        if normalized_side is None:
            return None

        price_decimal = _to_decimal(price)
        quantity_decimal = _to_decimal(quantity)
        if price_decimal is None or quantity_decimal is None:
            return None

        quantity_decimal = abs(quantity_decimal)
        if price_decimal <= 0 or quantity_decimal <= 0:
            return None

        realized_delta = tracker.apply_fill(normalized_side, price_decimal, quantity_decimal)
        tracker.realized_pnl += realized_delta

        return PnlUpdate(
            exchange=tracker.name,
            side=normalized_side,
            price=price_decimal,
            quantity=quantity_decimal,
            realized_delta=realized_delta,
            exchange_realized=tracker.realized_pnl,
            aggregated_realized=self.aggregated_realized_pnl
        )
