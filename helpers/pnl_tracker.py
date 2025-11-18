"""
PnL tracking utilities for hedge mode bots.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, Iterable, Optional


def _normalize_decimal(value: Any) -> Decimal:
    """
    Helper to ensure Decimal objects always have a consistent context.
    """
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


@dataclass(frozen=True)
class PnLUpdate:
    """Represents a single PnL update event."""
    exchange: str
    delta: Decimal
    per_exchange: Dict[str, Decimal]
    net: Decimal
    position: Decimal
    avg_entry_price: Decimal


class TradePnLTracker:
    """
    Tracks realized PnL for a single exchange by maintaining a running position
    and average entry price.
    """

    def __init__(self, name: str):
        self.name = name
        self.position = Decimal('0')
        self.avg_entry_price = Decimal('0')
        self.realized_pnl = Decimal('0')

    def reset(self) -> None:
        self.position = Decimal('0')
        self.avg_entry_price = Decimal('0')
        self.realized_pnl = Decimal('0')

    def record_fill(self, side: str, size: Decimal, price: Decimal) -> Decimal:
        """
        Record a trade fill and return the realized PnL delta produced by the fill.
        """
        if not side:
            return Decimal('0')

        normalized_side = side.strip().lower()
        if normalized_side not in {'buy', 'sell'}:
            return Decimal('0')

        if size <= 0 or price <= 0:
            return Decimal('0')

        signed_size = size if normalized_side == 'buy' else -size
        remaining = signed_size
        realized_delta = Decimal('0')

        while remaining != 0:
            if self.position == 0 or self.position * remaining > 0:
                # Increasing existing position (or opening a new one)
                abs_pos = abs(self.position)
                abs_remaining = abs(remaining)
                new_total = abs_pos + abs_remaining
                weighted_notional = (self.avg_entry_price * abs_pos) + (price * abs_remaining)
                self.avg_entry_price = weighted_notional / new_total
                self.position += remaining
                remaining = Decimal('0')
            else:
                # Closing existing position partially or fully
                abs_pos = abs(self.position)
                abs_remaining = abs(remaining)
                closing_qty = min(abs_pos, abs_remaining)
                direction = Decimal('1') if self.position > 0 else Decimal('-1')
                realized_delta += closing_qty * (price - self.avg_entry_price) * direction

                if self.position > 0:
                    self.position -= closing_qty
                else:
                    self.position += closing_qty

                if remaining > 0:
                    remaining -= closing_qty
                else:
                    remaining += closing_qty

                if self.position == 0:
                    self.avg_entry_price = Decimal('0')

        self.realized_pnl += realized_delta
        return realized_delta


class HedgePnLAggregator:
    """
    Aggregates realized PnL across multiple exchanges involved in a hedge.
    """

    def __init__(self, *exchange_names: str):
        if not exchange_names:
            raise ValueError("At least one exchange name must be provided for PnL tracking.")
        self._trackers: Dict[str, TradePnLTracker] = {
            name: TradePnLTracker(name) for name in exchange_names
        }

    def record_fill(
        self,
        exchange: str,
        side: str,
        size: Decimal,
        price: Decimal
    ) -> Optional[PnLUpdate]:
        tracker = self._trackers.get(exchange)
        if tracker is None:
            return None

        size_dec = _normalize_decimal(size)
        price_dec = _normalize_decimal(price)
        pnl_delta = tracker.record_fill(side, size_dec, price_dec)
        per_exchange_totals = {name: t.realized_pnl for name, t in self._trackers.items()}
        net_total = sum(per_exchange_totals.values())

        return PnLUpdate(
            exchange=exchange,
            delta=pnl_delta,
            per_exchange=per_exchange_totals,
            net=net_total,
            position=tracker.position,
            avg_entry_price=tracker.avg_entry_price
        )

    def totals(self) -> Dict[str, Decimal]:
        per_exchange = {name: tracker.realized_pnl for name, tracker in self._trackers.items()}
        per_exchange['net'] = sum(per_exchange.values())
        return per_exchange

    def exchanges(self) -> Iterable[str]:
        return self._trackers.keys()

    def reset(self) -> None:
        for tracker in self._trackers.values():
            tracker.reset()
