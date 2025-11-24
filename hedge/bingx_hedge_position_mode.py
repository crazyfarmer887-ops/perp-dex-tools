#!/usr/bin/env python3
"""
BingX hedge positioning helper.

This utility opens both long and short positions at market for the same contract size
and immediately places symmetric TP/SL limit orders around the blended average price.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import List, Optional, Tuple

import dotenv


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from exchanges.bingx import BingxClient  # noqa: E402


@dataclass
class OrderSnapshot:
    """Lightweight container for filled order data."""

    side: str
    avg_price: Decimal
    filled_size: Decimal


class Config:
    """Attribute-style wrapper expected by exchange clients."""

    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)


def _parse_decimal(value: str, label: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{label} '{value}' is not a valid decimal: {exc}") from exc
    if parsed <= 0:
        raise ValueError(f"{label} must be positive (got {parsed}).")
    return parsed


class BingxHedgePositionMode:
    """Core workflow for hedged BingX positioning with symmetric TP/SL targets."""

    def __init__(
        self,
        ticker: str,
        quantity: Decimal,
        roi_percent: Decimal,
        leverage: Decimal,
        quote_asset: str = "USDT",
        allow_non_reduce_only: bool = False,
    ):
        self.ticker = ticker.upper()
        self.quantity = quantity
        self.roi_percent = roi_percent
        self.leverage = leverage
        self.quote_asset = quote_asset.upper()
        if not self.quote_asset:
            raise ValueError("quote_asset must be provided (e.g., USDT or VST).")
        self.allow_non_reduce_only = allow_non_reduce_only

        self.logger = logging.getLogger("bingx_hedge_position")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
            self.logger.addHandler(handler)

        self.bingx_client: Optional[BingxClient] = None
        self.contract_id: Optional[str] = None
        self.tick_size: Decimal = Decimal("0.01")

    def _build_config(self) -> Config:
        return Config(
            {
                "ticker": self.ticker,
                "contract_id": "",
                "quantity": self.quantity,
                "tick_size": Decimal("0.01"),
                "direction": "buy",
                "close_order_side": "sell",
                "quote_asset": self.quote_asset,
            }
        )

    async def execute(self) -> None:
        """Entry point used by the CLI wrapper."""
        self.logger.info(
            "Initializing BingX hedge position mode (ticker=%s, quote=%s, qty=%s)",
            self.ticker,
            self.quote_asset,
            self.quantity,
        )

        self.bingx_client = BingxClient(self._build_config())

        try:
            await self._connect()
            long_order, short_order = await self._enter_both_sides()
            await self._place_tp_sl_orders(long_order, short_order)
            self.logger.info("✅ Completed hedge setup. Monitor TP/SL orders directly on BingX.")
        finally:
            await self._disconnect()

    async def _connect(self) -> None:
        assert self.bingx_client is not None
        await self.bingx_client.connect()
        contract_id, tick_size = await self.bingx_client.get_contract_attributes()
        self.contract_id = contract_id
        self.tick_size = tick_size
        self.bingx_client.config.contract_id = contract_id
        self.bingx_client.config.tick_size = tick_size
        self.logger.info(
            "Connected to BingX | contract=%s | quote=%s | tick_size=%s",
            contract_id,
            self.quote_asset,
            tick_size,
        )

    async def _disconnect(self) -> None:
        if self.bingx_client:
            try:
                await self.bingx_client.disconnect()
            except Exception:
                pass

    async def _enter_both_sides(self) -> Tuple[OrderSnapshot, OrderSnapshot]:
        """
        Open hedge by placing BUY and SELL market orders sequentially.
        Returns filled order snapshots for both directions.
        """
        assert self.bingx_client is not None and self.contract_id is not None

        long_order = await self._place_market_order("buy", "sell")
        short_order = await self._place_market_order("sell", "buy")

        self.logger.info(
            "Opened hedge | long %s @ %s | short %s @ %s",
            long_order.filled_size,
            long_order.avg_price,
            short_order.filled_size,
            short_order.avg_price,
        )

        return long_order, short_order

    async def _place_market_order(self, side: str, close_side: str) -> OrderSnapshot:
        assert self.bingx_client is not None and self.contract_id is not None

        self.bingx_client.config.direction = side
        self.bingx_client.config.close_order_side = close_side

        result = await self.bingx_client.place_market_order(self.contract_id, self.quantity, side)
        if not result.success or result.price is None:
            raise RuntimeError(f"Failed to place {side.upper()} market order: {result.error_message}")

        filled = result.filled_size or self.quantity
        if filled <= 0:
            raise RuntimeError(f"BingX returned non-positive fill size for {side.upper()} order.")

        price = Decimal(str(result.price))
        return OrderSnapshot(side=side, avg_price=price, filled_size=filled)

    def _compute_price_levels(self, long_avg: Decimal, short_avg: Decimal) -> Tuple[Decimal, Decimal, Decimal]:
        """
        Calculate blended mid price and symmetric TP/SL levels expressed as actual prices.
        """
        assert self.bingx_client is not None
        if long_avg <= 0 or short_avg <= 0:
            raise ValueError("Average prices must be positive to compute TP/SL.")

        base_price = (long_avg + short_avg) / Decimal("2")
        roi_fraction = (self.roi_percent / Decimal("100")) / self.leverage
        if roi_fraction <= 0:
            raise ValueError("ROI fraction must be positive.")

        raw_delta = base_price * roi_fraction
        delta = raw_delta.quantize(self.tick_size, rounding=ROUND_HALF_UP)
        if delta <= 0:
            delta = self.tick_size

        upper = self.bingx_client.round_to_tick(base_price + delta)
        lower = self.bingx_client.round_to_tick(base_price - delta)
        if lower <= 0:
            raise ValueError("Computed lower TP/SL level is non-positive. Check inputs.")

        self.logger.info(
            "Mid price %s | ROI target %s%% @ %sx | Δ=%s | upper=%s | lower=%s",
            base_price,
            self.roi_percent,
            self.leverage,
            delta,
            upper,
            lower,
        )

        return base_price, lower, upper

    def _is_reduce_only_rejection(self, error_message: Optional[str]) -> bool:
        if not error_message:
            return False
        lowered = error_message.lower()
        return "reduce only order" in lowered or '"code":101290' in lowered or "101290" in lowered

    async def _submit_tp_sl_order(
        self,
        side: str,
        price: Decimal,
        size: Decimal,
        label: str,
        reduce_only: bool,
    ):
        assert self.bingx_client is not None and self.contract_id is not None

        return await self.bingx_client.place_limit_order(
            contract_id=self.contract_id,
            quantity=size,
            side=side,
            price=price,
            reduce_only=reduce_only,
            post_only=False,
            time_in_force="GTC",
        )

    async def _place_tp_sl_orders(self, long_order: OrderSnapshot, short_order: OrderSnapshot) -> None:
        assert self.bingx_client is not None and self.contract_id is not None

        _, lower_price, upper_price = self._compute_price_levels(long_order.avg_price, short_order.avg_price)

        orders: List[Tuple[str, Decimal, Decimal, str]] = [
            ("sell", upper_price, long_order.filled_size, "LONG_TP"),
            ("sell", lower_price, long_order.filled_size, "LONG_SL"),
            ("buy", lower_price, short_order.filled_size, "SHORT_TP"),
            ("buy", upper_price, short_order.filled_size, "SHORT_SL"),
        ]

        for side, price, size, label in orders:
            if size <= 0:
                self.logger.warning("Skipping %s because size is non-positive (%s).", label, size)
                continue

            result = await self._submit_tp_sl_order(side, price, size, label, reduce_only=True)
            used_reduce_only = True

            if not result.success and self._is_reduce_only_rejection(result.error_message):
                if self.allow_non_reduce_only:
                    self.logger.warning(
                        "%s rejected due to reduce-only constraint; retrying without reduceOnly flag.",
                        label,
                    )
                    result = await self._submit_tp_sl_order(side, price, size, label, reduce_only=False)
                    used_reduce_only = False
                else:
                    raise RuntimeError(
                        f"Reduce-only order for {label} was rejected. Verify that your BingX account is in Hedge Mode "
                        f"or enable --allow-non-reduce-only to retry without reduceOnly."
                    )

            if result.success:
                self.logger.info(
                    "Placed %s | %s %s @ %s | reduce_only=%s | order_id=%s",
                    label,
                    side.upper(),
                    size,
                    price,
                    "Y" if used_reduce_only else "N",
                    result.order_id,
                )
            else:
                raise RuntimeError(f"Failed to place {label} order: {result.error_message}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open BingX hedge positions and set TP/SL orders based on ROI/leverage target."
    )
    parser.add_argument("--ticker", type=str, default="BTC", help="Contract ticker (default: BTC).")
    parser.add_argument("--size", type=str, required=True, help="Per-side contract size, e.g., 0.05")
    parser.add_argument(
        "--roi-percent",
        type=str,
        required=True,
        help="Target ROI percentage (e.g., 10 for 10%%) measured on leveraged PnL.",
    )
    parser.add_argument(
        "--leverage",
        type=str,
        required=True,
        help="Effective leverage to convert ROI to actual price move (e.g., 4 for 4x).",
    )
    parser.add_argument(
        "--quote-asset",
        type=str,
        default="USDT",
        help="Quote asset for the BingX contract (e.g., USDT, VST). Default: USDT.",
    )
    parser.add_argument(
        "--allow-non-reduce-only",
        action="store_true",
        help="If set, retry TP/SL orders without reduceOnly when BingX rejects them (useful for one-way accounts).",
    )
    parser.add_argument("--env-file", type=str, default=".env", help="Path to env file with API credentials.")
    return parser.parse_args()


async def _run_from_cli() -> None:
    args = parse_arguments()
    env_path = Path(args.env_file)
    if not env_path.exists():
        raise FileNotFoundError(f"Env file not found: {env_path}")
    dotenv.load_dotenv(env_path)

    quantity = _parse_decimal(args.size, "size")
    roi_percent = _parse_decimal(args.roi_percent, "roi-percent")
    leverage = _parse_decimal(args.leverage, "leverage")

    bot = BingxHedgePositionMode(
        ticker=args.ticker,
        quantity=quantity,
        roi_percent=roi_percent,
        leverage=leverage,
        quote_asset=args.quote_asset,
        allow_non_reduce_only=args.allow_non_reduce_only,
    )
    await bot.execute()


def main() -> None:
    try:
        asyncio.run(_run_from_cli())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as exc:
        print(f"Error: {exc}")
        raise


if __name__ == "__main__":
    main()
