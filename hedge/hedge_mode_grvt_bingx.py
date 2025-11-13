"""
Hedging bot that provides liquidity on GRVT and hedges exposure on BingX.
"""

import asyncio
import logging
import os
import signal
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Optional

from exchanges.grvt import GrvtClient
from exchanges.bingx import BingxClient


class Config:
    """Simple configuration wrapper."""

    def __init__(self, data: Dict[str, Any]):
        for key, value in data.items():
            setattr(self, key, value)


class HedgeBot:
    """Hedge GRVT post-only fills with BingX market orders."""

    def __init__(
        self,
        ticker: str,
        order_quantity: Decimal,
        fill_timeout: int = 5,
        iterations: int = 10,
        sleep_time: int = 0,
    ):
        self.ticker = ticker.upper()
        self.order_quantity = order_quantity
        self.fill_timeout = fill_timeout
        self.iterations = iterations
        self.sleep_time = sleep_time

        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.shutdown_requested = False

        self.grvt_client: Optional[GrvtClient] = None
        self.bingx_client: Optional[BingxClient] = None
        self.grvt_contract_id: str = ""
        self.bingx_contract_id: str = ""
        self.grvt_tick_size: Decimal = Decimal("0")
        self.bingx_tick_size: Decimal = Decimal("0")

        self.fill_queue: asyncio.Queue = asyncio.Queue()
        self._order_fill_tracker: Dict[str, Decimal] = {}
        self.bingx_position: Decimal = Decimal("0")

        self.logger = self._setup_logger()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def run(self) -> None:
        """Run hedging loop."""
        self.loop = asyncio.get_running_loop()
        self._setup_signal_handlers()

        try:
            await self._initialize_clients()
            await self._trading_loop()
        except asyncio.CancelledError:
            self.logger.info("Shutdown requested - cancelling tasks")
        except Exception as exc:
            self.logger.exception(f"Unexpected error in hedge bot: {exc}")
            raise
        finally:
            await self._shutdown()

    # ------------------------------------------------------------------ #
    # Initialization & shutdown
    # ------------------------------------------------------------------ #

    def _setup_logger(self) -> logging.Logger:
        os.makedirs("logs", exist_ok=True)
        logger = logging.getLogger(f"hedge_grvt_bingx_{self.ticker}")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        logger.propagate = False

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        log_file = os.path.join("logs", f"grvt_bingx_{self.ticker.lower()}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger

    def _setup_signal_handlers(self) -> None:
        """Register signal handlers for clean shutdown."""
        try:
            signal.signal(signal.SIGINT, self._request_shutdown)  # type: ignore[arg-type]
            signal.signal(signal.SIGTERM, self._request_shutdown)  # type: ignore[arg-type]
        except ValueError:
            # Signals are not supported on some platforms (e.g. Windows event loops, notebooks)
            self.logger.debug("Signal handling is not available in this environment")

    def _request_shutdown(self, signum, frame) -> None:
        """Signal handler."""
        if not self.shutdown_requested:
            self.logger.info("Shutdown signal received - stopping after current iteration")
            self.shutdown_requested = True

    async def _initialize_clients(self) -> None:
        """Instantiate and prepare exchange clients."""
        self.logger.info("Initialising GRVT and BingX clients")

        grvt_config = Config(
            {
                "ticker": self.ticker,
                "contract_id": "",
                "quantity": self.order_quantity,
                "tick_size": Decimal("0.01"),
                "direction": "buy",
                "close_order_side": "sell",
            }
        )
        self.grvt_client = GrvtClient(grvt_config)
        self.grvt_client.setup_order_update_handler(self._on_grvt_order_update)
        self.grvt_contract_id, self.grvt_tick_size = await self.grvt_client.get_contract_attributes()

        bingx_config = Config(
            {
                "ticker": self.ticker,
                "contract_id": "",
                "quantity": self.order_quantity,
                "tick_size": Decimal("0.01"),
                "direction": "buy",
                "close_order_side": "sell",
            }
        )
        self.bingx_client = BingxClient(bingx_config)
        self.bingx_client.setup_order_update_handler(self._on_bingx_order_update)
        self.bingx_contract_id, self.bingx_tick_size = await self.bingx_client.get_contract_attributes()

        await self.grvt_client.connect()
        await self.bingx_client.connect()

        self.logger.info(
            "Clients ready | GRVT contract %s (tick %s) | BingX contract %s (tick %s)",
            self.grvt_contract_id,
            self.grvt_tick_size,
            self.bingx_contract_id,
            self.bingx_tick_size,
        )

    async def _shutdown(self) -> None:
        """Disconnect exchange clients."""
        self.logger.info("Shutting down hedge bot")
        tasks = []
        if self.grvt_client:
            tasks.append(self.grvt_client.disconnect())
        if self.bingx_client:
            tasks.append(self.bingx_client.disconnect())

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # ------------------------------------------------------------------ #
    # Trading loop
    # ------------------------------------------------------------------ #

    async def _trading_loop(self) -> None:
        """Run alternating GRVT buy/sell cycles."""
        for iteration in range(1, self.iterations + 1):
            if self.shutdown_requested:
                self.logger.info("Shutdown requested - stopping before iteration %s", iteration)
                break

            self.logger.info("===== Iteration %s/%s =====", iteration, self.iterations)

            if not await self._execute_grvt_leg("buy"):
                break

            if self.sleep_time > 0:
                await asyncio.sleep(self.sleep_time)

            if not await self._execute_grvt_leg("sell"):
                break

            if self.sleep_time > 0:
                await asyncio.sleep(self.sleep_time)

    async def _execute_grvt_leg(self, side: str) -> bool:
        """Place GRVT order and hedge fills on BingX."""
        assert self.grvt_client is not None
        assert self.bingx_client is not None

        self._clear_fill_queue()

        self.logger.info("Placing GRVT post-only %s order for %s %s", side.upper(), self.order_quantity, self.ticker)
        order_result = await self.grvt_client.place_open_order(self.grvt_contract_id, self.order_quantity, side)

        if not order_result.success or not order_result.order_id:
            self.logger.error("Failed to place GRVT order: %s", order_result.error_message)
            return False

        self._order_fill_tracker[order_result.order_id] = Decimal("0")

        try:
            success = await self._consume_fills(order_result.order_id, side, self.order_quantity)
        finally:
            self._order_fill_tracker.pop(order_result.order_id, None)

        if not success:
            self.logger.warning("Hedging leg did not complete successfully, stopping bot")
            return False

        return True

    async def _consume_fills(self, order_id: str, side: str, expected_qty: Decimal) -> bool:
        """Consume fill events from queue and hedge each fill."""
        filled = Decimal("0")
        while filled < expected_qty:
            try:
                fill = await asyncio.wait_for(self.fill_queue.get(), timeout=self.fill_timeout)
            except asyncio.TimeoutError:
                self.logger.error("Timeout waiting for GRVT fill - cancelling order %s", order_id)
                await self.grvt_client.cancel_order(order_id)
                return False

            if fill.get("order_id") != order_id:
                # Unexpected fill (likely from a previous order); re-queue and continue
                self.logger.debug("Ignoring fill for order %s while waiting for %s", fill.get("order_id"), order_id)
                self.fill_queue.put_nowait(fill)
                await asyncio.sleep(0)
                continue

            quantity = fill.get("quantity", Decimal("0"))
            if quantity <= 0:
                continue

            filled += quantity
            await self._hedge_with_bingx(fill.get("side", side), quantity)

        return True

    async def _hedge_with_bingx(self, grvt_side: str, quantity: Decimal) -> None:
        """Place hedge order on BingX."""
        assert self.bingx_client is not None

        hedge_side = "sell" if grvt_side == "buy" else "buy"
        position_after = self._simulate_position_update(hedge_side, quantity)
        reduce_only = abs(position_after) < abs(self.bingx_position)

        self.logger.info(
            "Hedging on BingX: %s %s %s (reduce_only=%s)",
            hedge_side.upper(),
            quantity,
            self.ticker,
            reduce_only,
        )

        result = await self.bingx_client.place_market_order(
            self.bingx_contract_id,
            quantity,
            hedge_side,
            reduce_only=reduce_only,
        )

        if result.success:
            self.bingx_position = position_after
            self.logger.info(
                "BingX fill: %s %s @ %s | Position: %s",
                hedge_side.upper(),
                result.filled_size or quantity,
                result.price or "market",
                self.bingx_position,
            )
        else:
            self.logger.error("BingX hedge failed: %s", result.error_message or "unknown error")

    def _simulate_position_update(self, hedge_side: str, quantity: Decimal) -> Decimal:
        """Compute resulting position without mutating state."""
        if hedge_side == "buy":
            return self.bingx_position + quantity
        return self.bingx_position - quantity

    # ------------------------------------------------------------------ #
    # Event handlers
    # ------------------------------------------------------------------ #

    def _on_grvt_order_update(self, message: Dict[str, Any]) -> None:
        """Process GRVT order updates and push fills to queue."""
        try:
            contract_id = message.get("contract_id")
            if contract_id != self.grvt_contract_id:
                return

            order_id = message.get("order_id")
            status = message.get("status", "")
            side = (message.get("side") or "").lower()
            filled_size = Decimal(str(message.get("filled_size", "0")))

            if not order_id or filled_size <= 0:
                return

            previous = self._order_fill_tracker.get(order_id, Decimal("0"))
            delta = filled_size - previous
            if delta <= 0:
                return

            self._order_fill_tracker[order_id] = filled_size
            price = Decimal(str(message.get("price", "0")))

            fill_info = {
                "order_id": order_id,
                "side": side,
                "quantity": delta,
                "price": price,
                "status": status,
            }

            if self.loop:
                self.loop.call_soon_threadsafe(self.fill_queue.put_nowait, fill_info)
            else:
                self.fill_queue.put_nowait(fill_info)

            if status == "FILLED":
                self.logger.info("GRVT order %s filled: %s %s @ %s", order_id, side.upper(), filled_size, price)
        except (InvalidOperation, TypeError) as exc:
            self.logger.error("Failed to process GRVT order update: %s", exc)

    def _on_bingx_order_update(self, message: Dict[str, Any]) -> None:
        """Log BingX order updates (optional)."""
        status = message.get("status")
        order_type = message.get("order_type")
        order_id = message.get("order_id")

        if status in {"FILLED", "PARTIALLY_FILLED"}:
            self.logger.info(
                "BingX order update: %s %s %s %s @ %s",
                order_id,
                order_type,
                status,
                message.get("size"),
                message.get("price"),
            )

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #

    def _clear_fill_queue(self) -> None:
        while not self.fill_queue.empty():
            try:
                self.fill_queue.get_nowait()
            except asyncio.QueueEmpty:
                break


def parse_arguments():
    """CLI entry point compatibility (mirrors other hedge bots)."""
    import argparse

    parser = argparse.ArgumentParser(description="Hedge GRVT post-only fills with BingX market orders")
    parser.add_argument("--ticker", type=str, default="BTC", help="Ticker symbol (default: BTC)")
    parser.add_argument("--size", type=str, required=True, help="Order size per leg")
    parser.add_argument("--iter", type=int, required=True, help="Number of hedge iterations")
    parser.add_argument("--fill-timeout", type=int, default=5, help="Timeout (seconds) for GRVT fills")
    parser.add_argument("--sleep", type=int, default=0, help="Sleep time (seconds) between legs")
    return parser.parse_args()


async def main():
    args = parse_arguments()
    bot = HedgeBot(
        ticker=args.ticker,
        order_quantity=Decimal(args.size),
        fill_timeout=args.fill_timeout,
        iterations=args.iter,
        sleep_time=args.sleep,
    )
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
