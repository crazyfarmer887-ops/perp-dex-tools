import argparse
import asyncio
import contextlib
import logging
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Optional, Tuple

from exchanges.bingx import BingxClient
from exchanges.grvt import GrvtClient


@dataclass
class SimpleConfig:
    """Lightweight configuration object for exchange clients."""

    ticker: str
    contract_id: str
    quantity: Decimal
    tick_size: Decimal
    direction: str = "buy"
    close_order_side: str = "sell"


class HedgeBot:
    """
    Hedge trading bot that opens maker orders on GRVT and hedges fills on BingX.

    Workflow for each iteration:
        1. Place maker BUY on GRVT, hedge with SELL market on BingX when filled.
        2. Place maker SELL (close) on GRVT, hedge with BUY market on BingX when filled.
    """

    def __init__(
        self,
        ticker: str,
        order_quantity: Decimal,
        fill_timeout: int = 10,
        iterations: int = 10,
        sleep_time: int = 0,
    ):
        self.ticker = ticker.upper()
        self.order_quantity = order_quantity
        self.fill_timeout = fill_timeout
        self.iterations = iterations
        self.sleep_time = sleep_time

        # logging
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        log_path = logs_dir / f"grvt_bingx_{self.ticker.lower()}_hedge.log"

        self.logger = logging.getLogger(f"grvt_bingx_{self.ticker.lower()}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.propagate = False

        # Exchange clients
        self.grvt_client: Optional[GrvtClient] = None
        self.bingx_client: Optional[BingxClient] = None

        self.grvt_config = SimpleConfig(
            ticker=self.ticker,
            contract_id="",
            quantity=self.order_quantity,
            tick_size=Decimal("0.01"),
            direction="buy",
            close_order_side="sell",
        )

        self.bingx_config = SimpleConfig(
            ticker=self.ticker,
            contract_id="",
            quantity=self.order_quantity,
            tick_size=Decimal("0.01"),
            direction="sell",
            close_order_side="buy",
        )

        # Order tracking
        self._grvt_fill_event = asyncio.Event()
        self._pending_grvt_order_id: Optional[str] = None
        self._pending_grvt_status: Optional[str] = None
        self._pending_grvt_fill_size: Decimal = Decimal("0")
        self._pending_grvt_fill_price: Decimal = Decimal("0")

    # ------------------------------------------------------------------ #
    # Initialization
    # ------------------------------------------------------------------ #
    async def _initialize_clients(self) -> None:
        self.logger.info("Initializing GRVT and BingX clients...")

        self.grvt_client = GrvtClient(self.grvt_config)
        self.bingx_client = BingxClient(self.bingx_config)

        # Fetch contract metadata
        grvt_contract, grvt_tick = await self.grvt_client.get_contract_attributes()
        self.grvt_config.contract_id = grvt_contract
        self.grvt_config.tick_size = grvt_tick

        bingx_contract, bingx_tick = await self.bingx_client.get_contract_attributes()
        self.bingx_config.contract_id = bingx_contract
        self.bingx_config.tick_size = bingx_tick

        # Setup callbacks before connecting
        self.grvt_client.setup_order_update_handler(self._handle_grvt_order_update)

        await self.grvt_client.connect()
        await self.bingx_client.connect()

        # Allow websocket to subscribe
        await asyncio.sleep(2)

        self.logger.info("Clients initialized successfully.")

    # ------------------------------------------------------------------ #
    # GRVT callbacks
    # ------------------------------------------------------------------ #
    def _handle_grvt_order_update(self, message):
        if not self._pending_grvt_order_id:
            return

        if message.get("order_id") != self._pending_grvt_order_id:
            return

        status = message.get("status", "")
        filled_size = Decimal(str(message.get("filled_size", "0")))
        price = Decimal(str(message.get("price", "0")))

        self.logger.info(
            "[GRVT] Order %s status %s | filled=%s @ %s",
            self._pending_grvt_order_id,
            status,
            filled_size,
            price,
        )

        self._pending_grvt_status = status

        if status == "FILLED":
            self._pending_grvt_fill_size = filled_size
            self._pending_grvt_fill_price = price
            self._grvt_fill_event.set()
        elif status == "CANCELED":
            self._grvt_fill_event.set()

    # ------------------------------------------------------------------ #
    # Order helpers
    # ------------------------------------------------------------------ #
    async def _wait_for_grvt_fill(self) -> Tuple[Decimal, Decimal]:
        try:
            await asyncio.wait_for(self._grvt_fill_event.wait(), timeout=self.fill_timeout)
        except asyncio.TimeoutError:
            self.logger.warning("[GRVT] Fill timeout reached, cancelling order %s", self._pending_grvt_order_id)
            if self._pending_grvt_order_id:
                await self.grvt_client.cancel_order(self._pending_grvt_order_id)
            raise

        if self._pending_grvt_status != "FILLED":
            raise RuntimeError(f"GRVT order not filled (status: {self._pending_grvt_status})")

        return self._pending_grvt_fill_size, self._pending_grvt_fill_price

    async def _place_grvt_open(self, side: str) -> Tuple[Decimal, Decimal]:
        assert self.grvt_client is not None

        self._grvt_fill_event.clear()
        self._pending_grvt_status = None
        self._pending_grvt_fill_price = Decimal("0")
        self._pending_grvt_fill_size = Decimal("0")

        # Update config for direction
        self.grvt_config.direction = side
        self.grvt_config.close_order_side = "buy" if side == "sell" else "sell"

        order_result = await self.grvt_client.place_open_order(
            self.grvt_config.contract_id,
            self.order_quantity,
            side,
        )

        if not order_result.success:
            raise RuntimeError(f"Failed to place GRVT open order: {order_result.error_message}")

        self._pending_grvt_order_id = order_result.order_id
        if not self._pending_grvt_order_id:
            raise RuntimeError("GRVT open order did not return an order id")
        if order_result.status == "FILLED":
            self._pending_grvt_status = "FILLED"
            self._pending_grvt_fill_size = order_result.size or self.order_quantity
            self._pending_grvt_fill_price = order_result.price or Decimal("0")
            self._grvt_fill_event.set()

        return await self._wait_for_grvt_fill()

    async def _place_grvt_close(self, side: str) -> Tuple[Decimal, Decimal]:
        assert self.grvt_client is not None

        self._grvt_fill_event.clear()
        self._pending_grvt_status = None
        self._pending_grvt_fill_price = Decimal("0")
        self._pending_grvt_fill_size = Decimal("0")

        self.grvt_config.close_order_side = side

        target_price = await self.grvt_client.get_order_price(side)

        order_result = await self.grvt_client.place_close_order(
            self.grvt_config.contract_id,
            self.order_quantity,
            target_price,
            side,
        )

        if not order_result.success:
            raise RuntimeError(f"Failed to place GRVT close order: {order_result.error_message}")

        self._pending_grvt_order_id = order_result.order_id
        if not self._pending_grvt_order_id:
            raise RuntimeError("GRVT close order did not return an order id")
        if order_result.status == "FILLED":
            self._pending_grvt_status = "FILLED"
            self._pending_grvt_fill_size = order_result.size or self.order_quantity
            self._pending_grvt_fill_price = order_result.price or target_price
            self._grvt_fill_event.set()

        return await self._wait_for_grvt_fill()

    async def _hedge_on_bingx(self, side: str, quantity: Decimal) -> None:
        assert self.bingx_client is not None

        result = await self.bingx_client.place_market_order(
            self.bingx_config.contract_id,
            quantity,
            side,
        )

        if not result.success:
            raise RuntimeError(f"BingX hedge order failed: {result.error_message}")

        self.logger.info(
            "[BINGX] Market %s %s hedged @ %s",
            side.upper(),
            quantity,
            result.price or Decimal("0"),
        )

    # ------------------------------------------------------------------ #
    # Execution flow
    # ------------------------------------------------------------------ #
    async def _run_iteration(self, iteration: int) -> None:

        self.logger.info("========== Iteration %d ==========", iteration)

        # STEP 1: Open long on GRVT, hedge with short on BingX
        self.logger.info("Step 1: Opening GRVT long and hedging on BingX (sell).")
        filled_qty, fill_price = await self._place_grvt_open("buy")
        self.logger.info("[GRVT] Filled BUY %s @ %s", filled_qty, fill_price)

        await self._hedge_on_bingx("sell", filled_qty)

        if self.sleep_time:
            self.logger.info("Sleeping %s seconds before closing leg...", self.sleep_time)
            await asyncio.sleep(self.sleep_time)

        # STEP 2: Close long on GRVT, hedge with buy on BingX
        self.logger.info("Step 2: Closing GRVT long and hedging on BingX (buy).")
        close_qty, close_price = await self._place_grvt_close("sell")
        self.logger.info("[GRVT] Filled SELL %s @ %s", close_qty, close_price)

        await self._hedge_on_bingx("buy", close_qty)

    async def run(self) -> None:
        try:
            await self._initialize_clients()
            for idx in range(1, self.iterations + 1):
                await self._run_iteration(idx)
        except Exception as exc:
            self.logger.error("Hedge bot encountered an error: %s", exc)
            raise
        finally:
            await self._shutdown()

    async def _shutdown(self) -> None:
        self.logger.info("Shutting down hedge bot...")
        if self.grvt_client:
            with contextlib.suppress(Exception):
                await self.grvt_client.disconnect()
        if self.bingx_client:
            with contextlib.suppress(Exception):
                await self.bingx_client.disconnect()
        self.logger.info("Shutdown complete.")


def parse_arguments():
    parser = argparse.ArgumentParser(description="GRVT + BingX hedge trading bot")
    parser.add_argument("--ticker", type=str, default="BTC", help="Ticker symbol (default: BTC)")
    parser.add_argument("--size", type=str, required=True, help="Order size (base asset)")
    parser.add_argument("--iter", type=int, required=True, help="Number of iterations to run")
    parser.add_argument("--fill-timeout", type=int, default=10, help="Timeout for GRVT fill (seconds)")
    parser.add_argument("--sleep", type=int, default=0, help="Sleep time between open/close legs (seconds)")

    return parser.parse_args()


async def main():
    args = parse_arguments()
    quantity = Decimal(args.size)

    bot = HedgeBot(
        ticker=args.ticker,
        order_quantity=quantity,
        fill_timeout=args.fill_timeout,
        iterations=args.iter,
        sleep_time=args.sleep,
    )

    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
