import argparse
import asyncio
import logging
import os
import signal
import sys
import traceback
from decimal import Decimal
from typing import Any, Dict, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exchanges.grvt import GrvtClient  # noqa: E402
from exchanges.bingx import BingxClient  # noqa: E402


class Config:
    """Simple config wrapper to provide attribute access."""

    def __init__(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            setattr(self, key, value)


class HedgeBot:
    """Trading bot that places post-only orders on GRVT and hedges with market orders on BingX."""

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

        self.stop_flag = False
        self.grvt_client: Optional[GrvtClient] = None
        self.bingx_client: Optional[BingxClient] = None
        self.grvt_contract_id: Optional[str] = None
        self.bingx_contract_id: Optional[str] = None
        self.grvt_tick_size: Optional[Decimal] = None

        self.grvt_position = Decimal("0")
        self.bingx_position = Decimal("0")

        self.cycle_complete: Optional[asyncio.Event] = None
        self.hedge_lock = asyncio.Lock()
        self.active_cycle: Optional[Dict[str, Any]] = None

        os.makedirs("logs", exist_ok=True)
        self.log_filename = f"logs/grvt_bingx_{self.ticker}_hedge_mode.log"
        self.logger = logging.getLogger(f"hedge_grvt_bingx_{self.ticker}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        file_handler = logging.FileHandler(self.log_filename)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        file_handler.setLevel(logging.INFO)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter("%(levelname)s:%(message)s"))
        console_handler.setLevel(logging.INFO)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.propagate = False

    def setup_signal_handlers(self):
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

    def shutdown(self, signum=None, frame=None):
        if self.stop_flag:
            return

        self.stop_flag = True
        self.logger.info("\n🛑 Stopping hedge bot (signal: %s)", signum)
        if self.cycle_complete and not self.cycle_complete.is_set():
            self.cycle_complete.set()
        self.active_cycle = None

    async def initialize_grvt_client(self):
        config_dict = {
            "ticker": self.ticker,
            "contract_id": "",
            "quantity": self.order_quantity,
            "tick_size": Decimal("0.01"),
            "close_order_side": "sell",
        }
        config = Config(config_dict)
        try:
            self.grvt_client = GrvtClient(config)
        except Exception as exc:
            self.logger.error("❌ Failed to initialize GRVT client: %s", exc)
            raise

        self.logger.info("✅ GRVT client initialized")
        self.grvt_contract_id, self.grvt_tick_size = await self.grvt_client.get_contract_attributes()
        self.logger.info("✅ GRVT contract loaded: %s (tick: %s)", self.grvt_contract_id, self.grvt_tick_size)

    async def initialize_bingx_client(self):
        config_dict = {
            "ticker": self.ticker,
            "contract_id": "",
            "quantity": self.order_quantity,
            "tick_size": Decimal("0.01"),
            "close_order_side": "buy",
        }
        config = Config(config_dict)
        try:
            self.bingx_client = BingxClient(config)
        except Exception as exc:
            self.logger.error("❌ Failed to initialize BingX client: %s", exc)
            raise

        self.logger.info("✅ BingX client initialized")
        self.bingx_contract_id, _ = await self.bingx_client.get_contract_attributes()
        self.logger.info("✅ BingX contract loaded: %s", self.bingx_contract_id)

    async def setup_grvt_websocket(self):
        if not self.grvt_client or not self.grvt_contract_id:
            raise RuntimeError("GRVT client not initialized")

        def order_update_handler(order_data: Dict[str, Any]):
            if order_data.get("contract_id") != self.grvt_contract_id:
                return

            order_id = order_data.get("order_id")
            status = order_data.get("status")
            side = order_data.get("side")
            filled_size = order_data.get("filled_size")
            price = order_data.get("price")

            self.logger.info(
                "[GRVT] %s %s -> %s (filled=%s @ %s)",
                order_id,
                side,
                status,
                filled_size,
                price,
            )

            if not self.active_cycle:
                return

            if order_id != self.active_cycle.get("order_id"):
                return

            if status == "FILLED":
                asyncio.create_task(self._on_grvt_order_filled(order_data))
            elif status in {"CANCELED", "REJECTED"}:
                if self.cycle_complete and not self.cycle_complete.is_set():
                    self.cycle_complete.set()
                    self.active_cycle = None

        self.grvt_client.setup_order_update_handler(order_update_handler)
        await self.grvt_client.connect()
        self.logger.info("✅ GRVT WebSocket connected")

    async def _on_grvt_order_filled(self, order_data: Dict[str, Any]):
        async with self.hedge_lock:
            if not self.active_cycle or not self.bingx_client or not self.bingx_contract_id:
                return

            quantity = self._to_decimal(
                order_data.get("filled_size") or order_data.get("size") or self.order_quantity
            )
            side = order_data.get("side", "").lower()

            if quantity <= 0:
                self.logger.warning("⚠️ Received non-positive fill quantity, skipping hedge")
                if self.cycle_complete and not self.cycle_complete.is_set():
                    self.cycle_complete.set()
                    self.active_cycle = None
                return

            if side == "buy":
                self.grvt_position += quantity
            elif side == "sell":
                self.grvt_position -= quantity

            hedge_side = self.active_cycle.get("hedge_side")
            reduce_only = self.active_cycle.get("reduce_only", False)

            self.logger.info(
                "[STEP] Hedging on BingX -> %s %s (reduce_only=%s)",
                hedge_side,
                quantity,
                reduce_only,
            )

            result = await self.bingx_client.place_market_order(
                contract_id=self.bingx_contract_id,
                quantity=quantity,
                direction=hedge_side,
                reduce_only=reduce_only,
            )

            if result.success:
                if hedge_side == "buy":
                    self.bingx_position += quantity
                else:
                    self.bingx_position -= quantity

                self.logger.info(
                    "[BINGX] %s %s filled @ %s (order=%s)",
                    hedge_side.upper(),
                    quantity,
                    result.price,
                    result.order_id,
                )
            else:
                self.logger.error(
                    "❌ BingX hedge order failed: %s",
                    result.error_message,
                )

            if self.cycle_complete and not self.cycle_complete.is_set():
                self.cycle_complete.set()
            self.active_cycle = None

    async def trading_loop(self):
        self.logger.info("🚀 Starting GRVT ↔ BingX hedge bot for %s", self.ticker)

        await self.initialize_grvt_client()
        await self.initialize_bingx_client()

        if self.grvt_client:
            await self.setup_grvt_websocket()

        if self.bingx_client:
            await self.bingx_client.connect()

        self.cycle_complete = asyncio.Event()

        for iteration in range(1, self.iterations + 1):
            if self.stop_flag:
                break

            self.logger.info("===== Iteration %s / %s =====", iteration, self.iterations)

            await self.execute_cycle("buy", reduce_only=False)
            if self.stop_flag:
                break

            if self.sleep_time > 0:
                await asyncio.sleep(self.sleep_time)

            await self.execute_cycle("sell", reduce_only=True)

            if self.sleep_time > 0:
                await asyncio.sleep(self.sleep_time)

    async def execute_cycle(self, grvt_side: str, reduce_only: bool):
        if not self.grvt_client or not self.grvt_contract_id or self.stop_flag:
            return

        hedge_side = "sell" if grvt_side == "buy" else "buy"
        self.active_cycle = {
            "grvt_side": grvt_side,
            "hedge_side": hedge_side,
            "reduce_only": reduce_only,
            "order_id": None,
        }

        if self.cycle_complete:
            self.cycle_complete.clear()

        try:
            self.logger.info("[GRVT] Placing %s order for %s", grvt_side.upper(), self.order_quantity)
            order_result = await self.grvt_client.place_open_order(
                contract_id=self.grvt_contract_id,
                quantity=self.order_quantity,
                direction=grvt_side,
            )
        except Exception as exc:
            self.logger.error("❌ Failed to place GRVT order: %s", exc)
            self.active_cycle = None
            return

        if not order_result.success:
            self.logger.error("❌ GRVT order rejected: %s", order_result.error_message)
            self.active_cycle = None
            return

        self.active_cycle["order_id"] = order_result.order_id
        self.logger.info(
            "[GRVT] Order accepted (%s) status=%s @ %s",
            order_result.order_id,
            order_result.status,
            order_result.price,
        )

        if not self.cycle_complete:
            return

        timeout_seconds = max(self.fill_timeout, 5) * 6
        try:
            await asyncio.wait_for(self.cycle_complete.wait(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            self.logger.error("❌ Timeout waiting for order fill (GRVT order %s)", order_result.order_id)

            if order_result.order_id:
                cancel_result = await self.grvt_client.cancel_order(order_result.order_id)
                if cancel_result.success:
                    self.logger.info("[GRVT] Order %s cancelled", order_result.order_id)
                else:
                    self.logger.error("❌ Failed to cancel GRVT order %s: %s", order_result.order_id, cancel_result.error_message)
            self.active_cycle = None
        except Exception as exc:
            self.logger.error("❌ Error waiting for fill: %s", exc)
            self.active_cycle = None

    async def _cleanup(self):
        self.logger.info("🔄 Cleaning up resources...")

        try:
            if self.grvt_client:
                await self.grvt_client.disconnect()
        except Exception as exc:
            self.logger.error("⚠️ Error disconnecting GRVT client: %s", exc)

        try:
            if self.bingx_client:
                await self.bingx_client.disconnect()
        except Exception as exc:
            self.logger.error("⚠️ Error disconnecting BingX client: %s", exc)

    async def run(self):
        self.setup_signal_handlers()
        try:
            await self.trading_loop()
        except KeyboardInterrupt:
            self.logger.info("\n🛑 Received interrupt signal...")
        except Exception as exc:
            self.logger.error("❌ Unexpected error: %s", exc)
            self.logger.error(traceback.format_exc())
        finally:
            await self._cleanup()

    @staticmethod
    def _to_decimal(value: Any) -> Decimal:
        if value is None:
            return Decimal("0")
        try:
            return Decimal(str(value))
        except Exception:
            return Decimal("0")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Trading bot for GRVT and BingX hedging")
    parser.add_argument("--exchange", type=str, help="Exchange (grvt_bingx)")
    parser.add_argument("--ticker", type=str, default="BTC", help="Ticker symbol (default: BTC)")
    parser.add_argument("--size", type=str, help="Order size", required=False)
    parser.add_argument("--iter", type=int, help="Number of iterations", required=False)
    parser.add_argument("--fill-timeout", type=int, default=10, help="Fill timeout in seconds (default: 10)")
    parser.add_argument("--sleep", type=int, default=0, help="Sleep time between steps (default: 0)")
    return parser.parse_args()
