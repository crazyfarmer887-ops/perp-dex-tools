import asyncio
import logging
import os
import signal
import sys
import time
from decimal import Decimal
from typing import Any, Dict, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exchanges.grvt import GrvtClient
from exchanges.bingx import BingxClient


class Config:
    """Simple attribute-style config wrapper."""

    def __init__(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            setattr(self, key, value)


class HedgeBot:
    """
    Hedge bot that runs maker orders on GRVT and hedges fills on BingX.
    """

    def __init__(
        self,
        ticker: str,
        order_quantity: Decimal,
        fill_timeout: int = 10,
        iterations: int = 10,
        sleep_time: int = 0,
        take_profit_roi: Optional[Decimal] = None,
        stop_loss_roi: Optional[Decimal] = None,
    ):
        self.ticker = ticker.upper()
        self.order_quantity = order_quantity
        self.fill_timeout = fill_timeout
        self.iterations = iterations
        self.sleep_time = sleep_time
        self.take_profit_roi = take_profit_roi
        self.stop_loss_roi = stop_loss_roi
        self.position_tracker = {
            "long": {"size": Decimal("0"), "value": Decimal("0"), "average": None},
            "short": {"size": Decimal("0"), "value": Decimal("0"), "average": None},
        }
        self.last_roi_reason: Optional[str] = None
        self.roi_poll_interval = 1.0

        self.stop_flag = False
        self.loop: Optional[asyncio.AbstractEventLoop] = None

        self.grvt_client: Optional[GrvtClient] = None
        self.bingx_client: Optional[BingxClient] = None

        self.grvt_contract_id: Optional[str] = None
        self.grvt_tick_size: Optional[Decimal] = None
        self.bingx_contract_id: Optional[str] = None
        self.bingx_tick_size: Optional[Decimal] = None

        self.grvt_position = Decimal('0')
        self.bingx_position = Decimal('0')

        self.grvt_fill_event = asyncio.Event()
        self.last_grvt_fill: Optional[Dict[str, Any]] = None

        os.makedirs("logs", exist_ok=True)
        self.log_filename = f"logs/grvt_bingx_{self.ticker.lower()}_hedge_log.txt"

        self.logger = logging.getLogger(f"hedge_grvt_bingx_{self.ticker}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.logger.handlers.clear()

        file_handler = logging.FileHandler(self.log_filename)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self._log_roi_configuration()

    # ------------------------------------------------------------------ #
    # Initialization helpers
    # ------------------------------------------------------------------ #

    def setup_signal_handlers(self) -> None:
        def handler(signum, frame):
            self.logger.info("Received shutdown signal, stopping hedge bot...")
            self.stop_flag = True

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def _log_roi_configuration(self) -> None:
        if self.take_profit_roi is None and self.stop_loss_roi is None:
            return
        tp_text = f"{self.take_profit_roi}%" if self.take_profit_roi is not None else "N/A"
        sl_text = f"{self.stop_loss_roi}%" if self.stop_loss_roi is not None else "N/A"
        self.logger.info(f"🎯 ROI configuration -> Take Profit: {tp_text}, Stop Loss: {sl_text}")

    def _recompute_tracker_average(self, tracker_key: str) -> None:
        tracker = self.position_tracker[tracker_key]
        if tracker["size"] > 0:
            tracker["average"] = tracker["value"] / tracker["size"]
        else:
            tracker["size"] = Decimal("0")
            tracker["value"] = Decimal("0")
            tracker["average"] = None

    def _update_position_tracker(self, side: str, filled_size: Decimal, price: Decimal, prev_position: Decimal) -> None:
        if filled_size <= 0:
            return

        if side == "buy":
            if prev_position < 0:
                tracker = self.position_tracker["short"]
                tracker["size"] -= filled_size
                if tracker["size"] <= 0:
                    tracker["size"] = Decimal("0")
                    tracker["value"] = Decimal("0")
                    tracker["average"] = None
                else:
                    reference_avg = tracker["average"] if tracker["average"] is not None else price
                    tracker["value"] = reference_avg * tracker["size"]
                    self._recompute_tracker_average("short")
            else:
                tracker = self.position_tracker["long"]
                tracker["value"] += price * filled_size
                tracker["size"] += filled_size
                self._recompute_tracker_average("long")
        elif side == "sell":
            if prev_position > 0:
                tracker = self.position_tracker["long"]
                tracker["size"] -= filled_size
                if tracker["size"] <= 0:
                    tracker["size"] = Decimal("0")
                    tracker["value"] = Decimal("0")
                    tracker["average"] = None
                else:
                    reference_avg = tracker["average"] if tracker["average"] is not None else price
                    tracker["value"] = reference_avg * tracker["size"]
                    self._recompute_tracker_average("long")
            else:
                tracker = self.position_tracker["short"]
                tracker["value"] += price * filled_size
                tracker["size"] += filled_size
                self._recompute_tracker_average("short")

    def _compute_roi_targets(self, position_type: str):
        tracker = self.position_tracker[position_type]
        if tracker["average"] is None or tracker["size"] <= 0:
            return None, None, None

        avg_price = tracker["average"]
        tp_price = None
        sl_price = None

        if self.take_profit_roi is not None:
            if position_type == "long":
                tp_price = avg_price * (Decimal("1") + self.take_profit_roi / Decimal("100"))
            else:
                tp_price = avg_price * (Decimal("1") - self.take_profit_roi / Decimal("100"))
        if self.stop_loss_roi is not None:
            if position_type == "long":
                sl_price = avg_price * (Decimal("1") - self.stop_loss_roi / Decimal("100"))
            else:
                sl_price = avg_price * (Decimal("1") + self.stop_loss_roi / Decimal("100"))

        return avg_price, tp_price, sl_price

    async def _get_best_prices(self) -> Tuple[Decimal, Decimal]:
        if self.grvt_client is None or self.grvt_contract_id is None:
            raise RuntimeError("GRVT client not initialized")
        return await self.grvt_client.fetch_bbo_prices(self.grvt_contract_id)

    async def _wait_for_roi_target(self, position_type: str) -> str:
        avg_price, tp_price, sl_price = self._compute_roi_targets(position_type)
        if avg_price is None:
            return "none"

        self.logger.info(
            f"⏳ Waiting for ROI target ({position_type}) | "
            f"avg: {avg_price}, TP: {tp_price or 'N/A'}, SL: {sl_price or 'N/A'}"
        )

        trigger = "none"
        current_price: Optional[Decimal] = None

        while not self.stop_flag:
            try:
                best_bid, best_ask = await self._get_best_prices()
            except Exception as exc:
                self.logger.warning(f"⚠️ Failed to fetch prices for ROI monitoring: {exc}")
                await asyncio.sleep(self.roi_poll_interval)
                continue

            if position_type == "long":
                current_price = best_bid
                if current_price is None or current_price <= 0:
                    await asyncio.sleep(self.roi_poll_interval)
                    continue
                if tp_price is not None and current_price >= tp_price:
                    trigger = "take_profit"
                    break
                if sl_price is not None and current_price <= sl_price:
                    trigger = "stop_loss"
                    break
            else:
                current_price = best_ask
                if current_price is None or current_price <= 0:
                    await asyncio.sleep(self.roi_poll_interval)
                    continue
                if tp_price is not None and current_price <= tp_price:
                    trigger = "take_profit"
                    break
                if sl_price is not None and current_price >= sl_price:
                    trigger = "stop_loss"
                    break

            await asyncio.sleep(self.roi_poll_interval)

        if trigger != "none" and current_price is not None:
            self.logger.info(
                f"🎯 ROI trigger reached ({trigger}) for {position_type} position at price {current_price}"
            )
        return trigger

    async def _wait_for_roi_if_configured(self) -> None:
        if self.take_profit_roi is None and self.stop_loss_roi is None:
            self.last_roi_reason = "none"
            return

        if self.grvt_position > 0:
            self.last_roi_reason = await self._wait_for_roi_target("long")
        elif self.grvt_position < 0:
            self.last_roi_reason = await self._wait_for_roi_target("short")
        else:
            self.last_roi_reason = "none"

    def _build_grvt_config(self) -> Config:
        return Config({
            'ticker': self.ticker,
            'contract_id': '',
            'quantity': self.order_quantity,
            'tick_size': Decimal('0.01'),
            'direction': 'buy',
            'close_order_side': 'sell'
        })

    def _build_bingx_config(self) -> Config:
        return Config({
            'ticker': self.ticker,
            'contract_id': '',
            'quantity': self.order_quantity,
            'tick_size': Decimal('0.01'),
            'direction': 'buy',
            'close_order_side': 'sell'
        })

    def initialize_clients(self) -> None:
        if self.grvt_client is None:
            self.grvt_client = GrvtClient(self._build_grvt_config())
        if self.bingx_client is None:
            self.bingx_client = BingxClient(self._build_bingx_config())

    async def load_contract_metadata(self) -> None:
        assert self.grvt_client is not None
        assert self.bingx_client is not None

        self.grvt_contract_id, self.grvt_tick_size = await self.grvt_client.get_contract_attributes()
        self.bingx_contract_id, self.bingx_tick_size = await self.bingx_client.get_contract_attributes()

        self.logger.info(f"GRVT contract: {self.grvt_contract_id} (tick {self.grvt_tick_size})")
        self.logger.info(f"BingX contract: {self.bingx_contract_id} (tick {self.bingx_tick_size})")

    # ------------------------------------------------------------------ #
    # Order handling
    # ------------------------------------------------------------------ #

    def _handle_grvt_order_update(self, message: Dict[str, Any]) -> None:
        if self.grvt_contract_id is None:
            return
        if message.get('contract_id') != self.grvt_contract_id:
            return
        if message.get('order_type') != 'OPEN':
            return

        status = message.get('status')
        side = message.get('side', '').lower()
        filled_size = Decimal(str(message.get('filled_size', '0')))
        price = Decimal(str(message.get('price', '0')))
        order_id = message.get('order_id')

        if status == 'FILLED':
            prev_position = self.grvt_position
            if side == 'buy':
                self.grvt_position += filled_size
            else:
                self.grvt_position -= filled_size

            self._update_position_tracker(side, filled_size, price, prev_position)

            self.last_grvt_fill = {
                'order_id': order_id,
                'side': side,
                'size': filled_size,
                'price': price
            }

            self.logger.info(
                f"[GRVT] FILLED {side.upper()} {filled_size} @ {price} | Position={self.grvt_position}"
            )
            self.grvt_fill_event.set()

    async def setup_grvt_websocket(self) -> None:
        assert self.grvt_client is not None
        self.grvt_client.setup_order_update_handler(self._handle_grvt_order_update)
        await self.grvt_client.connect()

    async def setup_bingx(self) -> None:
        assert self.bingx_client is not None
        await self.bingx_client.connect()

    async def place_grvt_order(self, side: str) -> Optional[Dict[str, Any]]:
        assert self.grvt_client is not None
        assert self.grvt_contract_id is not None

        self.grvt_client.config.direction = side
        self.grvt_client.config.close_order_side = 'sell' if side == 'buy' else 'buy'

        self.grvt_fill_event.clear()
        self.last_grvt_fill = None

        order_result = await self.grvt_client.place_open_order(
            contract_id=self.grvt_contract_id,
            quantity=self.order_quantity,
            direction=side
        )

        if not order_result.success or not order_result.order_id:
            self.logger.error(f"[GRVT] Failed to place {side} order: {order_result.error_message}")
            return None

        self.logger.info(f"[GRVT] Order placed {order_result.order_id} ({side}) @ {order_result.price}")

        if order_result.status == 'FILLED':
            self.last_grvt_fill = {
                'order_id': order_result.order_id,
                'side': side,
                'size': order_result.size,
                'price': order_result.price
            }
            return self.last_grvt_fill

        try:
            await asyncio.wait_for(self.grvt_fill_event.wait(), timeout=self.fill_timeout)
        except asyncio.TimeoutError:
            self.logger.warning(f"[GRVT] {side} order {order_result.order_id} timed out, cancelling")
            cancel_result = await self.grvt_client.cancel_order(order_result.order_id)
            if not cancel_result.success:
                self.logger.error(f"[GRVT] Failed to cancel order {order_result.order_id}: {cancel_result.error_message}")
            return None

        return self.last_grvt_fill

    async def place_bingx_hedge(self, fill: Dict[str, Any]) -> None:
        assert self.bingx_client is not None
        assert self.bingx_contract_id is not None

        side = fill['side']
        size = fill['size']

        hedge_side = 'sell' if side == 'buy' else 'buy'
        self.bingx_client.config.direction = hedge_side
        self.bingx_client.config.close_order_side = 'buy' if hedge_side == 'sell' else 'sell'

        self.logger.info(f"[BINGX] Hedging {hedge_side} {size}")

        result = await self.bingx_client.place_market_order(
            contract_id=self.bingx_contract_id,
            quantity=size,
            side=hedge_side
        )

        if not result.success:
            self.logger.error(f"[BINGX] Hedge order failed: {result.error_message}")
            return

        if hedge_side == 'buy':
            self.bingx_position += size
        else:
            self.bingx_position -= size

        self.logger.info(
            f"[BINGX] {hedge_side.upper()} {size} @ {result.price} | Position={self.bingx_position}"
        )

    async def execute_cycle(self, side: str) -> None:
        fill = await self.place_grvt_order(side)
        if not fill or self.stop_flag:
            return

        await self.place_bingx_hedge(fill)

        if self.sleep_time > 0 and not self.stop_flag:
            await asyncio.sleep(self.sleep_time)

    # ------------------------------------------------------------------ #
    # Main run loop
    # ------------------------------------------------------------------ #

    async def trading_loop(self) -> None:
        self.logger.info(f"Starting GRVT+BingX hedge bot | ticker={self.ticker} | size={self.order_quantity}")

        self.initialize_clients()
        try:
            await self.load_contract_metadata()
            await self.setup_grvt_websocket()
            await self.setup_bingx()
        except Exception as exc:
            self.logger.error(f"Initialization failed: {exc}")
            self.stop_flag = True
            return

        await asyncio.sleep(2)

        iteration = 0
        while not self.stop_flag and iteration < self.iterations:
            iteration += 1
            self.logger.info(f"----- Iteration {iteration}/{self.iterations} -----")

            await self.execute_cycle('buy')
            if self.stop_flag:
                break

            await self._wait_for_roi_if_configured()
            if self.last_roi_reason and self.last_roi_reason != "none":
                self.logger.info(f"🎯 ROI condition met: {self.last_roi_reason}. Proceeding to close position.")
            if self.stop_flag:
                break

            await self.execute_cycle('sell')
            if self.stop_flag:
                break

        self.logger.info("Trading loop finished")

    async def cleanup(self) -> None:
        if self.grvt_client:
            try:
                await self.grvt_client.disconnect()
            except Exception as exc:
                self.logger.warning(f"Error disconnecting GRVT client: {exc}")
        if self.bingx_client:
            try:
                await self.bingx_client.disconnect()
            except Exception as exc:
                self.logger.warning(f"Error disconnecting BingX client: {exc}")

    async def run(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.setup_signal_handlers()

        start_time = time.time()
        try:
            await self.trading_loop()
        except Exception as exc:
            self.logger.error(f"Unexpected error: {exc}")
        finally:
            await self.cleanup()
            elapsed = time.time() - start_time
            self.logger.info(f"Hedge bot stopped after {elapsed:.1f}s")
