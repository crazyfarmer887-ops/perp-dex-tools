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
        tp_roi: Optional[Decimal] = None,
        sl_roi: Optional[Decimal] = None,
    ):
        self.ticker = ticker.upper()
        self.order_quantity = order_quantity
        self.fill_timeout = fill_timeout
        self.iterations = iterations
        self.sleep_time = sleep_time
        self.tp_roi = Decimal(tp_roi) if tp_roi is not None else None
        self.sl_roi = Decimal(sl_roi) if sl_roi is not None else None

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
        self.current_entry_price: Optional[Decimal] = None
        self.current_entry_side: Optional[str] = None
        self.current_entry_size: Optional[Decimal] = None
        self.current_entry_timestamp: Optional[float] = None
        self.current_take_profit_price: Optional[Decimal] = None
        self.current_stop_loss_price: Optional[Decimal] = None
        self.roi_poll_interval: float = 1.0
        self.max_roi_wait: float = max(self.fill_timeout * 60, 120)
        self.last_roi_reason: Optional[str] = None

        self.grvt_fill_event = asyncio.Event()
        self.last_grvt_fill: Optional[Dict[str, Any]] = None
        self.grvt_close_fill_event = asyncio.Event()
        self.active_close_order_id: Optional[str] = None
        self.active_close_order_reason: Optional[str] = None
        self.active_close_order_price: Optional[Decimal] = None
        self.active_close_order_size: Optional[Decimal] = None

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

    # ------------------------------------------------------------------ #
    # Initialization helpers
    # ------------------------------------------------------------------ #

    def setup_signal_handlers(self) -> None:
        def handler(signum, frame):
            self.logger.info("Received shutdown signal, stopping hedge bot...")
            self.stop_flag = True

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

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

        order_type = str(message.get('order_type', '')).upper()
        if order_type not in {'OPEN', 'CLOSE'}:
            return

        status = str(message.get('status', '')).upper()
        side = str(message.get('side', '')).lower()
        order_id = message.get('order_id')

        try:
            filled_size = Decimal(str(message.get('filled_size', '0')))
        except Exception:
            filled_size = Decimal('0')
        try:
            price = Decimal(str(message.get('price', '0')))
        except Exception:
            price = Decimal('0')

        if status == 'FILLED':
            if side == 'buy':
                self.grvt_position += filled_size
            elif side == 'sell':
                self.grvt_position -= filled_size

        if order_type == 'OPEN':
            if status == 'FILLED':
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
        else:
            self._handle_grvt_close_order_update(
                order_id=order_id,
                status=status,
                side=side,
                filled_size=filled_size,
                price=price
            )

    def _handle_grvt_close_order_update(
        self,
        order_id: Optional[str],
        status: str,
        side: str,
        filled_size: Decimal,
        price: Decimal
    ) -> None:
        if order_id is None:
            return

        if status == 'CANCELED':
            if order_id == self.active_close_order_id:
                self.logger.info(f"[GRVT] Close order {order_id} cancelled")
                self.active_close_order_id = None
                self.active_close_order_reason = None
                self.active_close_order_price = None
                self.active_close_order_size = None
            return

        if status not in {'FILLED', 'PARTIALLY_FILLED'}:
            return

        # Track remaining size for informational purposes
        if self.current_entry_size is not None and filled_size > 0:
            remaining = self.current_entry_size - filled_size
            self.current_entry_size = remaining if remaining > 0 else Decimal('0')

        if status == 'FILLED':
            reason = self.active_close_order_reason or self.last_roi_reason or 'manual_close'
            self.logger.info(
                f"[GRVT] CLOSE FILLED {side.upper()} {filled_size} @ {price} | "
                f"Position={self.grvt_position} | reason={reason}"
            )

            self.active_close_order_id = None
            self.active_close_order_reason = None
            self.active_close_order_price = None
            self.active_close_order_size = None

            self.grvt_close_fill_event.set()
            self._reset_entry_state(preserve_reason=True)

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

    def _reset_entry_state(self, preserve_reason: bool = False) -> None:
        self.current_entry_price = None
        self.current_entry_side = None
        self.current_entry_size = None
        self.current_entry_timestamp = None
        self.current_take_profit_price = None
        self.current_stop_loss_price = None
        if not preserve_reason:
            self.last_roi_reason = None
        self.active_close_order_id = None
        self.active_close_order_reason = None
        self.active_close_order_price = None
        self.active_close_order_size = None

    def _register_entry(self, fill: Dict[str, Any]) -> None:
        try:
            price = Decimal(str(fill.get('price')))
        except Exception:
            price = None
        if price is None or price <= 0:
            self.logger.warning("⚠️ Unable to register entry due to invalid price.")
            self._reset_entry_state()
            return

        try:
            size = Decimal(str(fill.get('size', '0')))
        except Exception:
            size = Decimal('0')

        side = str(fill.get('side', '')).lower()
        self.current_entry_price = price
        self.current_entry_side = side
        self.current_entry_size = size
        self.current_entry_timestamp = time.time()
        self.last_roi_reason = None
        self._update_roi_targets(side, price)

    def _update_roi_targets(self, side: str, entry_price: Decimal) -> None:
        self.current_take_profit_price = None
        self.current_stop_loss_price = None

        if entry_price <= 0:
            return

        hundred = Decimal('100')
        one = Decimal('1')
        messages = []

        if self.tp_roi is not None:
            tp_factor = self.tp_roi / hundred
            if side == 'buy':
                self.current_take_profit_price = entry_price * (one + tp_factor)
            else:
                self.current_take_profit_price = entry_price * (one - tp_factor)
            messages.append(f"TP @ {self.current_take_profit_price} ({self.tp_roi}% ROI)")

        if self.sl_roi is not None:
            sl_factor = self.sl_roi / hundred
            if side == 'buy':
                self.current_stop_loss_price = entry_price * (one - sl_factor)
            else:
                self.current_stop_loss_price = entry_price * (one + sl_factor)
            messages.append(f"SL @ {self.current_stop_loss_price} (-{self.sl_roi}% ROI)")

        if messages:
            self.logger.info(f"🎯 ROI targets set ({side.upper()}): {', '.join(messages)}")

    def _get_close_side(self) -> Optional[str]:
        if self.current_entry_side == 'buy':
            return 'sell'
        if self.current_entry_side == 'sell':
            return 'buy'
        return None

    async def _cancel_active_close_order(self) -> None:
        if not self.active_close_order_id or self.grvt_client is None:
            return

        order_id = self.active_close_order_id
        try:
            result = await self.grvt_client.cancel_order(order_id)
            if not result.success:
                self.logger.warning(
                    f"[GRVT] Failed to cancel close order {order_id}: {result.error_message}"
                )
        except Exception as exc:
            self.logger.warning(f"[GRVT] Error cancelling close order {order_id}: {exc}")
        finally:
            self.active_close_order_id = None
            self.active_close_order_reason = None
            self.active_close_order_price = None
            self.active_close_order_size = None

    async def _place_grvt_close_order(self, price: Decimal, reason: str) -> bool:
        if self.grvt_client is None or self.grvt_contract_id is None:
            self.logger.warning("⚠️ Cannot place close order without GRVT client/contract id.")
            return False
        if self.stop_flag:
            return False

        size = self.current_entry_size
        close_side = self._get_close_side()

        if size is None or size <= 0 or close_side is None:
            self.logger.warning("⚠️ No active entry to close; skipping close order placement.")
            return False
        if price <= 0:
            self.logger.warning("⚠️ Invalid close order price, skipping.")
            return False

        # Avoid duplicate placement for the same reason
        if self.active_close_order_id:
            if self.active_close_order_reason == reason:
                self.logger.debug(f"[GRVT] Close order already active for {reason}, skipping placement.")
                return True
            await self._cancel_active_close_order()

        try:
            order_result = await self.grvt_client.place_close_order(
                contract_id=self.grvt_contract_id,
                quantity=size,
                price=price,
                side=close_side
            )
        except Exception as exc:
            self.logger.error(f"[GRVT] Failed to place close order ({reason}): {exc}")
            return False

        if not order_result.success or not order_result.order_id:
            self.logger.error(
                f"[GRVT] Failed to place close order ({reason}): {order_result.error_message}"
            )
            return False

        reason_label = "take profit" if reason == 'take_profit' else "stop loss"
        order_price = order_result.price or price
        self.active_close_order_id = order_result.order_id
        self.active_close_order_reason = reason
        self.active_close_order_price = order_price
        self.active_close_order_size = size
        self.grvt_close_fill_event.clear()

        self.logger.info(
            f"[GRVT] Close order placed ({reason_label}) "
            f"{close_side.upper()} {size} @ {order_price} (id={order_result.order_id})"
        )

        if order_result.status == 'FILLED':
            # Immediate fill - trigger handler behavior
            self.grvt_close_fill_event.set()

        return True

    async def _wait_for_close_fill(self, timeout: Optional[float] = None) -> bool:
        if self.active_close_order_id is None:
            return True

        if self.grvt_close_fill_event.is_set():
            self.grvt_close_fill_event.clear()
            return True

        wait_timeout = timeout if timeout is not None else max(self.fill_timeout, 10)
        try:
            await asyncio.wait_for(self.grvt_close_fill_event.wait(), timeout=wait_timeout)
            self.grvt_close_fill_event.clear()
            return True
        except asyncio.TimeoutError:
            self.logger.warning(
                f"[GRVT] Close order {self.active_close_order_id} not filled within {wait_timeout}s"
            )
            return False

    async def wait_for_roi(self) -> None:
        if self.stop_flag:
            return
        if self.tp_roi is None and self.sl_roi is None:
            return
        if self.current_entry_price is None or self.current_entry_side is None:
            return
        if self.grvt_client is None or self.grvt_contract_id is None:
            self.logger.warning("⚠️ Cannot wait for ROI without GRVT client or contract id.")
            return

        entry_price = self.current_entry_price
        if entry_price <= 0:
            self.logger.warning("⚠️ Cannot evaluate ROI targets because entry price is non-positive.")
            return

        hundred = Decimal('100')
        start_time = time.time()
        self.logger.info("⏳ Waiting for ROI targets before executing opposite GRVT cycle...")

        while not self.stop_flag:
            if self.active_close_order_id:
                filled = await self._wait_for_close_fill(timeout=self.roi_poll_interval * 2)
                if filled:
                    return

            try:
                best_bid, best_ask = await self.grvt_client.fetch_bbo_prices(self.grvt_contract_id)
            except Exception as exc:
                self.logger.warning(f"⚠️ Failed to fetch GRVT prices while waiting for ROI: {exc}")
                await asyncio.sleep(self.roi_poll_interval)
                continue

            reference_price = None
            roi = None
            if self.current_entry_side == 'buy' and best_bid and best_bid > 0:
                reference_price = best_bid
                roi = (reference_price - entry_price) / entry_price * hundred
            elif self.current_entry_side == 'sell' and best_ask and best_ask > 0:
                reference_price = best_ask
                roi = (entry_price - reference_price) / entry_price * hundred

            if roi is not None:
                roi_float = float(roi)
                take_profit_hit = self.tp_roi is not None and roi >= self.tp_roi
                stop_loss_hit = self.sl_roi is not None and roi <= -self.sl_roi

                if take_profit_hit:
                    self.last_roi_reason = f"take_profit ({roi_float:.4f}%)"
                    tp_price = self.current_take_profit_price
                    if tp_price is not None and tp_price > 0:
                        self.logger.info(
                            f"🎯 ROI take profit reached: {roi_float:.4f}% (target {self.tp_roi}%). "
                            f"Placing close order @ {tp_price}"
                        )
                        placed = await self._place_grvt_close_order(tp_price, 'take_profit')
                        if placed:
                            filled = await self._wait_for_close_fill()
                            if filled:
                                self.logger.info("✅ Take profit close order filled; proceeding to next cycle.")
                                return
                    else:
                        self.logger.warning("⚠️ Take profit price unavailable; skipping close order placement.")

                if stop_loss_hit:
                    self.last_roi_reason = f"stop_loss ({roi_float:.4f}%)"
                    sl_price = self.current_stop_loss_price
                    if sl_price is not None and sl_price > 0:
                        self.logger.info(
                            f"🛑 ROI stop loss reached: {roi_float:.4f}% (threshold -{self.sl_roi}%). "
                            f"Placing close order @ {sl_price}"
                        )
                        placed = await self._place_grvt_close_order(sl_price, 'stop_loss')
                        if placed:
                            filled = await self._wait_for_close_fill()
                            if filled:
                                self.logger.info("✅ Stop loss close order filled; proceeding to next cycle.")
                                return
                    else:
                        self.logger.warning("⚠️ Stop loss price unavailable; skipping close order placement.")

            elapsed = time.time() - start_time
            if elapsed >= self.max_roi_wait:
                self.last_roi_reason = f"timeout ({elapsed:.1f}s)"
                self.logger.info(f"⏱️ ROI wait timed out after {elapsed:.1f}s; proceeding to next cycle.")
                return

            await asyncio.sleep(self.roi_poll_interval)

    async def execute_cycle(self, side: str) -> None:
        await self._cancel_active_close_order()

        fill = await self.place_grvt_order(side)
        if not fill or self.stop_flag:
            self._reset_entry_state()
            return

        self._register_entry(fill)
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

            await self.wait_for_roi()
            if self.stop_flag:
                break

            await self.execute_cycle('sell')
            if self.stop_flag:
                break

            await self.wait_for_roi()

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
