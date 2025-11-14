"""
Modular Trading Bot - Supports multiple exchanges
"""

import os
import time
import asyncio
import traceback
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional, Tuple

from exchanges import ExchangeFactory
from helpers import TradingLogger
from helpers.lark_bot import LarkBot
from helpers.telegram_bot import TelegramBot


@dataclass
class TradingConfig:
    """Configuration class for trading parameters."""
    ticker: str
    contract_id: str
    quantity: Decimal
    take_profit: Decimal
    tick_size: Decimal
    direction: str
    max_orders: int
    wait_time: int
    exchange: str
    grid_step: Decimal
    stop_price: Decimal
    pause_price: Decimal
    boost_mode: bool = False
    dual_sided: bool = False
    tp_roi: Optional[Decimal] = None
    sl_roi: Optional[Decimal] = None
    roi_poll_interval: float = 2.0
    roi_max_wait: Optional[int] = None

    @property
    def close_order_side(self) -> str:
        """Get the close order side based on bot direction."""
        return 'buy' if self.direction == "sell" else 'sell'


@dataclass
class OrderMonitor:
    """Thread-safe order monitoring state."""
    order_id: Optional[str] = None
    filled: bool = False
    filled_price: Optional[Decimal] = None
    filled_qty: Decimal = 0.0

    def reset(self):
        """Reset the monitor state."""
        self.order_id = None
        self.filled = False
        self.filled_price = None
        self.filled_qty = 0.0


class TradingBot:
    """Modular Trading Bot - Main trading logic supporting multiple exchanges."""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = TradingLogger(config.exchange, config.ticker, log_to_console=True)

        # Create exchange client
        try:
            self.exchange_client = ExchangeFactory.create_exchange(
                config.exchange,
                config
            )
        except ValueError as e:
            raise ValueError(f"Failed to create exchange client: {e}")

        # Trading state
        self.active_close_orders = []
        self.last_close_orders = 0
        self.last_open_order_time = 0
        self.last_log_time = 0
        self.current_order_status = None
        self.order_filled_event = asyncio.Event()
        self.order_canceled_event = asyncio.Event()
        self.shutdown_requested = False
        self.loop = None
        self.last_roi_reason: Optional[str] = None

        # Register order callback
        self._setup_websocket_handlers()

    async def graceful_shutdown(self, reason: str = "Unknown"):
        """Perform graceful shutdown of the trading bot."""
        self.logger.log(f"Starting graceful shutdown: {reason}", "INFO")
        self.shutdown_requested = True

        try:
            # Disconnect from exchange
            await self.exchange_client.disconnect()
            self.logger.log("Graceful shutdown completed", "INFO")

        except Exception as e:
            self.logger.log(f"Error during graceful shutdown: {e}", "ERROR")

    def _setup_websocket_handlers(self):
        """Setup WebSocket handlers for order updates."""
        def order_update_handler(message):
            """Handle order updates from WebSocket."""
            try:
                # Check if this is for our contract
                if message.get('contract_id') != self.config.contract_id:
                    return

                order_id = message.get('order_id')
                status = message.get('status')
                side = message.get('side', '')
                order_type = message.get('order_type', '')
                filled_size = Decimal(message.get('filled_size'))
                if order_type == "OPEN":
                    self.current_order_status = status

                if status == 'FILLED':
                    if order_type == "OPEN":
                        self.order_filled_amount = filled_size
                        # Ensure thread-safe interaction with asyncio event loop
                        if self.loop is not None:
                            self.loop.call_soon_threadsafe(self.order_filled_event.set)
                        else:
                            # Fallback (should not happen after run() starts)
                            self.order_filled_event.set()

                    self.logger.log(f"[{order_type}] [{order_id}] {status} "
                                    f"{message.get('size')} @ {message.get('price')}", "INFO")
                    self.logger.log_transaction(order_id, side, message.get('size'), message.get('price'), status)
                elif status == "CANCELED":
                    if order_type == "OPEN":
                        self.order_filled_amount = filled_size
                        if self.loop is not None:
                            self.loop.call_soon_threadsafe(self.order_canceled_event.set)
                        else:
                            self.order_canceled_event.set()

                        if self.order_filled_amount > 0:
                            self.logger.log_transaction(order_id, side, self.order_filled_amount, message.get('price'), status)
                            
                    # PATCH
                    if self.config.exchange == "extended":
                        self.logger.log(f"[{order_type}] [{order_id}] {status} "
                                        f"{Decimal(message.get('size')) - filled_size} @ {message.get('price')}", "INFO")
                    else:
                        self.logger.log(f"[{order_type}] [{order_id}] {status} "
                                        f"{message.get('size')} @ {message.get('price')}", "INFO")
                elif status == "PARTIALLY_FILLED":
                    self.logger.log(f"[{order_type}] [{order_id}] {status} "
                                    f"{filled_size} @ {message.get('price')}", "INFO")
                else:
                    self.logger.log(f"[{order_type}] [{order_id}] {status} "
                                    f"{message.get('size')} @ {message.get('price')}", "INFO")

            except Exception as e:
                self.logger.log(f"Error handling order update: {e}", "ERROR")
                self.logger.log(f"Traceback: {traceback.format_exc()}", "ERROR")

        # Setup order update handler
        self.exchange_client.setup_order_update_handler(order_update_handler)

    def _calculate_wait_time(self) -> Decimal:
        """Calculate wait time between orders."""
        cool_down_time = self.config.wait_time

        if len(self.active_close_orders) < self.last_close_orders:
            self.last_close_orders = len(self.active_close_orders)
            return 0

        self.last_close_orders = len(self.active_close_orders)
        if len(self.active_close_orders) >= self.config.max_orders:
            return 1

        if len(self.active_close_orders) / self.config.max_orders >= 2/3:
            cool_down_time = 2 * self.config.wait_time
        elif len(self.active_close_orders) / self.config.max_orders >= 1/3:
            cool_down_time = self.config.wait_time
        elif len(self.active_close_orders) / self.config.max_orders >= 1/6:
            cool_down_time = self.config.wait_time / 2
        else:
            cool_down_time = self.config.wait_time / 4

        # if the program detects active_close_orders during startup, it is necessary to consider cooldown_time
        if self.last_open_order_time == 0 and len(self.active_close_orders) > 0:
            self.last_open_order_time = time.time()

        if time.time() - self.last_open_order_time > cool_down_time:
            return 0
        else:
            return 1

    def _roi_enabled(self) -> bool:
        """Check whether ROI gating is enabled."""
        return self.config.tp_roi is not None or self.config.sl_roi is not None

    @staticmethod
    def _to_decimal(value: Optional[Decimal]) -> Decimal:
        if isinstance(value, Decimal):
            return value
        if value is None:
            return Decimal('0')
        try:
            return Decimal(str(value))
        except Exception:
            return Decimal('0')

    async def _resolve_entry_price(self, order_result) -> Decimal:
        """Resolve the best available fill price for ROI calculations."""
        filled_price = self._to_decimal(order_result.price)
        if not self._roi_enabled() or not order_result.order_id:
            return filled_price

        try:
            order_info = await self.exchange_client.get_order_info(order_result.order_id)
        except Exception as exc:
            self.logger.log(f"Failed to fetch order info for ROI calculation: {exc}", "WARNING")
            return filled_price

        if order_info and getattr(order_info, 'price', None):
            price_value = self._to_decimal(order_info.price)
            if price_value > 0:
                return price_value

        return filled_price

    def _determine_close_price(self, entry_price: Decimal, roi_reason: Optional[str]) -> Decimal:
        """Determine the close price based on ROI settings or static take profit."""
        entry_price = self._to_decimal(entry_price)
        if entry_price <= 0:
            return entry_price

        hundred = Decimal('100')
        one = Decimal('1')
        direction = self.config.direction

        def apply_percent(percent: Decimal, invert: bool = False) -> Decimal:
            factor = percent / hundred
            if direction == 'buy':
                return entry_price * ((one - factor) if invert else (one + factor))
            else:
                return entry_price * ((one + factor) if invert else (one - factor))

        if roi_reason == 'take_profit' and self.config.tp_roi is not None:
            return apply_percent(self.config.tp_roi, invert=False)

        if roi_reason == 'stop_loss' and self.config.sl_roi is not None:
            return apply_percent(self.config.sl_roi, invert=True)

        if self.config.tp_roi is not None and roi_reason in {None, 'timeout'} and self._roi_enabled():
            return apply_percent(self.config.tp_roi, invert=False)

        take_profit = self.config.take_profit
        if take_profit <= 0:
            return entry_price

        return apply_percent(take_profit, invert=False)

    async def _wait_for_roi_targets(self, entry_price: Decimal) -> Tuple[Optional[str], Optional[Decimal]]:
        """Wait until ROI targets are reached before placing closing orders."""
        if not self._roi_enabled():
            return None, None

        entry_price = self._to_decimal(entry_price)
        if entry_price <= 0:
            self.logger.log("Cannot evaluate ROI targets because entry price is non-positive.", "WARNING")
            return None, None

        self.logger.log("Waiting for ROI targets before placing close order...", "INFO")
        hundred = Decimal('100')
        poll_interval = max(self.config.roi_poll_interval, 0.1)
        start_time = time.time()
        last_reference: Optional[Decimal] = None

        while not self.shutdown_requested:
            try:
                best_bid, best_ask = await self.exchange_client.fetch_bbo_prices(self.config.contract_id)
            except Exception as exc:
                self.logger.log(f"Failed to fetch bid/ask while waiting for ROI: {exc}", "WARNING")
                await asyncio.sleep(poll_interval)
                continue

            reference_price = best_bid if self.config.direction == 'buy' else best_ask
            last_reference = reference_price
            roi = None

            if reference_price and reference_price > 0:
                if self.config.direction == 'buy':
                    roi = (reference_price - entry_price) / entry_price * hundred
                else:
                    roi = (entry_price - reference_price) / entry_price * hundred

            if roi is not None:
                roi_float = float(roi)
                if self.config.tp_roi is not None and roi >= self.config.tp_roi:
                    self.last_roi_reason = f"take_profit ({roi_float:.4f}%)"
                    self.logger.log(
                        f"ROI take profit reached: {roi_float:.4f}% (target {self.config.tp_roi}%)",
                        "INFO"
                    )
                    return "take_profit", reference_price

                if self.config.sl_roi is not None and roi <= -self.config.sl_roi:
                    self.last_roi_reason = f"stop_loss ({roi_float:.4f}%)"
                    self.logger.log(
                        f"ROI stop loss reached: {roi_float:.4f}% (threshold -{self.config.sl_roi}%)",
                        "WARNING"
                    )
                    return "stop_loss", reference_price

            if self.config.roi_max_wait is not None:
                elapsed = time.time() - start_time
                if elapsed >= self.config.roi_max_wait:
                    self.last_roi_reason = f"timeout ({elapsed:.1f}s)"
                    self.logger.log(
                        f"ROI wait timed out after {elapsed:.1f}s; proceeding with close order.",
                        "INFO"
                    )
                    return "timeout", last_reference

            await asyncio.sleep(poll_interval)

        return None, last_reference

    async def _execute_close_flow(self, quantity: Decimal, entry_price: Decimal) -> bool:
        """Place the appropriate close order, respecting ROI settings when enabled."""
        quantity = self._to_decimal(quantity)
        if quantity <= 0:
            self.logger.log("Skip closing flow because quantity is non-positive.", "WARNING")
            return False

        close_side = self.config.close_order_side

        if self.config.boost_mode:
            close_order_result = await self.exchange_client.place_market_order(
                self.config.contract_id,
                quantity,
                close_side
            )
            if not close_order_result.success:
                msg = f"[CLOSE] Failed to place boost-mode market order: {close_order_result.error_message}"
                self.logger.log(msg, "ERROR")
                raise Exception(msg)
            return True

        roi_reason = None
        if self._roi_enabled():
            roi_reason, _ = await self._wait_for_roi_targets(entry_price)

        close_price = self._determine_close_price(entry_price, roi_reason)
        if close_price <= 0:
            # Fallback to current best levels if ROI/entry data is invalid
            try:
                best_bid, best_ask = await self.exchange_client.fetch_bbo_prices(self.config.contract_id)
                close_price = best_bid if close_side == 'buy' else best_ask
            except Exception:
                close_price = entry_price if entry_price > 0 else self._to_decimal(self.config.tick_size)

        close_order_result = await self.exchange_client.place_close_order(
            self.config.contract_id,
            quantity,
            close_price,
            close_side
        )

        if self.config.exchange == "lighter":
            await asyncio.sleep(1)

        self.last_open_order_time = time.time()

        if not close_order_result.success:
            msg = f"[CLOSE] Failed to place close order: {close_order_result.error_message}"
            self.logger.log(msg, "ERROR")
            raise Exception(msg)

        return True

    async def _place_and_monitor_open_order(self) -> bool:
        """Place an order and monitor its execution."""
        try:
            # Reset state before placing order
            self.order_filled_event.clear()
            self.current_order_status = 'OPEN'
            self.order_filled_amount = Decimal('0')

            # Place the order
            order_result = await self.exchange_client.place_open_order(
                self.config.contract_id,
                self.config.quantity,
                self.config.direction
            )

            if not order_result.success:
                return False

            if order_result.status == 'FILLED':
                return await self._handle_order_result(order_result)
            elif not self.order_filled_event.is_set():
                try:
                    await asyncio.wait_for(self.order_filled_event.wait(), timeout=10)
                except asyncio.TimeoutError:
                    pass

            # Handle order result
            return await self._handle_order_result(order_result)

        except Exception as e:
            self.logger.log(f"Error placing order: {e}", "ERROR")
            self.logger.log(f"Traceback: {traceback.format_exc()}", "ERROR")
            return False

    async def _handle_order_result(self, order_result) -> bool:
        """Handle the result of an order placement."""
        order_id = order_result.order_id
        initial_order_price = self._to_decimal(order_result.price)
        entry_price = await self._resolve_entry_price(order_result)

        if self.order_filled_event.is_set() or order_result.status == 'FILLED':
            filled_qty = self._to_decimal(getattr(self, 'order_filled_amount', Decimal('0')))
            if filled_qty <= 0:
                filled_qty = self._to_decimal(order_result.size)
            if filled_qty <= 0:
                filled_qty = self.config.quantity

            await self._execute_close_flow(filled_qty, entry_price)
            return True

        new_order_price = await self.exchange_client.get_order_price(self.config.direction)

        def should_wait(direction: str, latest_price: Decimal, placed_price: Decimal) -> bool:
            if direction == "buy":
                return latest_price <= placed_price
            elif direction == "sell":
                return latest_price >= placed_price
            return False

        if self.config.exchange == "lighter":
            current_order_status = self.exchange_client.current_order.status
        else:
            order_info = await self.exchange_client.get_order_info(order_id)
            current_order_status = order_info.status

        while (
            should_wait(self.config.direction, new_order_price, initial_order_price)
            and current_order_status == "OPEN"
        ):
            self.logger.log(f"[OPEN] [{order_id}] Waiting for order to be filled @ {initial_order_price}", "INFO")
            await asyncio.sleep(5)
            if self.config.exchange == "lighter":
                current_order_status = self.exchange_client.current_order.status
            else:
                order_info = await self.exchange_client.get_order_info(order_id)
                if order_info is not None:
                    current_order_status = order_info.status
            new_order_price = await self.exchange_client.get_order_price(self.config.direction)

        self.order_canceled_event.clear()
        self.logger.log(f"[OPEN] [{order_id}] Cancelling order and placing a new order", "INFO")

        if self.config.exchange == "lighter":
            cancel_result = await self.exchange_client.cancel_order(order_id)
            start_time = time.time()
            while (time.time() - start_time < 10 and self.exchange_client.current_order.status != 'CANCELED' and
                    self.exchange_client.current_order.status != 'FILLED'):
                await asyncio.sleep(0.1)

            if self.exchange_client.current_order.status not in ['CANCELED', 'FILLED']:
                raise Exception(f"[OPEN] Error cancelling order: {self.exchange_client.current_order.status}")
            else:
                self.order_filled_amount = self.exchange_client.current_order.filled_size
        else:
            try:
                cancel_result = await self.exchange_client.cancel_order(order_id)
                if not cancel_result.success:
                    self.order_canceled_event.set()
                    self.logger.log(f"[CLOSE] Failed to cancel order {order_id}: {cancel_result.error_message}", "WARNING")
                else:
                    self.current_order_status = "CANCELED"

            except Exception as e:
                self.order_canceled_event.set()
                self.logger.log(f"[CLOSE] Error canceling order {order_id}: {e}", "ERROR")

            if self.config.exchange in {"backpack", "extended"}:
                self.order_filled_amount = cancel_result.filled_size
            else:
                if not self.order_canceled_event.is_set():
                    try:
                        await asyncio.wait_for(self.order_canceled_event.wait(), timeout=5)
                    except asyncio.TimeoutError:
                        order_info = await self.exchange_client.get_order_info(order_id)
                        self.order_filled_amount = order_info.filled_size

        if self.order_filled_amount > 0:
            filled_amount = self._to_decimal(self.order_filled_amount)
            await self._execute_close_flow(filled_amount, entry_price)

        return True

    async def _log_status_periodically(self):
        """Log status information periodically, including positions."""
        if time.time() - self.last_log_time > 60 or self.last_log_time == 0:
            print("--------------------------------")
            try:
                # Get active orders
                active_orders = await self.exchange_client.get_active_orders(self.config.contract_id)

                # Filter close orders
                self.active_close_orders = []
                for order in active_orders:
                    if order.side == self.config.close_order_side:
                        self.active_close_orders.append({
                            'id': order.order_id,
                            'price': order.price,
                            'size': order.size
                        })

                # Get positions
                position_amt = await self.exchange_client.get_account_positions()

                # Calculate active closing amount
                active_close_amount = sum(
                    Decimal(order.get('size', 0))
                    for order in self.active_close_orders
                    if isinstance(order, dict)
                )

                self.logger.log(f"Current Position: {position_amt} | Active closing amount: {active_close_amount} | "
                                f"Order quantity: {len(self.active_close_orders)}")
                self.last_log_time = time.time()
                # Check for position mismatch
                if abs(position_amt - active_close_amount) > (2 * self.config.quantity):
                    error_message = f"\n\nERROR: [{self.config.exchange.upper()}_{self.config.ticker.upper()}] "
                    error_message += "Position mismatch detected\n"
                    error_message += "###### ERROR ###### ERROR ###### ERROR ###### ERROR #####\n"
                    error_message += "Please manually rebalance your position and take-profit orders\n"
                    error_message += "请手动平衡当前仓位和正在关闭的仓位\n"
                    error_message += f"current position: {position_amt} | active closing amount: {active_close_amount} | "f"Order quantity: {len(self.active_close_orders)}\n"
                    error_message += "###### ERROR ###### ERROR ###### ERROR ###### ERROR #####\n"
                    self.logger.log(error_message, "ERROR")

                    await self.send_notification(error_message.lstrip())

                    if not self.shutdown_requested:
                        self.shutdown_requested = True

                    mismatch_detected = True
                else:
                    mismatch_detected = False

                return mismatch_detected

            except Exception as e:
                self.logger.log(f"Error in periodic status check: {e}", "ERROR")
                self.logger.log(f"Traceback: {traceback.format_exc()}", "ERROR")

            print("--------------------------------")

    async def _meet_grid_step_condition(self) -> bool:
        if self.active_close_orders:
            picker = min if self.config.direction == "buy" else max
            next_close_order = picker(self.active_close_orders, key=lambda o: o["price"])
            next_close_price = next_close_order["price"]

            best_bid, best_ask = await self.exchange_client.fetch_bbo_prices(self.config.contract_id)
            if best_bid <= 0 or best_ask <= 0 or best_bid >= best_ask:
                raise ValueError("No bid/ask data available")

            if self.config.direction == "buy":
                new_order_close_price = best_ask * (1 + self.config.take_profit/100)
                if next_close_price / new_order_close_price > 1 + self.config.grid_step/100:
                    return True
                else:
                    return False
            elif self.config.direction == "sell":
                new_order_close_price = best_bid * (1 - self.config.take_profit/100)
                if new_order_close_price / next_close_price > 1 + self.config.grid_step/100:
                    return True
                else:
                    return False
            else:
                raise ValueError(f"Invalid direction: {self.config.direction}")
        else:
            return True

    async def _check_price_condition(self) -> bool:
        stop_trading = False
        pause_trading = False

        if self.config.pause_price == self.config.stop_price == -1:
            return stop_trading, pause_trading

        best_bid, best_ask = await self.exchange_client.fetch_bbo_prices(self.config.contract_id)
        if best_bid <= 0 or best_ask <= 0 or best_bid >= best_ask:
            raise ValueError("No bid/ask data available")

        if self.config.stop_price != -1:
            if self.config.direction == "buy":
                if best_ask >= self.config.stop_price:
                    stop_trading = True
            elif self.config.direction == "sell":
                if best_bid <= self.config.stop_price:
                    stop_trading = True

        if self.config.pause_price != -1:
            if self.config.direction == "buy":
                if best_ask >= self.config.pause_price:
                    pause_trading = True
            elif self.config.direction == "sell":
                if best_bid <= self.config.pause_price:
                    pause_trading = True

        return stop_trading, pause_trading

    async def send_notification(self, message: str):
        lark_token = os.getenv("LARK_TOKEN")
        if lark_token:
            async with LarkBot(lark_token) as lark_bot:
                await lark_bot.send_text(message)

        telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if telegram_token and telegram_chat_id:
            with TelegramBot(telegram_token, telegram_chat_id) as tg_bot:
                tg_bot.send_text(message)

    async def run(self):
        """Main trading loop."""
        try:
            self.config.contract_id, self.config.tick_size = await self.exchange_client.get_contract_attributes()

            # Log current TradingConfig
            self.logger.log("=== Trading Configuration ===", "INFO")
            self.logger.log(f"Ticker: {self.config.ticker}", "INFO")
            self.logger.log(f"Contract ID: {self.config.contract_id}", "INFO")
            self.logger.log(f"Quantity: {self.config.quantity}", "INFO")
            self.logger.log(f"Take Profit: {self.config.take_profit}%", "INFO")
            self.logger.log(f"Direction: {self.config.direction}", "INFO")
            self.logger.log(f"Max Orders: {self.config.max_orders}", "INFO")
            self.logger.log(f"Wait Time: {self.config.wait_time}s", "INFO")
            self.logger.log(f"Exchange: {self.config.exchange}", "INFO")
            self.logger.log(f"Grid Step: {self.config.grid_step}%", "INFO")
            self.logger.log(f"Stop Price: {self.config.stop_price}", "INFO")
            self.logger.log(f"Pause Price: {self.config.pause_price}", "INFO")
            self.logger.log(f"Boost Mode: {self.config.boost_mode}", "INFO")
            self.logger.log(f"Dual-Sided Mode: {self.config.dual_sided}", "INFO")
            if self._roi_enabled():
                if self.config.tp_roi is not None:
                    self.logger.log(f"TP ROI: {self.config.tp_roi}%", "INFO")
                if self.config.sl_roi is not None:
                    self.logger.log(f"SL ROI: {self.config.sl_roi}%", "INFO")
                self.logger.log(f"ROI Poll Interval: {self.config.roi_poll_interval}s", "INFO")
                if self.config.roi_max_wait is not None:
                    self.logger.log(f"ROI Max Wait: {self.config.roi_max_wait}s", "INFO")
            self.logger.log("=============================", "INFO")

            # Capture the running event loop for thread-safe callbacks
            self.loop = asyncio.get_running_loop()
            # Connect to exchange
            await self.exchange_client.connect()

            # wait for connection to establish
            await asyncio.sleep(5)

            # Main trading loop
            while not self.shutdown_requested:
                # Update active orders
                active_orders = await self.exchange_client.get_active_orders(self.config.contract_id)

                # Filter close orders
                self.active_close_orders = []
                for order in active_orders:
                    if order.side == self.config.close_order_side:
                        self.active_close_orders.append({
                            'id': order.order_id,
                            'price': order.price,
                            'size': order.size
                        })

                # Periodic logging
                mismatch_detected = await self._log_status_periodically()

                stop_trading, pause_trading = await self._check_price_condition()
                if stop_trading:
                    msg = f"\n\nWARNING: [{self.config.exchange.upper()}_{self.config.ticker.upper()}] \n"
                    msg += "Stopped trading due to stop price triggered\n"
                    msg += "价格已经达到停止交易价格，脚本将停止交易\n"
                    await self.send_notification(msg.lstrip())
                    await self.graceful_shutdown(msg)
                    continue

                if pause_trading:
                    await asyncio.sleep(5)
                    continue

                if not mismatch_detected:
                    wait_time = self._calculate_wait_time()

                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        meet_grid_step_condition = await self._meet_grid_step_condition()
                        if not meet_grid_step_condition:
                            await asyncio.sleep(1)
                            continue

                        await self._place_and_monitor_open_order()
                        self.last_close_orders += 1

        except KeyboardInterrupt:
            self.logger.log("Bot stopped by user")
            await self.graceful_shutdown("User interruption (Ctrl+C)")
        except Exception as e:
            self.logger.log(f"Critical error: {e}", "ERROR")
            self.logger.log(f"Traceback: {traceback.format_exc()}", "ERROR")
            await self.graceful_shutdown(f"Critical error: {e}")
            raise
        finally:
            # Ensure all connections are closed even if graceful shutdown fails
            try:
                await self.exchange_client.disconnect()
            except Exception as e:
                self.logger.log(f"Error disconnecting from exchange: {e}", "ERROR")
