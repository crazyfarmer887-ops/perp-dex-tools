import asyncio
import os
import signal
import sys
import time
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple
import logging

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
        bingx_order_type: Optional[str] = None,
        bingx_limit_offset_ticks: Optional[Decimal] = None,
        bingx_attach_tp_sl: Optional[bool] = None,
        bingx_time_in_force: Optional[str] = None,
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
        self.pending_grvt_price: Optional[Tuple[str, Decimal]] = None

        config_warnings: List[str] = []

        def _coerce_decimal(value: Any, default: Decimal, label: str) -> Decimal:
            if value is None:
                return default
            try:
                return Decimal(str(value))
            except (InvalidOperation, ValueError):
                config_warnings.append(f"{label}='{value}' is invalid; falling back to {default}.")
                return default

        def _parse_bool(value: Optional[str], label: str) -> Optional[bool]:
            if value is None:
                return None
            normalized = value.strip().lower()
            if normalized in {'1', 'true', 't', 'yes', 'y', 'on'}:
                return True
            if normalized in {'0', 'false', 'f', 'no', 'n', 'off'}:
                return False
            config_warnings.append(f"{label}='{value}' is invalid; ignoring.")
            return None

        def _coerce_int(value: Any, default: int, label: str) -> int:
            if value is None:
                return default
            try:
                return int(str(value))
            except (ValueError, TypeError):
                config_warnings.append(f"{label}='{value}' is invalid; falling back to {default}.")
                return default

        def _coerce_float(value: Any, default: float, label: str) -> float:
            if value is None:
                return default
            try:
                return float(str(value))
            except (ValueError, TypeError):
                config_warnings.append(f"{label}='{value}' is invalid; falling back to {default}.")
                return default

        order_type_source = bingx_order_type or os.getenv('BINGX_HEDGE_ORDER_TYPE') or 'market'
        order_type_value = order_type_source.strip().lower()
        if order_type_value not in {'market', 'limit'}:
            config_warnings.append(
                f"BINGX hedge order type '{order_type_source}' is not supported; using 'market'."
            )
            order_type_value = 'market'
        self.bingx_hedge_order_type = order_type_value

        if bingx_limit_offset_ticks is not None:
            self.bingx_hedge_limit_offset_ticks = _coerce_decimal(
                bingx_limit_offset_ticks, Decimal('0'), 'bingx_limit_offset_ticks'
            )
        else:
            env_offset = os.getenv('BINGX_HEDGE_LIMIT_OFFSET_TICKS')
            self.bingx_hedge_limit_offset_ticks = _coerce_decimal(
                env_offset, Decimal('0'), 'BINGX_HEDGE_LIMIT_OFFSET_TICKS'
            )

        tif_source = bingx_time_in_force or os.getenv('BINGX_HEDGE_TIME_IN_FORCE')
        if tif_source:
            tif_value = tif_source.strip().upper()
            if tif_value in {'', 'NONE'}:
                tif_value = None
        else:
            tif_value = 'IOC' if self.bingx_hedge_order_type == 'limit' else None
        self.bingx_hedge_time_in_force = tif_value

        if bingx_attach_tp_sl is not None:
            attach_value = bool(bingx_attach_tp_sl)
        else:
            env_attach = _parse_bool(os.getenv('BINGX_HEDGE_ATTACH_TPSL'), 'BINGX_HEDGE_ATTACH_TPSL')
            if env_attach is None:
                attach_value = False
            else:
                attach_value = env_attach
        self.bingx_attach_tp_sl = attach_value

        default_cycle_retry_delay = float(self.sleep_time) if self.sleep_time > 0 else 3.0
        self.cycle_retry_delay = max(
            0.5,
            _coerce_float(
                os.getenv('GRVT_BINGX_CYCLE_RETRY_DELAY'),
                default_cycle_retry_delay,
                'GRVT_BINGX_CYCLE_RETRY_DELAY'
            )
        )
        self.max_cycle_retries = _coerce_int(
            os.getenv('GRVT_BINGX_MAX_CYCLE_RETRIES'),
            3,
            'GRVT_BINGX_MAX_CYCLE_RETRIES'
        )
        if self.max_cycle_retries < 0:
            config_warnings.append(
                f"GRVT_BINGX_MAX_CYCLE_RETRIES='{self.max_cycle_retries}' is invalid; using 0 (no limit)."
            )
            self.max_cycle_retries = 0

        self.hedge_retry_delay = max(
            0.5,
            _coerce_float(
                os.getenv('GRVT_BINGX_HEDGE_RETRY_DELAY'),
                1.0,
                'GRVT_BINGX_HEDGE_RETRY_DELAY'
            )
        )
        self.max_hedge_retries = _coerce_int(
            os.getenv('GRVT_BINGX_MAX_HEDGE_RETRIES'),
            3,
            'GRVT_BINGX_MAX_HEDGE_RETRIES'
        )
        if self.max_hedge_retries < 0:
            config_warnings.append(
                f"GRVT_BINGX_MAX_HEDGE_RETRIES='{self.max_hedge_retries}' is invalid; using 0 (no limit)."
            )
            self.max_hedge_retries = 0

        self.position_tolerance = _coerce_decimal(
            os.getenv('GRVT_BINGX_POSITION_TOLERANCE'),
            Decimal('0'),
            'GRVT_BINGX_POSITION_TOLERANCE'
        )
        if self.position_tolerance < 0:
            self.logger.warning(
                "GRVT_BINGX_POSITION_TOLERANCE='%s' is negative; using 0.",
                self.position_tolerance
            )
            self.position_tolerance = Decimal('0')

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

        for message in config_warnings:
            self.logger.warning(message)

        self.logger.info(
            "BingX hedge config | type=%s | limit_offset_ticks=%s | time_in_force=%s | attach_tp_sl=%s",
            self.bingx_hedge_order_type,
            self.bingx_hedge_limit_offset_ticks,
            self.bingx_hedge_time_in_force or 'DEFAULT',
            self.bingx_attach_tp_sl,
        )

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
        if message.get('order_type') != 'OPEN':
            return

        status = message.get('status')
        side = message.get('side', '').lower()
        filled_size = Decimal(str(message.get('filled_size', '0')))
        price = Decimal(str(message.get('price', '0')))
        order_id = message.get('order_id')

        if status == 'FILLED':
            if side == 'buy':
                self.grvt_position += filled_size
            else:
                self.grvt_position -= filled_size

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

    async def place_grvt_order(
        self,
        side: str,
        quantity: Optional[Decimal] = None,
        price_override: Optional[Decimal] = None
    ) -> Optional[Dict[str, Any]]:
        assert self.grvt_client is not None
        assert self.grvt_contract_id is not None

        self.grvt_client.config.direction = side
        self.grvt_client.config.close_order_side = 'sell' if side == 'buy' else 'buy'

        self.grvt_fill_event.clear()
        self.last_grvt_fill = None

        if price_override is not None:
            self.logger.info(f"[GRVT] Using override price {price_override} for {side} order")

        order_quantity = quantity if quantity is not None else self.order_quantity
        if order_quantity is None or order_quantity <= 0:
            self.logger.warning(
                "[GRVT] Invalid %s order quantity=%s; skipping order placement.",
                side,
                order_quantity
            )
            return None

        self.grvt_client.config.quantity = order_quantity

        order_result = await self.grvt_client.place_open_order(
            contract_id=self.grvt_contract_id,
            quantity=order_quantity,
            direction=side,
            price=price_override
        )

        if not order_result.success or not order_result.order_id:
            self.logger.error(f"[GRVT] Failed to place {side} order: {order_result.error_message}")
            return None

        self.logger.info(
            "[GRVT] Order placed %s (%s) qty=%s @ %s",
            order_result.order_id,
            side,
            order_quantity,
            order_result.price
        )

        if order_result.status == 'FILLED':
            if order_result.size is not None:
                if side == 'buy':
                    self.grvt_position += order_result.size
                else:
                    self.grvt_position -= order_result.size
            self.last_grvt_fill = {
                'order_id': order_result.order_id,
                'side': side,
                'size': order_result.size,
                'price': order_result.price
            }
            self.grvt_fill_event.set()
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

    async def place_bingx_hedge(self, fill: Dict[str, Any], is_closing: bool = False) -> bool:
        assert self.bingx_client is not None
        assert self.bingx_contract_id is not None

        side = str(fill['side']).lower()
        try:
            size = Decimal(str(fill['size']))
        except (InvalidOperation, ValueError, TypeError):
            self.logger.error(f"[BINGX] Invalid hedge size from fill: {fill.get('size')}")
            return False
        if size <= 0:
            self.logger.warning("[BINGX] Hedge size is non-positive; skipping hedge.")
            return False

        hedge_side = 'sell' if side == 'buy' else 'buy'
        self.bingx_client.config.direction = hedge_side
        self.bingx_client.config.close_order_side = 'buy' if hedge_side == 'sell' else 'sell'

        entry_price: Optional[Decimal]
        try:
            entry_price = Decimal(str(fill.get('price')))
        except (InvalidOperation, ValueError, TypeError):
            entry_price = None

        total_executed = Decimal('0')
        executed_orders: List[Tuple[str, Any]] = []

        # When closing positions, force limit orders only (no market fallback)
        if is_closing:
            self.logger.info("[BINGX] Closing position - using limit orders only (no market fallback)")
            limit_result = await self._place_bingx_limit_hedge(size, hedge_side, entry_price)
            if limit_result is not None:
                executed_orders.append(('limit', limit_result))
                filled_limit = self._extract_filled_size(limit_result)
                if filled_limit is None:
                    filled_limit = Decimal('0')
                if filled_limit > 0:
                    total_executed += filled_limit
                if filled_limit < size:
                    remaining = size - filled_limit
                    self.logger.warning(
                        "[BINGX] Limit hedge filled %s of %s; remaining %s will stay as open limit order.",
                        filled_limit,
                        size,
                        remaining
                    )
            else:
                self.logger.error("[BINGX] Failed to place limit hedge order for position close.")
                return False
        elif self.bingx_hedge_order_type == 'limit':
            limit_result = await self._place_bingx_limit_hedge(size, hedge_side, entry_price)
            if limit_result is not None:
                executed_orders.append(('limit', limit_result))
                filled_limit = self._extract_filled_size(limit_result)
                if filled_limit is None:
                    filled_limit = Decimal('0')
                if filled_limit > 0:
                    total_executed += filled_limit
                if filled_limit < size:
                    remaining = size - filled_limit
                    if remaining > 0:
                        self.logger.warning(
                            "[BINGX] Limit hedge filled %s of %s; executing market hedge for remaining %s.",
                            filled_limit,
                            size,
                            remaining
                        )
                        market_remaining = await self._place_bingx_market_hedge(
                            remaining,
                            hedge_side,
                            entry_price
                        )
                        if market_remaining is not None:
                            executed_orders.append(('market', market_remaining))
                            filled_market_remaining = self._extract_filled_size(market_remaining)
                            if filled_market_remaining is None or filled_market_remaining <= 0:
                                filled_market_remaining = remaining
                            total_executed += filled_market_remaining
            else:
                self.logger.info("[BINGX] Limit hedge unavailable; falling back to market order.")

        if self.bingx_hedge_order_type != 'limit' and total_executed == 0 and not is_closing:
            market_result = await self._place_bingx_market_hedge(size, hedge_side, entry_price)
            if market_result is not None:
                executed_orders.append(('market', market_result))
                filled_market = self._extract_filled_size(market_result)
                if filled_market is None or filled_market <= 0:
                    filled_market = size
                total_executed += filled_market
        elif total_executed == 0 and not is_closing:
            # Limit mode but nothing executed yet (e.g., limit failed completely)
            market_result = await self._place_bingx_market_hedge(size, hedge_side, entry_price)
            if market_result is not None:
                executed_orders.append(('market', market_result))
                filled_market = self._extract_filled_size(market_result)
                if filled_market is None or filled_market <= 0:
                    filled_market = size
                total_executed += filled_market

        if total_executed <= 0:
            self.logger.error("[BINGX] Failed to execute hedge order for %s %s.", hedge_side, size)
            return False

        for order_type, order_result in executed_orders:
            filled_size = self._extract_filled_size(order_result)
            if filled_size is None or filled_size <= 0:
                filled_size = order_result.size or Decimal('0')
            self.logger.info(
                "[BINGX] %s %s %s @ %s | status=%s",
                order_type.upper(),
                hedge_side.upper(),
                filled_size,
                order_result.price,
                order_result.status,
            )

        if hedge_side == 'buy':
            self.bingx_position += total_executed
        else:
            self.bingx_position -= total_executed

        self.logger.info(
            "[BINGX] Hedge complete | side=%s | executed=%s | position=%s",
            hedge_side.upper(),
            total_executed,
            self.bingx_position
        )
        return True

    async def _ensure_bingx_hedge(self, fill: Dict[str, Any], is_closing: bool = False) -> bool:
        attempts = 0
        while not self.stop_flag:
            attempts += 1

            hedge_success = await self.place_bingx_hedge(fill, is_closing=is_closing)
            if hedge_success:
                return True

            self.logger.warning(
                "[BINGX] Hedge attempt %s failed for %s %s. GRVT position=%s | BingX position=%s",
                attempts,
                fill.get('side'),
                fill.get('size'),
                self.grvt_position,
                self.bingx_position
            )

            if self.max_hedge_retries > 0 and attempts >= self.max_hedge_retries:
                self.logger.error(
                    "Exceeded max BingX hedge retries (%s); stopping to avoid unhedged exposure.",
                    self.max_hedge_retries
                )
                self.stop_flag = True
                return False

            await asyncio.sleep(self.hedge_retry_delay)

        return False

    async def _place_bingx_market_hedge(
        self,
        quantity: Decimal,
        hedge_side: str,
        entry_price: Optional[Decimal]
    ):
        assert self.bingx_client is not None
        assert self.bingx_contract_id is not None

        if quantity <= 0:
            return None

        take_profit_price, stop_loss_price = self._compute_tp_sl_targets(hedge_side, entry_price)
        self.logger.info(
            "[BINGX] Hedging %s %s via market order (tp=%s, sl=%s)",
            hedge_side,
            quantity,
            take_profit_price,
            stop_loss_price
        )

        try:
            result = await self.bingx_client.place_market_order(
                contract_id=self.bingx_contract_id,
                quantity=quantity,
                side=hedge_side,
                take_profit_price=take_profit_price,
                stop_loss_price=stop_loss_price,
                tp_sl_order_type='limit'
            )
        except Exception as exc:
            self.logger.error(f"[BINGX] Market hedge exception: {exc}")
            return None

        if not result.success:
            self.logger.error(f"[BINGX] Market hedge failed: {result.error_message}")
            return None

        return result

    def _compute_tp_sl_targets(
        self,
        hedge_side: str,
        entry_price: Optional[Decimal]
    ) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        if not self.bingx_attach_tp_sl:
            return None, None

        if entry_price is None or entry_price <= 0:
            self.logger.warning("⚠️ Cannot compute BingX TP/SL due to invalid entry price.")
            return None, None

        hundred = Decimal('100')
        take_profit_price: Optional[Decimal] = None
        stop_loss_price: Optional[Decimal] = None

        if self.tp_roi is not None:
            tp_factor = self.tp_roi / hundred
            if hedge_side == 'sell':
                take_profit_price = entry_price * (Decimal('1') - tp_factor)
            else:
                take_profit_price = entry_price * (Decimal('1') + tp_factor)

        if self.sl_roi is not None:
            sl_factor = self.sl_roi / hundred
            if hedge_side == 'sell':
                stop_loss_price = entry_price * (Decimal('1') + sl_factor)
            else:
                stop_loss_price = entry_price * (Decimal('1') - sl_factor)

        if take_profit_price is not None and take_profit_price <= 0:
            self.logger.warning("⚠️ Computed BingX take-profit price is non-positive; ignoring TP.")
            take_profit_price = None
        if stop_loss_price is not None and stop_loss_price <= 0:
            self.logger.warning("⚠️ Computed BingX stop-loss price is non-positive; ignoring SL.")
            stop_loss_price = None

        return take_profit_price, stop_loss_price

    async def _place_bingx_limit_hedge(
        self,
        quantity: Decimal,
        hedge_side: str,
        entry_price: Optional[Decimal]
    ):
        assert self.bingx_client is not None
        assert self.bingx_contract_id is not None

        try:
            best_bid, best_ask = await self.bingx_client.fetch_bbo_prices(self.bingx_contract_id)
        except Exception as exc:
            self.logger.warning(f"[BINGX] Failed to fetch BingX BBO for limit hedge: {exc}")
            return None

        tick_size = self.bingx_tick_size or getattr(self.bingx_client.config, 'tick_size', None) or Decimal('0.01')
        offset_ticks = self.bingx_hedge_limit_offset_ticks
        if offset_ticks < 0:
            offset_ticks = Decimal('0')
        price_offset = tick_size * offset_ticks

        if hedge_side == 'sell':
            if best_bid <= 0:
                self.logger.warning("[BINGX] Best bid is unavailable; cannot place limit hedge sell.")
                return None
            limit_price = best_bid - price_offset
        else:
            if best_ask <= 0:
                self.logger.warning("[BINGX] Best ask is unavailable; cannot place limit hedge buy.")
                return None
            limit_price = best_ask + price_offset

        if limit_price <= 0:
            self.logger.warning("[BINGX] Computed limit hedge price is non-positive; skipping limit hedge.")
            return None

        take_profit_price, stop_loss_price = self._compute_tp_sl_targets(hedge_side, entry_price)

        time_in_force = self.bingx_hedge_time_in_force
        if time_in_force and time_in_force.upper() == 'PO':
            self.logger.warning("BingX hedge time_in_force 'PO' conflicts with non-post-only limit; forcing 'IOC'.")
            time_in_force = 'IOC'

        self.logger.info(
            "[BINGX] Hedging %s %s via limit order @ %s (offset_ticks=%s, tif=%s, tp=%s, sl=%s)",
            hedge_side,
            quantity,
            limit_price,
            self.bingx_hedge_limit_offset_ticks,
            time_in_force or 'DEFAULT',
            take_profit_price,
            stop_loss_price,
        )

        try:
            return await self.bingx_client.place_limit_order(
                contract_id=self.bingx_contract_id,
                quantity=quantity,
                side=hedge_side,
                price=limit_price,
                reduce_only=False,
                post_only=False,
                time_in_force=time_in_force,
                take_profit_price=take_profit_price,
                stop_loss_price=stop_loss_price,
                tp_sl_order_type='limit'
            )
        except Exception as exc:
            self.logger.error(f"[BINGX] Limit hedge exception: {exc}")
            return None

    @staticmethod
    def _extract_filled_size(result) -> Optional[Decimal]:
        if result is None:
            return None
        filled = result.filled_size
        if filled is not None and isinstance(filled, Decimal):
            return filled
        if filled is not None:
            try:
                return Decimal(str(filled))
            except (InvalidOperation, ValueError, TypeError):
                return None
        if getattr(result, 'status', None) == 'FILLED' and getattr(result, 'size', None) is not None:
            try:
                return Decimal(str(result.size))
            except (InvalidOperation, ValueError, TypeError):
                return None
        return None

    def _reset_entry_state(self) -> None:
        self.current_entry_price = None
        self.current_entry_side = None
        self.current_entry_size = None
        self.current_entry_timestamp = None
        self.current_take_profit_price = None
        self.current_stop_loss_price = None
        self.last_roi_reason = None
        self.pending_grvt_price = None

    def _register_entry(self, fill: Dict[str, Any], previous_position: Decimal) -> None:
        tolerance = self.position_tolerance
        net_position = self.grvt_position

        if abs(net_position) <= tolerance:
            self.logger.info(
                "GRVT position flattened after %s fill; clearing ROI tracking.",
                str(fill.get('side', '')).upper()
            )
            self._reset_entry_state()
            return

        try:
            price = Decimal(str(fill.get('price')))
        except Exception:
            price = None

        side = 'buy' if net_position > 0 else 'sell'
        size = abs(net_position)

        flip_detected = (
            abs(previous_position) <= tolerance or
            (previous_position > 0 and net_position < 0) or
            (previous_position < 0 and net_position > 0)
        )

        if flip_detected:
            if price is None or price <= 0:
                self.logger.warning("⚠️ Unable to register new %s entry due to invalid fill price.", side.upper())
                self._reset_entry_state()
                return
            self.current_entry_price = price
        elif self.current_entry_price is None and price is not None and price > 0:
            self.current_entry_price = price

        self.current_entry_side = side
        self.current_entry_size = size
        self.current_entry_timestamp = time.time()
        self.last_roi_reason = None

        if self.current_entry_price is not None:
            self._update_roi_targets(side, self.current_entry_price)
        else:
            self.logger.warning("⚠️ ROI targets disabled due to missing entry price for %s position.", side.upper())

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

    def _target_position_for_side(self, side: str) -> Decimal:
        normalized = side.strip().lower()
        if normalized == 'buy':
            return self.order_quantity
        return -self.order_quantity

    def _compute_trade_delta(self, target_side: str) -> Decimal:
        target = self._target_position_for_side(target_side)
        return target - self.grvt_position

    def _schedule_next_grvt_order_price(self, trigger: str) -> None:
        if self.current_entry_side is None:
            return

        if trigger not in ('take_profit', 'stop_loss'):
            self.pending_grvt_price = None
            return

        target_price = (
            self.current_take_profit_price if trigger == 'take_profit' else self.current_stop_loss_price
        )

        if target_price is None or target_price <= 0:
            self.pending_grvt_price = None
            if target_price is not None and target_price <= 0:
                self.logger.warning("⚠️ Computed ROI target price is non-positive; skipping override.")
            return

        side = 'sell' if self.current_entry_side == 'buy' else 'buy'

        rounded_price = target_price
        if self.grvt_client is not None:
            try:
                rounded_price = self.grvt_client.round_to_tick(target_price)
            except Exception as exc:
                self.logger.warning(f"⚠️ Failed to round ROI target price: {exc}")
                rounded_price = target_price

        self.pending_grvt_price = (side, rounded_price)
        self.logger.info(
            f"📌 Scheduling next GRVT {side.upper()} order @ {rounded_price} due to {trigger.replace('_', ' ')}"
        )

    def _positions_are_flat(self) -> bool:
        tolerance = self.position_tolerance
        net_exposure = self.grvt_position + self.bingx_position
        return abs(net_exposure) <= tolerance

    async def wait_for_roi(self) -> None:
        if self.stop_flag:
            return
        if self.tp_roi is None and self.sl_roi is None:
            return
        if self.current_entry_price is None or self.current_entry_side is None:
            return
        if abs(self.grvt_position) <= self.position_tolerance:
            self.logger.info("No active GRVT position to monitor for ROI; skipping.")
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
                    self.logger.info(f"🎯 ROI take profit reached: {roi_float:.4f}% (target {self.tp_roi}%)")
                    self._schedule_next_grvt_order_price('take_profit')
                    return

                if stop_loss_hit:
                    self.last_roi_reason = f"stop_loss ({roi_float:.4f}%)"
                    self.logger.info(f"🛑 ROI stop loss reached: {roi_float:.4f}% (threshold -{self.sl_roi}%)")
                    self._schedule_next_grvt_order_price('stop_loss')
                    return

            elapsed = time.time() - start_time
            if elapsed >= self.max_roi_wait:
                self.last_roi_reason = f"timeout ({elapsed:.1f}s)"
                self.logger.info(f"⏱️ ROI wait timed out after {elapsed:.1f}s; proceeding to next cycle.")
                self.pending_grvt_price = None
                return

            await asyncio.sleep(self.roi_poll_interval)

    async def execute_cycle(self, side: str) -> bool:
        price_override = None
        if self.pending_grvt_price and self.pending_grvt_price[0] == side:
            price_override = self.pending_grvt_price[1]

        trade_delta = self._compute_trade_delta(side)
        if trade_delta == 0:
            self.logger.info(
                "Skipping %s cycle; GRVT position already at target (position=%s).",
                side.upper(),
                self.grvt_position
            )
            if self.pending_grvt_price and self.pending_grvt_price[0] == side:
                self.pending_grvt_price = None
            return True

        trade_side = 'buy' if trade_delta > 0 else 'sell'
        trade_quantity = abs(trade_delta)

        if trade_quantity <= 0:
            self.logger.warning("Computed non-positive trade quantity for %s cycle; aborting.", side.upper())
            return False

        if trade_side != side:
            self.logger.warning(
                "Target side %s requires executing %s to rebalance positions (delta=%s).",
                side.upper(),
                trade_side.upper(),
                trade_delta
            )

        if price_override is not None and self.pending_grvt_price and self.pending_grvt_price[0] != trade_side:
            self.logger.warning(
                "Pending GRVT price scheduled for %s but actual trade side is %s; ignoring override.",
                self.pending_grvt_price[0].upper(),
                trade_side.upper()
            )
            price_override = None

        previous_position = self.grvt_position

        # Detect if we're closing a position
        # If previous_position > 0 and we're selling, we're closing a long
        # If previous_position < 0 and we're buying, we're closing a short
        tolerance = self.position_tolerance
        is_closing = (
            (previous_position > tolerance and trade_side == 'sell') or
            (previous_position < -tolerance and trade_side == 'buy')
        )

        if is_closing:
            self.logger.info(
                "[GRVT] Closing position - using limit order (post-only) | previous_position=%s | trade_side=%s",
                previous_position,
                trade_side.upper()
            )

        fill = await self.place_grvt_order(trade_side, quantity=trade_quantity, price_override=price_override)
        if not fill or self.stop_flag:
            self._reset_entry_state()
            return False

        if price_override is not None and self.pending_grvt_price and self.pending_grvt_price[0] == trade_side:
            self.pending_grvt_price = None

        self._register_entry(fill, previous_position)
        hedge_success = await self._ensure_bingx_hedge(fill, is_closing=is_closing)
        if not hedge_success:
            self.logger.error(
                "Unable to complete BingX hedge for GRVT %s fill; halting cycle to avoid exposure.",
                trade_side.upper()
            )
            self._reset_entry_state()
            return False

        if self.sleep_time > 0 and not self.stop_flag:
            await asyncio.sleep(self.sleep_time)

        return True

    async def _run_cycle_phase(self, side: str) -> bool:
        attempt = 0
        while not self.stop_flag:
            attempt += 1
            success = await self.execute_cycle(side)
            if success:
                return True

            self.logger.warning(
                "Cycle %s attempt %s failed; GRVT position=%s | BingX position=%s",
                side.upper(),
                attempt,
                self.grvt_position,
                self.bingx_position
            )

            if self.max_cycle_retries > 0 and attempt >= self.max_cycle_retries:
                self.logger.error(
                    "Exceeded max retries (%s) for %s cycle; stopping to prevent compounding positions.",
                    self.max_cycle_retries,
                    side.upper()
                )
                self.stop_flag = True
                return False

            await asyncio.sleep(self.cycle_retry_delay)

        return False

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

            buy_completed = await self._run_cycle_phase('buy')
            if self.stop_flag or not buy_completed:
                break

            await self.wait_for_roi()
            if self.stop_flag:
                break

            sell_completed = await self._run_cycle_phase('sell')
            if self.stop_flag or not sell_completed:
                break

            await self.wait_for_roi()
            if self.stop_flag:
                break

            if not self._positions_are_flat():
                self.logger.error(
                    "Residual positions detected after iteration %s | GRVT=%s | BingX=%s. Halting to prevent compounding.",
                    iteration,
                    self.grvt_position,
                    self.bingx_position
                )
                self.stop_flag = True
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
