import asyncio
import logging
import os
import signal
import sys
import time
from decimal import Decimal
from typing import Any, Dict, Optional, Tuple

from rich import box
from rich.console import Console
from rich.live import Live
from rich.logging import RichHandler
from rich.table import Table

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exchanges.grvt import GrvtClient
from exchanges.bingx import BingxClient

from hedge.roi_utils import wait_for_roi_threshold


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
        roi_check_interval: float = 1.0,
        roi_timeout: Optional[float] = None,
    ):
        self.ticker = ticker.upper()
        self.order_quantity = order_quantity
        self.fill_timeout = fill_timeout
        self.iterations = iterations
        self.sleep_time = sleep_time
        self.tp_roi = tp_roi
        self.sl_roi = sl_roi
        self.roi_check_interval = roi_check_interval
        self.roi_timeout = roi_timeout

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
        self.last_fill_event = asyncio.Event()
        self.last_fill_price: Optional[Decimal] = None
        self.last_fill_side: Optional[str] = None

        os.makedirs("logs", exist_ok=True)
        self.log_filename = f"logs/grvt_bingx_{self.ticker.lower()}_hedge_log.txt"

        self.logger = logging.getLogger(f"hedge_grvt_bingx_{self.ticker}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.logger.handlers.clear()
        self.console = Console()

        file_handler = logging.FileHandler(self.log_filename)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)

        self.error_log_filename = f"logs/grvt_bingx_{self.ticker.lower()}_hedge_errors.txt"
        error_handler = logging.FileHandler(self.error_log_filename)
        error_handler.setLevel(logging.ERROR)
        error_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        error_handler.setFormatter(error_formatter)

        rich_handler = RichHandler(
            console=self.console,
            show_time=False,
            show_path=False,
            rich_tracebacks=False,
            markup=False
        )
        rich_handler.setLevel(logging.INFO)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
        self.logger.addHandler(rich_handler)

        self.live: Optional[Live] = None
        self.total_grvt_volume = Decimal('0')
        self.total_bingx_volume = Decimal('0')
        self.position_tolerance = Decimal('0.0001')
        self.max_bingx_retries = 5

        self.bingx_fill_event = asyncio.Event()
        self.last_bingx_fill: Optional[Dict[str, Any]] = None
        self.active_bingx_order_id: Optional[str] = None

        self.ui_state: Dict[str, Any] = {
            'phase': 'INIT',
            'iteration': 0,
            'grvt_position': Decimal('0'),
            'bingx_position': Decimal('0'),
            'grvt_volume': Decimal('0'),
            'bingx_volume': Decimal('0'),
            'last_action': 'idle',
            'last_message': '',
            'status': 'Idle'
        }

    # ------------------------------------------------------------------ #
    # Initialization helpers
    # ------------------------------------------------------------------ #

    def setup_signal_handlers(self) -> None:
        def handler(signum, frame):
            self.logger.info("Received shutdown signal, stopping hedge bot...")
            self.stop_flag = True

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def _fmt_decimal(self, value: Optional[Decimal]) -> str:
        if value is None:
            return "-"
        normalized = value.normalize()
        return format(normalized, 'f')

    def _render_ui(self) -> Table:
        table = Table(
            title=f"GRVT + BingX Hedge | {self.ticker}",
            box=box.SIMPLE_HEAVY,
            show_lines=True
        )
        table.add_column("Metric", justify="left", style="cyan", no_wrap=True)
        table.add_column("Value", justify="right", style="white")

        table.add_row("Status", self.ui_state['status'])
        table.add_row("Phase", self.ui_state['phase'])
        table.add_row("Iteration", f"{self.ui_state['iteration']}/{self.iterations}")
        table.add_row("GRVT Position", self._fmt_decimal(self.grvt_position))
        table.add_row("BingX Position", self._fmt_decimal(self.bingx_position))
        table.add_row("GRVT Volume", self._fmt_decimal(self.total_grvt_volume))
        table.add_row("BingX Volume", self._fmt_decimal(self.total_bingx_volume))
        table.add_row("Last Action", self.ui_state['last_action'])
        table.add_row("Message", self.ui_state['last_message'])
        return table

    def _refresh_ui(self) -> None:
        if self.live:
            self.live.update(self._render_ui(), refresh=True)

    def _update_ui(self, **kwargs: Any) -> None:
        self.ui_state.update(kwargs)
        self._refresh_ui()

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

    async def fetch_grvt_bbo_prices(self) -> Tuple[Decimal, Decimal]:
        assert self.grvt_client is not None
        assert self.grvt_contract_id is not None
        return await self.grvt_client.fetch_bbo_prices(self.grvt_contract_id)

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
            self.total_grvt_volume += filled_size

            self.last_grvt_fill = {
                'order_id': order_id,
                'side': side,
                'size': filled_size,
                'price': price
            }

            self.logger.info(
                f"[GRVT] FILLED {side.upper()} {filled_size} @ {price} | Position={self.grvt_position}"
            )
            self._update_ui(
                grvt_position=self.grvt_position,
                grvt_volume=self.total_grvt_volume,
                last_action=f"GRVT {side} fill",
                last_message=f"{filled_size} @ {price}",
                status="Active"
            )
            self.grvt_fill_event.set()

    def _handle_bingx_order_update(self, message: Dict[str, Any]) -> None:
        if self.bingx_contract_id is None:
            return
        if message.get('contract_id') != self.bingx_contract_id:
            return
        if message.get('order_type') != 'CLOSE':
            return

        status = message.get('status')
        order_id = message.get('order_id')
        side = message.get('side', '').lower()
        size = Decimal(str(message.get('size', '0')))
        price = Decimal(str(message.get('price', '0')))

        if status == 'FILLED':
            if side == 'buy':
                self.bingx_position += size
            else:
                self.bingx_position -= size
            self.total_bingx_volume += size
            self.last_bingx_fill = {
                'order_id': order_id,
                'side': side,
                'size': size,
                'price': price
            }
            if order_id == self.active_bingx_order_id:
                if not self.bingx_fill_event.is_set():
                    self.bingx_fill_event.set()
                self.active_bingx_order_id = None

            self.logger.info(
                f"[BINGX] FILLED {side.upper()} {size} @ {price} | Position={self.bingx_position}"
            )
            self._update_ui(
                bingx_position=self.bingx_position,
                bingx_volume=self.total_bingx_volume,
                last_action=f"BINGX {side} fill",
                last_message=f"{size} @ {price}",
                status="Hedging"
            )

    async def setup_grvt_websocket(self) -> None:
        assert self.grvt_client is not None
        self.grvt_client.setup_order_update_handler(self._handle_grvt_order_update)
        await self.grvt_client.connect()

    async def setup_bingx(self) -> None:
        assert self.bingx_client is not None
        self.bingx_client.setup_order_update_handler(self._handle_bingx_order_update)
        await self.bingx_client.connect()

    async def place_grvt_order(self, side: str) -> Optional[Dict[str, Any]]:
        assert self.grvt_client is not None
        assert self.grvt_contract_id is not None

        self.grvt_client.config.direction = side
        self.grvt_client.config.close_order_side = 'sell' if side == 'buy' else 'buy'

        self.grvt_fill_event.clear()
        self.last_grvt_fill = None
        self._update_ui(
            phase=f"{side.upper()} cycle",
            last_action=f"GRVT {side} order",
            last_message="Submitting...",
            status="Submitting"
        )

        order_result = await self.grvt_client.place_open_order(
            contract_id=self.grvt_contract_id,
            quantity=self.order_quantity,
            direction=side
        )

        if not order_result.success or not order_result.order_id:
            self.logger.error(f"[GRVT] Failed to place {side} order: {order_result.error_message}")
            self._update_ui(
                last_action="GRVT order failed",
                last_message=order_result.error_message or "Unknown error",
                status="Error"
            )
            return None

        self.logger.info(f"[GRVT] Order placed {order_result.order_id} ({side}) @ {order_result.price}")
        self._update_ui(
            last_action=f"GRVT {side} order placed",
            last_message=f"id={order_result.order_id}",
            status="Waiting fill"
        )

        if order_result.status == 'FILLED':
            self.last_grvt_fill = {
                'order_id': order_result.order_id,
                'side': side,
                'size': order_result.size,
                'price': order_result.price
            }
            self.last_fill_price = order_result.price
            self.last_fill_side = side
            if not self.last_fill_event.is_set():
                self.last_fill_event.set()
            return self.last_grvt_fill

        try:
            await asyncio.wait_for(self.grvt_fill_event.wait(), timeout=self.fill_timeout)
        except asyncio.TimeoutError:
            self.logger.warning(f"[GRVT] {side} order {order_result.order_id} timed out, cancelling")
            cancel_result = await self.grvt_client.cancel_order(order_result.order_id)
            if not cancel_result.success:
                self.logger.error(f"[GRVT] Failed to cancel order {order_result.order_id}: {cancel_result.error_message}")
            self._update_ui(
                last_action="GRVT timeout",
                last_message=f"Order {order_result.order_id} cancelled",
                status="Error"
            )
            return None

        if self.last_grvt_fill:
            self.last_fill_price = self.last_grvt_fill.get('price')
            self.last_fill_side = self.last_grvt_fill.get('side')
            if not self.last_fill_event.is_set():
                self.last_fill_event.set()

        return self.last_grvt_fill

    async def wait_for_last_fill(self, expected_side: Optional[str], timeout: Optional[float] = None) -> Optional[Decimal]:
        """Wait for the latest GRVT fill captured by websocket."""
        if timeout is None:
            timeout = float(self.fill_timeout)
        try:
            await asyncio.wait_for(self.last_fill_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            self.logger.warning("Timeout waiting for GRVT fill confirmation")
            return None

        price = Decimal(str(self.last_fill_price)) if self.last_fill_price is not None else None
        side = self.last_fill_side
        self.last_fill_event.clear()

        if expected_side and side != expected_side:
            self.logger.warning(f"Expected fill side {expected_side}, but received {side}")
        return price

    async def place_bingx_hedge(self, fill: Dict[str, Any]) -> bool:
        assert self.bingx_client is not None
        assert self.bingx_contract_id is not None

        side = fill['side']
        target_size = Decimal(str(fill['size']))

        hedge_side = 'sell' if side == 'buy' else 'buy'
        self.bingx_client.config.direction = hedge_side
        self.bingx_client.config.close_order_side = 'buy' if hedge_side == 'sell' else 'sell'
        reduce_only = (side == 'sell')  # close GRVT sell -> BingX buy closes short

        filled_total = Decimal('0')
        attempt = 0

        self._update_ui(
            last_action=f"BINGX hedge {hedge_side}",
            last_message=f"Target {target_size}",
            status="Hedging"
        )

        while filled_total < target_size and not self.stop_flag:
            remaining = target_size - filled_total
            attempt += 1
            if attempt > self.max_bingx_retries:
                self.logger.error("[BINGX] Max hedge attempts exceeded")
                break

            self.logger.info(f"[BINGX] Hedging attempt {attempt}: {hedge_side} {remaining}")
            order_result = await self.bingx_client.place_bbo_close_order(
                contract_id=self.bingx_contract_id,
                quantity=remaining,
                side=hedge_side,
                reduce_only=reduce_only
            )

            if not order_result.success or not order_result.order_id:
                self.logger.error(f"[BINGX] Hedge order failed: {order_result.error_message}")
                self._update_ui(
                    last_action="BINGX hedge failed",
                    last_message=order_result.error_message or "Unknown error",
                    status="Error"
                )
                return False

            self.active_bingx_order_id = order_result.order_id

            if order_result.status == 'FILLED':
                synthetic_message = {
                    'order_id': order_result.order_id,
                    'side': hedge_side,
                    'order_type': 'CLOSE',
                    'status': 'FILLED',
                    'size': str(remaining),
                    'price': str(order_result.price or Decimal('0')),
                    'contract_id': self.bingx_contract_id,
                    'filled_size': str(remaining)
                }
                self._handle_bingx_order_update(synthetic_message)
                filled_total = target_size
                break

            self.bingx_fill_event.clear()
            fill_confirmed = await self._wait_for_bingx_fill(order_result.order_id)
            if fill_confirmed:
                filled_total += remaining
                self.active_bingx_order_id = None
                continue

            # Timeout or mismatch - check current status
            info = await self.bingx_client.get_order_info(order_result.order_id)
            if info and info.status == 'FILLED':
                filled_total = target_size
                self.active_bingx_order_id = None
                continue

            partial_filled = Decimal('0')
            if info:
                partial_filled = info.filled_size
                filled_total = min(target_size, filled_total + partial_filled)

            await self.bingx_client.cancel_order(order_result.order_id)
            self.active_bingx_order_id = None
            self.logger.warning(
                f"[BINGX] Repricing hedge order {order_result.order_id}, "
                f"filled={filled_total}, remaining={target_size - filled_total}"
            )

        success = filled_total >= target_size
        if not success:
            self._update_ui(
                last_action="BINGX hedge incomplete",
                last_message="Continuing without full hedge",
                status="Warning"
            )
            self.logger.warning("[BINGX] Unable to complete hedge; continuing per relaxed mode")
            return False

        self.logger.info("[BINGX] Hedge cycle completed")
        self._update_ui(
            last_action="BINGX hedge complete",
            last_message=f"{target_size}",
            status="Hedged"
        )
        return True

    async def _wait_for_bingx_fill(self, order_id: str) -> bool:
        timeout = float(self.fill_timeout)
        if timeout <= 0:
            timeout = 5.0
        try:
            await asyncio.wait_for(self.bingx_fill_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return False
        finally:
            if self.bingx_fill_event.is_set():
                self.bingx_fill_event.clear()

        if self.last_bingx_fill and self.last_bingx_fill.get('order_id') == order_id:
            return True
        return False

    async def execute_cycle(self, side: str) -> Optional[Dict[str, Any]]:
        fill = await self.place_grvt_order(side)
        if not fill or self.stop_flag:
            return None

        await self.place_bingx_hedge(fill)
        if self.stop_flag:
            return None

        if self.sleep_time > 0 and not self.stop_flag:
            await asyncio.sleep(self.sleep_time)

        return fill

    # ------------------------------------------------------------------ #
    # Main run loop
    # ------------------------------------------------------------------ #

    async def trading_loop(self) -> None:
        self.logger.info(f"Starting GRVT+BingX hedge bot | ticker={self.ticker} | size={self.order_quantity}")
        self._update_ui(status="Connecting", phase="Initialization")

        self.initialize_clients()
        try:
            await self.load_contract_metadata()
            await self.setup_grvt_websocket()
            await self.setup_bingx()
            self._update_ui(status="Ready", phase="Idle")
        except Exception as exc:
            self.logger.error(f"Initialization failed: {exc}")
            self.stop_flag = True
            self._update_ui(status="Error", last_action="Init failed", last_message=str(exc))
            return

        await asyncio.sleep(2)

        iteration = 0
        while not self.stop_flag and iteration < self.iterations:
            iteration += 1
            self.logger.info(f"----- Iteration {iteration}/{self.iterations} -----")
            self._update_ui(iteration=iteration, phase="BUY cycle", status="Running")

            await self.execute_cycle('buy')
            if self.stop_flag:
                break

            entry_avg_price = await self.wait_for_last_fill(expected_side='buy')

            if entry_avg_price is not None and (self.tp_roi is not None or self.sl_roi is not None):
                self.logger.info(f"Waiting for ROI thresholds based on entry price {entry_avg_price} before sell cycle")
                self._update_ui(
                    phase="ROI wait",
                    last_action="Waiting ROI",
                    last_message=f"Entry {entry_avg_price}",
                    status="Monitoring"
                )
                trigger_reason = await wait_for_roi_threshold(
                    tp_roi=self.tp_roi,
                    sl_roi=self.sl_roi,
                    entry_side='buy',
                    avg_entry_price=entry_avg_price,
                    fetch_bbo_prices=self.fetch_grvt_bbo_prices,
                    logger=self.logger,
                    check_interval=self.roi_check_interval,
                    timeout=self.roi_timeout,
                    stop_condition=lambda: self.stop_flag
                )
                if trigger_reason:
                    self.logger.info(f"ROI wait finished due to: {trigger_reason}")
                    self._update_ui(last_action="ROI trigger", last_message=trigger_reason, status="Running")

            if self.stop_flag:
                break

            self._update_ui(phase="SELL cycle", status="Running")
            await self.execute_cycle('sell')
            if self.stop_flag:
                break

            await self.wait_for_last_fill(expected_side='sell')

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
        with Live(self._render_ui(), console=self.console, refresh_per_second=4) as live:
            self.live = live
            try:
                await self.trading_loop()
            except Exception as exc:
                self.logger.error(f"Unexpected error: {exc}")
                self._update_ui(status="Error", last_action="Runtime error", last_message=str(exc))
            finally:
                await self.cleanup()
                elapsed = time.time() - start_time
                self.logger.info(f"Hedge bot stopped after {elapsed:.1f}s")
                self._update_ui(status="Stopped", last_action="Shutdown", last_message=f"{elapsed:.1f}s elapsed")
                self.live = None
