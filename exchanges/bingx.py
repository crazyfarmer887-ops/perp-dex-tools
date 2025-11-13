"""
BingX exchange client implementation using ccxt.
"""

import asyncio
import os
from contextlib import suppress
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Optional, Set, Tuple

import ccxt.async_support as ccxt_async  # type: ignore
from ccxt.base.errors import (ExchangeError, NetworkError, RequestTimeout)  # type: ignore

from .base import BaseExchangeClient, OrderInfo, OrderResult, query_retry
from helpers.logger import TradingLogger


def _to_decimal(value: Any, default: str = "0") -> Decimal:
    """Safely convert a value to Decimal."""
    if value is None:
        return Decimal(default)
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return Decimal(default)


class BingxClient(BaseExchangeClient):
    """BingX exchange client implementation."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.api_key = os.getenv("BINGX_API_KEY")
        self.secret_key = os.getenv("BINGX_SECRET_KEY")
        self.recv_window = int(os.getenv("BINGX_RECV_WINDOW", "5000"))
        self.testnet = os.getenv("BINGX_TESTNET", "false").lower() in ("1", "true", "yes")

        if not self.api_key or not self.secret_key:
            raise ValueError("BINGX_API_KEY and BINGX_SECRET_KEY must be set in environment variables")

        self.logger = TradingLogger(exchange="bingx", ticker=self.config.ticker, log_to_console=False)

        self.client = ccxt_async.bingx(
            {
                "apiKey": self.api_key,
                "secret": self.secret_key,
                "enableRateLimit": True,
                "options": {"defaultType": "swap", "recvWindow": self.recv_window},
            }
        )

        # Enable sandbox if requested
        with suppress(Exception):
            self.client.set_sandbox_mode(self.testnet)

        self._order_update_handler = None
        self._monitor_tasks: Set[asyncio.Task] = set()
        self._poll_interval = float(os.getenv("BINGX_ORDER_POLL_INTERVAL", "1"))
        self._loop = None

    def _validate_config(self) -> None:
        """Validate required configuration."""
        required_env_vars = ["BINGX_API_KEY", "BINGX_SECRET_KEY"]
        missing = [var for var in required_env_vars if not os.getenv(var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {missing}")

    async def connect(self) -> None:
        """Connect to BingX (load markets)."""
        await self.client.load_markets()
        self._loop = asyncio.get_running_loop()

    async def disconnect(self) -> None:
        """Disconnect and cleanup."""
        for task in list(self._monitor_tasks):
            task.cancel()
        if self._monitor_tasks:
            await asyncio.gather(*self._monitor_tasks, return_exceptions=True)
        await self.client.close()
        self._monitor_tasks.clear()

    def get_exchange_name(self) -> str:
        return "bingx"

    def setup_order_update_handler(self, handler) -> None:
        """Store order update handler for polling notifications."""
        self._order_update_handler = handler

    # --- Internal helpers -------------------------------------------------

    def _map_status(self, status: Optional[str], filled: Decimal, remaining: Decimal) -> str:
        if not status:
            return "UNKNOWN"
        status = status.lower()
        if status in {"open", "new"}:
            if filled > 0 and remaining > 0:
                return "PARTIALLY_FILLED"
            return "OPEN"
        if status in {"closed", "filled"}:
            return "FILLED"
        if status in {"canceled", "cancelled"}:
            if filled > 0:
                return "FILLED"
            return "CANCELED"
        if status in {"pending", "accepted"}:
            return "PENDING"
        if status in {"expired"}:
            return "CANCELED"
        return status.upper()

    def _determine_order_type(self, side: str) -> str:
        close_side = getattr(self.config, "close_order_side", None)
        if close_side and side == close_side:
            return "CLOSE"
        return "OPEN"

    def _notify_order_update(self, info: OrderInfo) -> None:
        if not self._order_update_handler:
            return

        message = {
            "order_id": info.order_id,
            "side": info.side,
            "order_type": self._determine_order_type(info.side),
            "status": info.status,
            "size": str(info.size),
            "price": str(info.price),
            "contract_id": self.config.contract_id,
            "filled_size": str(info.filled_size or Decimal("0")),
        }

        try:
            self._order_update_handler(message)
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.log(f"Error in order update handler: {exc}", "ERROR")

    def _start_order_monitor(self, order_id: str) -> None:
        if not self._order_update_handler:
            return

        async def monitor():
            try:
                last_status = None
                while True:
                    order_info = await self.get_order_info(order_id)
                    if not order_info:
                        break

                    status_changed = order_info.status != last_status
                    fill_changed = order_info.filled_size and order_info.filled_size > 0
                    if status_changed or fill_changed:
                        self._notify_order_update(order_info)
                        last_status = order_info.status

                    if order_info.status in {"FILLED", "CANCELED"}:
                        break

                    await asyncio.sleep(self._poll_interval)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.log(f"Order monitor error for {order_id}: {exc}", "ERROR")

        task = asyncio.create_task(monitor())
        self._monitor_tasks.add(task)
        task.add_done_callback(lambda t: self._monitor_tasks.discard(t))

    def _parse_order(self, order: Dict[str, Any]) -> OrderInfo:
        size = _to_decimal(order.get("amount"))
        price = _to_decimal(order.get("price") or order.get("average"))
        remaining = _to_decimal(order.get("remaining"))
        filled = _to_decimal(order.get("filled"))

        status = self._map_status(order.get("status"), filled, remaining)
        return OrderInfo(
            order_id=order.get("id", ""),
            side=(order.get("side") or "").lower(),
            size=size,
            price=price,
            status=status,
            filled_size=filled,
            remaining_size=remaining,
            cancel_reason=order.get("info", {}).get("reason", ""),
        )

    def _with_precision(self, symbol: str, amount: Decimal, price: Optional[Decimal] = None) -> Tuple[str, Optional[str]]:
        raw_amount = float(amount)
        amount_precise = self.client.amount_to_precision(symbol, raw_amount)
        price_precise: Optional[str] = None
        if price is not None:
            price_precise = self.client.price_to_precision(symbol, float(price))
        return amount_precise, price_precise

    async def _create_limit_order(
        self, contract_id: str, side: str, quantity: Decimal, price: Decimal, reduce_only: bool
    ) -> OrderInfo:
        amount_precise, price_precise = self._with_precision(contract_id, quantity, price)
        params = {
            "reduceOnly": reduce_only,
            "timeInForce": "PostOnly",
        }

        try:
            order = await self.client.create_order(
                contract_id,
                type="limit",
                side=side,
                amount=amount_precise,
                price=price_precise,
                params=params,
            )
        except (ExchangeError, NetworkError, RequestTimeout) as exc:
            raise Exception(f"BingX limit order failed: {exc}")

        return self._parse_order(order)

    # --- Public API -------------------------------------------------------

    @query_retry(reraise=True)
    async def fetch_bbo_prices(self, contract_id: str) -> Tuple[Decimal, Decimal]:
        order_book = await self.client.fetch_order_book(contract_id, limit=5)
        bids = order_book.get("bids") or []
        asks = order_book.get("asks") or []

        best_bid = _to_decimal(bids[0][0]) if bids else Decimal("0")
        best_ask = _to_decimal(asks[0][0]) if asks else Decimal("0")
        return best_bid, best_ask

    async def place_open_order(self, contract_id: str, quantity: Decimal, direction: str) -> OrderResult:
        best_bid, best_ask = await self.fetch_bbo_prices(contract_id)
        if best_bid <= 0 or best_ask <= 0 or best_bid >= best_ask:
            return OrderResult(success=False, error_message="Invalid bid/ask prices")

        if direction == "buy":
            target_price = min(best_bid, best_ask - self.config.tick_size)
        elif direction == "sell":
            target_price = max(best_ask, best_bid + self.config.tick_size)
        else:
            return OrderResult(success=False, error_message=f"Unsupported side: {direction}")

        target_price = self.round_to_tick(max(target_price, self.config.tick_size))

        try:
            order_info = await self._create_limit_order(contract_id, direction, quantity, target_price, reduce_only=False)
        except Exception as exc:
            self.logger.log(f"[OPEN] Error placing order: {exc}", "ERROR")
            return OrderResult(success=False, error_message=str(exc))

        if order_info.order_id:
            self._start_order_monitor(order_info.order_id)

        return OrderResult(
            success=True,
            order_id=order_info.order_id,
            side=order_info.side,
            size=order_info.size,
            price=order_info.price,
            status=order_info.status,
        )

    async def place_close_order(self, contract_id: str, quantity: Decimal, price: Decimal, side: str) -> OrderResult:
        adjusted_price = self.round_to_tick(max(price, self.config.tick_size))

        # Ensure maker behavior by nudging price if needed
        best_bid, best_ask = await self.fetch_bbo_prices(contract_id)
        if side == "sell" and adjusted_price < best_ask:
            adjusted_price = self.round_to_tick(max(best_ask, best_bid + self.config.tick_size))
        elif side == "buy" and adjusted_price > best_bid:
            adjusted_price = self.round_to_tick(min(best_bid, best_ask - self.config.tick_size))

        try:
            order_info = await self._create_limit_order(contract_id, side, quantity, adjusted_price, reduce_only=True)
        except Exception as exc:
            self.logger.log(f"[CLOSE] Error placing order: {exc}", "ERROR")
            return OrderResult(success=False, error_message=str(exc))

        if order_info.order_id:
            self._start_order_monitor(order_info.order_id)

        return OrderResult(
            success=True,
            order_id=order_info.order_id,
            side=order_info.side,
            size=order_info.size,
            price=order_info.price,
            status=order_info.status,
        )

    async def cancel_order(self, order_id: str) -> OrderResult:
        try:
            order = await self.client.cancel_order(order_id, symbol=self.config.contract_id)
            info = self._parse_order(order)
            self._notify_order_update(info)
            return OrderResult(success=True, status=info.status, filled_size=info.filled_size)
        except (ExchangeError, NetworkError, RequestTimeout) as exc:
            return OrderResult(success=False, error_message=str(exc))

    @query_retry(reraise=True)
    async def get_order_info(self, order_id: str) -> Optional[OrderInfo]:
        order = await self.client.fetch_order(order_id, symbol=self.config.contract_id)
        if not order:
            return None
        return self._parse_order(order)

    @query_retry(reraise=True)
    async def get_active_orders(self, contract_id: str) -> List[OrderInfo]:
        orders = await self.client.fetch_open_orders(symbol=contract_id)
        return [self._parse_order(order) for order in orders]

    @query_retry(reraise=True)
    async def get_account_positions(self) -> Decimal:
        try:
            positions = await self.client.fetch_positions(symbols=[self.config.contract_id])
        except AttributeError:
            positions = await self.client.fetch_positions()

        for position in positions:
            if position.get("symbol") != self.config.contract_id:
                continue
            contracts = position.get("contracts") or position.get("positionAmt") or position.get("size")
            return abs(_to_decimal(contracts))

        return Decimal("0")

    async def place_market_order(self, contract_id: str, quantity: Decimal, direction: str) -> OrderResult:
        amount_precise, _ = self._with_precision(contract_id, quantity)
        params = {"reduceOnly": direction == self.config.close_order_side}

        try:
            order = await self.client.create_order(
                contract_id,
                type="market",
                side=direction,
                amount=amount_precise,
                price=None,
                params=params,
            )
        except (ExchangeError, NetworkError, RequestTimeout) as exc:
            return OrderResult(success=False, error_message=str(exc))

        info = self._parse_order(order)
        self._notify_order_update(info)
        return OrderResult(
            success=True,
            order_id=info.order_id,
            side=info.side,
            size=info.size,
            price=info.price,
            status=info.status,
        )

    async def get_order_price(self, direction: str) -> Decimal:
        best_bid, best_ask = await self.fetch_bbo_prices(self.config.contract_id)
        if best_bid <= 0 or best_ask <= 0:
            raise ValueError("Invalid bid/ask prices")

        if direction == "buy":
            price = min(best_bid, best_ask - self.config.tick_size)
        elif direction == "sell":
            price = max(best_ask, best_bid + self.config.tick_size)
        else:
            raise ValueError("Invalid direction")

        return self.round_to_tick(max(price, self.config.tick_size))

    async def get_contract_attributes(self) -> Tuple[str, Decimal]:
        await self.client.load_markets()
        ticker = self.config.ticker.upper()
        for market in self.client.markets.values():
            if not market.get("swap"):
                continue
            if market.get("base") == ticker and market.get("quote") == "USDT":
                self.config.contract_id = market["symbol"]

                tick_size = market.get("precision", {}).get("price")
                if isinstance(tick_size, int):
                    self.config.tick_size = Decimal("1") / (Decimal("10") ** tick_size)
                elif tick_size:
                    self.config.tick_size = _to_decimal(tick_size, "0.0001")
                else:
                    self.config.tick_size = _to_decimal(
                        market.get("limits", {}).get("price", {}).get("min"), "0.0001"
                    )

                min_amount = market.get("limits", {}).get("amount", {}).get("min")
                if min_amount and self.config.quantity < Decimal(str(min_amount)):
                    raise ValueError(
                        f"Order quantity is less than min quantity: {self.config.quantity} < {Decimal(str(min_amount))}"
                    )

                return self.config.contract_id, self.config.tick_size

        raise ValueError(f"USDT perpetual contract not found for ticker: {ticker}")
