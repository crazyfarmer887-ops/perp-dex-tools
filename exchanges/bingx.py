"""
BingX exchange client implementation using CCXT.
"""

import os
import asyncio
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

import ccxt  # type: ignore

from .base import BaseExchangeClient, OrderInfo, OrderResult, query_retry
from helpers.logger import TradingLogger


class BingxClient(BaseExchangeClient):
    """BingX exchange client implementation."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.api_key = os.getenv("BINGX_API_KEY")
        self.secret_key = os.getenv("BINGX_SECRET_KEY")

        # Initialize CCXT client
        self.client = ccxt.bingx(
            {
                "apiKey": self.api_key,
                "secret": self.secret_key,
                "enableRateLimit": True,
                "options": {
                    "defaultType": "swap",
                    "recvWindow": 5000,
                },
            }
        )

        self.logger = TradingLogger(exchange="bingx", ticker=self.config.ticker, log_to_console=False)
        self._market: Optional[Dict[str, Any]] = None
        self._order_update_handler = None
        self._polling_task: Optional[asyncio.Task] = None
        self._polling_active = False
        self._tracked_status: Dict[str, str] = {}

    # --------------------------------------------------------------------- #
    # Base class overrides
    # --------------------------------------------------------------------- #

    def _validate_config(self) -> None:
        """Validate BingX configuration."""
        required_env_vars = ["BINGX_API_KEY", "BINGX_SECRET_KEY"]
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")

    async def connect(self) -> None:
        """Prepare BingX client (load markets and start polling)."""
        await self._ensure_market()

        if self._order_update_handler and not self._polling_active:
            self._polling_active = True
            self._polling_task = asyncio.create_task(self._poll_orders())

    async def disconnect(self) -> None:
        """Disconnect BingX client."""
        self._polling_active = False
        if self._polling_task and not self._polling_task.done():
            self._polling_task.cancel()
            try:
                await self._polling_task
            except asyncio.CancelledError:
                pass
        # Close CCXT session
        try:
            await self._run_in_executor(self.client.close)
        except Exception as exc:
            self.logger.log(f"Error closing BingX client: {exc}", "ERROR")

    async def place_open_order(self, contract_id: str, quantity: Decimal, direction: str) -> OrderResult:
        """Place a post-only open order."""
        await self._ensure_market()

        best_bid, best_ask = await self.fetch_bbo_prices(contract_id)
        tick = self.config.tick_size

        if direction.lower() == "buy":
            price = best_ask - tick if best_ask > 0 else best_bid
            if price <= 0:
                price = best_bid
        elif direction.lower() == "sell":
            price = best_bid + tick if best_bid > 0 else best_ask
            if price <= 0:
                price = best_ask
        else:
            raise ValueError("Invalid direction for BingX order (expected 'buy' or 'sell')")

        price = self.round_to_tick(price)
        params = {"postOnly": True}
        order = await self._create_ccxt_order(contract_id, "limit", direction.lower(), quantity, price, params)
        order_info = self._ccxt_order_to_order_info(order)
        self._tracked_status[order_info.order_id] = order_info.status
        self._emit_order_update(order_info)

        return self._order_info_to_result(order_info)

    async def place_close_order(self, contract_id: str, quantity: Decimal, price: Decimal, side: str) -> OrderResult:
        """Place a reduce-only limit order for closing positions."""
        await self._ensure_market()

        params = {"reduceOnly": True}
        price = self.round_to_tick(price)
        order = await self._create_ccxt_order(contract_id, "limit", side.lower(), quantity, price, params)
        order_info = self._ccxt_order_to_order_info(order)
        self._tracked_status[order_info.order_id] = order_info.status
        self._emit_order_update(order_info)
        return self._order_info_to_result(order_info)

    async def cancel_order(self, order_id: str) -> OrderResult:
        """Cancel an existing order."""
        await self._ensure_market()

        try:
            await self._run_in_executor(self.client.cancel_order, order_id, self.config.contract_id)
            self._tracked_status.pop(order_id, None)
            return OrderResult(success=True, order_id=order_id, status="CANCELED")
        except Exception as exc:
            return OrderResult(success=False, order_id=order_id, status="ERROR", error_message=str(exc))

    @query_retry(reraise=True)
    async def get_order_info(self, order_id: str) -> Optional[OrderInfo]:
        """Fetch order information."""
        await self._ensure_market()
        order = await self._run_in_executor(self.client.fetch_order, order_id, self.config.contract_id)
        if not order:
            return None
        return self._ccxt_order_to_order_info(order)

    @query_retry(reraise=True)
    async def get_active_orders(self, contract_id: str) -> List[OrderInfo]:
        """Get active (open) orders."""
        await self._ensure_market()
        orders = await self._run_in_executor(self.client.fetch_open_orders, contract_id)
        return [self._ccxt_order_to_order_info(order) for order in orders]

    @query_retry(reraise=True)
    async def get_account_positions(self) -> Decimal:
        """Get absolute position size for the configured contract."""
        await self._ensure_market()
        positions = await self._run_in_executor(self.client.fetch_positions)

        for position in positions:
            symbol = position.get("symbol") or position.get("info", {}).get("symbol")
            if symbol != self.config.contract_id:
                continue

            size = (
                position.get("contracts")
                or position.get("positionAmt")
                or position.get("info", {}).get("positionAmt")
                or 0
            )

            try:
                return abs(Decimal(str(size)))
            except (InvalidOperation, TypeError):
                return Decimal("0")

        return Decimal("0")

    def setup_order_update_handler(self, handler) -> None:
        """Register order update handler (polling-based)."""
        self._order_update_handler = handler

    def get_exchange_name(self) -> str:
        """Return canonical exchange name."""
        return "bingx"

    # --------------------------------------------------------------------- #
    # Public helpers
    # --------------------------------------------------------------------- #

    async def get_contract_attributes(self) -> Tuple[str, Decimal]:
        """Resolve BingX contract ID and tick size."""
        await self._ensure_market()

        min_qty = self._extract_min_quantity(self._market)
        if min_qty and self.config.quantity < min_qty:
            raise ValueError(
                f"Order quantity is less than min quantity: {self.config.quantity} < {min_qty}"
            )

        return self.config.contract_id, self.config.tick_size

    async def place_market_order(
        self,
        contract_id: str,
        quantity: Decimal,
        side: str,
        reduce_only: bool = False,
    ) -> OrderResult:
        """Place a market order, typically used for hedging."""
        await self._ensure_market()

        params = {"reduceOnly": reduce_only} if reduce_only else {}
        order = await self._create_ccxt_order(contract_id, "market", side.lower(), quantity, None, params)
        order_info = self._ccxt_order_to_order_info(order)
        # Market orders are usually filled immediately
        self._emit_order_update(order_info)
        return self._order_info_to_result(order_info)

    async def fetch_bbo_prices(self, contract_id: str) -> Tuple[Decimal, Decimal]:
        """Fetch best bid and offer for the specified contract."""
        await self._ensure_market()
        order_book = await self._run_in_executor(self.client.fetch_order_book, contract_id, 20)

        bids = order_book.get("bids", [])
        asks = order_book.get("asks", [])

        best_bid = Decimal(str(bids[0][0])) if bids else Decimal("0")
        best_ask = Decimal(str(asks[0][0])) if asks else Decimal("0")

        return best_bid, best_ask

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    async def _run_in_executor(self, func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    async def _ensure_market(self) -> None:
        """Ensure market metadata is loaded."""
        if self._market is not None and self.config.contract_id and self.config.tick_size:
            return

        markets = await self._run_in_executor(self.client.load_markets)
        target_market = self._locate_market(markets)
        if not target_market:
            raise ValueError(f"BingX market not found for ticker {self.config.ticker}")

        self._market = target_market
        self.config.contract_id = target_market["symbol"]
        self.config.tick_size = self._determine_tick_size(target_market)

    def _locate_market(self, markets: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Locate the swap market for the configured ticker."""
        for market in markets.values():
            if not market.get("swap"):
                continue
            if market.get("base") != self.config.ticker:
                continue
            if market.get("quote") != "USDT":
                continue
            return market
        return None

    def _determine_tick_size(self, market: Dict[str, Any]) -> Decimal:
        """Determine the tick size from market metadata."""
        info = market.get("info", {})

        candidates = [
            info.get("tickSize"),
            info.get("priceTickSize"),
            info.get("minPriceIncrement"),
            market.get("limits", {}).get("price", {}).get("min"),
        ]

        for candidate in candidates:
            if candidate:
                try:
                    tick = Decimal(str(candidate))
                    if tick > 0:
                        return tick
                except (InvalidOperation, TypeError):
                    continue

        precision = market.get("precision", {}).get("price")
        if precision is not None:
            try:
                return Decimal("1") / (Decimal("10") ** Decimal(str(precision)))
            except (InvalidOperation, TypeError):
                pass

        # Fallback tick size
        return Decimal("0.01")

    def _extract_min_quantity(self, market: Optional[Dict[str, Any]]) -> Optional[Decimal]:
        if not market:
            return None
        min_amount = market.get("limits", {}).get("amount", {}).get("min")
        if not min_amount:
            return None
        try:
            min_qty = Decimal(str(min_amount))
            return min_qty if min_qty > 0 else None
        except (InvalidOperation, TypeError):
            return None

    async def _create_ccxt_order(
        self,
        contract_id: str,
        order_type: str,
        side: str,
        quantity: Decimal,
        price: Optional[Decimal],
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create an order via CCXT with executor handling."""
        params = params or {}
        amount = float(quantity)
        price_value = float(price) if price is not None else None
        return await self._run_in_executor(
            self.client.create_order,
            contract_id,
            order_type,
            side,
            amount,
            price_value,
            params,
        )

    def _ccxt_order_to_order_info(self, order: Dict[str, Any]) -> OrderInfo:
        """Convert CCXT order structure to OrderInfo."""
        order_id = order.get("id") or order.get("orderId") or order.get("clientOrderId")
        if not order_id:
            raise ValueError("Unable to determine order id from BingX order response")

        side = (order.get("side") or "").lower()
        amount = Decimal(str(order.get("amount") or order.get("origQty") or 0))
        filled = Decimal(str(order.get("filled") or order.get("executedQty") or 0))
        if amount == 0 and filled > 0:
            amount = filled

        price_value = order.get("price") or order.get("average") or order.get("avgPrice") or 0
        price = Decimal(str(price_value)) if price_value else Decimal("0")

        remaining = Decimal(str(order.get("remaining") or 0))
        if remaining == 0 and amount > 0:
            remaining = amount - filled
        if remaining < 0:
            remaining = Decimal("0")

        status_raw = (order.get("status") or "").lower()
        status_map = {
            "open": "OPEN",
            "closed": "FILLED",
            "canceled": "CANCELED",
            "cancelled": "CANCELED",
            "partial": "PARTIALLY_FILLED",
            "filled": "FILLED",
            "new": "OPEN",
        }
        status = status_map.get(status_raw, status_raw.upper() if status_raw else "UNKNOWN")

        return OrderInfo(
            order_id=str(order_id),
            side=side,
            size=amount,
            price=price,
            status=status,
            filled_size=filled,
            remaining_size=remaining,
        )

    def _order_info_to_result(self, order_info: OrderInfo) -> OrderResult:
        """Convert OrderInfo to OrderResult."""
        return OrderResult(
            success=True,
            order_id=order_info.order_id,
            side=order_info.side,
            size=order_info.size,
            price=order_info.price,
            status=order_info.status,
            filled_size=order_info.filled_size,
        )

    def _emit_order_update(self, order_info: OrderInfo) -> None:
        """Invoke registered order update handler."""
        if not self._order_update_handler:
            return

        order_type = "OPEN"
        if hasattr(self.config, "direction"):
            order_type = "OPEN" if order_info.side == getattr(self.config, "direction") else "CLOSE"
        elif hasattr(self.config, "close_order_side") and self.config.close_order_side:
            order_type = "OPEN" if order_info.side != self.config.close_order_side else "CLOSE"

        message = {
            "order_id": order_info.order_id,
            "side": order_info.side,
            "order_type": order_type,
            "status": order_info.status,
            "size": str(order_info.size),
            "price": str(order_info.price),
            "contract_id": self.config.contract_id,
            "filled_size": str(order_info.filled_size),
        }

        try:
            self._order_update_handler(message)
        except Exception as exc:
            self.logger.log(f"Error invoking order update handler: {exc}", "ERROR")

    async def _poll_orders(self) -> None:
        """Poll BingX for order status updates."""
        while self._polling_active:
            try:
                active_orders = await self.get_active_orders(self.config.contract_id)
                current_ids = {order.order_id for order in active_orders}

                for order in active_orders:
                    previous_status = self._tracked_status.get(order.order_id)
                    if previous_status != order.status:
                        self._tracked_status[order.order_id] = order.status
                        self._emit_order_update(order)

                finished_ids = [order_id for order_id in list(self._tracked_status.keys()) if order_id not in current_ids]
                for order_id in finished_ids:
                    order_info = await self.get_order_info(order_id)
                    if order_info:
                        self._tracked_status.pop(order_id, None)
                        self._emit_order_update(order_info)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self.logger.log(f"Order polling error: {exc}", "ERROR")
                await asyncio.sleep(2)
            finally:
                await asyncio.sleep(1)
