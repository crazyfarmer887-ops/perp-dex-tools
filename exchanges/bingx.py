"""
BingX exchange client implementation using ccxt.
"""

from __future__ import annotations

import os
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import ccxt.async_support as ccxt  # type: ignore

from .base import BaseExchangeClient, OrderInfo, OrderResult, query_retry
from helpers.logger import TradingLogger


class BingxClient(BaseExchangeClient):
    """BingX exchange client implementation."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.api_key = os.getenv("BINGX_API_KEY")
        self.secret_key = os.getenv("BINGX_SECRET_KEY")
        self.sub_account = os.getenv("BINGX_SUB_ACCOUNT")
        self.default_type = os.getenv("BINGX_DEFAULT_TYPE", "swap")

        self.logger = TradingLogger(exchange="bingx", ticker=self.config.ticker, log_to_console=False)
        self._order_update_handler = None
        self._markets_loaded = False

        exchange_options: Dict[str, Any] = {
            "apiKey": self.api_key,
            "secret": self.secret_key,
            "enableRateLimit": True,
            "options": {
                "defaultType": self.default_type,
            },
        }

        if self.sub_account:
            exchange_options["headers"] = {"X-BX-SUBACCT": self.sub_account}

        self.exchange = ccxt.bingx(exchange_options)

    def _validate_config(self) -> None:
        required_env_vars = ["BINGX_API_KEY", "BINGX_SECRET_KEY"]
        missing_vars = [env for env in required_env_vars if not os.getenv(env)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    async def _ensure_markets_loaded(self) -> None:
        if not self._markets_loaded:
            await self.exchange.load_markets()
            self._markets_loaded = True

    async def connect(self) -> None:
        """BingX does not require an explicit WebSocket connection for REST operations."""
        try:
            await self._ensure_markets_loaded()
        except Exception as exc:
            self.logger.log(f"Error loading BingX markets: {exc}", "ERROR")
            raise

    async def disconnect(self) -> None:
        """Close the ccxt client session."""
        try:
            await self.exchange.close()
        except Exception as exc:
            self.logger.log(f"Error closing BingX client: {exc}", "ERROR")

    def get_exchange_name(self) -> str:
        return "bingx"

    def setup_order_update_handler(self, handler) -> None:
        """Store the handler for future WebSocket support."""
        self._order_update_handler = handler
        self.logger.log(
            "Order update handler registered, but BingX WebSocket streaming is not implemented yet.",
            "WARNING",
        )

    async def _to_exchange_amount(self, contract_id: str, quantity: Decimal) -> Decimal:
        await self._ensure_markets_loaded()
        amount_str = self.exchange.amount_to_precision(contract_id, float(quantity))
        return Decimal(str(amount_str))

    async def _to_exchange_price(self, contract_id: str, price: Decimal) -> Decimal:
        await self._ensure_markets_loaded()
        price_str = self.exchange.price_to_precision(contract_id, float(price))
        return Decimal(str(price_str))

    @query_retry(reraise=True)
    async def fetch_bbo_prices(self, contract_id: str) -> Tuple[Decimal, Decimal]:
        await self._ensure_markets_loaded()
        order_book = await self.exchange.fetch_order_book(contract_id, limit=5)

        bids = order_book.get("bids", [])
        asks = order_book.get("asks", [])

        best_bid = Decimal(str(bids[0][0])) if bids else Decimal("0")
        best_ask = Decimal(str(asks[0][0])) if asks else Decimal("0")

        if best_bid <= 0 or best_ask <= 0:
            raise ValueError("Invalid bid/ask data from BingX")

        return best_bid, best_ask

    async def place_open_order(self, contract_id: str, quantity: Decimal, direction: str) -> OrderResult:
        try:
            if direction not in {"buy", "sell"}:
                raise ValueError(f"Invalid direction: {direction}")

            best_bid, best_ask = await self.fetch_bbo_prices(contract_id)

            if direction == "buy":
                price = best_ask - self.config.tick_size
            else:
                price = best_bid + self.config.tick_size

            price = self.round_to_tick(price)
            amount_dec = await self._to_exchange_amount(contract_id, quantity)
            price_dec = await self._to_exchange_price(contract_id, price)

            order = await self.exchange.create_order(
                symbol=contract_id,
                type="limit",
                side=direction,
                amount=float(amount_dec),
                price=float(price_dec),
                params={"postOnly": True, "reduceOnly": False},
            )

            return self._to_order_result(order, direction, amount_dec, price_dec)
        except Exception as exc:
            self.logger.log(f"[OPEN] Error placing BingX order: {exc}", "ERROR")
            return OrderResult(success=False, error_message=str(exc))

    async def place_close_order(self, contract_id: str, quantity: Decimal, price: Decimal, side: str) -> OrderResult:
        try:
            if side not in {"buy", "sell"}:
                raise ValueError(f"Invalid side: {side}")

            best_bid, best_ask = await self.fetch_bbo_prices(contract_id)

            if side == "sell" and price <= best_bid:
                adjusted_price = best_bid + self.config.tick_size
            elif side == "buy" and price >= best_ask:
                adjusted_price = best_ask - self.config.tick_size
            else:
                adjusted_price = price

            adjusted_price = self.round_to_tick(adjusted_price)

            amount_dec = await self._to_exchange_amount(contract_id, quantity)
            price_dec = await self._to_exchange_price(contract_id, adjusted_price)

            order = await self.exchange.create_order(
                symbol=contract_id,
                type="limit",
                side=side,
                amount=float(amount_dec),
                price=float(price_dec),
                params={"postOnly": True, "reduceOnly": True},
            )
            return self._to_order_result(order, side, amount_dec, price_dec)
        except Exception as exc:
            self.logger.log(f"[CLOSE] Error placing BingX order: {exc}", "ERROR")
            return OrderResult(success=False, error_message=str(exc))

    async def place_market_order(
        self,
        contract_id: str,
        quantity: Decimal,
        direction: str,
        reduce_only: bool = False,
    ) -> OrderResult:
        try:
            if direction not in {"buy", "sell"}:
                raise ValueError(f"Invalid direction: {direction}")

            amount_dec = await self._to_exchange_amount(contract_id, quantity)
            order = await self.exchange.create_order(
                symbol=contract_id,
                type="market",
                side=direction,
                amount=float(amount_dec),
                params={"reduceOnly": reduce_only},
            )

            filled_price = order.get("average") or order.get("price")
            price_decimal = Decimal(str(filled_price)) if filled_price else None
            return OrderResult(
                success=True,
                order_id=order.get("id"),
                side=direction,
                size=amount_dec,
                price=price_decimal,
                status=(order.get("status") or "FILLED").upper(),
                filled_size=Decimal(str(order.get("filled", amount_dec))),
            )
        except Exception as exc:
            self.logger.log(f"[MARKET] Error placing BingX order: {exc}", "ERROR")
            return OrderResult(success=False, error_message=str(exc))

    async def cancel_order(self, order_id: str) -> OrderResult:
        try:
            await self._ensure_markets_loaded()
            await self.exchange.cancel_order(order_id, self.config.contract_id)
            return OrderResult(success=True, order_id=order_id, status="CANCELED")
        except Exception as exc:
            self.logger.log(f"Error canceling BingX order {order_id}: {exc}", "ERROR")
            return OrderResult(success=False, error_message=str(exc))

    @query_retry()
    async def get_order_info(self, order_id: str) -> Optional[OrderInfo]:
        await self._ensure_markets_loaded()
        order = await self.exchange.fetch_order(order_id, self.config.contract_id)
        if not order:
            return None
        return self._to_order_info(order)

    @query_retry(default_return=[])
    async def get_active_orders(self, contract_id: str) -> List[OrderInfo]:
        await self._ensure_markets_loaded()
        orders = await self.exchange.fetch_open_orders(symbol=contract_id)
        return [self._to_order_info(order) for order in orders]

    @query_retry(default_return=Decimal("0"))
    async def get_account_positions(self) -> Decimal:
        await self._ensure_markets_loaded()
        positions = await self.exchange.fetch_positions()

        for position in positions:
            if position.get("symbol") == self.config.contract_id:
                contracts = position.get("contracts")
                if contracts is None:
                    contracts = position.get("info", {}).get("positionAmt", 0)
                return abs(Decimal(str(contracts or "0")))
        return Decimal("0")

    async def get_contract_attributes(self) -> Tuple[str, Decimal]:
        await self._ensure_markets_loaded()

        markets = self.exchange.markets
        ticker = self.config.ticker

        for symbol, market in markets.items():
            if (
                market.get("swap")
                and market.get("linear")
                and market.get("base") == ticker
                and market.get("quote") == "USDT"
            ):
                tick_size = self._extract_tick_size(market)
                min_amount = self._extract_min_amount(market)

                self.config.contract_id = symbol
                self.config.tick_size = tick_size

                if self.config.quantity < min_amount:
                    raise ValueError(
                        f"Order quantity is less than min quantity: {self.config.quantity} < {min_amount}"
                    )
                return symbol, tick_size

        raise ValueError(f"No BingX USDT perpetual contract found for ticker {ticker}")

    def _extract_tick_size(self, market: Dict[str, Any]) -> Decimal:
        precision = market.get("precision", {})
        price_precision = precision.get("price")

        if isinstance(price_precision, int):
            return Decimal("1") / (Decimal("10") ** price_precision)
        if isinstance(price_precision, float):
            return Decimal(str(price_precision))

        limits = market.get("limits", {}).get("price", {})
        if limits.get("min"):
            return Decimal(str(limits["min"]))

        info_precision = market.get("info", {}).get("pricePrecision")
        if info_precision is not None:
            return Decimal("1") / (Decimal("10") ** int(info_precision))

        raise ValueError(f"Unable to determine tick size for market {market.get('symbol')}")

    def _extract_min_amount(self, market: Dict[str, Any]) -> Decimal:
        limits = market.get("limits", {}).get("amount", {})
        if limits.get("min"):
            return Decimal(str(limits["min"]))

        info_min = market.get("info", {}).get("minTradeNum")
        if info_min:
            return Decimal(str(info_min))

        raise ValueError(f"Unable to determine minimum amount for market {market.get('symbol')}")

    def _to_order_result(
        self,
        order: Dict[str, Any],
        direction: str,
        size: Decimal,
        price: Decimal,
    ) -> OrderResult:
        status = (order.get("status") or "NEW").upper()
        filled = Decimal(str(order.get("filled", 0)))
        return OrderResult(
            success=True,
            order_id=order.get("id"),
            side=direction,
            size=size,
            price=price,
            status=status,
            filled_size=filled,
        )

    def _to_order_info(self, order: Dict[str, Any]) -> OrderInfo:
        status = (order.get("status") or "NEW").upper()
        side = order.get("side", "").lower()

        return OrderInfo(
            order_id=order.get("id", ""),
            side=side,
            size=Decimal(str(order.get("amount", 0))),
            price=Decimal(str(order.get("price", 0))),
            status=status,
            filled_size=Decimal(str(order.get("filled", 0))),
            remaining_size=Decimal(str(order.get("remaining", 0))),
            cancel_reason=order.get("info", {}).get("reason", ""),
        )
