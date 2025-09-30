"""
GRVT exchange client implementation.
"""

import os
import asyncio
from decimal import Decimal
from typing import Dict, Any, List, Optional, Tuple

from .base import BaseExchangeClient, OrderResult, OrderInfo
from helpers.logger import TradingLogger

from grvt_pysdk.grvt_ccxt_pro import GrvtCcxtPro
from grvt_pysdk.grvt_ccxt_ws import GrvtCcxtWS
from grvt_pysdk.grvt_ccxt_env import GrvtEnv, GrvtWSEndpointType


class GRVTWebSocketManager:
    """WebSocket manager for GRVT order updates."""

    def __init__(self, env: GrvtEnv, api_key: str, private_key: str, trading_account_id: str, symbol: str, order_update_callback):
        self.env = env
        self.api_key = api_key
        self.private_key = private_key
        self.trading_account_id = trading_account_id
        self.symbol = symbol
        self.order_update_callback = order_update_callback
        self.ws_client: Optional[GrvtCcxtWS] = None
        self.logger = None
        self.connected_event = asyncio.Event()

    def set_logger(self, logger):
        """Set the logger instance."""
        self.logger = logger

    async def connect(self):
        """Connect to GRVT WebSocket and subscribe to channels."""
        try:
            loop = asyncio.get_event_loop()
            params = {
                "api_key": self.api_key,
                "trading_account_id": self.trading_account_id,
                "private_key": self.private_key,
                "api_ws_version": "v1",
            }
            self.ws_client = GrvtCcxtWS(self.env, loop, self.logger, parameters=params)
            await self.ws_client.initialize()

            # Subscribe to order and fill updates
            await self.ws_client.subscribe(
                stream="order",
                callback=self._handle_message,
                ws_end_point_type=GrvtWSEndpointType.TRADE_DATA_RPC_FULL,
                params={"instrument": self.symbol},
            )
            await self.ws_client.subscribe(
                stream="fill",
                callback=self._handle_message,
                ws_end_point_type=GrvtWSEndpointType.TRADE_DATA_RPC_FULL,
                params={"instrument": self.symbol},
            )

            if self.logger:
                self.logger.log(f"Successfully sent subscription requests for {self.symbol}", "INFO")

            # Signal that the connection and subscription process is complete
            self.connected_event.set()

        except Exception as e:
            if self.logger:
                self.logger.log(f"GRVT WebSocket connection error: {e}", "ERROR", send_lark=True)
            self.connected_event.set()  # Set event on error to prevent blocking
            raise

    async def _handle_message(self, message: dict):
        """Handle incoming WebSocket messages."""
        try:
            if self.order_update_callback:
                await self.order_update_callback(message)
        except Exception as e:
            if self.logger:
                self.logger.log(f"Error handling GRVT WebSocket message: {e}", "ERROR")

    async def disconnect(self):
        """Disconnect from WebSocket."""
        if self.ws_client:
            await self.ws_client.close()
            if self.logger:
                self.logger.log("GRVT WebSocket disconnected", "INFO")


class GRVTClient(BaseExchangeClient):
    """GRVT exchange client implementation."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize GRVT client."""
        super().__init__(config)
        self.logger = TradingLogger(exchange="grvt", ticker=self.config.ticker, log_to_console=True)
        self._validate_config()

        self.api_key = os.getenv("GRVT_API_KEY")
        self.private_key = os.getenv("GRVT_PRIVATE_KEY")
        self.sub_account_id = os.getenv("GRVT_SUB_ACCOUNT_ID")
        self.env_name = os.getenv("GRVT_ENV", "prod")
        self.env = GrvtEnv(self.env_name)

        params = {
            "api_key": self.api_key,
            "trading_account_id": self.sub_account_id,
            "private_key": self.private_key,
        }
        self.client = GrvtCcxtPro(self.env, self.logger, parameters=params)
        self._order_update_handler = None
        self.ws_manager = None

    def _validate_config(self) -> None:
        """Validate GRVT configuration."""
        required_env_vars = ['GRVT_API_KEY', 'GRVT_SUB_ACCOUNT_ID', 'GRVT_PRIVATE_KEY']
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables for GRVT: {missing_vars}")

    async def connect(self) -> None:
        """Connect to GRVT, load markets, and start WebSocket."""
        self.logger.log("Connecting to GRVT and loading markets...", "INFO")
        await self.client.load_markets()
        self.logger.log("Successfully loaded GRVT markets.", "INFO")

        self.ws_manager = GRVTWebSocketManager(
            env=self.env,
            api_key=self.api_key,
            private_key=self.private_key,
            trading_account_id=self.sub_account_id,
            symbol=self.config.contract_id,
            order_update_callback=self._handle_websocket_update
        )
        self.ws_manager.set_logger(self.logger)

        try:
            asyncio.create_task(self.ws_manager.connect())
            self.logger.log("Waiting for GRVT WebSocket connection to be established...", "INFO")
            await asyncio.wait_for(self.ws_manager.connected_event.wait(), timeout=30)
            self.logger.log("GRVT WebSocket connection established.", "INFO")
        except asyncio.TimeoutError:
            self.logger.log("Timeout waiting for GRVT WebSocket connection.", "ERROR", send_lark=True)
            raise
        except Exception as e:
            self.logger.log(f"Error connecting to GRVT WebSocket: {e}", "ERROR", send_lark=True)
            raise

    async def disconnect(self) -> None:
        """Disconnect from GRVT."""
        self.logger.log("Disconnecting from GRVT...", "INFO")
        if self.ws_manager:
            await self.ws_manager.disconnect()

    def get_exchange_name(self) -> str:
        """Get the exchange name."""
        return "grvt"

    def setup_order_update_handler(self, handler) -> None:
        """Setup order update handler for WebSocket."""
        self._order_update_handler = handler

    async def _handle_websocket_update(self, message: dict):
        """Handle and standardize order updates from WebSocket."""
        try:
            params = message.get("params", {})
            channel = params.get("channel", "")
            data = params.get("data", {})

            if not data or not channel or not self._order_update_handler:
                return

            if channel.startswith("order") or channel.startswith("fill"):
                order_id = data.get('order_id')
                if not order_id:
                    return

                grvt_status = data.get("status")
                status = ""

                if channel.startswith("fill"):
                    if Decimal(data.get('remaining_quantity', '0')) == 0:
                        status = 'FILLED'
                    else:
                        status = 'PARTIALLY_FILLED'
                elif grvt_status in ["CANCELED", "FILLED", "REJECTED", "EXPIRED", "PARTIALLY_FILLED"]:
                    status = grvt_status
                elif grvt_status in ["PENDING", "NEW"]:
                    status = "OPEN"
                else:
                    self.logger.log(f"Unhandled GRVT order status: {grvt_status}", "WARNING")
                    return

                order_side = data.get('side', '').lower()
                is_close_order = (order_side == self.config.close_order_side)
                order_type = "CLOSE" if is_close_order else "OPEN"

                await self._order_update_handler({
                    'order_id': order_id,
                    'side': order_side,
                    'order_type': order_type,
                    'status': status,
                    'size': Decimal(data.get('quantity', '0')),
                    'price': Decimal(data.get('price', '0')),
                    'contract_id': data.get('instrument_name'),
                    'filled_size': Decimal(data.get('cumulative_filled_quantity', '0'))
                })
        except Exception as e:
            self.logger.log(f"Error handling GRVT WebSocket update: {e} - Data: {message}", "ERROR")

    async def get_contract_attributes(self) -> Tuple[str, Decimal]:
        """Get contract ID and tick size for a ticker."""
        ticker = self.config.ticker
        if not ticker:
            raise ValueError("Ticker is empty")

        perp_symbol = f"{ticker.upper()}_USDT_Perp"

        markets = await self.client.fetch_markets()
        market = next((m for m in markets if m['id'] == perp_symbol), None)

        if market:
            self.config.contract_id = market['id']
            self.config.tick_size = Decimal(str(market['precision']['price']))
            min_quantity = Decimal(str(market['limits']['amount']['min']))
            if self.config.quantity < min_quantity:
                 raise ValueError(f"Order quantity {self.config.quantity} is less than min quantity {min_quantity}")
        else:
            raise ValueError(f"Market not found for ticker {ticker} (expected symbol like {perp_symbol})")

        return self.config.contract_id, self.config.tick_size

    async def _fetch_bbo_prices(self, contract_id: str) -> Tuple[Decimal, Decimal]:
        order_book = await self.client.fetch_order_book(contract_id, limit=1)
        best_bid = Decimal(order_book['bids'][0][0]) if order_book['bids'] else Decimal(0)
        best_ask = Decimal(order_book['asks'][0][0]) if order_book['asks'] else Decimal(0)
        return best_bid, best_ask

    async def place_open_order(self, contract_id: str, quantity: Decimal, direction: str) -> OrderResult:
        """Place an open order."""
        try:
            best_bid, best_ask = await self._fetch_bbo_prices(contract_id)
            if best_bid <= 0 or best_ask <= 0:
                return OrderResult(success=False, error_message='Invalid bid/ask prices')

            if direction == 'buy':
                price = best_ask - self.config.tick_size
            else:
                price = best_bid + self.config.tick_size

            order = await self.client.create_order(
                symbol=contract_id, order_type="limit", side=direction,
                amount=float(quantity), price=float(self.round_to_tick(price)),
                params={'post_only': True}
            )
            return OrderResult(success=True, order_id=order['id'], side=order['side'],
                size=Decimal(order['amount']), price=Decimal(order['price']), status=order['status'])
        except Exception as e:
            self.logger.log(f"Error placing open order: {e}", "ERROR")
            return OrderResult(success=False, error_message=str(e))

    async def place_close_order(self, contract_id: str, quantity: Decimal, price: Decimal, side: str) -> OrderResult:
        """Place a close order."""
        try:
            order = await self.client.create_order(
                symbol=contract_id, order_type="limit", side=side,
                amount=float(quantity), price=float(self.round_to_tick(price)),
                params={'post_only': True}
            )
            return OrderResult(success=True, order_id=order['id'], side=order['side'],
                size=Decimal(order['amount']), price=Decimal(order['price']), status=order['status'])
        except Exception as e:
            self.logger.log(f"Error placing close order: {e}", "ERROR")
            return OrderResult(success=False, error_message=str(e))

    async def cancel_order(self, order_id: str) -> OrderResult:
        """Cancel an order."""
        try:
            result = await self.client.cancel_order(id=order_id, symbol=self.config.contract_id)
            return OrderResult(success=True, order_id=result.get('id'))
        except Exception as e:
            self.logger.log(f"Error canceling order {order_id}: {e}", "ERROR")
            return OrderResult(success=False, order_id=order_id, error_message=str(e))

    async def get_order_info(self, order_id: str) -> Optional[OrderInfo]:
        """Get order information."""
        try:
            order = await self.client.fetch_order(id=order_id, symbol=self.config.contract_id)
            return OrderInfo(order_id=order['id'], side=order['side'], size=Decimal(order['amount']),
                price=Decimal(order['price']), status=order['status'], filled_size=Decimal(order['filled']),
                remaining_size=Decimal(order['remaining']))
        except Exception as e:
            self.logger.log(f"Error fetching order info for {order_id}: {e}", "ERROR")
            return None

    async def get_active_orders(self, contract_id: str) -> List[OrderInfo]:
        """Get active orders for a contract."""
        try:
            orders = await self.client.fetch_open_orders(symbol=contract_id)
            return [OrderInfo(order_id=o['id'], side=o['side'], size=Decimal(o['amount']),
                    price=Decimal(o['price']), status=o['status'], filled_size=Decimal(o['filled']),
                    remaining_size=Decimal(o['remaining'])) for o in orders]
        except Exception as e:
            self.logger.log(f"Error fetching active orders for {contract_id}: {e}", "ERROR")
            return []

    async def get_account_positions(self) -> Decimal:
        """Get account positions."""
        try:
            positions = await self.client.fetch_positions(symbols=[self.config.contract_id])
            for position in positions:
                if position.get('symbol') == self.config.contract_id:
                    size = Decimal(position.get('contracts', '0'))
                    if position.get('side') == 'short':
                        return -size
                    return size
            return Decimal(0)
        except Exception as e:
            self.logger.log(f"Error fetching account positions: {e}", "ERROR", send_lark=True)
            return Decimal(0)