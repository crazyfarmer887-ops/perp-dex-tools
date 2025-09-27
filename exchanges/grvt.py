import os
import asyncio
import logging
from decimal import Decimal
from typing import Dict, Any, List, Optional, Tuple

from pysdk.grvt_ccxt_pro import GrvtCcxtPro
from pysdk.grvt_ccxt_env import GrvtEnv

from .base import BaseExchangeClient, OrderResult, OrderInfo

logger = logging.getLogger(__name__)


class GRVTClient(BaseExchangeClient):
    """
    GRVT exchange client implementation using ccxt-pro compatible SDK.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.grvt_client: Optional[GrvtCcxtPro] = None
        self.order_update_handler = None
        self.env = GrvtEnv(os.getenv("GRVT_ENV", "testnet"))

    def _validate_config(self) -> None:
        """
        Validates the necessary configuration for the GRVT client.
        """
        required_keys = [
            "GRVT_API_KEY",
            "GRVT_PRIVATE_KEY",
            "GRVT_TRADING_ACCOUNT_ID",
        ]
        for key in required_keys:
            if key not in os.environ:
                raise ValueError(f"Missing required environment variable: {key}")

    async def connect(self) -> None:
        """
        Connects to the GRVT exchange and loads markets.
        """
        self._validate_config()
        params = {
            "api_key": os.getenv("GRVT_API_KEY"),
            "private_key": os.getenv("GRVT_PRIVATE_KEY"),
            "trading_account_id": os.getenv("GRVT_TRADING_ACCOUNT_ID"),
        }
        self.grvt_client = GrvtCcxtPro(self.env, logger, parameters=params)
        await self.grvt_client.load_markets()
        logger.info("Successfully connected to GRVT exchange and loaded markets.")

    async def disconnect(self) -> None:
        """
        Disconnects from the GRVT exchange.
        """
        if self.grvt_client:
            await self.grvt_client.close()
            logger.info("Successfully disconnected from GRVT exchange.")
        self.grvt_client = None

    def get_exchange_name(self) -> str:
        return "grvt"

    def _order_update_handler(self, order: dict):
        """Processes a single order update from the WebSocket stream."""
        if not self.order_update_handler:
            return

        try:
            status = order.get('status')
            side = order.get('side')
            filled_size = Decimal(str(order.get('filled', '0')))
            size = Decimal(str(order.get('amount', '0')))

            # Determine order type ('OPEN' or 'CLOSE')
            order_type = "CLOSE" if side == self.config.close_order_side else "OPEN"

            # Map exchange status to internal bot status
            if status == 'closed':
                mapped_status = 'FILLED'
            elif status == 'canceled':
                mapped_status = 'CANCELED'
            elif status == 'open':
                mapped_status = 'PARTIALLY_FILLED' if filled_size > 0 else 'OPEN'
            else:
                logger.warning(f"Unknown GRVT order status received: {status}")
                return

            # Construct the message payload for the trading bot
            bot_message = {
                'order_id': order.get('id'),
                'side': side,
                'order_type': order_type,
                'status': mapped_status,
                'size': str(size),
                'price': str(order.get('price', '0')),
                'contract_id': order.get('symbol'),
                'filled_size': str(filled_size),
            }

            # Pass the processed message to the bot's handler
            self.order_update_handler(bot_message)

        except Exception as e:
            logger.error(f"Error processing GRVT order update: {e}", exc_info=True)

    async def _order_watcher(self):
        """Continuously watches for order updates from the exchange."""
        logger.info("Starting GRVT order watcher...")
        while True:
            try:
                # watch_orders is a long-polling method that waits for updates
                if not self.grvt_client or not self.grvt_client.markets:
                    await self.connect()
                orders = await self.grvt_client.watch_orders(symbol=self.config.contract_id)
                for order in orders:
                    self._order_update_handler(order)
            except asyncio.CancelledError:
                logger.info("GRVT order watcher cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in GRVT order watcher: {e}", exc_info=True)
                # Wait before retrying to prevent spamming on connection issues
                await asyncio.sleep(5)

    def setup_order_update_handler(self, handler) -> None:
        """Sets up the handler for order updates and starts the watcher task."""
        self.order_update_handler = handler
        logger.info("Order update handler set up for GRVT.")
        asyncio.create_task(self._order_watcher())

    def _from_sdk_order(self, order: dict) -> OrderInfo:
        """Converts a ccxt-style order dict to an OrderInfo object."""
        return OrderInfo(
            order_id=order['id'],
            side=order['side'],
            size=Decimal(str(order['amount'])),
            price=Decimal(str(order.get('price', '0'))),
            status=order['status'],
            filled_size=Decimal(str(order['filled'])),
            remaining_size=Decimal(str(order['remaining'])),
        )

    async def place_open_order(self, contract_id: str, quantity: Decimal, direction: str) -> OrderResult:
        if not self.grvt_client:
            raise ConnectionError("GRVT client not connected.")
        try:
            ticker = await self.grvt_client.fetch_ticker(contract_id)
            last_price = Decimal(str(ticker['last']))

            price_tick = self.grvt_client.markets[contract_id]['precision']['price']
            price_adjustment = Decimal(str(price_tick)) * Decimal('1')

            if direction.lower() == 'buy':
                price = last_price - price_adjustment
            else:
                price = last_price + price_adjustment

            price = self.round_to_tick(price)

            order = await self.grvt_client.create_order(
                symbol=contract_id, order_type='limit', side=direction.lower(),
                amount=float(quantity), price=float(price),
                params={"time_in_force": "GTC"}
            )
            logger.info(f"Placed open order on GRVT: {order}")
            return OrderResult(
                success=True, order_id=order['id'], side=order['side'],
                size=Decimal(str(order['amount'])), price=Decimal(str(order['price'])),
                status=order['status']
            )
        except Exception as e:
            logger.error(f"Error placing open order on GRVT: {e}", exc_info=True)
            return OrderResult(success=False, error_message=str(e))

    async def place_close_order(self, contract_id: str, quantity: Decimal, price: Decimal, side: str) -> OrderResult:
        if not self.grvt_client:
            raise ConnectionError("GRVT client not connected.")
        try:
            order = await self.grvt_client.create_order(
                symbol=contract_id, order_type='limit', side=side.lower(),
                amount=float(quantity), price=float(price),
                params={"time_in_force": "GTC"}
            )
            logger.info(f"Placed close order on GRVT: {order}")
            return OrderResult(
                success=True, order_id=order['id'], side=order['side'],
                size=Decimal(str(order['amount'])), price=Decimal(str(order['price'])),
                status=order['status']
            )
        except Exception as e:
            logger.error(f"Error placing close order on GRVT: {e}", exc_info=True)
            return OrderResult(success=False, error_message=str(e))

    async def cancel_order(self, order_id: str) -> OrderResult:
        if not self.grvt_client:
            raise ConnectionError("GRVT client not connected.")
        try:
            response = await self.grvt_client.cancel_order(id=order_id)
            logger.info(f"Canceled order on GRVT: {response}")
            return OrderResult(success=True, order_id=order_id)
        except Exception as e:
            logger.error(f"Error canceling order on GRVT: {e}", exc_info=True)
            return OrderResult(success=False, order_id=order_id, error_message=str(e))

    async def get_order_info(self, order_id: str) -> Optional[OrderInfo]:
        if not self.grvt_client:
            raise ConnectionError("GRVT client not connected.")
        try:
            order = await self.grvt_client.fetch_order(id=order_id)
            return self._from_sdk_order(order) if order else None
        except Exception as e:
            logger.error(f"Error getting order info from GRVT for order {order_id}: {e}", exc_info=True)
            return None

    async def get_active_orders(self, contract_id: str) -> List[OrderInfo]:
        if not self.grvt_client:
            raise ConnectionError("GRVT client not connected.")
        try:
            open_orders = await self.grvt_client.fetch_open_orders(symbol=contract_id)
            return [self._from_sdk_order(o) for o in open_orders]
        except Exception as e:
            logger.error(f"Error getting active orders from GRVT for {contract_id}: {e}", exc_info=True)
            return []

    async def get_account_positions(self) -> Decimal:
        if not self.grvt_client:
            raise ConnectionError("GRVT client not connected.")

        contract_id = self.config['contract_id']
        try:
            positions = await self.grvt_client.fetch_positions(symbols=[contract_id])
            if not positions:
                return Decimal("0.0")

            position = positions[0]
            size = Decimal(str(position.get('contracts', '0')))
            if position.get('side') == 'short':
                size = -size
            return size
        except Exception as e:
            logger.error(f"Error getting account positions from GRVT: {e}", exc_info=True)
            return Decimal("0.0")

    async def place_market_order(self, contract_id: str, quantity: Decimal, side: str) -> OrderResult:
        if not self.grvt_client:
            raise ConnectionError("GRVT client not connected.")
        try:
            order = await self.grvt_client.create_order(
                symbol=contract_id, order_type='market',
                side=side.lower(), amount=float(quantity)
            )
            logger.info(f"Placed market order on GRVT: {order}")
            return OrderResult(
                success=True, order_id=order['id'], side=order['side'],
                size=Decimal(str(order['amount'])), price=Decimal(str(order.get('average', '0'))),
                status=order['status'], filled_size=Decimal(str(order.get('filled', '0')))
            )
        except Exception as e:
            logger.error(f"Error placing market order on GRVT: {e}", exc_info=True)
            return OrderResult(success=False, error_message=str(e))

    async def get_contract_attributes(self) -> Tuple[str, Decimal]:
        """Get contract ID and tick size for a ticker."""
        ticker = self.config.ticker
        if not ticker:
            raise ValueError("Ticker is not configured.")

        # In GRVT, the ticker is the contract_id
        contract_id = ticker
        self.config.contract_id = contract_id

        if not self.grvt_client or not self.grvt_client.markets:
            await self.connect()

        market = self.grvt_client.markets.get(contract_id)
        if not market:
            raise ValueError(f"Market {contract_id} not found.")

        # Extract tick size (price precision)
        tick_size = Decimal(str(market['precision']['price']))
        self.config.tick_size = tick_size

        return contract_id, tick_size

    async def fetch_bbo_prices(self, contract_id: str) -> Tuple[Decimal, Decimal]:
        """Fetches the best bid and offer prices for a given contract."""
        if not self.grvt_client:
            raise ConnectionError("GRVT client not connected.")
        try:
            ticker = await self.grvt_client.fetch_ticker(contract_id)
            best_bid = Decimal(str(ticker['bid']))
            best_ask = Decimal(str(ticker['ask']))

            if best_bid <= 0 or best_ask <= 0:
                raise ValueError("Invalid bid/ask prices received from GRVT.")

            return best_bid, best_ask
        except Exception as e:
            logger.error(f"Error fetching BBO prices from GRVT for {contract_id}: {e}", exc_info=True)
            raise ValueError(f"Could not fetch BBO prices for {contract_id} from GRVT.")