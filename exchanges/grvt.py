"""
GRVT exchange client implementation.
"""

import os
import asyncio
import time
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from typing import Dict, Any, List, Optional, Tuple
from pysdk.grvt_ccxt import GrvtCcxt
from pysdk.grvt_ccxt_ws import GrvtCcxtWS
from pysdk.grvt_ccxt_env import GrvtEnv, GrvtWSEndpointType

from .base import BaseExchangeClient, OrderResult, OrderInfo, query_retry
from helpers.logger import TradingLogger


class GrvtClient(BaseExchangeClient):
    """GRVT exchange client implementation."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize GRVT client."""
        super().__init__(config)

        # GRVT credentials from environment
        self.trading_account_id = os.getenv('GRVT_TRADING_ACCOUNT_ID')
        self.private_key = os.getenv('GRVT_PRIVATE_KEY')
        self.api_key = os.getenv('GRVT_API_KEY')
        self.environment = os.getenv('GRVT_ENVIRONMENT', 'prod')

        if not self.trading_account_id or not self.private_key or not self.api_key:
            raise ValueError(
                "GRVT_TRADING_ACCOUNT_ID, GRVT_PRIVATE_KEY, and GRVT_API_KEY must be set in environment variables"
            )

        # Convert environment string to proper enum
        env_map = {
            'prod': GrvtEnv.PROD,
            'testnet': GrvtEnv.TESTNET,
            'staging': GrvtEnv.STAGING,
            'dev': GrvtEnv.DEV
        }
        self.env = env_map.get(self.environment.lower(), GrvtEnv.PROD)

        # Initialize logger
        self.logger = TradingLogger(exchange="grvt", ticker=self.config.ticker, log_to_console=False)

        # Initialize GRVT clients
        self._initialize_grvt_clients()

        self._order_update_handler = None
        self._ws_client = None
        self._order_update_callback = None
        
        # Enterprise-grade position management: async locks and state tracking
        self._order_lock = asyncio.Lock()  # Prevent concurrent order placement
        self._position_lock = asyncio.Lock()  # Prevent concurrent position reads/writes
        self._last_position_check: Optional[Decimal] = None
        self._last_position_check_time: float = 0.0
        self._position_cache_ttl: float = 0.1  # Cache position for 100ms to prevent excessive API calls
        
        # Strict position tolerance: 0.001% precision
        self._position_tolerance_percent = Decimal('0.001')  # 0.001%
        self._min_position_change = Decimal('0.00001')  # Minimum position change to consider significant

    def _initialize_grvt_clients(self) -> None:
        """Initialize the GRVT REST and WebSocket clients."""
        try:
            # Parameters for GRVT SDK
            parameters = {
                'trading_account_id': self.trading_account_id,
                'private_key': self.private_key,
                'api_key': self.api_key
            }

            # Initialize REST client
            self.rest_client = GrvtCcxt(
                env=self.env,
                parameters=parameters
            )

        except Exception as e:
            raise ValueError(f"Failed to initialize GRVT client: {e}")

    def _validate_config(self) -> None:
        """Validate GRVT configuration."""
        required_env_vars = ['GRVT_TRADING_ACCOUNT_ID', 'GRVT_PRIVATE_KEY', 'GRVT_API_KEY']
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")

    async def connect(self) -> None:
        """Connect to GRVT WebSocket."""
        try:
            # Initialize WebSocket client - match the working test implementation
            loop = asyncio.get_running_loop()

            # Import logger from pysdk like in the test file
            from pysdk.grvt_ccxt_logging_selector import logger

            # Parameters for GRVT SDK - match test file structure
            parameters = {
                'api_key': self.api_key,
                'trading_account_id': self.trading_account_id,
                'api_ws_version': 'v1',
                'private_key': self.private_key
            }

            self._ws_client = GrvtCcxtWS(
                env=self.env,
                loop=loop,
                logger=logger,  # Add logger parameter like in test file
                parameters=parameters
            )

            # Initialize and connect
            await self._ws_client.initialize()
            await asyncio.sleep(2)  # Wait for connection to establish

            # If an order update callback was set before connect, subscribe now
            if self._order_update_callback is not None:
                asyncio.create_task(self._subscribe_to_orders(self._order_update_callback))
                self.logger.log(f"Deferred subscription started for {self.config.contract_id}", "INFO")

        except Exception as e:
            self.logger.log(f"Error connecting to GRVT WebSocket: {e}", "ERROR")
            raise

    async def disconnect(self) -> None:
        """Disconnect from GRVT."""
        try:
            if self._ws_client:
                await self._ws_client.__aexit__()
        except Exception as e:
            self.logger.log(f"Error during GRVT disconnect: {e}", "ERROR")

    def get_exchange_name(self) -> str:
        """Get the exchange name."""
        return "grvt"

    def setup_order_update_handler(self, handler) -> None:
        """Setup order update handler for WebSocket."""
        self._order_update_handler = handler

        async def order_update_callback(message: Dict[str, Any]):
            """Handle order updates from WebSocket - match working test implementation."""
            # Log raw message for debugging
            self.logger.log(f"Received WebSocket message: {message}", "DEBUG")
            self.logger.log("**************************************************", "DEBUG")
            try:
                # Parse the message structure - match the working test implementation exactly
                if 'feed' in message:
                    data = message.get('feed', {})
                    leg = data.get('legs', [])[0] if data.get('legs') else None

                    if isinstance(data, dict) and leg:
                        contract_id = leg.get('instrument', '')
                        if contract_id != self.config.contract_id:
                            return

                        order_state = data.get('state', {})
                        # Extract order data using the exact structure from test
                        order_id = data.get('order_id', '')
                        status = order_state.get('status', '')
                        side = 'buy' if leg.get('is_buying_asset') else 'sell'
                        size = leg.get('size', '0')
                        price = leg.get('limit_price', '0')
                        filled_size = order_state.get('traded_size')[0] if order_state.get('traded_size') else '0'

                        if order_id and status:
                            # Determine order type based on side
                            if side == self.config.close_order_side:
                                order_type = "CLOSE"
                            else:
                                order_type = "OPEN"

                            # Map GRVT status to our status
                            status_map = {
                                'OPEN': 'OPEN',
                                'FILLED': 'FILLED',
                                'CANCELLED': 'CANCELED',
                                'REJECTED': 'CANCELED'
                            }
                            mapped_status = status_map.get(status, status)

                            # Handle partially filled orders
                            if status == 'OPEN' and Decimal(filled_size) > 0:
                                mapped_status = "PARTIALLY_FILLED"

                            if mapped_status in ['OPEN', 'PARTIALLY_FILLED', 'FILLED', 'CANCELED']:
                                if self._order_update_handler:
                                    self._order_update_handler({
                                        'order_id': order_id,
                                        'side': side,
                                        'order_type': order_type,
                                        'status': mapped_status,
                                        'size': size,
                                        'price': price,
                                        'contract_id': contract_id,
                                        'filled_size': filled_size
                                    })
                            else:
                                self.logger.log(f"Ignoring order update with status: {mapped_status}", "DEBUG")
                        else:
                            self.logger.log(f"Order update missing order_id or status: {data}", "DEBUG")
                    else:
                        self.logger.log(f"Order update data is not dict or missing legs: {data}", "DEBUG")
                else:
                    # Handle other message types (position, fill, etc.)
                    method = message.get('method', 'unknown')
                    self.logger.log(f"Received non-order message: {method}", "DEBUG")

            except Exception as e:
                self.logger.log(f"Error handling order update: {e}", "ERROR")
                self.logger.log(f"Message that caused error: {message}", "ERROR")

        # Store callback for use after connect
        self._order_update_callback = order_update_callback

        # Subscribe immediately if WebSocket is already initialized; otherwise defer to connect()
        if self._ws_client:
            try:
                asyncio.create_task(self._subscribe_to_orders(self._order_update_callback))
                self.logger.log(f"Successfully initiated subscription to order updates for {self.config.contract_id}", "INFO")
            except Exception as e:
                self.logger.log(f"Error subscribing to order updates: {e}", "ERROR")
                raise
        else:
            self.logger.log("WebSocket not ready yet; will subscribe after connect()", "INFO")

    async def _subscribe_to_orders(self, callback):
        """Subscribe to order updates asynchronously."""
        try:
            await self._ws_client.subscribe(
                stream="order",
                callback=callback,
                ws_end_point_type=GrvtWSEndpointType.TRADE_DATA_RPC_FULL,
                params={"instrument": self.config.contract_id}
            )
            await asyncio.sleep(0)  # Small delay like in test file
            self.logger.log(f"Successfully subscribed to order updates for {self.config.contract_id}", "INFO")
        except Exception as e:
            self.logger.log(f"Error in subscription task: {e}", "ERROR")

    @query_retry(reraise=True)
    async def fetch_bbo_prices(self, contract_id: str) -> Tuple[Decimal, Decimal]:
        """Fetch best bid and offer prices for a contract."""
        # Get order book from GRVT
        order_book = self.rest_client.fetch_order_book(contract_id, limit=10)

        if not order_book or 'bids' not in order_book or 'asks' not in order_book:
            raise ValueError(f"Unable to get order book: {order_book}")

        bids = order_book.get('bids', [])
        asks = order_book.get('asks', [])

        best_bid = Decimal(bids[0]['price']) if bids and len(bids) > 0 else Decimal(0)
        best_ask = Decimal(asks[0]['price']) if asks and len(asks) > 0 else Decimal(0)

        return best_bid, best_ask

    async def place_post_only_order(self, contract_id: str, quantity: Decimal, price: Decimal,
                                    side: str) -> OrderResult:
        """Place a post only order with GRVT using official SDK."""

        # Place the order using GRVT SDK
        order_result = self.rest_client.create_limit_order(
            symbol=contract_id,
            side=side,
            amount=quantity,
            price=price,
            params={
                'post_only': True,
                'order_duration_secs': 30 * 86400 - 1, # GRVT SDK: signature expired cap is 30 days (default 1 day)
            }
        )
        if not order_result:
            raise Exception(f"[OPEN] Error placing order")

        client_order_id = order_result.get('metadata').get('client_order_id')
        order_status = order_result.get('state').get('status')
        order_status_start_time = time.time()
        order_info = await self.get_order_info(client_order_id=client_order_id)
        if order_info is not None:
            order_status = order_info.status

        while order_status in ['PENDING'] and time.time() - order_status_start_time < 10:
            # Check order status after a short delay
            await asyncio.sleep(0.05)
            order_info = await self.get_order_info(client_order_id=client_order_id)
            if order_info is not None:
                order_status = order_info.status

        if order_status == 'PENDING':
            raise Exception('Paradex Server Error: Order not processed after 10 seconds')
        else:
            return order_info

    async def get_order_price(self, direction: str) -> Decimal:
        """Get the price of an order with GRVT using official SDK."""
        best_bid, best_ask = await self.fetch_bbo_prices(self.config.contract_id)
        if best_bid <= 0 or best_ask <= 0:
            raise ValueError("Invalid bid/ask prices")

        if direction == 'buy':
            return best_ask - self.config.tick_size
        elif direction == 'sell':
            return best_bid + self.config.tick_size
        else:
            raise ValueError("Invalid direction")

    async def _validate_position_before_order(
        self, 
        contract_id: str, 
        quantity: Decimal, 
        side: str,
        order_type: str
    ) -> Tuple[bool, str]:
        """
        Enterprise-grade position validation with 0.001% precision.
        Returns (is_valid, error_message)
        """
        try:
            # Get current position and active orders
            current_position = await self.get_account_positions()
            active_orders = await self.get_active_orders(contract_id)
            
            # Calculate total pending close orders
            pending_close_amount = Decimal('0')
            active_open_orders_count = 0
            
            for order in active_orders:
                if order.side == self.config.close_order_side:
                    pending_close_amount += order.size
                elif order.side == self.config.direction:
                    active_open_orders_count += 1
            
            # Strict validation: Check if we already have an open order
            if order_type == "OPEN" and active_open_orders_count > 0:
                return False, f"Active open order already exists (count: {active_open_orders_count})"
            
            # Strict validation: Check position balance with 0.001% tolerance
            if order_type == "CLOSE":
                # Calculate expected position after all pending closes
                expected_position_after_close = current_position - pending_close_amount
                
                # Check if we're trying to close more than we have (with 0.001% tolerance)
                tolerance_amount = current_position * (self._position_tolerance_percent / Decimal('100'))
                if tolerance_amount < self._min_position_change:
                    tolerance_amount = self._min_position_change
                
                if quantity > expected_position_after_close + tolerance_amount:
                    return False, (
                        f"Position insufficient: trying to close {quantity}, "
                        f"but available is {expected_position_after_close} "
                        f"(current: {current_position}, pending: {pending_close_amount}, "
                        f"tolerance: {tolerance_amount})"
                    )
                
                # Check if position is already flat (within tolerance)
                if current_position <= tolerance_amount:
                    return False, (
                        f"Position already flat: {current_position} "
                        f"(tolerance: {tolerance_amount})"
                    )
            
            return True, ""
            
        except Exception as e:
            self.logger.log(f"Error in position validation: {e}", "ERROR")
            return False, f"Position validation error: {str(e)}"

    async def place_open_order(
        self,
        contract_id: str,
        quantity: Decimal,
        direction: str,
        price: Optional[Decimal] = None
    ) -> OrderResult:
        """Place an open order with GRVT - Enterprise-grade with strict position checking."""
        # Acquire lock to prevent concurrent order placement
        async with self._order_lock:
            attempt = 0
            max_retries = 20  # Prevent infinite loops
            
            while attempt < max_retries:
                attempt += 1
                
                # Strict position validation before placing order
                is_valid, error_msg = await self._validate_position_before_order(
                    contract_id, quantity, direction, "OPEN"
                )
                if not is_valid:
                    self.logger.log(f"[OPEN] Position validation failed: {error_msg}", "ERROR")
                    return OrderResult(success=False, error_message=f"Position validation failed: {error_msg}")
                
                if attempt % 5 == 0:
                    self.logger.log(f"[OPEN] Attempt {attempt} to place order", "INFO")
                    active_orders = await self.get_active_orders(contract_id)
                    active_open_orders = 0
                    for order in active_orders:
                        if order.side == self.config.direction:
                            active_open_orders += 1
                    if active_open_orders > 1:
                        self.logger.log(f"[OPEN] ERROR: Active open orders abnormal: {active_open_orders}", "ERROR")
                        raise Exception(f"[OPEN] ERROR: Active open orders abnormal: {active_open_orders}")

                # Get current market prices
                best_bid, best_ask = await self.fetch_bbo_prices(contract_id)

                if best_bid <= 0 or best_ask <= 0:
                    return OrderResult(success=False, error_message='Invalid bid/ask prices')

                # Determine order side and price
                if direction == 'buy':
                    if price is not None:
                        order_price = Decimal(price)
                        if order_price >= best_ask:
                            order_price = best_ask - self.config.tick_size
                    else:
                        order_price = best_ask - self.config.tick_size
                elif direction == 'sell':
                    if price is not None:
                        order_price = Decimal(price)
                        if order_price <= best_bid:
                            order_price = best_bid + self.config.tick_size
                    else:
                        order_price = best_bid + self.config.tick_size
                else:
                    raise Exception(f"[OPEN] Invalid direction: {direction}")

                if order_price <= 0:
                    return OrderResult(success=False, error_message='Calculated order price is non-positive')

                order_price = self.round_to_tick(order_price)

                if direction == 'buy' and order_price >= best_ask:
                    order_price = self.round_to_tick(best_ask - self.config.tick_size)
                elif direction == 'sell' and order_price <= best_bid:
                    order_price = self.round_to_tick(best_bid + self.config.tick_size)

                # Place the order using GRVT SDK
                try:
                    order_info = await self.place_post_only_order(contract_id, quantity, order_price, direction)
                except Exception as e:
                    self.logger.log(f"[OPEN] Error placing order: {e}", "ERROR")
                    await asyncio.sleep(0.1)  # Small delay before retry
                    continue

                order_status = order_info.status
                order_id = order_info.order_id

                if order_status == 'REJECTED':
                    await asyncio.sleep(0.1)  # Small delay before retry
                    continue
                if order_status in ['OPEN', 'FILLED']:
                    # Invalidate position cache after successful order
                    async with self._position_lock:
                        self._last_position_check = None
                    
                    return OrderResult(
                        success=True,
                        order_id=order_id,
                        side=direction,
                        size=quantity,
                        price=order_price,
                        status=order_status
                    )
                elif order_status == 'PENDING':
                    raise Exception("[OPEN] Order not processed after 10 seconds")
                else:
                    raise Exception(f"[OPEN] Unexpected order status: {order_status}")
            
            return OrderResult(success=False, error_message=f"Failed to place open order after {max_retries} attempts")

    async def place_close_order(self, contract_id: str, quantity: Decimal, price: Decimal, side: str) -> OrderResult:
        """Place a close order with GRVT - Enterprise-grade with strict position checking."""
        # Acquire lock to prevent concurrent order placement
        async with self._order_lock:
            attempt = 0
            max_retries = 20  # Prevent infinite loops
            active_close_orders = await self._get_active_close_orders(contract_id)
            
            while attempt < max_retries:
                attempt += 1
                
                # Strict position validation before placing order
                is_valid, error_msg = await self._validate_position_before_order(
                    contract_id, quantity, side, "CLOSE"
                )
                if not is_valid:
                    self.logger.log(f"[CLOSE] Position validation failed: {error_msg}", "ERROR")
                    return OrderResult(success=False, error_message=f"Position validation failed: {error_msg}")
                
                if attempt % 5 == 0:
                    self.logger.log(f"[CLOSE] Attempt {attempt} to place order", "INFO")
                    current_close_orders = await self._get_active_close_orders(contract_id)

                    if current_close_orders - active_close_orders > 1:
                        self.logger.log(f"[CLOSE] ERROR: Active close orders abnormal: "
                                        f"{active_close_orders}, {current_close_orders}", "ERROR")
                        raise Exception(f"[CLOSE] ERROR: Active close orders abnormal: "
                                        f"{active_close_orders}, {current_close_orders}")
                    else:
                        active_close_orders = current_close_orders

                # Adjust price to ensure maker order
                best_bid, best_ask = await self.fetch_bbo_prices(contract_id)

                if side == 'sell' and price <= best_bid:
                    adjusted_price = best_bid + self.config.tick_size
                elif side == 'buy' and price >= best_ask:
                    adjusted_price = best_ask - self.config.tick_size
                else:
                    adjusted_price = price

                adjusted_price = self.round_to_tick(adjusted_price)
                
                try:
                    order_info = await self.place_post_only_order(contract_id, quantity, adjusted_price, side)
                except Exception as e:
                    self.logger.log(f"[CLOSE] Error placing order: {e}", "ERROR")
                    await asyncio.sleep(0.1)  # Small delay before retry
                    continue

                order_status = order_info.status
                order_id = order_info.order_id

                if order_status == 'REJECTED':
                    await asyncio.sleep(0.1)  # Small delay before retry
                    continue
                if order_status in ['OPEN', 'FILLED']:
                    # Invalidate position cache after successful order
                    async with self._position_lock:
                        self._last_position_check = None
                    
                    return OrderResult(
                        success=True,
                        order_id=order_id,
                        side=side,
                        size=quantity,
                        price=adjusted_price,
                        status=order_status
                    )
                elif order_status == 'PENDING':
                    raise Exception("[CLOSE] Order not processed after 10 seconds")
                else:
                    raise Exception(f"[CLOSE] Unexpected order status: {order_status}")
            
            return OrderResult(success=False, error_message=f"Failed to place close order after {max_retries} attempts")

    async def cancel_order(self, order_id: str) -> OrderResult:
        """Cancel an order with GRVT."""
        try:
            # Cancel the order using GRVT SDK
            cancel_result = self.rest_client.cancel_order(id=order_id)

            if cancel_result:
                return OrderResult(success=True)
            else:
                return OrderResult(success=False, error_message='Failed to cancel order')

        except Exception as e:
            return OrderResult(success=False, error_message=str(e))

    @query_retry(reraise=True)
    async def get_order_info(self, order_id: str = None, client_order_id: str = None) -> Optional[OrderInfo]:
        """Get order information from GRVT."""
        # Get order information using GRVT SDK
        if order_id is not None:
            order_data = self.rest_client.fetch_order(id=order_id)
        elif client_order_id is not None:
            order_data = self.rest_client.fetch_order(params={'client_order_id': client_order_id})
        else:
            raise ValueError("Either order_id or client_order_id must be provided")

        if not order_data or 'result' not in order_data:
            raise ValueError(f"Unable to get order info: {order_id}")

        order = order_data['result']
        legs = order.get('legs', [])
        if not legs:
            raise ValueError(f"Unable to get order info: {order_id}")

        leg = legs[0]  # Get first leg
        state = order.get('state', {})

        return OrderInfo(
            order_id=order.get('order_id', ''),
            side=leg.get('is_buying_asset', False) and 'buy' or 'sell',
            size=Decimal(leg.get('size', 0)),
            price=Decimal(leg.get('limit_price', 0)),
            status=state.get('status', ''),
            filled_size=(Decimal(state.get('traded_size', ['0'])[0])
                         if isinstance(state.get('traded_size'), list) else Decimal(0)),
            remaining_size=(Decimal(state.get('book_size', ['0'])[0])
                            if isinstance(state.get('book_size'), list) else Decimal(0))
        )

    async def _get_active_close_orders(self, contract_id: str) -> int:
        """Get active close orders for a contract using official SDK."""
        active_orders = await self.get_active_orders(contract_id)
        active_close_orders = 0
        for order in active_orders:
            if order.side == self.config.close_order_side:
                active_close_orders += 1
        return active_close_orders

    @query_retry(reraise=True)
    async def get_active_orders(self, contract_id: str) -> List[OrderInfo]:
        """Get active orders for a contract."""
        # Get active orders using GRVT SDK
        orders = self.rest_client.fetch_open_orders(symbol=contract_id)

        if not orders:
            return []

        order_list = []
        for order in orders:
            legs = order.get('legs', [])
            if not legs:
                continue

            leg = legs[0]  # Get first leg
            state = order.get('state', {})

            order_list.append(OrderInfo(
                order_id=order.get('order_id', ''),
                side=leg.get('is_buying_asset', False) and 'buy' or 'sell',
                size=Decimal(leg.get('size', 0)),
                price=Decimal(leg.get('limit_price', 0)),
                status=state.get('status', ''),
                filled_size=(Decimal(state.get('traded_size', ['0'])[0])
                             if isinstance(state.get('traded_size'), list) else Decimal(0)),
                remaining_size=(Decimal(state.get('book_size', ['0'])[0])
                                if isinstance(state.get('book_size'), list) else Decimal(0))
            ))

        return order_list

    @query_retry(reraise=True)
    async def get_account_positions(self) -> Decimal:
        """Get account positions with caching to prevent excessive API calls."""
        async with self._position_lock:
            current_time = time.time()
            
            # Use cached position if still valid (within TTL)
            if (self._last_position_check is not None and 
                current_time - self._last_position_check_time < self._position_cache_ttl):
                return self._last_position_check
            
            # Get positions using GRVT SDK
            positions = self.rest_client.fetch_positions()

            position_amt = Decimal(0)
            for position in positions:
                if position.get('instrument') == self.config.contract_id:
                    position_amt = abs(Decimal(position.get('size', 0)))
                    break
            
            # Update cache
            self._last_position_check = position_amt
            self._last_position_check_time = current_time
            
            return position_amt

    @query_retry(default_return=Decimal('0'))
    async def get_signed_position(self) -> Decimal:
        """Fetch the current signed position size for the configured contract with caching."""
        async with self._position_lock:
            current_time = time.time()
            
            # Use cached position if still valid (within TTL)
            # Note: We can't use absolute position cache for signed position, but we can optimize
            # by checking if we recently fetched positions
            if current_time - self._last_position_check_time < self._position_cache_ttl:
                # If we have cached absolute position, we can infer signed position
                # But for accuracy, we'll fetch fresh data for signed position
                pass
            
            positions = self.rest_client.fetch_positions()

            for position in positions:
                if position.get('instrument') != self.config.contract_id:
                    continue
                try:
                    signed_pos = Decimal(str(position.get('size', '0')))
                    # Update cache time to prevent excessive API calls
                    self._last_position_check_time = current_time
                    return signed_pos
                except (InvalidOperation, ValueError, TypeError):
                    return Decimal('0')

            return Decimal('0')

    async def get_contract_attributes(self) -> Tuple[str, Decimal]:
        """Get contract ID and tick size for a ticker."""
        ticker = self.config.ticker
        if not ticker:
            raise ValueError("Ticker is empty")

        # Get markets from GRVT
        markets = self.rest_client.fetch_markets()

        for market in markets:
            if (market.get('base') == ticker and
                    market.get('quote') == 'USDT' and
                    market.get('kind') == 'PERPETUAL'):

                self.config.contract_id = market.get('instrument', '')
                self.config.tick_size = Decimal(market.get('tick_size', 0))

                # Validate minimum quantity
                min_size = Decimal(market.get('min_size', 0))
                if self.config.quantity < min_size:
                    raise ValueError(
                        f"Order quantity is less than min quantity: {self.config.quantity} < {min_size}"
                    )

                return self.config.contract_id, self.config.tick_size

        raise ValueError(f"Contract not found for ticker: {ticker}")
