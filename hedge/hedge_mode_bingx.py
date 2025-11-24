"""
BingX Hedge Position Mode
Creates simultaneous long and short positions with TP/SL based on average entry price
"""

import asyncio
import os
import signal
import sys
import time
from decimal import Decimal
from typing import Optional, Tuple, Dict, Any
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exchanges.bingx import BingxClient


class Config:
    """Simple attribute-style config wrapper for BingX client."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            setattr(self, key, value)


class BingxHedgeBot:
    """
    BingX hedge bot that opens simultaneous long and short positions
    with TP/SL orders based on average entry price.
    """
    
    def __init__(
        self,
        ticker: str,
        order_quantity: Decimal,
        tp_roi: Optional[Decimal] = None,
        sl_roi: Optional[Decimal] = None,
        iterations: int = 1,
        sleep_time: int = 0,
    ):
        """
        Initialize BingX hedge bot.
        
        Args:
            ticker: Trading pair ticker (e.g., 'BTC')
            order_quantity: Position size for each side
            tp_roi: Take profit ROI percentage (e.g., 10 for 10%)
            sl_roi: Stop loss ROI percentage (e.g., 10 for 10%)
            iterations: Number of trading cycles to run
            sleep_time: Sleep time between iterations in seconds
        """
        self.ticker = ticker.upper()
        self.order_quantity = Decimal(str(order_quantity))
        self.tp_roi = Decimal(str(tp_roi)) if tp_roi is not None else None
        self.sl_roi = Decimal(str(sl_roi)) if sl_roi is not None else None
        self.iterations = iterations
        self.sleep_time = sleep_time
        
        self.stop_flag = False
        self.bingx_client: Optional[BingxClient] = None
        self.contract_id: Optional[str] = None
        self.tick_size: Optional[Decimal] = None
        
        # Position tracking
        self.long_position: Optional[Dict[str, Any]] = None
        self.short_position: Optional[Dict[str, Any]] = None
        self.average_entry_price: Optional[Decimal] = None
        
        # Logging setup
        os.makedirs("logs", exist_ok=True)
        self.log_filename = f"logs/bingx_{self.ticker.lower()}_hedge_log.txt"
        
        self.logger = logging.getLogger(f"bingx_hedge_{self.ticker}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(self.log_filename)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(
            f"BingX Hedge Bot initialized | ticker={self.ticker} | "
            f"quantity={self.order_quantity} | tp_roi={self.tp_roi}% | sl_roi={self.sl_roi}%"
        )
    
    def setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def handler(signum, frame):
            self.logger.info("Received shutdown signal, stopping hedge bot...")
            self.stop_flag = True
        
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)
    
    def _build_config(self) -> Config:
        """Build configuration for BingX client."""
        return Config({
            'ticker': self.ticker,
            'contract_id': '',
            'quantity': self.order_quantity,
            'tick_size': Decimal('0.01'),
            'direction': 'buy',
            'close_order_side': 'sell'
        })
    
    def initialize_client(self) -> None:
        """Initialize BingX client."""
        if self.bingx_client is None:
            self.bingx_client = BingxClient(self._build_config())
            self.logger.info("BingX client initialized")
    
    async def load_contract_metadata(self) -> None:
        """Load contract metadata from BingX."""
        assert self.bingx_client is not None
        
        self.contract_id, self.tick_size = await self.bingx_client.get_contract_attributes()
        self.logger.info(f"Contract loaded: {self.contract_id} (tick size: {self.tick_size})")
    
    async def setup_bingx(self) -> None:
        """Connect to BingX and setup client."""
        assert self.bingx_client is not None
        await self.bingx_client.connect()
        self.logger.info("Connected to BingX")
    
    def calculate_tp_sl_prices(self, average_price: Decimal) -> Tuple[Decimal, Decimal, Decimal, Decimal]:
        """
        Calculate TP/SL prices based on average entry price.
        
        Args:
            average_price: Average entry price of long and short positions
            
        Returns:
            Tuple of (long_tp, long_sl, short_tp, short_sl)
        """
        hundred = Decimal('100')
        one = Decimal('1')
        
        # Calculate TP/SL for long position
        long_tp = average_price
        long_sl = average_price
        if self.tp_roi is not None:
            tp_factor = self.tp_roi / hundred
            long_tp = average_price * (one + tp_factor)
        if self.sl_roi is not None:
            sl_factor = self.sl_roi / hundred
            long_sl = average_price * (one - sl_factor)
        
        # Calculate TP/SL for short position (inverse of long)
        short_tp = average_price
        short_sl = average_price
        if self.tp_roi is not None:
            tp_factor = self.tp_roi / hundred
            short_tp = average_price * (one - tp_factor)
        if self.sl_roi is not None:
            sl_factor = self.sl_roi / hundred
            short_sl = average_price * (one + sl_factor)
        
        # Round to tick size
        if self.bingx_client:
            long_tp = self.bingx_client.round_to_tick(long_tp)
            long_sl = self.bingx_client.round_to_tick(long_sl)
            short_tp = self.bingx_client.round_to_tick(short_tp)
            short_sl = self.bingx_client.round_to_tick(short_sl)
        
        return long_tp, long_sl, short_tp, short_sl
    
    async def open_hedge_positions(self) -> bool:
        """
        Open simultaneous long and short positions with market orders.
        
        Returns:
            True if both positions opened successfully, False otherwise
        """
        assert self.bingx_client is not None
        assert self.contract_id is not None
        
        self.logger.info(f"Opening hedge positions with size {self.order_quantity}...")
        
        # Place long and short market orders simultaneously
        long_task = self.bingx_client.place_market_order(
            contract_id=self.contract_id,
            quantity=self.order_quantity,
            side='buy'
        )
        
        short_task = self.bingx_client.place_market_order(
            contract_id=self.contract_id,
            quantity=self.order_quantity,
            side='sell'
        )
        
        # Execute both orders in parallel
        long_result, short_result = await asyncio.gather(long_task, short_task)
        
        # Check results
        if not long_result.success:
            self.logger.error(f"Failed to open long position: {long_result.error_message}")
            return False
        
        if not short_result.success:
            self.logger.error(f"Failed to open short position: {short_result.error_message}")
            # Try to close the long position if short failed
            await self.close_position('buy')
            return False
        
        # Store position info
        self.long_position = {
            'order_id': long_result.order_id,
            'side': 'buy',
            'size': long_result.filled_size or self.order_quantity,
            'price': long_result.price,
            'status': long_result.status
        }
        
        self.short_position = {
            'order_id': short_result.order_id,
            'side': 'sell',
            'size': short_result.filled_size or self.order_quantity,
            'price': short_result.price,
            'status': short_result.status
        }
        
        # Calculate average entry price
        long_price = long_result.price or Decimal('0')
        short_price = short_result.price or Decimal('0')
        
        if long_price > 0 and short_price > 0:
            self.average_entry_price = (long_price + short_price) / Decimal('2')
        else:
            self.logger.error("Invalid entry prices, cannot calculate average")
            return False
        
        self.logger.info(
            f"✅ Hedge positions opened | "
            f"Long: {self.order_quantity} @ {long_price} | "
            f"Short: {self.order_quantity} @ {short_price} | "
            f"Average: {self.average_entry_price}"
        )
        
        return True
    
    async def place_tp_sl_orders(self) -> bool:
        """
        Place TP/SL limit orders for both positions based on average entry price.
        
        Returns:
            True if all orders placed successfully, False otherwise
        """
        assert self.bingx_client is not None
        assert self.contract_id is not None
        
        if self.average_entry_price is None or self.average_entry_price <= 0:
            self.logger.error("Cannot place TP/SL orders: invalid average entry price")
            return False
        
        if self.tp_roi is None and self.sl_roi is None:
            self.logger.info("No TP/SL ROI configured, skipping TP/SL orders")
            return True
        
        # Calculate TP/SL prices
        long_tp, long_sl, short_tp, short_sl = self.calculate_tp_sl_prices(self.average_entry_price)
        
        self.logger.info(
            f"Placing TP/SL orders based on average price {self.average_entry_price}..."
        )
        
        tasks = []
        order_descriptions = []
        
        # Long position TP (sell limit order)
        if self.tp_roi is not None and long_tp > 0:
            tasks.append(
                self.bingx_client.place_limit_order(
                    contract_id=self.contract_id,
                    quantity=self.order_quantity,
                    side='sell',
                    price=long_tp,
                    reduce_only=True,
                    post_only=False,
                    time_in_force='GTC'
                )
            )
            order_descriptions.append(f"Long TP: SELL @ {long_tp}")
        
        # Long position SL (sell limit order)
        if self.sl_roi is not None and long_sl > 0:
            tasks.append(
                self.bingx_client.place_limit_order(
                    contract_id=self.contract_id,
                    quantity=self.order_quantity,
                    side='sell',
                    price=long_sl,
                    reduce_only=True,
                    post_only=False,
                    time_in_force='GTC'
                )
            )
            order_descriptions.append(f"Long SL: SELL @ {long_sl}")
        
        # Short position TP (buy limit order)
        if self.tp_roi is not None and short_tp > 0:
            tasks.append(
                self.bingx_client.place_limit_order(
                    contract_id=self.contract_id,
                    quantity=self.order_quantity,
                    side='buy',
                    price=short_tp,
                    reduce_only=True,
                    post_only=False,
                    time_in_force='GTC'
                )
            )
            order_descriptions.append(f"Short TP: BUY @ {short_tp}")
        
        # Short position SL (buy limit order)
        if self.sl_roi is not None and short_sl > 0:
            tasks.append(
                self.bingx_client.place_limit_order(
                    contract_id=self.contract_id,
                    quantity=self.order_quantity,
                    side='buy',
                    price=short_sl,
                    reduce_only=True,
                    post_only=False,
                    time_in_force='GTC'
                )
            )
            order_descriptions.append(f"Short SL: BUY @ {short_sl}")
        
        if not tasks:
            self.logger.warning("No TP/SL orders to place")
            return True
        
        # Execute all TP/SL orders in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check results
        success_count = 0
        for i, (result, description) in enumerate(zip(results, order_descriptions)):
            if isinstance(result, Exception):
                self.logger.error(f"❌ {description} - Exception: {result}")
            elif not result.success:
                self.logger.error(f"❌ {description} - Error: {result.error_message}")
            else:
                self.logger.info(f"✅ {description} - Order ID: {result.order_id}")
                success_count += 1
        
        if success_count == len(tasks):
            self.logger.info(
                f"🎯 All TP/SL orders placed successfully | "
                f"Long: TP={long_tp}, SL={long_sl} | "
                f"Short: TP={short_tp}, SL={short_sl}"
            )
            return True
        else:
            self.logger.warning(
                f"⚠️ Only {success_count}/{len(tasks)} TP/SL orders placed successfully"
            )
            return success_count > 0
    
    async def close_position(self, side: str) -> bool:
        """
        Close a position with market order.
        
        Args:
            side: 'buy' to close long, 'sell' to close short
            
        Returns:
            True if closed successfully, False otherwise
        """
        assert self.bingx_client is not None
        assert self.contract_id is not None
        
        # Determine close side (opposite of position)
        close_side = 'sell' if side == 'buy' else 'buy'
        
        self.logger.info(f"Closing {side} position with {close_side} market order...")
        
        result = await self.bingx_client.place_market_order(
            contract_id=self.contract_id,
            quantity=self.order_quantity,
            side=close_side
        )
        
        if result.success:
            self.logger.info(f"✅ Position closed: {close_side} {self.order_quantity} @ {result.price}")
            return True
        else:
            self.logger.error(f"❌ Failed to close position: {result.error_message}")
            return False
    
    async def close_all_positions(self) -> None:
        """Close all open positions."""
        self.logger.info("Closing all positions...")
        
        # Get current positions
        long_pos = await self.bingx_client.get_signed_position() if self.bingx_client else Decimal('0')
        
        if abs(long_pos) > Decimal('0.001'):
            if long_pos > 0:
                await self.close_position('buy')
            else:
                await self.close_position('sell')
        else:
            self.logger.info("No open positions to close")
    
    async def wait_for_tp_sl(self, timeout: int = 300) -> None:
        """
        Wait for TP/SL orders to be filled.
        
        Args:
            timeout: Maximum time to wait in seconds
        """
        if self.tp_roi is None and self.sl_roi is None:
            return
        
        self.logger.info(f"⏳ Waiting for TP/SL orders to fill (timeout: {timeout}s)...")
        
        start_time = time.time()
        poll_interval = 2.0
        
        while not self.stop_flag and (time.time() - start_time) < timeout:
            # Check current position
            position = await self.bingx_client.get_signed_position() if self.bingx_client else Decimal('0')
            
            if abs(position) < Decimal('0.001'):
                self.logger.info("✅ All positions closed (TP/SL filled or manually closed)")
                break
            
            elapsed = time.time() - start_time
            if elapsed % 30 < poll_interval:  # Log every 30 seconds
                self.logger.info(
                    f"Position still open: {position} | Elapsed: {elapsed:.0f}s"
                )
            
            await asyncio.sleep(poll_interval)
        
        if not self.stop_flag:
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                self.logger.info(f"⏱️ TP/SL wait timed out after {elapsed:.0f}s")
    
    async def execute_hedge_cycle(self) -> bool:
        """
        Execute one complete hedge cycle.
        
        Returns:
            True if cycle completed successfully, False otherwise
        """
        try:
            # Open hedge positions
            if not await self.open_hedge_positions():
                self.logger.error("Failed to open hedge positions")
                return False
            
            # Place TP/SL orders
            if not await self.place_tp_sl_orders():
                self.logger.warning("Some TP/SL orders failed, but continuing...")
            
            # Wait for TP/SL to fill
            await self.wait_for_tp_sl()
            
            # Close any remaining positions
            await self.close_all_positions()
            
            return True
            
        except Exception as exc:
            self.logger.error(f"Error during hedge cycle: {exc}")
            return False
    
    async def trading_loop(self) -> None:
        """Main trading loop."""
        self.logger.info(
            f"Starting BingX hedge bot | ticker={self.ticker} | "
            f"size={self.order_quantity} | iterations={self.iterations}"
        )
        
        self.initialize_client()
        
        try:
            await self.load_contract_metadata()
            await self.setup_bingx()
        except Exception as exc:
            self.logger.error(f"Initialization failed: {exc}")
            self.stop_flag = True
            return
        
        # Small delay to ensure connection is stable
        await asyncio.sleep(2)
        
        iteration = 0
        while not self.stop_flag and iteration < self.iterations:
            iteration += 1
            self.logger.info(f"===== Iteration {iteration}/{self.iterations} =====")
            
            # Execute hedge cycle
            success = await self.execute_hedge_cycle()
            
            if not success:
                self.logger.error(f"Hedge cycle {iteration} failed")
                if iteration < self.iterations:
                    self.logger.info(f"Waiting {self.sleep_time}s before next iteration...")
                    await asyncio.sleep(self.sleep_time)
            else:
                self.logger.info(f"✅ Hedge cycle {iteration} completed successfully")
                if iteration < self.iterations and self.sleep_time > 0:
                    self.logger.info(f"Waiting {self.sleep_time}s before next iteration...")
                    await asyncio.sleep(self.sleep_time)
        
        self.logger.info("Trading loop finished")
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.bingx_client:
            try:
                # Try to close any remaining positions
                await self.close_all_positions()
            except Exception as exc:
                self.logger.warning(f"Error closing positions during cleanup: {exc}")
            
            try:
                await self.bingx_client.disconnect()
            except Exception as exc:
                self.logger.warning(f"Error disconnecting BingX client: {exc}")
    
    async def run(self) -> None:
        """Main run method."""
        self.setup_signal_handlers()
        
        start_time = time.time()
        try:
            await self.trading_loop()
        except Exception as exc:
            self.logger.error(f"Unexpected error: {exc}")
        finally:
            await self.cleanup()
            elapsed = time.time() - start_time
            self.logger.info(f"BingX hedge bot stopped after {elapsed:.1f}s")


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='BingX Hedge Position Mode Bot')
    parser.add_argument('ticker', type=str, help='Trading pair ticker (e.g., BTC)')
    parser.add_argument('quantity', type=Decimal, help='Position size for each side')
    parser.add_argument('--tp-roi', type=Decimal, default=None, 
                       help='Take profit ROI percentage (e.g., 10 for 10%%)')
    parser.add_argument('--sl-roi', type=Decimal, default=None,
                       help='Stop loss ROI percentage (e.g., 10 for 10%%)')
    parser.add_argument('--iterations', type=int, default=1,
                       help='Number of trading cycles (default: 1)')
    parser.add_argument('--sleep', type=int, default=0,
                       help='Sleep time between iterations in seconds (default: 0)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.quantity <= 0:
        print("Error: Quantity must be positive")
        sys.exit(1)
    
    if args.tp_roi is not None and args.tp_roi <= 0:
        print("Error: TP ROI must be positive")
        sys.exit(1)
    
    if args.sl_roi is not None and args.sl_roi <= 0:
        print("Error: SL ROI must be positive")
        sys.exit(1)
    
    # Create and run bot
    bot = BingxHedgeBot(
        ticker=args.ticker,
        order_quantity=args.quantity,
        tp_roi=args.tp_roi,
        sl_roi=args.sl_roi,
        iterations=args.iterations,
        sleep_time=args.sleep
    )
    
    await bot.run()


if __name__ == '__main__':
    asyncio.run(main())