import asyncio
import os
import signal
import sys
import time
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Optional
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exchanges.bingx import BingxClient


class Config:
    """Simple attribute-style config wrapper."""

    def __init__(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            setattr(self, key, value)


class BingXHedgeBot:
    """
    BingX hedge bot that opens both long and short positions simultaneously,
    then sets TP/SL based on the average entry price.
    """

    def __init__(
        self,
        ticker: str,
        order_quantity: Decimal,
        iterations: int = 10,
        sleep_time: int = 0,
        tp_roi: Optional[Decimal] = None,
        sl_roi: Optional[Decimal] = None,
    ):
        self.ticker = ticker.upper()
        self.order_quantity = order_quantity
        self.iterations = iterations
        self.sleep_time = sleep_time
        self.tp_roi = Decimal(tp_roi) if tp_roi is not None else None
        self.sl_roi = Decimal(sl_roi) if sl_roi is not None else None

        self.stop_flag = False
        self.loop: Optional[asyncio.AbstractEventLoop] = None

        self.bingx_client: Optional[BingxClient] = None
        self.bingx_contract_id: Optional[str] = None
        self.bingx_tick_size: Optional[Decimal] = None

        self.long_position = Decimal('0')
        self.short_position = Decimal('0')
        
        os.makedirs("logs", exist_ok=True)
        self.log_filename = f"logs/bingx_{self.ticker.lower()}_hedge_log.txt"

        self.logger = logging.getLogger(f"hedge_bingx_{self.ticker}")
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

        self.logger.info(
            "BingX Hedge Bot initialized | ticker=%s | quantity=%s | tp_roi=%s | sl_roi=%s",
            self.ticker,
            self.order_quantity,
            self.tp_roi,
            self.sl_roi
        )

    def setup_signal_handlers(self) -> None:
        def handler(signum, frame):
            self.logger.info("Received shutdown signal, stopping hedge bot...")
            self.stop_flag = True

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def _build_bingx_config(self) -> Config:
        return Config({
            'ticker': self.ticker,
            'contract_id': '',
            'quantity': self.order_quantity,
            'tick_size': Decimal('0.01'),
            'direction': 'buy',
            'close_order_side': 'sell'
        })

    def initialize_client(self) -> None:
        if self.bingx_client is None:
            self.bingx_client = BingxClient(self._build_bingx_config())

    async def load_contract_metadata(self) -> None:
        assert self.bingx_client is not None
        self.bingx_contract_id, self.bingx_tick_size = await self.bingx_client.get_contract_attributes()
        self.logger.info(f"BingX contract: {self.bingx_contract_id} (tick {self.bingx_tick_size})")

    async def setup_bingx(self) -> None:
        assert self.bingx_client is not None
        await self.bingx_client.connect()

    async def place_market_order(self, side: str, quantity: Decimal) -> Optional[Dict[str, Any]]:
        """Place a market order on BingX."""
        assert self.bingx_client is not None
        assert self.bingx_contract_id is not None

        self.logger.info(f"[BINGX] Placing market {side.upper()} order: {quantity}")

        try:
            result = await self.bingx_client.place_market_order(
                contract_id=self.bingx_contract_id,
                quantity=quantity,
                side=side
            )
        except Exception as exc:
            self.logger.error(f"[BINGX] Market {side} order exception: {exc}")
            return None

        if not result.success:
            self.logger.error(f"[BINGX] Market {side} order failed: {result.error_message}")
            return None

        self.logger.info(
            f"[BINGX] Market {side.upper()} order filled: {result.size} @ {result.price}"
        )

        return {
            'order_id': result.order_id,
            'side': side,
            'size': result.size,
            'price': result.price,
            'status': result.status
        }

    async def place_limit_tp_order(self, side: str, quantity: Decimal, price: Decimal) -> Optional[Dict[str, Any]]:
        """Place a limit TP order on BingX."""
        assert self.bingx_client is not None
        assert self.bingx_contract_id is not None

        close_side = 'sell' if side == 'buy' else 'buy'
        
        self.logger.info(
            f"[BINGX] Placing limit {close_side.upper()} TP order: {quantity} @ {price}"
        )

        try:
            result = await self.bingx_client.place_limit_order(
                contract_id=self.bingx_contract_id,
                quantity=quantity,
                side=close_side,
                price=price,
                reduce_only=True,
                post_only=False,
                time_in_force='GTC'
            )
        except Exception as exc:
            self.logger.error(f"[BINGX] Limit {close_side} TP order exception: {exc}")
            return None

        if not result.success:
            self.logger.error(f"[BINGX] Limit {close_side} TP order failed: {result.error_message}")
            return None

        self.logger.info(
            f"[BINGX] Limit {close_side.upper()} TP order placed: {result.order_id}"
        )

        return {
            'order_id': result.order_id,
            'side': close_side,
            'size': result.size,
            'price': result.price,
            'status': result.status
        }

    async def open_hedge_positions(self) -> bool:
        """
        Open both long and short positions at market price,
        then set TP orders based on the average entry price.
        """
        self.logger.info("🔵 Opening hedge positions (long + short)...")

        # Place both market orders simultaneously
        long_task = self.place_market_order('buy', self.order_quantity)
        short_task = self.place_market_order('sell', self.order_quantity)

        long_result, short_result = await asyncio.gather(long_task, short_task)

        if not long_result or not short_result:
            self.logger.error("Failed to open both positions. Aborting.")
            return False

        long_price = long_result['price']
        short_price = short_result['price']
        long_size = long_result['size']
        short_size = short_result['size']

        self.logger.info(
            f"✅ Positions opened | LONG @ {long_price} | SHORT @ {short_price}"
        )

        # Calculate average entry price
        average_price = (long_price + short_price) / Decimal('2')
        self.logger.info(f"📊 Average entry price: {average_price}")

        # Update position tracking
        self.long_position += long_size
        self.short_position += short_size

        # Calculate and place TP orders if configured
        if self.tp_roi is not None and self.tp_roi > 0:
            hundred = Decimal('100')
            tp_factor = self.tp_roi / hundred

            # Long TP = average price * (1 + roi%)
            long_tp_price = average_price * (Decimal('1') + tp_factor)
            # Short TP = average price * (1 - roi%)
            short_tp_price = average_price * (Decimal('1') - tp_factor)

            # Round to tick size
            if self.bingx_client:
                try:
                    long_tp_price = self.bingx_client.round_to_tick(long_tp_price)
                    short_tp_price = self.bingx_client.round_to_tick(short_tp_price)
                except Exception as exc:
                    self.logger.warning(f"⚠️ Failed to round TP prices: {exc}")

            self.logger.info(
                f"🎯 TP targets | LONG @ {long_tp_price} (+{self.tp_roi}%) | SHORT @ {short_tp_price} (+{self.tp_roi}%)"
            )

            # Place TP orders
            long_tp_task = self.place_limit_tp_order('buy', long_size, long_tp_price)
            short_tp_task = self.place_limit_tp_order('sell', short_size, short_tp_price)

            long_tp_result, short_tp_result = await asyncio.gather(long_tp_task, short_tp_task)

            if not long_tp_result or not short_tp_result:
                self.logger.warning("⚠️ Failed to place some TP orders")
                return False

            self.logger.info("✅ TP orders placed successfully")

        return True

    async def wait_for_positions_close(self) -> None:
        """Wait for both positions to be closed by TP orders."""
        if self.tp_roi is None:
            self.logger.info("No TP configured, skipping wait.")
            return

        self.logger.info("⏳ Waiting for positions to close via TP orders...")
        
        poll_interval = 5.0  # Check every 5 seconds
        max_wait = 3600.0  # Wait up to 1 hour
        start_time = time.time()

        while not self.stop_flag:
            elapsed = time.time() - start_time
            if elapsed >= max_wait:
                self.logger.warning(f"⏱️ Position wait timed out after {elapsed:.1f}s")
                break

            # Check positions
            try:
                long_pos = await self.bingx_client.get_signed_position()
                
                if abs(long_pos) <= Decimal('0.001'):
                    self.logger.info("✅ Positions closed")
                    self.long_position = Decimal('0')
                    self.short_position = Decimal('0')
                    return
                    
                self.logger.info(f"📊 Current position: {long_pos} (waiting for close...)")
                
            except Exception as exc:
                self.logger.warning(f"Error checking positions: {exc}")

            await asyncio.sleep(poll_interval)

    async def close_all_positions(self) -> None:
        """Close all open positions at market price."""
        assert self.bingx_client is not None
        
        self.logger.info("🔴 Closing all positions...")
        
        try:
            position = await self.bingx_client.get_signed_position()
            
            if abs(position) <= Decimal('0.001'):
                self.logger.info("No positions to close")
                return
            
            # Determine close side
            side = 'sell' if position > 0 else 'buy'
            quantity = abs(position)
            
            self.logger.info(f"Closing position: {side.upper()} {quantity}")
            
            result = await self.bingx_client.place_market_order(
                contract_id=self.bingx_contract_id,
                quantity=quantity,
                side=side
            )
            
            if result.success:
                self.logger.info(f"✅ Position closed @ {result.price}")
                self.long_position = Decimal('0')
                self.short_position = Decimal('0')
            else:
                self.logger.error(f"Failed to close position: {result.error_message}")
                
        except Exception as exc:
            self.logger.error(f"Error closing positions: {exc}")

    async def trading_loop(self) -> None:
        self.logger.info(
            f"Starting BingX hedge bot | ticker={self.ticker} | size={self.order_quantity} | iterations={self.iterations}"
        )

        self.initialize_client()
        try:
            await self.load_contract_metadata()
            await self.setup_bingx()
        except Exception as exc:
            self.logger.error(f"Initialization failed: {exc}")
            self.stop_flag = True
            return

        await asyncio.sleep(2)

        iteration = 0
        while not self.stop_flag and iteration < self.iterations:
            iteration += 1
            self.logger.info(f"===== Iteration {iteration}/{self.iterations} =====")

            # Open hedge positions
            success = await self.open_hedge_positions()
            if not success or self.stop_flag:
                self.logger.error("Failed to open hedge positions, stopping.")
                break

            # Wait for positions to close
            await self.wait_for_positions_close()
            
            if self.stop_flag:
                break

            # Sleep before next iteration
            if self.sleep_time > 0 and iteration < self.iterations:
                self.logger.info(f"💤 Sleeping for {self.sleep_time}s before next iteration...")
                await asyncio.sleep(self.sleep_time)

        self.logger.info("Trading loop finished")

    async def cleanup(self) -> None:
        if self.bingx_client:
            try:
                # Try to close any remaining positions
                await self.close_all_positions()
            except Exception as exc:
                self.logger.warning(f"Error during cleanup: {exc}")
            
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
            self.logger.error(f"Unexpected error: {exc}", exc_info=True)
        finally:
            await self.cleanup()
            elapsed = time.time() - start_time
            self.logger.info(f"BingX hedge bot stopped after {elapsed:.1f}s")


def main():
    """Main entry point for the BingX hedge bot."""
    import sys
    
    # Parse command line arguments
    if len(sys.argv) < 3:
        print("Usage: python hedge_mode_bingx.py <TICKER> <QUANTITY> [ITERATIONS] [TP_ROI] [SL_ROI]")
        print("Example: python hedge_mode_bingx.py BTC 0.01 10 10 5")
        sys.exit(1)
    
    ticker = sys.argv[1]
    quantity = Decimal(sys.argv[2])
    iterations = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    tp_roi = Decimal(sys.argv[4]) if len(sys.argv) > 4 else Decimal('10')
    sl_roi = Decimal(sys.argv[5]) if len(sys.argv) > 5 else None
    
    bot = BingXHedgeBot(
        ticker=ticker,
        order_quantity=quantity,
        iterations=iterations,
        tp_roi=tp_roi,
        sl_roi=sl_roi
    )
    
    asyncio.run(bot.run())


if __name__ == "__main__":
    main()
