import asyncio
import argparse
import logging
import os
import sys
import time
from decimal import Decimal
from typing import Optional, Dict, Any, Tuple

# Add workspace root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exchanges.bingx import BingxClient
from helpers.logger import TradingLogger

class BingxHedgePositionBot:
    """
    BingX Hedge Position Mode Bot.
    Opens both Long and Short positions at market price.
    Calculates average entry price and sets TP/SL limit orders based on ROI/Leverage.
    """

    def __init__(
        self,
        ticker: str,
        amount: Decimal,
        leverage: int,
        roi: Decimal,
    ):
        self.ticker = ticker.upper()
        self.amount = amount
        self.leverage = leverage
        self.roi = roi
        
        # Configuration for BingxClient
        self.config = {
            'ticker': self.ticker,
            'quantity': self.amount,
            # These are required by BingxClient but might be overridden
            'contract_id': f"{self.ticker}-USDT", 
            'tick_size': Decimal('0.1'), # Will be updated
            'close_order_side': 'sell' 
        }
        
        self.client = BingxClient(self.config)
        self.logger = logging.getLogger(f"bingx_hedge_{self.ticker}")
        self._setup_logger()

    def _setup_logger(self):
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    async def initialize(self):
        await self.client.connect()
        # Update contract ID and tick size
        await self.client.get_contract_attributes()
        self.logger.info(f"Initialized. Contract: {self.client.config.contract_id}, Tick Size: {self.client.config.tick_size}")

    async def _place_dual_position(self) -> Tuple[Decimal, Decimal]:
        """
        Places market Long and Short orders.
        Returns (long_entry_price, short_entry_price)
        """
        self.logger.info(f"Placing Dual Positions (Long & Short) for {self.amount} {self.ticker}...")
        
        # We need to ensure we are using Hedge Mode parameters (positionSide)
        # BingX Swap (Standard/Perpetual) via CCXT usually requires 'positionSide' for Hedge Mode.
        
        tasks = [
            self.client.exchange.create_order(
                symbol=self.client.config.contract_id,
                type='market',
                side='buy',
                amount=float(self.amount),
                params={'positionSide': 'LONG'}
            ),
            self.client.exchange.create_order(
                symbol=self.client.config.contract_id,
                type='market',
                side='sell',
                amount=float(self.amount),
                params={'positionSide': 'SHORT'}
            )
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        long_order = results[0]
        short_order = results[1]
        
        long_price = Decimal('0')
        short_price = Decimal('0')

        if isinstance(long_order, Exception):
            self.logger.error(f"Failed to open LONG: {long_order}")
        else:
            long_price = Decimal(str(long_order.get('average') or long_order.get('price') or 0))
            self.logger.info(f"LONG Opened: {long_order.get('id')} @ {long_price}")

        if isinstance(short_order, Exception):
            self.logger.error(f"Failed to open SHORT: {short_order}")
        else:
            short_price = Decimal(str(short_order.get('average') or short_order.get('price') or 0))
            self.logger.info(f"SHORT Opened: {short_order.get('id')} @ {short_price}")

        return long_price, short_price

    def _calculate_targets(self, avg_price: Decimal) -> Tuple[Decimal, Decimal]:
        """
        Calculates Upper and Lower target prices based on ROI and Leverage.
        Delta = Price * (ROI% / Leverage)
        Upper = Avg + Delta
        Lower = Avg - Delta
        """
        # ROI is in percentage (e.g. 10 for 10%)
        # If ROI=10, Leverage=10 -> Price movement needed = 1%
        
        price_movement_pct = (self.roi / Decimal('100')) / Decimal(self.leverage)
        delta = avg_price * price_movement_pct
        
        # Round delta to tick size
        tick_size = self.client.config.tick_size
        delta = (delta / tick_size).quantize(Decimal('1')) * tick_size
        
        upper_price = avg_price + delta
        lower_price = avg_price - delta
        
        return upper_price, lower_price

    async def _place_closing_orders(self, upper_price: Decimal, lower_price: Decimal):
        """
        Places closing orders.
        User Logic: "tpsl limit... apply tpsl to both"
        
        LONG Position:
          - TP: Sell Limit @ Upper (Close Long)
          - SL: Sell Stop (Trigger) @ Lower (Close Long)
        
        SHORT Position:
          - TP: Buy Limit @ Lower (Close Short)
          - SL: Buy Stop (Trigger) @ Upper (Close Short)
          
        However, user said "set as limit". 
        If strict limit orders:
          - Sell Limit @ Upper (TP Long)
          - Buy Limit @ Lower (TP Short)
          
        SL via Limit order is tricky because if we place Sell Limit @ Lower (below market), it fills immediately.
        So SL MUST be a Trigger Order (Stop Limit/Market).
        "This can be set as limit price" probably refers to the execution price of the TP/SL or the TP orders themselves.
        
        We will place:
        1. Limit Sell (Long TP) @ Upper
        2. Limit Buy (Short TP) @ Lower
        3. Stop Limit Sell (Long SL) @ Lower (Trigger=Lower, Price=Lower)
        4. Stop Limit Buy (Short SL) @ Upper (Trigger=Upper, Price=Upper)
        """
        self.logger.info(f"Placing Closing Orders... Upper: {upper_price}, Lower: {lower_price}")
        
        # BingX Trigger Orders via CCXT
        # For Stop Loss, we use params to specify trigger price and type.
        
        orders = []
        
        # 1. LONG TP (Limit Sell @ Upper)
        orders.append(self.client.exchange.create_order(
            symbol=self.client.config.contract_id,
            type='limit',
            side='sell',
            amount=float(self.amount),
            price=float(upper_price),
            params={'positionSide': 'LONG', 'reduceOnly': True}
        ))
        
        # 2. SHORT TP (Limit Buy @ Lower)
        orders.append(self.client.exchange.create_order(
            symbol=self.client.config.contract_id,
            type='limit',
            side='buy',
            amount=float(self.amount),
            price=float(lower_price),
            params={'positionSide': 'SHORT', 'reduceOnly': True}
        ))
        
        # 3. LONG SL (Stop Limit Sell @ Lower)
        # BingX API usually requires 'stopPrice'
        orders.append(self.client.exchange.create_order(
            symbol=self.client.config.contract_id,
            type='limit', # Using limit order with stop price
            side='sell',
            amount=float(self.amount),
            price=float(lower_price), # Limit Price
            params={
                'positionSide': 'LONG', 
                'stopPrice': float(lower_price), # Trigger Price
                'stop': 'loss', # CCXT unified param? Or BingX specific
                # BingX might need 'type': 'STOP_LIMIT' or similar if CCXT doesn't map it automatically
                # For safety, let's trust CCXT mapping for 'stopPrice' or check BingxClient implementation
                # BingxClient _build_tp_sl_payload uses 'STOP' type. 
                # But here we are placing a standalone order, not attaching to position.
                # Standalone trigger order: type='STOP_LIMIT' or 'STOP'
                'type': 'STOP', # Try STOP or STOP_MARKET/STOP_LIMIT
                'workingType': 'MARK_PRICE',
                'reduceOnly': True
            }
        ))
        
        # 4. SHORT SL (Stop Limit Buy @ Upper)
        orders.append(self.client.exchange.create_order(
            symbol=self.client.config.contract_id,
            type='limit',
            side='buy',
            amount=float(self.amount),
            price=float(upper_price),
            params={
                'positionSide': 'SHORT',
                'stopPrice': float(upper_price),
                'type': 'STOP',
                'workingType': 'MARK_PRICE',
                'reduceOnly': True
            }
        ))

        results = await asyncio.gather(*orders, return_exceptions=True)
        
        for i, res in enumerate(results):
            desc = ["Long TP", "Short TP", "Long SL", "Short SL"][i]
            if isinstance(res, Exception):
                self.logger.error(f"Failed to place {desc}: {res}")
            else:
                self.logger.info(f"{desc} Placed: {res.get('id')}")

    async def run(self):
        try:
            await self.initialize()
            
            # 1. Enter Positions
            long_px, short_px = await self._place_dual_position()
            
            if long_px == 0 or short_px == 0:
                self.logger.error("Failed to open both positions. Aborting TP/SL setup.")
                return

            # 2. Average Price
            avg_entry = (long_px + short_px) / Decimal('2')
            self.logger.info(f"Avg Entry Price: {avg_entry} (L: {long_px}, S: {short_px})")
            
            # 3. Calculate Targets
            upper, lower = self._calculate_targets(avg_entry)
            self.logger.info(f"Targets calculated (ROI {self.roi}% w/ {self.leverage}x) -> Delta: {upper - avg_entry}")
            
            # 4. Place Closing Orders
            await self._place_closing_orders(upper, lower)
            
            self.logger.info("Done.")
            
        except Exception as e:
            self.logger.error(f"Error in execution: {e}")
        finally:
            await self.client.disconnect()

async def main():
    parser = argparse.ArgumentParser(description='BingX Hedge Mode Bot')
    parser.add_argument('--ticker', type=str, required=True, help='Trading Pair (e.g. BTC)')
    parser.add_argument('--amount', type=str, required=True, help='Amount in BTC/ETH etc')
    parser.add_argument('--leverage', type=int, required=True, help='Leverage used for calculation')
    parser.add_argument('--roi', type=str, required=True, help='Target ROI %')
    
    args = parser.parse_args()
    
    bot = BingxHedgePositionBot(
        ticker=args.ticker,
        amount=Decimal(args.amount),
        leverage=args.leverage,
        roi=Decimal(args.roi)
    )
    
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())
