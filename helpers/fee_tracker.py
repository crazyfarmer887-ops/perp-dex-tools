"""
Fee tracking and reporting module for trading bot.
Tracks maker/taker fees, calculates total fees, and generates reports.
"""

import os
import csv
from datetime import datetime
from decimal import Decimal
from typing import Dict, Optional, Tuple
from collections import defaultdict
import pytz


class FeeTracker:
    """Tracks trading fees and generates reports."""

    # Exchange fee structures (maker/taker fees in percentage)
    # These are typical fees - actual fees may vary based on VIP levels, rebates, etc.
    EXCHANGE_FEES = {
        'edgex': {'maker': Decimal('0.02'), 'taker': Decimal('0.05')},  # VIP1: 0.02% maker, 0.05% taker
        'backpack': {'maker': Decimal('0.02'), 'taker': Decimal('0.04')},
        'extended': {'maker': Decimal('0.02'), 'taker': Decimal('0.05')},
        'aster': {'maker': Decimal('0.02'), 'taker': Decimal('0.04')},
        'grvt': {'maker': Decimal('0.02'), 'taker': Decimal('0.05')},
        'apex': {'maker': Decimal('0.02'), 'taker': Decimal('0.05')},
        'paradex': {'maker': Decimal('0.02'), 'taker': Decimal('0.05')},
        'lighter': {'maker': Decimal('0.02'), 'taker': Decimal('0.05')},
        'bingx': {'maker': Decimal('0.02'), 'taker': Decimal('0.04')},
    }

    def __init__(self, exchange: str, ticker: str):
        """Initialize fee tracker."""
        self.exchange = exchange.lower()
        self.ticker = ticker.upper()
        self.timezone = pytz.timezone(os.getenv('TIMEZONE', 'Asia/Shanghai'))
        
        # Get fee structure for this exchange
        self.maker_fee_rate = self.EXCHANGE_FEES.get(self.exchange, {}).get('maker', Decimal('0.02'))
        self.taker_fee_rate = self.EXCHANGE_FEES.get(self.exchange, {}).get('taker', Decimal('0.05'))
        
        # Fee tracking data
        self.fees_by_type = defaultdict(Decimal)  # {'maker': total, 'taker': total}
        self.fees_by_order_type = defaultdict(Decimal)  # {'OPEN': total, 'CLOSE': total}
        self.total_volume = Decimal('0')
        self.total_fees = Decimal('0')
        self.transaction_count = 0
        
        # Setup fee log file
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        logs_dir = os.path.join(project_root, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        account_name = os.getenv('ACCOUNT_NAME')
        fee_file_name = f"{self.exchange}_{self.ticker}_fees.csv"
        if account_name:
            fee_file_name = f"{self.exchange}_{self.ticker}_{account_name}_fees.csv"
        
        self.fee_log_file = os.path.join(logs_dir, fee_file_name)
        self._initialize_fee_log()

    def _initialize_fee_log(self):
        """Initialize fee log file with headers."""
        if not os.path.isfile(self.fee_log_file):
            with open(self.fee_log_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    'Timestamp', 'OrderID', 'OrderType', 'Side', 'Quantity', 
                    'Price', 'Notional', 'FeeType', 'FeeRate', 'FeeAmount', 'Status'
                ])

    def calculate_fee(
        self, 
        quantity: Decimal, 
        price: Decimal, 
        fee_type: str = 'maker',
        order_type: Optional[str] = None
    ) -> Decimal:
        """
        Calculate fee for a transaction.
        
        Args:
            quantity: Order quantity
            price: Order price
            fee_type: 'maker' or 'taker'
            order_type: 'OPEN' or 'CLOSE' (optional, for tracking)
        
        Returns:
            Calculated fee amount
        """
        notional = quantity * price
        fee_rate = self.maker_fee_rate if fee_type.lower() == 'maker' else self.taker_fee_rate
        fee_amount = notional * fee_rate / Decimal('100')
        
        # Track fees
        self.fees_by_type[fee_type.lower()] += fee_amount
        if order_type:
            self.fees_by_order_type[order_type.upper()] += fee_amount
        
        self.total_volume += notional
        self.total_fees += fee_amount
        self.transaction_count += 1
        
        return fee_amount

    def log_fee(
        self,
        order_id: str,
        order_type: str,
        side: str,
        quantity: Decimal,
        price: Decimal,
        fee_type: str,
        status: str
    ):
        """Log a fee transaction to CSV file."""
        try:
            timestamp = datetime.now(self.timezone).strftime("%Y-%m-%d %H:%M:%S")
            notional = quantity * price
            fee_rate = self.maker_fee_rate if fee_type.lower() == 'maker' else self.taker_fee_rate
            fee_amount = self.calculate_fee(quantity, price, fee_type, order_type)
            
            row = [
                timestamp,
                order_id,
                order_type,
                side,
                quantity,
                price,
                notional,
                fee_type.upper(),
                fee_rate,
                fee_amount,
                status
            ]
            
            with open(self.fee_log_file, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(row)
                
        except Exception as e:
            print(f"Failed to log fee: {e}")

    def get_fee_summary(self) -> Dict:
        """Get summary of fees."""
        return {
            'exchange': self.exchange,
            'ticker': self.ticker,
            'total_volume': self.total_volume,
            'total_fees': self.total_fees,
            'maker_fees': self.fees_by_type.get('maker', Decimal('0')),
            'taker_fees': self.fees_by_type.get('taker', Decimal('0')),
            'open_order_fees': self.fees_by_order_type.get('OPEN', Decimal('0')),
            'close_order_fees': self.fees_by_order_type.get('CLOSE', Decimal('0')),
            'transaction_count': self.transaction_count,
            'maker_fee_rate': self.maker_fee_rate,
            'taker_fee_rate': self.taker_fee_rate,
            'avg_fee_per_transaction': self.total_fees / self.transaction_count if self.transaction_count > 0 else Decimal('0'),
            'fee_percentage': (self.total_fees / self.total_volume * 100) if self.total_volume > 0 else Decimal('0')
        }

    def print_summary(self):
        """Print fee summary to console."""
        summary = self.get_fee_summary()
        
        print("\n" + "="*60)
        print(f"Fee Summary - {summary['exchange'].upper()} {summary['ticker']}")
        print("="*60)
        print(f"Total Volume: ${summary['total_volume']:,.2f}")
        print(f"Total Fees: ${summary['total_fees']:,.4f}")
        print(f"  - Maker Fees: ${summary['maker_fees']:,.4f}")
        print(f"  - Taker Fees: ${summary['taker_fees']:,.4f}")
        print(f"  - Open Order Fees: ${summary['open_order_fees']:,.4f}")
        print(f"  - Close Order Fees: ${summary['close_order_fees']:,.4f}")
        print(f"Transaction Count: {summary['transaction_count']}")
        print(f"Average Fee per Transaction: ${summary['avg_fee_per_transaction']:,.4f}")
        print(f"Fee Percentage: {summary['fee_percentage']:.4f}%")
        print(f"Maker Fee Rate: {summary['maker_fee_rate']}%")
        print(f"Taker Fee Rate: {summary['taker_fee_rate']}%")
        print("="*60 + "\n")

    def reset(self):
        """Reset fee tracking data."""
        self.fees_by_type.clear()
        self.fees_by_order_type.clear()
        self.total_volume = Decimal('0')
        self.total_fees = Decimal('0')
        self.transaction_count = 0
