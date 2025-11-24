#!/usr/bin/env python3
"""
Test script for BingX Hedge Mode
This script tests the basic functionality without actually placing orders
"""

import sys
import os
import asyncio
from decimal import Decimal

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hedge.hedge_mode_bingx import BingxHedgeBot


def test_tp_sl_calculation():
    """Test TP/SL price calculation logic."""
    print("Testing TP/SL calculation...")
    
    # Create bot instance
    bot = BingxHedgeBot(
        ticker="BTC",
        order_quantity=Decimal("0.001"),
        tp_roi=Decimal("10"),
        sl_roi=Decimal("10"),
        iterations=1
    )
    
    # Test with average price of 95,000
    average_price = Decimal("95000")
    long_tp, long_sl, short_tp, short_sl = bot.calculate_tp_sl_prices(average_price)
    
    print(f"Average Price: {average_price}")
    print(f"Long TP: {long_tp} (Expected: ~104,500)")
    print(f"Long SL: {long_sl} (Expected: ~85,500)")
    print(f"Short TP: {short_tp} (Expected: ~85,500)")
    print(f"Short SL: {short_sl} (Expected: ~104,500)")
    
    # Verify calculations
    expected_long_tp = average_price * Decimal("1.1")
    expected_long_sl = average_price * Decimal("0.9")
    expected_short_tp = average_price * Decimal("0.9")
    expected_short_sl = average_price * Decimal("1.1")
    
    assert abs(long_tp - expected_long_tp) < Decimal("1"), f"Long TP calculation error"
    assert abs(long_sl - expected_long_sl) < Decimal("1"), f"Long SL calculation error"
    assert abs(short_tp - expected_short_tp) < Decimal("1"), f"Short TP calculation error"
    assert abs(short_sl - expected_short_sl) < Decimal("1"), f"Short SL calculation error"
    
    print("✅ TP/SL calculation test passed!\n")


def test_bot_initialization():
    """Test bot initialization with various parameters."""
    print("Testing bot initialization...")
    
    # Test with minimal parameters
    bot1 = BingxHedgeBot(
        ticker="BTC",
        order_quantity=Decimal("0.001")
    )
    assert bot1.ticker == "BTC"
    assert bot1.order_quantity == Decimal("0.001")
    assert bot1.tp_roi is None
    assert bot1.sl_roi is None
    print("✅ Minimal initialization test passed!")
    
    # Test with full parameters
    bot2 = BingxHedgeBot(
        ticker="ETH",
        order_quantity=Decimal("0.1"),
        tp_roi=Decimal("5"),
        sl_roi=Decimal("3"),
        iterations=3,
        sleep_time=60
    )
    assert bot2.ticker == "ETH"
    assert bot2.order_quantity == Decimal("0.1")
    assert bot2.tp_roi == Decimal("5")
    assert bot2.sl_roi == Decimal("3")
    assert bot2.iterations == 3
    assert bot2.sleep_time == 60
    print("✅ Full initialization test passed!\n")


async def test_config_loading():
    """Test configuration loading (without actual connection)."""
    print("Testing configuration loading...")
    
    # Check if environment variables exist
    has_api_key = os.getenv('BINGX_API_KEY') is not None
    has_api_secret = os.getenv('BINGX_API_SECRET') is not None
    
    if has_api_key and has_api_secret:
        print("✅ BingX API credentials found in environment")
    else:
        print("⚠️ BingX API credentials not found - actual trading will not work")
    
    environment = os.getenv('BINGX_ENVIRONMENT', 'prod')
    print(f"Environment: {environment}\n")


def main():
    """Main test function."""
    print("=" * 50)
    print("BingX Hedge Mode Test Suite")
    print("=" * 50)
    print()
    
    try:
        # Run synchronous tests
        test_bot_initialization()
        test_tp_sl_calculation()
        
        # Run async tests
        asyncio.run(test_config_loading())
        
        print("=" * 50)
        print("✅ All tests passed successfully!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()