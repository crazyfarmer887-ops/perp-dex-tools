# Examples and Tutorials

## Table of Contents

1. [Getting Started Tutorial](#getting-started-tutorial)
2. [Basic Examples](#basic-examples)
3. [Advanced Examples](#advanced-examples)
4. [Strategy Examples](#strategy-examples)
5. [Integration Examples](#integration-examples)
6. [Troubleshooting Examples](#troubleshooting-examples)

---

## Getting Started Tutorial

### Tutorial 1: First Trading Bot

This tutorial walks you through setting up and running your first trading bot.

#### Step 1: Install Dependencies

```bash
# Clone repository
git clone <repository-url>
cd perp-dex-tools

# Create virtual environment
python3 -m venv env
source env/bin/activate  # Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Step 2: Configure Environment

Create `.env` file:

```bash
# Copy example
cp env_example.txt .env

# Edit with your credentials
nano .env
```

Add your EdgeX credentials:

```bash
# EdgeX Configuration
EDGEX_ACCOUNT_ID=your_account_id_here
EDGEX_STARK_PRIVATE_KEY=0x1234...your_key_here
EDGEX_BASE_URL=https://pro.edgex.exchange
EDGEX_WS_URL=wss://quote.edgex.exchange

# Optional: Telegram notifications
TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
TELEGRAM_CHAT_ID=123456789
```

#### Step 3: Test Connection

Create test script `test_connection.py`:

```python
import asyncio
import os
import dotenv
from exchanges import ExchangeFactory
from decimal import Decimal

dotenv.load_dotenv()

async def test_connection():
    """Test exchange connection."""
    
    class SimpleConfig:
        def __init__(self):
            self.ticker = "ETH"
            self.exchange = "edgex"
            self.quantity = Decimal("0.01")
    
    config = SimpleConfig()
    
    try:
        # Create client
        client = ExchangeFactory.create_exchange("edgex", config)
        print("✓ Client created")
        
        # Connect
        await client.connect()
        print("✓ Connected to exchange")
        
        # Get contract info
        contract_id, tick_size = await client.get_contract_attributes()
        print(f"✓ Contract: {contract_id}, Tick: {tick_size}")
        
        # Get position
        position = await client.get_account_positions()
        print(f"✓ Current position: {position}")
        
        # Disconnect
        await client.disconnect()
        print("✓ Disconnected")
        
        print("\n✅ Connection test successful!")
        
    except Exception as e:
        print(f"\n❌ Connection test failed: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(test_connection())
```

Run the test:

```bash
python test_connection.py
```

#### Step 4: Run First Bot

Start with conservative settings:

```bash
python runbot.py \
  --exchange edgex \
  --ticker ETH \
  --quantity 0.01 \
  --take-profit 0.05 \
  --max-orders 5 \
  --wait-time 600
```

#### Step 5: Monitor Operation

Open another terminal to watch logs:

```bash
# Watch activity log
tail -f logs/edgex_ETH_activity.log

# Watch transactions
tail -f logs/edgex_ETH_orders.csv
```

#### Step 6: Stop the Bot

Press `Ctrl+C` in the bot terminal. The bot will gracefully shutdown.

#### Step 7: Review Results

Check the logs:

```bash
# View recent activity
tail -n 50 logs/edgex_ETH_activity.log

# Count filled orders
grep FILLED logs/edgex_ETH_orders.csv | wc -l

# View all transactions
cat logs/edgex_ETH_orders.csv
```

---

## Basic Examples

### Example 1: Simple ETH Trading

Basic ETH trading with default parameters.

```bash
python runbot.py \
  --exchange edgex \
  --ticker ETH \
  --quantity 0.1 \
  --take-profit 0.02 \
  --max-orders 40 \
  --wait-time 450
```

**What it does:**
- Trades ETH perpetual on EdgeX
- Places 0.1 ETH orders
- Takes 0.02% profit
- Allows up to 40 concurrent orders
- Waits 450 seconds between orders (dynamically adjusted)

### Example 2: BTC Selling Strategy

Sell BTC with conservative settings.

```bash
python runbot.py \
  --exchange backpack \
  --ticker BTC \
  --direction sell \
  --quantity 0.05 \
  --take-profit 0.03 \
  --max-orders 30 \
  --wait-time 600
```

**What it does:**
- Shorts BTC on Backpack
- Places 0.05 BTC sell orders
- Takes 0.03% profit on close
- Maximum 30 concurrent short positions
- Waits 600 seconds between orders

### Example 3: Multiple Tickers

Run bots for multiple trading pairs.

**Terminal 1 - ETH:**
```bash
python runbot.py \
  --exchange edgex \
  --ticker ETH \
  --quantity 0.1 \
  --max-orders 40
```

**Terminal 2 - BTC:**
```bash
python runbot.py \
  --exchange edgex \
  --ticker BTC \
  --quantity 0.05 \
  --max-orders 30
```

**Terminal 3 - SOL:**
```bash
python runbot.py \
  --exchange edgex \
  --ticker SOL \
  --quantity 1.0 \
  --max-orders 25
```

### Example 4: Basic Hedge Mode

Simple hedge trading with Backpack and Lighter.

```bash
python hedge_mode.py \
  --exchange backpack \
  --ticker BTC \
  --size 0.002 \
  --iter 10 \
  --fill-timeout 5
```

**What it does:**
- Places maker orders on Backpack
- Hedges with market orders on Lighter
- Executes 10 complete cycles
- 5-second timeout for order fills

---

## Advanced Examples

### Example 5: Grid Trading Strategy

Use grid step to space out orders.

```bash
python runbot.py \
  --exchange edgex \
  --ticker ETH \
  --quantity 0.1 \
  --take-profit 0.02 \
  --max-orders 50 \
  --wait-time 300 \
  --grid-step 0.5
```

**Advanced configuration:**
- Grid step: 0.5% minimum distance between close orders
- High max orders: 50 for dense grid
- Short wait time: 300 seconds for faster fills
- **Result**: Orders spread evenly across price range

### Example 6: Range Trading with Stop/Pause

Trade within a specific price range.

```bash
python runbot.py \
  --exchange backpack \
  --ticker ETH \
  --direction buy \
  --quantity 0.15 \
  --take-profit 0.03 \
  --max-orders 35 \
  --wait-time 400 \
  --pause-price 3200 \
  --stop-price 3500
```

**Strategy:**
- Buy ETH below 3200
- Pause if price reaches 3200
- Resume if price drops back below 3200
- Stop completely if price hits 3500
- **Use case**: Range-bound market trading

### Example 7: High-Frequency Boost Mode

Maximum volume generation with boost mode.

```bash
python runbot.py \
  --exchange backpack \
  --ticker ETH \
  --direction buy \
  --quantity 0.2 \
  --max-orders 80 \
  --wait-time 120 \
  --boost
```

**Configuration:**
- Boost mode: Maker open + taker close
- High max orders: 80
- Very short wait: 120 seconds
- Larger size: 0.2 ETH
- **Goal**: Maximum trading volume

### Example 8: ROI-Based Hedge Trading

Hedge trading with profit targets.

```bash
python hedge_mode.py \
  --exchange apex \
  --ticker BTC \
  --size 0.05 \
  --iter 20 \
  --tp-roi 0.4 \
  --sl-roi 0.2 \
  --fill-timeout 10
```

**Advanced features:**
- Take profit: 0.4% ROI target
- Stop loss: 0.2% ROI limit
- Extended timeout: 10 seconds
- Waits for ROI targets before closing
- **Strategy**: Profit-optimized hedge trading

### Example 9: Multi-Exchange Arbitrage

Run bots on multiple exchanges simultaneously.

**Terminal 1 - EdgeX:**
```bash
python runbot.py \
  --exchange edgex \
  --ticker ETH \
  --quantity 0.1 \
  --max-orders 40
```

**Terminal 2 - Backpack:**
```bash
python runbot.py \
  --exchange backpack \
  --ticker ETH \
  --quantity 0.1 \
  --max-orders 40
```

**Terminal 3 - GRVT:**
```bash
python runbot.py \
  --exchange grvt \
  --ticker ETH \
  --quantity 0.1 \
  --max-orders 40
```

**Strategy:**
- Same ticker across exchanges
- Capture price differences
- Diversify exchange exposure

### Example 10: Dynamic Parameter Adjustment

Script to adjust parameters based on market conditions.

```python
# dynamic_bot.py
import asyncio
import os
from decimal import Decimal
from trading_bot import TradingBot, TradingConfig

async def get_market_volatility():
    """Calculate market volatility (simplified)."""
    # In practice, fetch real market data
    return Decimal("0.5")  # 0.5% volatility

async def run_dynamic_bot():
    """Run bot with dynamic parameters."""
    
    # Check volatility
    volatility = await get_market_volatility()
    
    # Adjust parameters based on volatility
    if volatility > Decimal("1.0"):
        # High volatility: Conservative
        take_profit = Decimal("0.05")
        grid_step = Decimal("1.0")
        max_orders = 20
        wait_time = 900
    elif volatility > Decimal("0.5"):
        # Medium volatility: Balanced
        take_profit = Decimal("0.03")
        grid_step = Decimal("0.5")
        max_orders = 40
        wait_time = 450
    else:
        # Low volatility: Aggressive
        take_profit = Decimal("0.02")
        grid_step = Decimal("0.3")
        max_orders = 60
        wait_time = 300
    
    # Create config
    config = TradingConfig(
        ticker="ETH",
        contract_id="",
        tick_size=Decimal(0),
        quantity=Decimal("0.1"),
        take_profit=take_profit,
        direction="buy",
        max_orders=max_orders,
        wait_time=wait_time,
        exchange="edgex",
        grid_step=grid_step,
        stop_price=Decimal("-1"),
        pause_price=Decimal("-1"),
        boost_mode=False
    )
    
    print(f"Starting bot with volatility-adjusted parameters:")
    print(f"  Volatility: {volatility}%")
    print(f"  Take Profit: {take_profit}%")
    print(f"  Grid Step: {grid_step}%")
    print(f"  Max Orders: {max_orders}")
    print(f"  Wait Time: {wait_time}s")
    
    # Run bot
    bot = TradingBot(config)
    await bot.run()

if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()
    asyncio.run(run_dynamic_bot())
```

Run it:
```bash
python dynamic_bot.py
```

---

## Strategy Examples

### Strategy 1: Trend Following

Follow the market trend with dynamic stops.

```bash
# Bullish trend
python runbot.py \
  --exchange edgex \
  --ticker BTC \
  --direction buy \
  --quantity 0.05 \
  --take-profit 0.02 \
  --max-orders 50 \
  --wait-time 300 \
  --stop-price 68000  # Exit if trend reverses

# Bearish trend
python runbot.py \
  --exchange edgex \
  --ticker BTC \
  --direction sell \
  --quantity 0.05 \
  --take-profit 0.02 \
  --max-orders 50 \
  --wait-time 300 \
  --stop-price 62000  # Exit if trend reverses
```

### Strategy 2: Mean Reversion

Trade around a central price with wide grid.

```bash
python runbot.py \
  --exchange backpack \
  --ticker ETH \
  --direction buy \
  --quantity 0.1 \
  --take-profit 0.05 \
  --max-orders 30 \
  --wait-time 600 \
  --grid-step 1.5 \
  --pause-price 3200
```

**Concept:**
- Buy when price is low
- Pause if price gets too high
- Wide grid for mean reversion
- Resume when price drops back

### Strategy 3: Scalping

High-frequency trading for small profits.

```bash
python runbot.py \
  --exchange backpack \
  --ticker ETH \
  --quantity 0.05 \
  --take-profit 0.01 \
  --max-orders 100 \
  --wait-time 60 \
  --grid-step 0.1 \
  --boost
```

**Configuration:**
- Very tight take profit: 0.01%
- Many orders: 100
- Short wait: 60 seconds
- Tight grid: 0.1%
- Boost mode for speed

### Strategy 4: Pyramid Trading

Gradually build position with increasing size.

```python
# pyramid_strategy.py
import asyncio
from decimal import Decimal
from trading_bot import TradingBot, TradingConfig

async def run_pyramid():
    """Run pyramid trading strategy."""
    
    base_quantity = Decimal("0.05")
    
    configs = [
        # Level 1: Small position
        TradingConfig(
            ticker="BTC",
            contract_id="",
            tick_size=Decimal(0),
            quantity=base_quantity,
            take_profit=Decimal("0.05"),
            direction="buy",
            max_orders=10,
            wait_time=600,
            exchange="edgex",
            grid_step=Decimal("2.0"),
            stop_price=Decimal("62000"),
            pause_price=Decimal("-1"),
            boost_mode=False
        ),
        # Level 2: Medium position
        TradingConfig(
            ticker="BTC",
            contract_id="",
            tick_size=Decimal(0),
            quantity=base_quantity * 2,
            take_profit=Decimal("0.03"),
            direction="buy",
            max_orders=20,
            wait_time=450,
            exchange="edgex",
            grid_step=Decimal("1.0"),
            stop_price=Decimal("62000"),
            pause_price=Decimal("-1"),
            boost_mode=False
        ),
        # Level 3: Large position
        TradingConfig(
            ticker="BTC",
            contract_id="",
            tick_size=Decimal(0),
            quantity=base_quantity * 4,
            take_profit=Decimal("0.02"),
            direction="buy",
            max_orders=30,
            wait_time=300,
            exchange="edgex",
            grid_step=Decimal("0.5"),
            stop_price=Decimal("62000"),
            pause_price=Decimal("-1"),
            boost_mode=False
        ),
    ]
    
    # Run all levels (in production, run in separate processes)
    tasks = [TradingBot(config).run() for config in configs]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()
    asyncio.run(run_pyramid())
```

### Strategy 5: Pairs Trading

Trade two correlated assets.

**Terminal 1 - Long ETH:**
```bash
python runbot.py \
  --exchange edgex \
  --ticker ETH \
  --direction buy \
  --quantity 0.15 \
  --max-orders 40
```

**Terminal 2 - Short BTC:**
```bash
python runbot.py \
  --exchange edgex \
  --ticker BTC \
  --direction sell \
  --quantity 0.05 \
  --max-orders 40
```

**Concept:**
- Long on one asset
- Short on correlated asset
- Profit from spread convergence

---

## Integration Examples

### Example 11: Custom Exchange Integration

Implement and use a custom exchange.

```python
# exchanges/myexchange.py
from decimal import Decimal
from typing import List, Optional, Tuple
from .base import BaseExchangeClient, OrderResult, OrderInfo

class MyExchangeClient(BaseExchangeClient):
    """Custom exchange implementation."""
    
    def _validate_config(self) -> None:
        """Validate configuration."""
        required = ['api_key', 'api_secret']
        for field in required:
            if not hasattr(self.config, field):
                raise ValueError(f"Missing: {field}")
    
    async def connect(self) -> None:
        """Connect to exchange."""
        print(f"Connecting to MyExchange...")
        # Implement WebSocket connection
        self.connected = True
    
    async def disconnect(self) -> None:
        """Disconnect from exchange."""
        print("Disconnecting from MyExchange...")
        self.connected = False
    
    async def get_contract_attributes(self) -> Tuple[str, Decimal]:
        """Get contract info."""
        contract_id = f"{self.config.ticker}-PERP"
        tick_size = Decimal("0.01")
        return contract_id, tick_size
    
    async def place_open_order(
        self, 
        contract_id: str, 
        quantity: Decimal, 
        direction: str
    ) -> OrderResult:
        """Place opening order."""
        # Implement order placement
        return OrderResult(
            success=True,
            order_id="custom_123",
            side=direction,
            size=quantity,
            price=Decimal("2000"),
            status="OPEN"
        )
    
    # Implement other required methods...
    
    def get_exchange_name(self) -> str:
        return "myexchange"

# Register exchange
from exchanges.factory import ExchangeFactory
ExchangeFactory.register_exchange('myexchange', MyExchangeClient)
```

Use it:

```bash
python runbot.py \
  --exchange myexchange \
  --ticker ETH \
  --quantity 0.1
```

### Example 12: Custom Notification System

Implement custom notifications.

```python
# custom_notifier.py
import asyncio
import aiohttp
from typing import Optional

class WebhookNotifier:
    """Send notifications to custom webhook."""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def send_alert(self, title: str, message: str, level: str = "info"):
        """Send alert to webhook."""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        payload = {
            "title": title,
            "message": message,
            "level": level,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            async with self.session.post(self.webhook_url, json=payload) as response:
                return await response.json()
        except Exception as e:
            print(f"Failed to send webhook: {e}")
            return None

# Usage with trading bot
from trading_bot import TradingBot, TradingConfig

class NotifyingBot(TradingBot):
    """Bot with custom webhook notifications."""
    
    def __init__(self, config: TradingConfig, webhook_url: str):
        super().__init__(config)
        self.webhook_url = webhook_url
    
    async def send_notification(self, message: str):
        """Override to add webhook notifications."""
        # Call parent
        await super().send_notification(message)
        
        # Send to webhook
        async with WebhookNotifier(self.webhook_url) as notifier:
            await notifier.send_alert(
                title=f"{self.config.exchange.upper()} Alert",
                message=message,
                level="warning"
            )

# Run it
async def main():
    config = TradingConfig(...)  # Your config
    bot = NotifyingBot(config, "https://your-webhook.com/api/notify")
    await bot.run()

asyncio.run(main())
```

### Example 13: Database Integration

Store trading data in database.

```python
# db_logger.py
import asyncio
import sqlite3
from decimal import Decimal
from datetime import datetime
from helpers import TradingLogger

class DatabaseLogger(TradingLogger):
    """Logger that stores data in SQLite."""
    
    def __init__(self, exchange: str, ticker: str, log_to_console: bool = False):
        super().__init__(exchange, ticker, log_to_console)
        
        # Create database
        self.db_path = f"logs/{exchange}_{ticker}_trades.db"
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                order_id TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                status TEXT NOT NULL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS activity_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                level TEXT NOT NULL,
                message TEXT NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def log_transaction(
        self, 
        order_id: str, 
        side: str, 
        quantity: Decimal, 
        price: Decimal, 
        status: str
    ):
        """Log transaction to database."""
        # Call parent to log to CSV
        super().log_transaction(order_id, side, quantity, price, status)
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO trades (timestamp, order_id, side, quantity, price, status)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(self.timezone).isoformat(),
            order_id,
            side,
            float(quantity),
            float(price),
            status
        ))
        
        conn.commit()
        conn.close()
    
    def log(self, message: str, level: str = "INFO"):
        """Log message to database."""
        # Call parent
        super().log(message, level)
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO activity_log (timestamp, level, message)
            VALUES (?, ?, ?)
        """, (
            datetime.now(self.timezone).isoformat(),
            level,
            message
        ))
        
        conn.commit()
        conn.close()
    
    def get_trade_stats(self):
        """Get trading statistics from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total_trades,
                SUM(quantity) as total_volume,
                AVG(price) as avg_price,
                COUNT(CASE WHEN status = 'FILLED' THEN 1 END) as filled_count
            FROM trades
        """)
        
        result = cursor.fetchone()
        conn.close()
        
        return {
            'total_trades': result[0],
            'total_volume': result[1],
            'avg_price': result[2],
            'filled_count': result[3]
        }

# Usage
logger = DatabaseLogger("edgex", "ETH", log_to_console=True)

# Log trades
logger.log_transaction("order1", "buy", Decimal("0.1"), Decimal("2000"), "FILLED")
logger.log("Trade executed", "INFO")

# Get stats
stats = logger.get_trade_stats()
print(f"Statistics: {stats}")
```

---

## Troubleshooting Examples

### Example 14: Debug Connection Issues

Script to diagnose connection problems.

```python
# debug_connection.py
import asyncio
import os
import dotenv
from exchanges import ExchangeFactory

dotenv.load_dotenv()

async def debug_exchange_connection(exchange: str, ticker: str):
    """Debug exchange connection issues."""
    
    print(f"\n=== Debugging {exchange.upper()} Connection ===\n")
    
    # Check environment variables
    print("1. Checking environment variables...")
    if exchange == "edgex":
        account_id = os.getenv("EDGEX_ACCOUNT_ID")
        private_key = os.getenv("EDGEX_STARK_PRIVATE_KEY")
        base_url = os.getenv("EDGEX_BASE_URL")
        ws_url = os.getenv("EDGEX_WS_URL")
        
        print(f"   EDGEX_ACCOUNT_ID: {'✓ Set' if account_id else '✗ Missing'}")
        print(f"   EDGEX_STARK_PRIVATE_KEY: {'✓ Set' if private_key else '✗ Missing'}")
        print(f"   EDGEX_BASE_URL: {base_url or '✗ Missing'}")
        print(f"   EDGEX_WS_URL: {ws_url or '✗ Missing'}")
        
        if not all([account_id, private_key]):
            print("\n❌ Missing required environment variables!")
            return
    
    # Create client
    print("\n2. Creating exchange client...")
    try:
        from decimal import Decimal
        
        class TestConfig:
            def __init__(self):
                self.ticker = ticker
                self.exchange = exchange
                self.quantity = Decimal("0.01")
        
        config = TestConfig()
        client = ExchangeFactory.create_exchange(exchange, config)
        print("   ✓ Client created successfully")
    except Exception as e:
        print(f"   ✗ Failed to create client: {e}")
        return
    
    # Test connection
    print("\n3. Testing connection...")
    try:
        await client.connect()
        print("   ✓ Connected successfully")
    except Exception as e:
        print(f"   ✗ Connection failed: {e}")
        import traceback
        print(f"\nTraceback:\n{traceback.format_exc()}")
        return
    
    # Get contract info
    print("\n4. Getting contract information...")
    try:
        contract_id, tick_size = await client.get_contract_attributes()
        print(f"   ✓ Contract ID: {contract_id}")
        print(f"   ✓ Tick Size: {tick_size}")
    except Exception as e:
        print(f"   ✗ Failed to get contract info: {e}")
    
    # Get position
    print("\n5. Getting account position...")
    try:
        position = await client.get_account_positions()
        print(f"   ✓ Current Position: {position}")
    except Exception as e:
        print(f"   ✗ Failed to get position: {e}")
    
    # Get active orders
    print("\n6. Getting active orders...")
    try:
        contract_id, _ = await client.get_contract_attributes()
        orders = await client.get_active_orders(contract_id)
        print(f"   ✓ Active Orders: {len(orders)}")
        for order in orders[:5]:  # Show first 5
            print(f"      - {order.side} {order.size} @ {order.price} ({order.status})")
    except Exception as e:
        print(f"   ✗ Failed to get orders: {e}")
    
    # Disconnect
    print("\n7. Disconnecting...")
    try:
        await client.disconnect()
        print("   ✓ Disconnected successfully")
    except Exception as e:
        print(f"   ✗ Disconnection error: {e}")
    
    print("\n✅ Debug complete!\n")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python debug_connection.py <exchange> <ticker>")
        print("Example: python debug_connection.py edgex ETH")
        sys.exit(1)
    
    exchange = sys.argv[1].lower()
    ticker = sys.argv[2].upper()
    
    asyncio.run(debug_exchange_connection(exchange, ticker))
```

Run it:
```bash
python debug_connection.py edgex ETH
```

### Example 15: Position Reconciliation

Script to check and fix position mismatches.

```python
# reconcile_position.py
import asyncio
import os
import dotenv
from decimal import Decimal
from exchanges import ExchangeFactory

dotenv.load_dotenv()

async def reconcile_position(exchange: str, ticker: str):
    """Check and report position status."""
    
    print(f"\n=== Position Reconciliation: {exchange.upper()} {ticker} ===\n")
    
    # Create client
    class Config:
        def __init__(self):
            self.ticker = ticker
            self.exchange = exchange
            self.quantity = Decimal("0.01")
    
    client = ExchangeFactory.create_exchange(exchange, Config())
    await client.connect()
    
    # Get contract info
    contract_id, _ = await client.get_contract_attributes()
    
    # Get current position
    print("1. Current Position:")
    position = await client.get_account_positions()
    print(f"   Total Position: {position}")
    
    # Get active orders
    print("\n2. Active Orders:")
    orders = await client.get_active_orders(contract_id)
    
    open_orders = [o for o in orders if o.side == "buy" or o.side == "sell"]
    close_orders = []
    
    # Categorize orders (simplified - adjust based on your strategy)
    for order in orders:
        if "close" in order.order_id.lower() or order.side != ("buy" if position > 0 else "sell"):
            close_orders.append(order)
    
    print(f"   Total Orders: {len(orders)}")
    print(f"   Open Orders: {len(open_orders)}")
    print(f"   Close Orders: {len(close_orders)}")
    
    # Calculate close order quantity
    close_qty = sum(o.size for o in close_orders)
    print(f"   Total Close Quantity: {close_qty}")
    
    # Check for mismatch
    print("\n3. Position Analysis:")
    difference = abs(position) - close_qty
    
    if abs(difference) < Decimal("0.001"):
        print("   ✓ Position matches close orders")
        print("   Status: HEALTHY")
    else:
        print(f"   ✗ Position mismatch detected!")
        print(f"   Position: {position}")
        print(f"   Close Orders: {close_qty}")
        print(f"   Difference: {difference}")
        print("   Status: NEEDS ATTENTION")
        
        print("\n4. Recommended Actions:")
        if difference > 0:
            print(f"   - Place additional close orders for {difference}")
            print(f"   - Or manually close {difference} position")
        else:
            print(f"   - Cancel {abs(difference)} in close orders")
            print(f"   - Or manually adjust position")
    
    # List all orders for review
    if len(orders) > 0:
        print("\n5. Order Details:")
        for i, order in enumerate(orders[:10], 1):
            print(f"   {i}. {order.order_id}")
            print(f"      Side: {order.side}, Size: {order.size}, Price: {order.price}")
            print(f"      Status: {order.status}")
    
    await client.disconnect()
    print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python reconcile_position.py <exchange> <ticker>")
        sys.exit(1)
    
    exchange = sys.argv[1].lower()
    ticker = sys.argv[2].upper()
    
    asyncio.run(reconcile_position(exchange, ticker))
```

Run it:
```bash
python reconcile_position.py edgex ETH
```

---

## Conclusion

These examples and tutorials cover:

✅ Getting started from scratch  
✅ Basic trading configurations  
✅ Advanced strategies  
✅ Custom integrations  
✅ Troubleshooting tools  

### Next Steps

1. **Start Simple**: Begin with Tutorial 1
2. **Experiment**: Try different parameters
3. **Monitor**: Watch logs and performance
4. **Optimize**: Adjust based on results
5. **Scale**: Gradually increase positions
6. **Customize**: Build your own strategies

### Additional Resources

- **API Documentation**: See `API_DOCUMENTATION.md`
- **Exchange Details**: See `EXCHANGE_IMPLEMENTATIONS.md`
- **Usage Guide**: See `USAGE_GUIDE.md`
- **Helper Utils**: See `HELPER_UTILITIES.md`

### Community and Support

- Review main README files
- Check exchange-specific guides
- Test in development environment first
- Start with small positions
- Monitor regularly

Happy trading! 🚀

---

**Document Version**: 1.0  
**Last Updated**: 2025-11-23  
**Status**: Production Ready
