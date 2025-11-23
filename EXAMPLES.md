# Examples and Code Snippets

Comprehensive collection of practical examples for using the Perp DEX Trading Bot.

---

## Table of Contents

1. [Basic Trading Examples](#basic-trading-examples)
2. [Advanced Trading Strategies](#advanced-trading-strategies)
3. [Hedge Mode Examples](#hedge-mode-examples)
4. [Programmatic Usage](#programmatic-usage)
5. [Custom Extensions](#custom-extensions)
6. [Notification Examples](#notification-examples)
7. [Error Handling Examples](#error-handling-examples)
8. [Multi-Bot Orchestration](#multi-bot-orchestration)

---

## Basic Trading Examples

### Example 1: Simple Long Strategy on EdgeX

**Command Line:**
```bash
python runbot.py \
  --exchange edgex \
  --ticker ETH \
  --quantity 0.1 \
  --take-profit 0.02 \
  --direction buy \
  --max-orders 40 \
  --wait-time 450
```

**What it does:**
- Opens long (buy) positions on ETH
- Each order is 0.1 ETH
- Takes profit at 0.02% above entry
- Maintains up to 40 concurrent positions
- Waits ~450 seconds between new orders

**Expected behavior:**
```
[EDGEX_ETH] === Trading Configuration ===
[EDGEX_ETH] Ticker: ETH
[EDGEX_ETH] Direction: buy
[EDGEX_ETH] Quantity: 0.1
[EDGEX_ETH] Take Profit: 0.02%
--------------------------------
[EDGEX_ETH] Current Position: 0.3 | Active closing amount: 0.3 | Order quantity: 3
--------------------------------
[EDGEX_ETH] [OPEN] [order_123] FILLED 0.1 @ 2000.50
[EDGEX_ETH] [CLOSE] [order_124] OPEN 0.1 @ 2000.90
```

---

### Example 2: Short Strategy with Grid Control

**Command Line:**
```bash
python runbot.py \
  --exchange backpack \
  --ticker BTC \
  --quantity 0.05 \
  --take-profit 0.03 \
  --direction sell \
  --max-orders 30 \
  --wait-time 600 \
  --grid-step 0.5
```

**What it does:**
- Opens short (sell) positions on BTC
- Each order is 0.05 BTC
- Takes profit at 0.03% below entry
- Maintains up to 30 concurrent positions
- Ensures 0.5% spacing between close orders

**Grid step in action:**
```
# Current close orders at: 99500, 99000, 98500
# Market price: 100000

# New order would close at 99700
# Distance to nearest (99500): 0.2% < 0.5%
# Result: Skip this order, wait for better spacing

# New order would close at 99000 or below
# Distance to nearest: >= 0.5%
# Result: Place order
```

---

### Example 3: Conservative Trading with Stop Price

**Command Line:**
```bash
python runbot.py \
  --exchange edgex \
  --ticker ETH \
  --quantity 0.05 \
  --take-profit 0.02 \
  --direction buy \
  --max-orders 20 \
  --wait-time 600 \
  --stop-price 4000
```

**What it does:**
- Small position size (0.05 ETH)
- Limited exposure (20 max orders)
- Stops trading if ETH reaches $4000
- Long wait time (600s) for patience

**Stop price behavior:**
```python
# Market price monitoring
while trading:
    best_bid, best_ask = await get_prices()
    
    # For buy direction
    if direction == 'buy' and best_ask >= stop_price:
        await graceful_shutdown("Stop price reached")
        send_notification("Trading stopped: ETH reached $4000")
        break
```

---

### Example 4: Boost Mode for High Volume

**Command Line:**
```bash
python runbot.py \
  --exchange backpack \
  --ticker ETH \
  --direction buy \
  --quantity 0.2 \
  --boost
```

**What it does:**
- Places maker order to open position
- Immediately closes with taker/market order
- Maximizes trading volume
- Higher fees but faster execution

**Trade flow:**
```
1. Place limit buy @ 2000.50 (maker)
   Status: FILLED
   Fee: 0.02% maker fee

2. Immediately place market sell @ 2000.45 (taker)
   Status: FILLED
   Fee: 0.05% taker fee

3. Total cost: ~0.07% + slippage
4. Repeat immediately
```

**Use case:** Trading competitions, volume requirements

---

## Advanced Trading Strategies

### Example 5: Range-Bound Trading

**Scenario:** Trade ETH only between $2500-$3500

**Command Line:**
```bash
python runbot.py \
  --exchange edgex \
  --ticker ETH \
  --quantity 0.1 \
  --take-profit 0.02 \
  --direction buy \
  --max-orders 40 \
  --wait-time 450 \
  --pause-price 3500 \
  --stop-price 3800
```

**What happens:**
- Normal trading below $3500
- Pauses at $3500 (resumes if drops back)
- Exits completely at $3800

**Programmatic approach:**
```python
import asyncio
from decimal import Decimal
from trading_bot import TradingBot, TradingConfig

async def range_bound_strategy():
    config = TradingConfig(
        ticker='ETH',
        contract_id='',
        quantity=Decimal('0.1'),
        take_profit=Decimal('0.02'),
        tick_size=Decimal('0.01'),
        direction='buy',
        max_orders=40,
        wait_time=450,
        exchange='edgex',
        grid_step=Decimal('0.5'),
        stop_price=Decimal('3800'),
        pause_price=Decimal('3500'),
        boost_mode=False
    )
    
    bot = TradingBot(config)
    await bot.run()

asyncio.run(range_bound_strategy())
```

---

### Example 6: Adaptive Wait Time Strategy

**Goal:** Adjust wait time based on market volatility

**Implementation:**
```python
from trading_bot import TradingBot, TradingConfig
from decimal import Decimal
import asyncio

class AdaptiveBot(TradingBot):
    """Bot with adaptive wait time based on volatility."""
    
    async def _calculate_volatility(self) -> Decimal:
        """Calculate recent price volatility."""
        # Fetch recent price data
        prices = await self.fetch_recent_prices(lookback=100)
        
        # Calculate standard deviation
        mean = sum(prices) / len(prices)
        variance = sum((p - mean) ** 2 for p in prices) / len(prices)
        volatility = variance ** 0.5
        
        return Decimal(str(volatility))
    
    def _calculate_wait_time(self) -> Decimal:
        """Override with volatility-based wait time."""
        base_wait = super()._calculate_wait_time()
        
        if base_wait > 0:
            return base_wait
        
        # Get volatility
        volatility = asyncio.create_task(self._calculate_volatility())
        vol_value = volatility.result()
        
        # High volatility = longer wait
        if vol_value > Decimal('50'):
            return Decimal('900')  # 15 minutes
        elif vol_value > Decimal('25'):
            return Decimal('600')  # 10 minutes
        else:
            return Decimal('450')  # 7.5 minutes

# Usage
config = TradingConfig(...)
adaptive_bot = AdaptiveBot(config)
await adaptive_bot.run()
```

---

### Example 7: Multi-Timeframe Strategy

**Goal:** Trade on multiple timeframes simultaneously

**Implementation:**
```python
import asyncio
from decimal import Decimal
from trading_bot import TradingBot, TradingConfig

async def multi_timeframe_strategy():
    """Run scalping and swing bots simultaneously."""
    
    # Scalping bot: Fast, small profits
    scalp_config = TradingConfig(
        ticker='ETH',
        contract_id='',
        quantity=Decimal('0.05'),
        take_profit=Decimal('0.01'),  # 0.01% profit
        tick_size=Decimal('0.01'),
        direction='buy',
        max_orders=20,
        wait_time=180,  # 3 minutes
        exchange='edgex',
        grid_step=Decimal('0.2'),
        stop_price=Decimal('-1'),
        pause_price=Decimal('-1'),
        boost_mode=False
    )
    
    # Swing bot: Slow, larger profits
    swing_config = TradingConfig(
        ticker='ETH',
        contract_id='',
        quantity=Decimal('0.2'),
        take_profit=Decimal('0.1'),  # 0.1% profit
        tick_size=Decimal('0.01'),
        direction='buy',
        max_orders=10,
        wait_time=1800,  # 30 minutes
        exchange='edgex',
        grid_step=Decimal('1.0'),
        stop_price=Decimal('-1'),
        pause_price=Decimal('-1'),
        boost_mode=False
    )
    
    scalp_bot = TradingBot(scalp_config)
    swing_bot = TradingBot(swing_config)
    
    # Run both concurrently
    try:
        await asyncio.gather(
            scalp_bot.run(),
            swing_bot.run()
        )
    except KeyboardInterrupt:
        await scalp_bot.graceful_shutdown("User stop")
        await swing_bot.graceful_shutdown("User stop")

asyncio.run(multi_timeframe_strategy())
```

---

## Hedge Mode Examples

### Example 8: Basic Hedge Trading

**Command Line:**
```bash
python hedge_mode.py \
  --exchange backpack \
  --ticker BTC \
  --size 0.05 \
  --iter 20 \
  --fill-timeout 5
```

**What it does:**
1. Places maker buy on Backpack
2. Hedges with market sell on Lighter
3. Waits briefly
4. Places maker sell on Backpack (close)
5. Closes hedge with market buy on Lighter
6. Repeats 20 times

**Expected output:**
```
Starting hedge mode for backpack exchange...
Ticker: BTC, Size: 0.05, Iterations: 20
--------------------------------------------------
=== Iteration 1/20 ===
[Backpack] Placing maker BUY order for 0.05 BTC
[Backpack] Order filled at 100000.00
[Lighter] Placing hedge SELL order for 0.05 BTC
[Lighter] Hedge filled at 99995.00
[Backpack] Current position: +0.05 BTC
[Lighter] Current position: -0.05 BTC
[Backpack] Placing maker SELL order for 0.05 BTC
[Backpack] Order filled at 100020.00
[Lighter] Closing hedge with BUY order for 0.05 BTC
[Lighter] Hedge closed at 100005.00
=== Iteration 1/20 Complete ===
Profit: $5.00
```

---

### Example 9: Hedge Mode with ROI Targets

**Command Line:**
```bash
python hedge_mode.py \
  --exchange apex \
  --ticker BTC \
  --size 0.05 \
  --iter 10 \
  --tp-roi 0.5 \
  --sl-roi 0.3 \
  --sleep 10
```

**What it does:**
- Opens hedged position
- Monitors PnL continuously
- Closes at +0.5% profit OR -0.3% loss
- Waits 10 seconds between steps

**ROI monitoring logic:**
```python
# After opening hedge
entry_price_primary = 100000.00
entry_price_hedge = 99995.00
average_entry = (entry_price_primary + entry_price_hedge) / 2  # 99997.50

tp_target = average_entry * (1 + 0.005)  # 100497.50
sl_target = average_entry * (1 - 0.003)  # 99697.50

# Monitor until target hit
while True:
    current_price = get_current_price()
    
    if current_price >= tp_target:
        logger.log("Take profit target reached!")
        break
    elif current_price <= sl_target:
        logger.log("Stop loss triggered!")
        break
    
    await asyncio.sleep(1)

# Close positions
```

---

### Example 10: Position Cleanup (GRVT + BingX)

**Scenario:** You have existing hedge positions to close

**Command Line:**
```bash
python hedge_mode.py \
  --exchange grvt_bingx \
  --ticker BTC \
  --size 0.05 \
  --iter 1 \
  --position-close
```

**What it does:**
```python
# 1. Check current positions
grvt_position = await grvt_client.get_position()  # +0.15 BTC (long)
bingx_position = await bingx_client.get_position()  # -0.15 BTC (short)

# 2. Place opposite limit orders
# On GRVT: Sell 0.15 BTC at best_bid + tick
await grvt_client.place_limit_order(
    side='sell',
    size=0.15,
    price=best_bid + tick_size
)

# On BingX: Buy 0.15 BTC at best_ask - tick
await bingx_client.place_limit_order(
    side='buy',
    size=0.15,
    price=best_ask - tick_size
)

# 3. Wait for fills
# 4. Verify positions closed
```

---

## Programmatic Usage

### Example 11: Creating a Custom Bot

**File: `custom_bot.py`**

```python
import asyncio
import os
from decimal import Decimal
from trading_bot import TradingBot, TradingConfig
from helpers import TradingLogger

class CustomBot:
    """Custom trading bot with additional features."""
    
    def __init__(self):
        self.logger = TradingLogger('custom', 'ETH', log_to_console=True)
        self.bots = []
    
    async def initialize_bots(self):
        """Create multiple bot instances."""
        
        # Conservative bot
        conservative_config = TradingConfig(
            ticker='ETH',
            contract_id='',
            quantity=Decimal('0.05'),
            take_profit=Decimal('0.02'),
            tick_size=Decimal('0.01'),
            direction='buy',
            max_orders=20,
            wait_time=600,
            exchange='edgex',
            grid_step=Decimal('0.5'),
            stop_price=Decimal('-1'),
            pause_price=Decimal('-1'),
            boost_mode=False
        )
        conservative_bot = TradingBot(conservative_config)
        self.bots.append(('conservative', conservative_bot))
        
        # Aggressive bot
        aggressive_config = TradingConfig(
            ticker='ETH',
            contract_id='',
            quantity=Decimal('0.15'),
            take_profit=Decimal('0.01'),
            tick_size=Decimal('0.01'),
            direction='buy',
            max_orders=50,
            wait_time=300,
            exchange='edgex',
            grid_step=Decimal('0.3'),
            stop_price=Decimal('-1'),
            pause_price=Decimal('-1'),
            boost_mode=False
        )
        aggressive_bot = TradingBot(aggressive_config)
        self.bots.append(('aggressive', aggressive_bot))
    
    async def run(self):
        """Run all bots concurrently."""
        await self.initialize_bots()
        
        tasks = []
        for name, bot in self.bots:
            self.logger.log(f"Starting {name} bot", "INFO")
            tasks.append(bot.run())
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            self.logger.log("Stopping all bots", "INFO")
            for name, bot in self.bots:
                await bot.graceful_shutdown("User interrupt")

# Usage
if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()
    
    custom_bot = CustomBot()
    asyncio.run(custom_bot.run())
```

**Run:**
```bash
python custom_bot.py
```

---

### Example 12: Bot with Market Condition Detection

**Implementation:**

```python
from trading_bot import TradingBot, TradingConfig
from decimal import Decimal
import asyncio
from typing import Literal

class SmartBot(TradingBot):
    """Bot that adapts to market conditions."""
    
    async def detect_market_condition(self) -> Literal['trending', 'ranging', 'volatile']:
        """Detect current market condition."""
        # Fetch recent candles
        candles = await self.exchange_client.fetch_recent_candles(limit=50)
        
        # Calculate indicators
        highs = [c['high'] for c in candles]
        lows = [c['low'] for c in candles]
        closes = [c['close'] for c in candles]
        
        # Calculate ATR (Average True Range)
        atr = self._calculate_atr(highs, lows, closes)
        
        # Calculate trend strength
        sma_20 = sum(closes[-20:]) / 20
        sma_50 = sum(closes[-50:]) / 50
        
        # Detect condition
        if abs(sma_20 - sma_50) > atr * 2:
            return 'trending'
        elif atr < sum(closes) / len(closes) * 0.01:
            return 'ranging'
        else:
            return 'volatile'
    
    async def _place_and_monitor_open_order(self) -> bool:
        """Override to adapt strategy based on conditions."""
        condition = await self.detect_market_condition()
        
        # Adjust take profit based on condition
        original_tp = self.config.take_profit
        
        if condition == 'volatile':
            # Larger TP in volatile markets
            self.config.take_profit = original_tp * Decimal('1.5')
            self.logger.log("Volatile market: Increased take profit", "INFO")
        elif condition == 'ranging':
            # Tighter TP in ranging markets
            self.config.take_profit = original_tp * Decimal('0.7')
            self.logger.log("Ranging market: Decreased take profit", "INFO")
        
        # Place order with adapted strategy
        result = await super()._place_and_monitor_open_order()
        
        # Restore original TP
        self.config.take_profit = original_tp
        
        return result

# Usage
config = TradingConfig(...)
smart_bot = SmartBot(config)
await smart_bot.run()
```

---

## Custom Extensions

### Example 13: Custom Exchange Client

**File: `exchanges/custom_exchange.py`**

```python
from exchanges.base import BaseExchangeClient, OrderResult, OrderInfo
from decimal import Decimal
from typing import List, Optional, Tuple
import aiohttp

class CustomExchangeClient(BaseExchangeClient):
    """Custom exchange implementation."""
    
    def __init__(self, config):
        super().__init__(config)
        self.session = None
        self.api_url = "https://api.customexchange.com"
    
    def _validate_config(self):
        """Validate required configuration."""
        required_keys = ['ticker', 'api_key', 'api_secret']
        for key in required_keys:
            if not hasattr(self.config, key):
                raise ValueError(f"Missing required config: {key}")
    
    async def connect(self):
        """Establish connection."""
        self.session = aiohttp.ClientSession()
        # Authenticate
        await self._authenticate()
    
    async def disconnect(self):
        """Close connection."""
        if self.session:
            await self.session.close()
    
    async def place_open_order(
        self,
        contract_id: str,
        quantity: Decimal,
        direction: str
    ) -> OrderResult:
        """Place an open order."""
        try:
            # Get current price
            price = await self.get_order_price(direction)
            
            # Place order via API
            async with self.session.post(
                f"{self.api_url}/orders",
                json={
                    'symbol': contract_id,
                    'side': direction,
                    'type': 'limit',
                    'quantity': str(quantity),
                    'price': str(price)
                },
                headers=self._get_auth_headers()
            ) as response:
                data = await response.json()
                
                if response.status == 200:
                    return OrderResult(
                        success=True,
                        order_id=data['orderId'],
                        side=direction,
                        size=quantity,
                        price=price,
                        status='OPEN'
                    )
                else:
                    return OrderResult(
                        success=False,
                        error_message=data.get('message', 'Unknown error')
                    )
        
        except Exception as e:
            return OrderResult(
                success=False,
                error_message=str(e)
            )
    
    # Implement other required methods...
    
    async def get_contract_attributes(self) -> Tuple[str, Decimal]:
        """Get contract ID and tick size."""
        async with self.session.get(
            f"{self.api_url}/instruments/{self.config.ticker}",
            headers=self._get_auth_headers()
        ) as response:
            data = await response.json()
            return data['symbol'], Decimal(str(data['tickSize']))
    
    def get_exchange_name(self) -> str:
        return "custom"

# Register the exchange
from exchanges import ExchangeFactory
ExchangeFactory.register_exchange('custom', CustomExchangeClient)
```

**Usage:**
```python
from exchanges import ExchangeFactory
from decimal import Decimal

config = {
    'ticker': 'ETH',
    'api_key': 'your_key',
    'api_secret': 'your_secret',
    'quantity': Decimal('0.1'),
    'direction': 'buy'
}

exchange = ExchangeFactory.create_exchange('custom', config)
await exchange.connect()
```

---

### Example 14: Custom Notification Handler

**File: `helpers/custom_notifier.py`**

```python
import aiohttp
from typing import Optional

class DiscordNotifier:
    """Send notifications to Discord webhook."""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()
    
    async def send_message(self, message: str, username: str = "Trading Bot"):
        """Send message to Discord."""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        payload = {
            'username': username,
            'content': message
        }
        
        async with self.session.post(self.webhook_url, json=payload) as response:
            return response.status == 204
    
    async def send_embed(
        self,
        title: str,
        description: str,
        color: int = 0x00ff00,
        fields: Optional[list] = None
    ):
        """Send rich embed to Discord."""
        embed = {
            'title': title,
            'description': description,
            'color': color,
            'fields': fields or []
        }
        
        payload = {'embeds': [embed]}
        
        async with self.session.post(self.webhook_url, json=payload) as response:
            return response.status == 204

# Usage in bot
async def send_trade_notification(order_info):
    webhook = "https://discord.com/api/webhooks/..."
    
    async with DiscordNotifier(webhook) as notifier:
        await notifier.send_embed(
            title="Trade Executed",
            description=f"Order {order_info.order_id} filled",
            color=0x00ff00,
            fields=[
                {'name': 'Side', 'value': order_info.side, 'inline': True},
                {'name': 'Size', 'value': str(order_info.size), 'inline': True},
                {'name': 'Price', 'value': str(order_info.price), 'inline': True}
            ]
        )
```

---

## Notification Examples

### Example 15: Comprehensive Telegram Alerts

**Implementation:**

```python
import os
from helpers.telegram_bot import TelegramBot
from helpers import TradingLogger

class TradingNotifier:
    """Centralized notification system."""
    
    def __init__(self, exchange: str, ticker: str):
        self.exchange = exchange
        self.ticker = ticker
        self.logger = TradingLogger(exchange, ticker)
        
        # Setup Telegram
        token = os.getenv('TELEGRAM_BOT_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.telegram = TelegramBot(token, chat_id) if token and chat_id else None
    
    def notify_startup(self, config):
        """Send startup notification."""
        message = f"""
<b>🚀 Bot Started</b>

<b>Exchange:</b> {self.exchange.upper()}
<b>Ticker:</b> {self.ticker}
<b>Direction:</b> {config.direction}
<b>Quantity:</b> {config.quantity}
<b>Take Profit:</b> {config.take_profit}%
<b>Max Orders:</b> {config.max_orders}
        """
        self._send_telegram(message)
        self.logger.log("Bot started", "INFO")
    
    def notify_order_filled(self, order_info):
        """Send order fill notification."""
        emoji = "🟢" if order_info.side == 'buy' else "🔴"
        message = f"""
{emoji} <b>Order Filled</b>

<b>Side:</b> {order_info.side.upper()}
<b>Size:</b> {order_info.size}
<b>Price:</b> ${order_info.price}
<b>Order ID:</b> {order_info.order_id}
        """
        self._send_telegram(message)
    
    def notify_position_mismatch(self, position, active_close):
        """Send position mismatch alert."""
        message = f"""
⚠️ <b>Position Mismatch Alert</b>

<b>Current Position:</b> {position}
<b>Active Close Amount:</b> {active_close}
<b>Difference:</b> {abs(position - active_close)}

<b>Action Required:</b> Manual reconciliation needed
        """
        self._send_telegram(message)
        self.logger.log("Position mismatch detected", "ERROR")
    
    def notify_stop_price(self, current_price, stop_price):
        """Send stop price notification."""
        message = f"""
🛑 <b>Stop Price Triggered</b>

<b>Current Price:</b> ${current_price}
<b>Stop Price:</b> ${stop_price}

Bot has stopped trading.
        """
        self._send_telegram(message)
    
    def notify_daily_summary(self, trades_count, total_volume, pnl):
        """Send daily summary."""
        emoji = "📈" if pnl >= 0 else "📉"
        message = f"""
{emoji} <b>Daily Summary</b>

<b>Trades:</b> {trades_count}
<b>Volume:</b> {total_volume} {self.ticker}
<b>PnL:</b> ${pnl:.2f}
<b>Exchange:</b> {self.exchange.upper()}
        """
        self._send_telegram(message)
    
    def _send_telegram(self, message: str):
        """Send message via Telegram."""
        if self.telegram:
            with self.telegram as bot:
                bot.send_text(message, parse_mode="HTML")

# Usage in bot
notifier = TradingNotifier('edgex', 'ETH')
notifier.notify_startup(config)

# In order handler
if order.status == 'FILLED':
    notifier.notify_order_filled(order)

# In position check
if mismatch_detected:
    notifier.notify_position_mismatch(position, active_close_amount)
```

---

### Example 16: Multi-Channel Notifications

**Send to both Telegram and Lark:**

```python
import os
import asyncio
from helpers.telegram_bot import TelegramBot
from helpers.lark_bot import LarkBot

class MultiChannelNotifier:
    """Send notifications to multiple channels."""
    
    def __init__(self):
        # Telegram
        self.tg_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.tg_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        # Lark
        self.lark_token = os.getenv('LARK_TOKEN')
    
    async def send_notification(self, message: str):
        """Send to all configured channels."""
        tasks = []
        
        # Telegram
        if self.tg_token and self.tg_chat_id:
            tasks.append(self._send_telegram(message))
        
        # Lark
        if self.lark_token:
            tasks.append(self._send_lark(message))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _send_telegram(self, message: str):
        """Send via Telegram."""
        def sync_send():
            with TelegramBot(self.tg_token, self.tg_chat_id) as bot:
                return bot.send_text(message)
        
        # Run in executor since Telegram bot is sync
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, sync_send)
    
    async def _send_lark(self, message: str):
        """Send via Lark."""
        async with LarkBot(self.lark_token) as bot:
            return await bot.send_text(message)

# Usage
notifier = MultiChannelNotifier()

# In bot
await notifier.send_notification("Critical alert: Position mismatch!")
```

---

## Error Handling Examples

### Example 17: Retry with Exponential Backoff

**Implementation:**

```python
from exchanges.base import query_retry
import asyncio
from decimal import Decimal

class RobustExchangeClient:
    """Exchange client with comprehensive error handling."""
    
    @query_retry(
        default_return=None,
        exception_type=(ConnectionError, TimeoutError, aiohttp.ClientError),
        max_attempts=5,
        min_wait=2,
        max_wait=30,
        reraise=False
    )
    async def place_order_with_retry(
        self,
        contract_id: str,
        quantity: Decimal,
        price: Decimal,
        side: str
    ):
        """Place order with automatic retry."""
        try:
            # Attempt to place order
            result = await self.exchange_api.place_order(
                contract_id=contract_id,
                quantity=quantity,
                price=price,
                side=side
            )
            
            if not result['success']:
                # Treat API errors as retryable
                raise ConnectionError(f"API error: {result['message']}")
            
            return result
        
        except aiohttp.ClientError as e:
            self.logger.log(f"Network error placing order: {e}", "WARNING")
            raise
        
        except Exception as e:
            self.logger.log(f"Unexpected error: {e}", "ERROR")
            raise

# Usage
client = RobustExchangeClient()
result = await client.place_order_with_retry(
    contract_id="ETH-PERP",
    quantity=Decimal('0.1'),
    price=Decimal('2000.00'),
    side='buy'
)

if result is None:
    # All retries failed
    await bot.send_notification("Critical: Unable to place orders")
    await bot.graceful_shutdown("Order placement failed")
```

---

### Example 18: Graceful Error Recovery

**Implementation:**

```python
from trading_bot import TradingBot
import asyncio
import traceback

class ResilientBot(TradingBot):
    """Bot with enhanced error recovery."""
    
    def __init__(self, config):
        super().__init__(config)
        self.error_count = 0
        self.max_consecutive_errors = 5
    
    async def run(self):
        """Run with error recovery."""
        while not self.shutdown_requested:
            try:
                await super().run()
                # Reset error count on successful run
                self.error_count = 0
            
            except ConnectionError as e:
                self.error_count += 1
                self.logger.log(f"Connection error #{self.error_count}: {e}", "ERROR")
                
                if self.error_count >= self.max_consecutive_errors:
                    await self.send_notification(
                        f"Critical: {self.error_count} consecutive connection errors"
                    )
                    await self.graceful_shutdown("Too many connection errors")
                    break
                
                # Wait before retry
                wait_time = min(60 * (2 ** self.error_count), 600)  # Exponential backoff, max 10 min
                self.logger.log(f"Retrying in {wait_time} seconds...", "INFO")
                await asyncio.sleep(wait_time)
                
                # Reconnect
                try:
                    await self.exchange_client.disconnect()
                    await asyncio.sleep(5)
                    await self.exchange_client.connect()
                    self.logger.log("Reconnected successfully", "INFO")
                except Exception as reconnect_error:
                    self.logger.log(f"Reconnection failed: {reconnect_error}", "ERROR")
            
            except Exception as e:
                self.logger.log(f"Unexpected error: {e}", "ERROR")
                self.logger.log(f"Traceback: {traceback.format_exc()}", "ERROR")
                
                await self.send_notification(f"Critical error: {e}")
                await self.graceful_shutdown("Unexpected error")
                break

# Usage
config = TradingConfig(...)
resilient_bot = ResilientBot(config)
await resilient_bot.run()
```

---

## Multi-Bot Orchestration

### Example 19: Portfolio Bot Manager

**File: `portfolio_manager.py`**

```python
import asyncio
from decimal import Decimal
from trading_bot import TradingBot, TradingConfig
from helpers import TradingLogger
from typing import Dict, List

class PortfolioManager:
    """Manage multiple bots across different pairs and exchanges."""
    
    def __init__(self):
        self.bots: Dict[str, TradingBot] = {}
        self.logger = TradingLogger('portfolio', 'MULTI', log_to_console=True)
    
    def add_bot(self, name: str, config: TradingConfig):
        """Add a bot to the portfolio."""
        bot = TradingBot(config)
        self.bots[name] = bot
        self.logger.log(f"Added bot: {name}", "INFO")
    
    async def run_all(self):
        """Run all bots concurrently."""
        self.logger.log(f"Starting {len(self.bots)} bots...", "INFO")
        
        tasks = []
        for name, bot in self.bots.items():
            task = asyncio.create_task(
                self._run_bot_with_monitoring(name, bot)
            )
            tasks.append(task)
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            self.logger.log("Stopping all bots...", "INFO")
            await self.stop_all()
    
    async def _run_bot_with_monitoring(self, name: str, bot: TradingBot):
        """Run bot with monitoring."""
        try:
            self.logger.log(f"[{name}] Starting...", "INFO")
            await bot.run()
        except Exception as e:
            self.logger.log(f"[{name}] Error: {e}", "ERROR")
            await bot.send_notification(f"Bot {name} encountered error: {e}")
    
    async def stop_all(self):
        """Stop all bots gracefully."""
        for name, bot in self.bots.items():
            self.logger.log(f"[{name}] Stopping...", "INFO")
            await bot.graceful_shutdown("Portfolio manager stop")
    
    async def get_portfolio_status(self) -> Dict:
        """Get status of all bots."""
        status = {}
        
        for name, bot in self.bots.items():
            try:
                position = await bot.exchange_client.get_account_positions()
                active_orders = await bot.exchange_client.get_active_orders(
                    bot.config.contract_id
                )
                
                status[name] = {
                    'position': float(position),
                    'active_orders': len(active_orders),
                    'exchange': bot.config.exchange,
                    'ticker': bot.config.ticker
                }
            except Exception as e:
                status[name] = {'error': str(e)}
        
        return status

# Usage
async def main():
    manager = PortfolioManager()
    
    # Add ETH bot on EdgeX
    eth_config = TradingConfig(
        ticker='ETH',
        contract_id='',
        quantity=Decimal('0.1'),
        take_profit=Decimal('0.02'),
        tick_size=Decimal('0.01'),
        direction='buy',
        max_orders=40,
        wait_time=450,
        exchange='edgex',
        grid_step=Decimal('0.5'),
        stop_price=Decimal('-1'),
        pause_price=Decimal('-1'),
        boost_mode=False
    )
    manager.add_bot('ETH_EdgeX', eth_config)
    
    # Add BTC bot on Backpack
    btc_config = TradingConfig(
        ticker='BTC',
        contract_id='',
        quantity=Decimal('0.05'),
        take_profit=Decimal('0.02'),
        tick_size=Decimal('0.01'),
        direction='buy',
        max_orders=30,
        wait_time=500,
        exchange='backpack',
        grid_step=Decimal('0.5'),
        stop_price=Decimal('-1'),
        pause_price=Decimal('-1'),
        boost_mode=False
    )
    manager.add_bot('BTC_Backpack', btc_config)
    
    # Run all
    await manager.run_all()

if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()
    asyncio.run(main())
```

**Run:**
```bash
python portfolio_manager.py
```

---

### Example 20: Scheduled Bot Operations

**File: `scheduled_bot.py`**

```python
import asyncio
from datetime import datetime, time
from trading_bot import TradingBot, TradingConfig

class ScheduledBot:
    """Bot that runs only during specific hours."""
    
    def __init__(self, config: TradingConfig, start_hour: int, end_hour: int):
        self.bot = TradingBot(config)
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.running = False
    
    async def run(self):
        """Run bot with schedule."""
        while True:
            now = datetime.now().time()
            current_hour = now.hour
            
            # Check if within trading hours
            if self.start_hour <= current_hour < self.end_hour:
                if not self.running:
                    print(f"Starting bot at {now}")
                    self.running = True
                    # Start bot in background
                    asyncio.create_task(self._run_bot())
            else:
                if self.running:
                    print(f"Stopping bot at {now}")
                    await self.bot.graceful_shutdown("Outside trading hours")
                    self.running = False
            
            # Check every minute
            await asyncio.sleep(60)
    
    async def _run_bot(self):
        """Run the actual bot."""
        try:
            await self.bot.run()
        except Exception as e:
            print(f"Bot error: {e}")
            self.running = False

# Usage: Trade only between 8 AM and 6 PM
config = TradingConfig(...)
scheduled_bot = ScheduledBot(config, start_hour=8, end_hour=18)
await scheduled_bot.run()
```

---

## Summary

This examples document provides:

✅ **20 comprehensive examples** covering:
- Basic trading strategies
- Advanced techniques
- Hedge mode operations
- Programmatic usage
- Custom extensions
- Notifications
- Error handling
- Multi-bot orchestration

✅ **Real-world scenarios** with:
- Complete code implementations
- Command-line examples
- Expected outputs
- Best practices

✅ **Copy-paste ready code** for:
- Quick starts
- Custom modifications
- Production deployment

For more information, see:
- [API_DOCUMENTATION.md](API_DOCUMENTATION.md) - Complete API reference
- [FUNCTION_REFERENCE.md](FUNCTION_REFERENCE.md) - Quick function lookup
- [QUICK_START.md](QUICK_START.md) - Getting started guide

---

**Happy Trading!** 🚀
