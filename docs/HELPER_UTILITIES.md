# Helper Utilities Documentation

## Table of Contents

1. [Overview](#overview)
2. [TradingLogger](#tradinglogger)
3. [TelegramBot](#telegrambot)
4. [LarkBot](#larkbot)
5. [Utility Functions](#utility-functions)
6. [Best Practices](#best-practices)
7. [Examples](#examples)

---

## Overview

The helper utilities provide essential functionality for logging, notifications, and common operations. These utilities are designed to be reusable, well-tested, and easy to integrate.

### Available Utilities

| Utility | Purpose | Location |
|---------|---------|----------|
| TradingLogger | Structured logging and transaction tracking | `helpers/logger.py` |
| TelegramBot | Telegram notifications | `helpers/telegram_bot.py` |
| LarkBot | Lark/Feishu notifications | `helpers/lark_bot.py` |

---

## TradingLogger

Comprehensive logging system with file and console output, transaction tracking, and timezone support.

**Location**: `helpers/logger.py`

### Class Definition

```python
class TradingLogger:
    """Enhanced logging with structured output and error handling."""
    
    def __init__(
        self, 
        exchange: str, 
        ticker: str, 
        log_to_console: bool = False
    ):
        """
        Initialize trading logger.
        
        Args:
            exchange: Exchange name (e.g., "edgex", "backpack")
            ticker: Trading pair symbol (e.g., "ETH", "BTC")
            log_to_console: Whether to output logs to console
        """
```

### Features

- **Dual Output**: File and optional console logging
- **Timezone Support**: Configurable timezone for timestamps
- **Transaction Tracking**: Separate CSV file for trade records
- **Multi-Account Support**: Automatic account name in filenames
- **Log Rotation**: Manual log cleanup (future: automatic rotation)
- **Structured Format**: Consistent timestamp and level formatting

### Methods

#### `__init__(exchange, ticker, log_to_console=False)`

Initialize the logger.

**Parameters:**
- `exchange` (str): Exchange name
- `ticker` (str): Trading pair symbol
- `log_to_console` (bool): Enable console output (default: False)

**Example:**
```python
from helpers import TradingLogger

# File logging only
logger = TradingLogger("edgex", "ETH")

# File + console logging
logger = TradingLogger("edgex", "ETH", log_to_console=True)
```

#### `log(message, level="INFO")`

Log a message with specified level.

**Parameters:**
- `message` (str): Log message
- `level` (str): Log level - "DEBUG", "INFO", "WARNING", "ERROR"

**Example:**
```python
logger.log("Starting trading bot", "INFO")
logger.log("Failed to place order", "ERROR")
logger.log("WebSocket connected", "DEBUG")
logger.log("Rate limit approaching", "WARNING")
```

#### `log_transaction(order_id, side, quantity, price, status)`

Log a trade transaction to CSV file.

**Parameters:**
- `order_id` (str): Order identifier
- `side` (str): "buy" or "sell"
- `quantity` (Decimal): Trade quantity
- `price` (Decimal): Execution price
- `status` (str): Order status ("FILLED", "CANCELED", etc.)

**Example:**
```python
from decimal import Decimal

logger.log_transaction(
    order_id="abc123",
    side="buy",
    quantity=Decimal("0.1"),
    price=Decimal("2000.50"),
    status="FILLED"
)
```

### Output Files

#### Activity Log

**Filename Format:**
- Single account: `{exchange}_{ticker}_activity.log`
- Multi-account: `{exchange}_{ticker}_{account_name}_activity.log`

**Location:** `logs/` directory (auto-created)

**Format:**
```
2025-11-23 10:30:15.123 - INFO - [EDGEX_ETH] Order placed successfully
2025-11-23 10:30:16.456 - INFO - [EDGEX_ETH] [OPEN] [abc123] FILLED 0.1 @ 2000.5
2025-11-23 10:30:17.789 - ERROR - [EDGEX_ETH] Failed to connect to WebSocket
```

#### Transaction Log (CSV)

**Filename Format:**
- Single account: `{exchange}_{ticker}_orders.csv`
- Multi-account: `{exchange}_{ticker}_{account_name}_orders.csv`

**Location:** `logs/` directory

**Columns:**
```csv
Timestamp,OrderID,Side,Quantity,Price,Status
2025-11-23 10:30:16,abc123,buy,0.1,2000.5,FILLED
2025-11-23 10:31:22,def456,sell,0.1,2001.0,FILLED
```

### Configuration

#### Environment Variables

```bash
# Timezone for log timestamps (default: Asia/Shanghai)
TIMEZONE=Asia/Shanghai

# Account name for multi-account setups
ACCOUNT_NAME=main_account
```

#### Supported Timezones

Common timezone values:
- `Asia/Shanghai`
- `America/New_York`
- `Europe/London`
- `Asia/Tokyo`
- `America/Los_Angeles`
- `UTC`

### Implementation Details

#### Logger Setup

```python
def _setup_logger(self, log_to_console: bool) -> logging.Logger:
    """Setup the logger with proper configuration."""
    logger = logging.getLogger(f"trading_bot_{self.exchange}_{self.ticker}")
    logger.setLevel(logging.INFO)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Create timezone-aware formatter
    class TimeZoneFormatter(logging.Formatter):
        def __init__(self, fmt=None, datefmt=None, tz=None):
            super().__init__(fmt=fmt, datefmt=datefmt)
            self.tz = tz
        
        def formatTime(self, record, datefmt=None):
            dt = datetime.fromtimestamp(record.created, tz=self.tz)
            if datefmt:
                return dt.strftime(datefmt)
            return dt.isoformat()
    
    formatter = TimeZoneFormatter(
        "%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        tz=self.timezone
    )
    
    # Add file handler
    file_handler = logging.FileHandler(self.debug_log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Add console handler if requested
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger
```

### Usage Examples

#### Basic Usage

```python
from helpers import TradingLogger
from decimal import Decimal

# Create logger
logger = TradingLogger("edgex", "ETH", log_to_console=True)

# Log messages
logger.log("Bot started", "INFO")
logger.log("Connecting to exchange", "INFO")
logger.log("WebSocket connected", "DEBUG")

# Log transaction
logger.log_transaction(
    order_id="order123",
    side="buy",
    quantity=Decimal("0.1"),
    price=Decimal("2000.50"),
    status="FILLED"
)

logger.log("Bot stopped", "INFO")
```

#### Multi-Account Setup

```python
import os
os.environ['ACCOUNT_NAME'] = 'main_account'

logger = TradingLogger("edgex", "ETH")
# Creates: logs/edgex_ETH_main_account_activity.log
# Creates: logs/edgex_ETH_main_account_orders.csv
```

#### Custom Timezone

```python
import os
os.environ['TIMEZONE'] = 'America/New_York'

logger = TradingLogger("edgex", "ETH")
# All timestamps will be in New York timezone
```

#### Integration with Trading Bot

```python
from trading_bot import TradingBot, TradingConfig
from helpers import TradingLogger
from decimal import Decimal

class CustomBot(TradingBot):
    def __init__(self, config: TradingConfig):
        super().__init__(config)
        
        # Additional custom logger
        self.custom_logger = TradingLogger(
            config.exchange, 
            config.ticker,
            log_to_console=True
        )
    
    async def _place_and_monitor_open_order(self) -> bool:
        self.custom_logger.log("Placing order", "INFO")
        result = await super()._place_and_monitor_open_order()
        if result:
            self.custom_logger.log("Order placed successfully", "INFO")
        else:
            self.custom_logger.log("Order placement failed", "ERROR")
        return result
```

---

## TelegramBot

Send notifications via Telegram for important events and alerts.

**Location**: `helpers/telegram_bot.py`

### Class Definition

```python
class TelegramBot:
    """Send notifications via Telegram Bot API."""
    
    def __init__(
        self, 
        token: str, 
        chat_id: str, 
        base_url: Optional[str] = None
    ):
        """
        Initialize Telegram bot.
        
        Args:
            token: Telegram bot token from BotFather
            chat_id: Telegram chat ID to send messages to
            base_url: Optional custom API base URL
        """
```

### Features

- **Simple API**: Easy-to-use text messaging
- **Context Manager**: Automatic resource cleanup
- **SSL Support**: Secure connections via certifi
- **Error Handling**: Graceful error handling and logging
- **Custom API URL**: Support for proxy or custom endpoints
- **HTML/Markdown**: Rich text formatting support

### Methods

#### `__init__(token, chat_id, base_url=None)`

Initialize Telegram bot.

**Parameters:**
- `token` (str): Bot token from BotFather
- `chat_id` (str): Target chat ID
- `base_url` (Optional[str]): Custom API URL

**Example:**
```python
from helpers.telegram_bot import TelegramBot

bot = TelegramBot(
    token="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11",
    chat_id="123456789"
)
```

#### `send_text(content, parse_mode="HTML")`

Send a text message.

**Parameters:**
- `content` (str): Message text
- `parse_mode` (str): "HTML" or "Markdown"

**Returns:** API response dictionary

**Example:**
```python
# Plain text
response = bot.send_text("Trading bot started!")

# HTML formatting
response = bot.send_text(
    "<b>Alert:</b> Position mismatch detected!",
    parse_mode="HTML"
)

# Markdown formatting
response = bot.send_text(
    "*Alert:* Position mismatch detected!",
    parse_mode="Markdown"
)
```

#### Context Manager Support

Use `with` statement for automatic cleanup:

```python
with TelegramBot(token, chat_id) as bot:
    bot.send_text("Hello!")
    bot.send_text("Trading started")
# Automatically closes session
```

### Configuration

#### Get Bot Token

1. Open Telegram and search for `@BotFather`
2. Send `/newbot` command
3. Follow instructions to create bot
4. Copy the bot token provided

#### Get Chat ID

Method 1 - Via Bot:
1. Search for `@userinfobot` in Telegram
2. Start the bot
3. Your chat ID will be displayed

Method 2 - Via API:
1. Send a message to your bot
2. Visit: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
3. Look for `"chat":{"id":123456789}`

#### Environment Variables

```bash
# Telegram configuration
TELEGRAM_BOT_TOKEN=123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11
TELEGRAM_CHAT_ID=123456789
```

### Usage Examples

#### Basic Notification

```python
from helpers.telegram_bot import TelegramBot
import os

token = os.getenv("TELEGRAM_BOT_TOKEN")
chat_id = os.getenv("TELEGRAM_CHAT_ID")

with TelegramBot(token, chat_id) as bot:
    bot.send_text("Trading bot started successfully!")
```

#### Rich Formatting

```python
with TelegramBot(token, chat_id) as bot:
    message = """
<b>Trading Alert</b>

Exchange: <code>EdgeX</code>
Ticker: <code>ETH</code>
Status: <b>Position Mismatch</b>

Position: 1.5 ETH
Active Orders: 1.2 ETH
<b>Difference: 0.3 ETH</b>

Please review manually.
    """
    bot.send_text(message, parse_mode="HTML")
```

#### Error Notifications

```python
try:
    # Trading logic
    pass
except Exception as e:
    with TelegramBot(token, chat_id) as bot:
        bot.send_text(f"<b>ERROR:</b> {str(e)}", parse_mode="HTML")
```

#### Integration with Trading Bot

```python
import os
from trading_bot import TradingBot

class CustomBot(TradingBot):
    async def send_notification(self, message: str):
        """Override notification method."""
        # Call parent implementation
        await super().send_notification(message)
        
        # Send via Telegram
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        
        if token and chat_id:
            with TelegramBot(token, chat_id) as bot:
                bot.send_text(f"[{self.config.exchange.upper()}] {message}")
```

---

## LarkBot

Send notifications via Lark (Feishu) for team collaboration and alerts.

**Location**: `helpers/lark_bot.py`

### Class Definition

```python
class LarkBot:
    """Send notifications via Lark/Feishu webhook."""
    
    def __init__(
        self, 
        token: str, 
        base_url: Optional[str] = None
    ):
        """
        Initialize Lark bot.
        
        Args:
            token: Lark webhook token
            base_url: Optional custom webhook base URL
        """
```

### Features

- **Async API**: Built on aiohttp for async operations
- **Context Manager**: Automatic async resource cleanup
- **SSL Support**: Secure connections via certifi
- **Error Handling**: Graceful error handling
- **Webhook-based**: No bot registration required
- **Team Collaboration**: Send to group chats

### Methods

#### `__init__(token, base_url=None)`

Initialize Lark bot.

**Parameters:**
- `token` (str): Webhook token from Lark
- `base_url` (Optional[str]): Custom webhook URL

**Example:**
```python
from helpers.lark_bot import LarkBot

async def main():
    async with LarkBot(token="your-webhook-token") as bot:
        await bot.send_text("Hello from bot!")
```

#### `send_text(content)`

Send a text message (async).

**Parameters:**
- `content` (str): Message text

**Returns:** API response dictionary

**Example:**
```python
async with LarkBot(token) as bot:
    response = await bot.send_text("Trading alert!")
    print(response)
```

#### Context Manager Support (Async)

Use `async with` statement:

```python
async with LarkBot(token) as bot:
    await bot.send_text("Hello!")
    await bot.send_text("Trading started")
# Automatically closes session
```

### Configuration

#### Get Webhook Token

1. Open Lark/Feishu app
2. Go to group chat
3. Click "..." → "Settings" → "Bots"
4. Add "Custom Bot"
5. Copy webhook URL
6. Extract token from URL: `https://www.feishu.cn/flow/api/trigger-webhook/{TOKEN}`

#### Environment Variables

```bash
# Lark configuration
LARK_TOKEN=your-webhook-token
```

### Usage Examples

#### Basic Notification

```python
import asyncio
import os
from helpers.lark_bot import LarkBot

async def notify():
    token = os.getenv("LARK_TOKEN")
    async with LarkBot(token) as bot:
        await bot.send_text("Trading bot started!")

asyncio.run(notify())
```

#### Multiple Messages

```python
async def send_updates():
    token = os.getenv("LARK_TOKEN")
    async with LarkBot(token) as bot:
        await bot.send_text("Starting bot...")
        await bot.send_text("Connected to exchange")
        await bot.send_text("Bot running successfully")

asyncio.run(send_updates())
```

#### Error Handling

```python
async def notify_error(error_message: str):
    token = os.getenv("LARK_TOKEN")
    if not token:
        print("Lark token not configured")
        return
    
    try:
        async with LarkBot(token) as bot:
            message = f"ERROR: {error_message}"
            response = await bot.send_text(message)
            if response.get("code") == 0:
                print("Notification sent successfully")
            else:
                print(f"Failed to send: {response}")
    except Exception as e:
        print(f"Lark notification error: {e}")
```

#### Integration with Trading Bot

```python
import os
from trading_bot import TradingBot
from helpers.lark_bot import LarkBot

class CustomBot(TradingBot):
    async def send_notification(self, message: str):
        """Override notification method."""
        # Call parent implementation
        await super().send_notification(message)
        
        # Send via Lark
        lark_token = os.getenv("LARK_TOKEN")
        if lark_token:
            async with LarkBot(lark_token) as bot:
                formatted_message = f"""
Trading Alert - {self.config.exchange.upper()}

{message}

Ticker: {self.config.ticker}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                """
                await bot.send_text(formatted_message)
```

---

## Utility Functions

### Retry Decorator

**Location**: `exchanges/base.py`

Automatic retry with exponential backoff for API calls.

```python
from exchanges.base import query_retry

@query_retry(
    default_return=None,
    exception_type=Exception,
    max_attempts=5,
    min_wait=1,
    max_wait=10,
    reraise=False
)
async def unreliable_api_call():
    """This will retry up to 5 times with exponential backoff."""
    # API call that might fail
    pass
```

**Parameters:**
- `default_return`: Return value if all retries fail
- `exception_type`: Exception types to retry on
- `max_attempts`: Maximum retry attempts
- `min_wait`: Minimum wait time (seconds)
- `max_wait`: Maximum wait time (seconds)
- `reraise`: Whether to reraise exception after retries

**Example:**
```python
from decimal import Decimal
from exchanges.base import query_retry

class MyClient:
    @query_retry(
        default_return=Decimal("0"),
        exception_type=(ConnectionError, TimeoutError),
        max_attempts=3,
        min_wait=2,
        max_wait=10,
        reraise=True
    )
    async def get_balance(self) -> Decimal:
        # Fetch balance from API
        response = await self.api.get("/balance")
        return Decimal(response['balance'])
```

---

## Best Practices

### Logging

1. **Use Appropriate Log Levels**
   ```python
   logger.log("Order placed", "INFO")  # Normal operations
   logger.log("Debugging data", "DEBUG")  # Debug information
   logger.log("Rate limit approaching", "WARNING")  # Warnings
   logger.log("Failed to connect", "ERROR")  # Errors
   ```

2. **Include Context**
   ```python
   logger.log(f"Order {order_id} filled at {price}", "INFO")
   logger.log(f"Position: {position}, Active orders: {len(orders)}", "INFO")
   ```

3. **Log Transactions**
   ```python
   # Always log filled orders
   if order_status == "FILLED":
       logger.log_transaction(order_id, side, quantity, price, status)
   ```

4. **Use Console Logging Sparingly**
   ```python
   # Development/Testing
   logger = TradingLogger("edgex", "ETH", log_to_console=True)
   
   # Production
   logger = TradingLogger("edgex", "ETH", log_to_console=False)
   ```

### Notifications

1. **Configure Notifications**
   ```bash
   # Use for important alerts only
   TELEGRAM_BOT_TOKEN=...
   TELEGRAM_CHAT_ID=...
   ```

2. **Send Important Events Only**
   ```python
   # Good: Critical alerts
   await bot.send_notification("Position mismatch detected")
   await bot.send_notification("Bot stopped due to error")
   
   # Avoid: Routine operations
   # await bot.send_notification("Order placed")  # Too frequent
   ```

3. **Use Formatting**
   ```python
   # Make notifications readable
   message = """
   <b>Alert</b>: Position Mismatch
   
   Exchange: EdgeX
   Position: 1.5 ETH
   Expected: 1.2 ETH
   """
   bot.send_text(message, parse_mode="HTML")
   ```

4. **Handle Errors Gracefully**
   ```python
   try:
       with TelegramBot(token, chat_id) as bot:
           bot.send_text("Alert!")
   except Exception as e:
       logger.log(f"Notification failed: {e}", "WARNING")
       # Continue execution
   ```

### Error Handling

1. **Use Try-Except Blocks**
   ```python
   try:
       logger.log_transaction(order_id, side, qty, price, status)
   except Exception as e:
       logger.log(f"Failed to log transaction: {e}", "ERROR")
   ```

2. **Log Errors with Context**
   ```python
   try:
       result = await api_call()
   except Exception as e:
       logger.log(f"API call failed: {e}", "ERROR")
       logger.log(f"Traceback: {traceback.format_exc()}", "ERROR")
   ```

3. **Use Retry Decorator**
   ```python
   @query_retry(max_attempts=3)
   async def fetch_data():
       # Automatically retries on failure
       return await api.get_data()
   ```

---

## Examples

### Complete Trading Bot with Logging

```python
import asyncio
from decimal import Decimal
from trading_bot import TradingBot, TradingConfig
from helpers import TradingLogger

async def main():
    # Create logger
    logger = TradingLogger("edgex", "ETH", log_to_console=True)
    logger.log("Initializing trading bot", "INFO")
    
    # Create config
    config = TradingConfig(
        ticker="ETH",
        contract_id="",
        tick_size=Decimal(0),
        quantity=Decimal("0.1"),
        take_profit=Decimal("0.02"),
        direction="buy",
        max_orders=40,
        wait_time=450,
        exchange="edgex",
        grid_step=Decimal("0.5"),
        stop_price=Decimal("-1"),
        pause_price=Decimal("-1"),
        boost_mode=False
    )
    
    # Create and run bot
    bot = TradingBot(config)
    
    try:
        logger.log("Starting trading bot", "INFO")
        await bot.run()
    except KeyboardInterrupt:
        logger.log("Bot stopped by user", "INFO")
    except Exception as e:
        logger.log(f"Bot error: {e}", "ERROR")
    finally:
        logger.log("Bot shutdown complete", "INFO")

if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()
    asyncio.run(main())
```

### Multi-Channel Notifications

```python
import os
from helpers.telegram_bot import TelegramBot
from helpers.lark_bot import LarkBot

async def send_alert(message: str):
    """Send alert via all configured channels."""
    
    # Telegram
    tg_token = os.getenv("TELEGRAM_BOT_TOKEN")
    tg_chat = os.getenv("TELEGRAM_CHAT_ID")
    if tg_token and tg_chat:
        with TelegramBot(tg_token, tg_chat) as bot:
            bot.send_text(f"<b>Alert:</b> {message}", parse_mode="HTML")
    
    # Lark
    lark_token = os.getenv("LARK_TOKEN")
    if lark_token:
        async with LarkBot(lark_token) as bot:
            await bot.send_text(f"Alert: {message}")

# Usage
await send_alert("Position mismatch detected!")
```

### Custom Logger with Notifications

```python
from helpers import TradingLogger
from helpers.telegram_bot import TelegramBot
import os

class NotifyingLogger(TradingLogger):
    """Logger that sends critical errors via Telegram."""
    
    def __init__(self, exchange: str, ticker: str, log_to_console: bool = False):
        super().__init__(exchange, ticker, log_to_console)
        self.tg_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.tg_chat = os.getenv("TELEGRAM_CHAT_ID")
    
    def log(self, message: str, level: str = "INFO"):
        """Log and send critical errors to Telegram."""
        super().log(message, level)
        
        if level == "ERROR" and self.tg_token and self.tg_chat:
            try:
                with TelegramBot(self.tg_token, self.tg_chat) as bot:
                    bot.send_text(
                        f"<b>ERROR</b> [{self.exchange.upper()}_{self.ticker.upper()}]\n\n{message}",
                        parse_mode="HTML"
                    )
            except Exception as e:
                # Don't fail on notification error
                super().log(f"Failed to send Telegram notification: {e}", "WARNING")

# Usage
logger = NotifyingLogger("edgex", "ETH", log_to_console=True)
logger.log("Critical error occurred!", "ERROR")  # Sends Telegram notification
```

---

**Document Version**: 1.0  
**Last Updated**: 2025-11-23  
**Compatibility**: All current implementations
