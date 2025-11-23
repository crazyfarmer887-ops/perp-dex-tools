# API Documentation

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Trading Bot API](#trading-bot-api)
5. [Exchange Client API](#exchange-client-api)
6. [Hedge Mode API](#hedge-mode-api)
7. [Helper Utilities](#helper-utilities)
8. [Configuration](#configuration)
9. [Examples](#examples)

---

## Overview

This trading bot is a modular, multi-exchange automated trading system that supports both standard trading and hedge mode strategies. The system is designed with a clean architecture that allows for easy addition of new exchanges and trading strategies.

### Key Features

- **Multi-Exchange Support**: EdgeX, Backpack, Paradex, Aster, Lighter, GRVT, Extended, Apex, BingX
- **Modular Architecture**: Easy to extend with new exchanges and strategies
- **Two Trading Modes**: Standard trading mode and hedge mode
- **Real-time WebSocket Integration**: Live order updates and market data
- **Comprehensive Logging**: File and console logging with transaction tracking
- **Notification Support**: Telegram and Lark notifications
- **Risk Management**: Configurable position limits, grid steps, and stop/pause prices

---

## Architecture

### Directory Structure

```
perp-dex-tools/
├── exchanges/           # Exchange client implementations
│   ├── base.py         # Base exchange interface
│   ├── factory.py      # Exchange factory for dynamic client creation
│   ├── edgex.py        # EdgeX implementation
│   ├── backpack.py     # Backpack implementation
│   ├── paradex.py      # Paradex implementation
│   ├── aster.py        # Aster implementation
│   ├── lighter.py      # Lighter implementation
│   ├── grvt.py         # GRVT implementation
│   ├── extended.py     # Extended implementation
│   ├── apex.py         # Apex implementation
│   └── bingx.py        # BingX implementation
├── hedge/              # Hedge mode implementations
│   ├── hedge_mode_bp.py       # Backpack hedge mode
│   ├── hedge_mode_ext.py      # Extended hedge mode
│   ├── hedge_mode_apex.py     # Apex hedge mode
│   ├── hedge_mode_grvt.py     # GRVT hedge mode
│   ├── hedge_mode_grvt_bingx.py  # GRVT + BingX hedge mode
│   └── hedge_mode_edgex.py    # EdgeX hedge mode
├── helpers/            # Utility modules
│   ├── logger.py       # Trading logger
│   ├── telegram_bot.py # Telegram notifications
│   └── lark_bot.py     # Lark notifications
├── trading_bot.py      # Main trading bot logic
├── runbot.py          # Entry point for standard trading
└── hedge_mode.py      # Entry point for hedge mode
```

### Design Patterns

1. **Factory Pattern**: `ExchangeFactory` creates exchange clients dynamically
2. **Abstract Base Class**: `BaseExchangeClient` defines the interface for all exchanges
3. **Dependency Injection**: Configuration objects are passed to clients
4. **Event-Driven**: WebSocket handlers for real-time order updates

---

## Core Components

### 1. TradingConfig

Data class that holds all trading configuration parameters.

**Location**: `trading_bot.py`

```python
@dataclass
class TradingConfig:
    ticker: str              # Trading pair symbol (e.g., "ETH", "BTC")
    contract_id: str         # Exchange-specific contract identifier
    quantity: Decimal        # Order quantity
    take_profit: Decimal     # Take profit percentage
    tick_size: Decimal       # Price tick size
    direction: str           # Trading direction: "buy" or "sell"
    max_orders: int          # Maximum concurrent orders
    wait_time: int           # Seconds to wait between orders
    exchange: str            # Exchange name
    grid_step: Decimal       # Minimum distance between close orders (%)
    stop_price: Decimal      # Price to stop trading (-1 for disabled)
    pause_price: Decimal     # Price to pause trading (-1 for disabled)
    boost_mode: bool         # Enable boost mode (Aster/Backpack only)
```

**Properties**:
- `close_order_side`: Returns the opposite side for closing positions

**Example**:
```python
config = TradingConfig(
    ticker="ETH",
    contract_id="ETH-PERP",
    quantity=Decimal("0.1"),
    take_profit=Decimal("0.02"),
    tick_size=Decimal("0.01"),
    direction="buy",
    max_orders=40,
    wait_time=450,
    exchange="edgex",
    grid_step=Decimal("0.5"),
    stop_price=Decimal("-1"),
    pause_price=Decimal("-1"),
    boost_mode=False
)
```

### 2. OrderResult

Standardized structure for order operation results.

**Location**: `exchanges/base.py`

```python
@dataclass
class OrderResult:
    success: bool                      # Whether operation succeeded
    order_id: Optional[str] = None    # Order identifier
    side: Optional[str] = None        # "buy" or "sell"
    size: Optional[Decimal] = None    # Order size
    price: Optional[Decimal] = None   # Order price
    status: Optional[str] = None      # Order status
    error_message: Optional[str] = None  # Error details if failed
    filled_size: Optional[Decimal] = None  # Filled quantity
```

**Example**:
```python
# Successful order
result = OrderResult(
    success=True,
    order_id="123456",
    side="buy",
    size=Decimal("0.1"),
    price=Decimal("2000.50"),
    status="FILLED",
    filled_size=Decimal("0.1")
)

# Failed order
result = OrderResult(
    success=False,
    error_message="Insufficient balance"
)
```

### 3. OrderInfo

Structure for querying order information.

**Location**: `exchanges/base.py`

```python
@dataclass
class OrderInfo:
    order_id: str              # Order identifier
    side: str                  # "buy" or "sell"
    size: Decimal             # Total order size
    price: Decimal            # Order price
    status: str               # Order status
    filled_size: Decimal = 0.0      # Filled quantity
    remaining_size: Decimal = 0.0   # Remaining quantity
    cancel_reason: str = ''         # Cancellation reason if applicable
```

---

## Trading Bot API

### TradingBot

Main trading bot class that orchestrates the trading strategy.

**Location**: `trading_bot.py`

#### Constructor

```python
def __init__(self, config: TradingConfig)
```

**Parameters**:
- `config`: TradingConfig object with all trading parameters

**Example**:
```python
from trading_bot import TradingBot, TradingConfig
from decimal import Decimal

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

bot = TradingBot(config)
```

#### Public Methods

##### run()

Main trading loop that executes the trading strategy.

```python
async def run(self) -> None
```

**Behavior**:
1. Connects to exchange
2. Retrieves contract attributes
3. Starts main trading loop:
   - Monitors active orders
   - Places new orders based on strategy
   - Handles position management
   - Checks stop/pause conditions
4. Handles graceful shutdown on errors or interruption

**Example**:
```python
import asyncio

bot = TradingBot(config)
await bot.run()

# Or from a script
asyncio.run(bot.run())
```

##### graceful_shutdown()

Performs graceful shutdown of the trading bot.

```python
async def graceful_shutdown(self, reason: str = "Unknown") -> None
```

**Parameters**:
- `reason`: Reason for shutdown (for logging)

**Example**:
```python
await bot.graceful_shutdown("Market conditions changed")
```

##### send_notification()

Sends notifications via configured channels (Telegram/Lark).

```python
async def send_notification(self, message: str) -> None
```

**Parameters**:
- `message`: Notification message text

**Example**:
```python
await bot.send_notification("Position mismatch detected!")
```

#### Private Methods

These methods are used internally by the bot but understanding them helps with customization.

##### _place_and_monitor_open_order()

Places an order and monitors its execution.

```python
async def _place_and_monitor_open_order(self) -> bool
```

**Returns**: True if order was successfully placed and handled

##### _handle_order_result()

Handles the result of an order placement (filled, partially filled, or cancelled).

```python
async def _handle_order_result(self, order_result: OrderResult) -> bool
```

##### _calculate_wait_time()

Calculates dynamic wait time based on active orders.

```python
def _calculate_wait_time(self) -> Decimal
```

**Logic**:
- No wait if close orders decreased
- 1 second if max orders reached
- 2x wait time if ≥2/3 of max orders
- 1x wait time if ≥1/3 of max orders
- 0.5x wait time if ≥1/6 of max orders
- 0.25x wait time if <1/6 of max orders

##### _meet_grid_step_condition()

Checks if new order meets grid step requirement.

```python
async def _meet_grid_step_condition(self) -> bool
```

**Returns**: True if new order satisfies grid step distance

##### _check_price_condition()

Checks if stop or pause price conditions are met.

```python
async def _check_price_condition(self) -> Tuple[bool, bool]
```

**Returns**: (stop_trading, pause_trading) boolean tuple

##### _log_status_periodically()

Logs position and order status every 60 seconds.

```python
async def _log_status_periodically(self) -> bool
```

**Returns**: True if position mismatch detected

---

## Exchange Client API

### BaseExchangeClient

Abstract base class that all exchange clients must implement.

**Location**: `exchanges/base.py`

#### Abstract Methods

All exchange implementations must provide these methods:

##### connect()

Establish connection to the exchange (WebSocket, etc.).

```python
async def connect(self) -> None
```

##### disconnect()

Close connection to the exchange.

```python
async def disconnect(self) -> None
```

##### place_open_order()

Place an opening order (entry order).

```python
async def place_open_order(
    self, 
    contract_id: str, 
    quantity: Decimal, 
    direction: str
) -> OrderResult
```

**Parameters**:
- `contract_id`: Exchange-specific contract identifier
- `quantity`: Order size
- `direction`: "buy" or "sell"

**Returns**: OrderResult with order details

##### place_close_order()

Place a closing order (take profit order).

```python
async def place_close_order(
    self, 
    contract_id: str, 
    quantity: Decimal, 
    price: Decimal, 
    side: str
) -> OrderResult
```

**Parameters**:
- `contract_id`: Exchange-specific contract identifier
- `quantity`: Order size
- `price`: Limit price
- `side`: "buy" or "sell"

**Returns**: OrderResult with order details

##### place_market_order()

Place a market order (immediate execution).

```python
async def place_market_order(
    self, 
    contract_id: str, 
    quantity: Decimal, 
    side: str
) -> OrderResult
```

**Parameters**:
- `contract_id`: Exchange-specific contract identifier
- `quantity`: Order size
- `side`: "buy" or "sell"

**Returns**: OrderResult with order details

##### cancel_order()

Cancel an existing order.

```python
async def cancel_order(self, order_id: str) -> OrderResult
```

**Parameters**:
- `order_id`: Order identifier to cancel

**Returns**: OrderResult indicating success/failure

##### get_order_info()

Retrieve information about a specific order.

```python
async def get_order_info(self, order_id: str) -> Optional[OrderInfo]
```

**Parameters**:
- `order_id`: Order identifier

**Returns**: OrderInfo object or None if not found

##### get_active_orders()

Get all active orders for a contract.

```python
async def get_active_orders(self, contract_id: str) -> List[OrderInfo]
```

**Parameters**:
- `contract_id`: Exchange-specific contract identifier

**Returns**: List of OrderInfo objects

##### get_account_positions()

Get current account positions (net position).

```python
async def get_account_positions(self) -> Decimal
```

**Returns**: Net position size (positive for long, negative for short)

##### get_contract_attributes()

Get contract-specific attributes.

```python
async def get_contract_attributes(self) -> Tuple[str, Decimal]
```

**Returns**: Tuple of (contract_id, tick_size)

##### get_order_price()

Get the price at which to place an order based on direction.

```python
async def get_order_price(self, direction: str) -> Decimal
```

**Parameters**:
- `direction`: "buy" or "sell"

**Returns**: Recommended order price

##### fetch_bbo_prices()

Fetch best bid and best offer prices.

```python
async def fetch_bbo_prices(self, contract_id: str) -> Tuple[Decimal, Decimal]
```

**Parameters**:
- `contract_id`: Exchange-specific contract identifier

**Returns**: Tuple of (best_bid, best_ask)

##### setup_order_update_handler()

Setup handler for WebSocket order updates.

```python
def setup_order_update_handler(self, handler: Callable) -> None
```

**Parameters**:
- `handler`: Callback function to handle order updates

##### get_exchange_name()

Get the exchange name.

```python
def get_exchange_name(self) -> str
```

**Returns**: Exchange name string

#### Utility Methods

##### round_to_tick()

Round a price to the exchange's tick size.

```python
def round_to_tick(self, price: Decimal) -> Decimal
```

**Parameters**:
- `price`: Price to round

**Returns**: Rounded price

**Example**:
```python
# With tick_size = 0.01
client.round_to_tick(Decimal("2000.567"))  # Returns Decimal("2000.57")
```

##### query_retry()

Decorator for automatic retry with exponential backoff.

```python
@query_retry(
    default_return=None,
    exception_type=Exception,
    max_attempts=5,
    min_wait=1,
    max_wait=10,
    reraise=False
)
async def my_method(self):
    pass
```

**Parameters**:
- `default_return`: Value to return if all retries fail
- `exception_type`: Exception types to retry on
- `max_attempts`: Maximum retry attempts
- `min_wait`: Minimum wait time (seconds)
- `max_wait`: Maximum wait time (seconds)
- `reraise`: Whether to reraise exception after all retries fail

---

## Hedge Mode API

### HedgeBot

Trading bot that places maker orders on one exchange and hedges with market orders on another.

**Common Implementations**:
- `hedge.hedge_mode_bp.HedgeBot`: Backpack + Lighter
- `hedge.hedge_mode_ext.HedgeBot`: Extended + Lighter
- `hedge.hedge_mode_apex.HedgeBot`: Apex + Lighter
- `hedge.hedge_mode_grvt.HedgeBot`: GRVT + Lighter
- `hedge.hedge_mode_grvt_bingx.HedgeBot`: GRVT + BingX
- `hedge.hedge_mode_edgex.HedgeBot`: EdgeX + Lighter

#### Constructor

```python
def __init__(
    self,
    ticker: str,
    order_quantity: Decimal,
    fill_timeout: int = 5,
    iterations: int = 20,
    sleep_time: int = 0,
    tp_roi: Optional[Decimal] = None,
    sl_roi: Optional[Decimal] = None,
)
```

**Parameters**:
- `ticker`: Trading pair symbol (e.g., "BTC", "ETH")
- `order_quantity`: Size of each order
- `fill_timeout`: Timeout in seconds for maker orders
- `iterations`: Number of trading cycles to execute
- `sleep_time`: Pause duration after each trade (seconds)
- `tp_roi`: Take profit ROI percentage (optional)
- `sl_roi`: Stop loss ROI percentage (optional)

**Example**:
```python
from hedge.hedge_mode_bp import HedgeBot
from decimal import Decimal

bot = HedgeBot(
    ticker="BTC",
    order_quantity=Decimal("0.05"),
    fill_timeout=5,
    iterations=20,
    sleep_time=0,
    tp_roi=Decimal("0.4"),
    sl_roi=Decimal("0.2")
)
```

#### Public Methods

##### run()

Execute the hedge trading strategy.

```python
async def run(self) -> None
```

**Workflow**:
1. Initialize exchange connections
2. For each iteration:
   - Place maker order on primary exchange
   - Wait for fill or timeout
   - Place hedge market order on secondary exchange
   - Wait for target ROI if configured
   - Place maker close order on primary exchange
   - Place hedge close order on secondary exchange
3. Log results and cleanup

**Example**:
```python
import asyncio

bot = HedgeBot(
    ticker="BTC",
    order_quantity=Decimal("0.05"),
    iterations=10
)

asyncio.run(bot.run())
```

##### cleanup()

Clean up resources and close connections.

```python
async def cleanup(self) -> None
```

**Example**:
```python
try:
    await bot.run()
finally:
    await bot.cleanup()
```

##### close_positions_with_limit_orders()

*(GRVT + BingX only)* Close existing hedge positions using limit orders.

```python
async def close_positions_with_limit_orders(self) -> None
```

**Usage**:
```bash
python hedge_mode.py --exchange grvt_bingx --ticker BTC --size 0.05 --iter 1 --position-close
```

---

## Helper Utilities

### TradingLogger

Comprehensive logging system with file and console output.

**Location**: `helpers/logger.py`

#### Constructor

```python
def __init__(
    self, 
    exchange: str, 
    ticker: str, 
    log_to_console: bool = False
)
```

**Parameters**:
- `exchange`: Exchange name
- `ticker`: Trading pair symbol
- `log_to_console`: Whether to output to console

**Example**:
```python
from helpers import TradingLogger

logger = TradingLogger("edgex", "ETH", log_to_console=True)
```

#### Methods

##### log()

Log a message with specified level.

```python
def log(self, message: str, level: str = "INFO") -> None
```

**Parameters**:
- `message`: Log message
- `level`: Log level ("DEBUG", "INFO", "WARNING", "ERROR")

**Example**:
```python
logger.log("Order placed successfully", "INFO")
logger.log("Failed to connect", "ERROR")
```

##### log_transaction()

Log a trade transaction to CSV file.

```python
def log_transaction(
    self, 
    order_id: str, 
    side: str, 
    quantity: Decimal, 
    price: Decimal, 
    status: str
) -> None
```

**Parameters**:
- `order_id`: Order identifier
- `side`: "buy" or "sell"
- `quantity`: Trade quantity
- `price`: Trade price
- `status`: Order status

**Example**:
```python
logger.log_transaction(
    order_id="123456",
    side="buy",
    quantity=Decimal("0.1"),
    price=Decimal("2000.50"),
    status="FILLED"
)
```

**Output Files**:
- Activity log: `logs/{exchange}_{ticker}_activity.log`
- Transaction CSV: `logs/{exchange}_{ticker}_orders.csv`
- With account name: `logs/{exchange}_{ticker}_{account_name}_*.log/csv`

### TelegramBot

Send notifications via Telegram.

**Location**: `helpers/telegram_bot.py`

#### Constructor

```python
def __init__(self, token: str, chat_id: str, base_url: Optional[str] = None)
```

**Parameters**:
- `token`: Telegram bot token
- `chat_id`: Telegram chat ID
- `base_url`: Optional custom API base URL

**Example**:
```python
from helpers.telegram_bot import TelegramBot

with TelegramBot(token="YOUR_TOKEN", chat_id="YOUR_CHAT_ID") as bot:
    bot.send_text("Trading started!")
```

#### Methods

##### send_text()

Send a text message.

```python
def send_text(self, content: str, parse_mode: str = "HTML") -> Dict[str, Any]
```

**Parameters**:
- `content`: Message text
- `parse_mode`: Telegram parse mode ("HTML" or "Markdown")

**Returns**: API response dictionary

**Example**:
```python
with TelegramBot(token, chat_id) as bot:
    result = bot.send_text("<b>Alert:</b> Position mismatch!")
    print(result)
```

### LarkBot

Send notifications via Lark (Feishu).

**Location**: `helpers/lark_bot.py`

#### Constructor

```python
def __init__(self, token: str, base_url: Optional[str] = None)
```

**Parameters**:
- `token`: Lark webhook token
- `base_url`: Optional custom webhook base URL

**Example**:
```python
import asyncio
from helpers.lark_bot import LarkBot

async def notify():
    async with LarkBot(token="YOUR_TOKEN") as bot:
        await bot.send_text("Trading started!")

asyncio.run(notify())
```

#### Methods

##### send_text()

Send a text message.

```python
async def send_text(self, content: str) -> Dict[str, Any]
```

**Parameters**:
- `content`: Message text

**Returns**: API response dictionary

**Example**:
```python
async with LarkBot(token) as bot:
    result = await bot.send_text("Alert: Position mismatch!")
    print(result)
```

---

## Configuration

### Environment Variables

All configuration is loaded from `.env` files. You can use multiple `.env` files for different accounts.

#### General Configuration

```bash
# Account identification (optional, for multi-account setups)
ACCOUNT_NAME=main_account

# Timezone for logging (default: Asia/Shanghai)
TIMEZONE=Asia/Shanghai
```

#### Notification Configuration

```bash
# Telegram notifications (optional)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Lark notifications (optional)
LARK_TOKEN=your_lark_webhook_token
```

#### Exchange-Specific Configuration

##### EdgeX

```bash
EDGEX_ACCOUNT_ID=your_account_id
EDGEX_STARK_PRIVATE_KEY=your_private_key
EDGEX_BASE_URL=https://pro.edgex.exchange
EDGEX_WS_URL=wss://quote.edgex.exchange
```

##### Backpack

```bash
BACKPACK_PUBLIC_KEY=your_api_key
BACKPACK_SECRET_KEY=your_api_secret
```

##### Paradex

```bash
PARADEX_L1_ADDRESS=your_l1_address
PARADEX_L2_PRIVATE_KEY=your_l2_private_key
```

##### Aster

```bash
ASTER_API_KEY=your_api_key
ASTER_SECRET_KEY=your_api_secret
```

##### Lighter

```bash
API_KEY_PRIVATE_KEY=your_private_key
LIGHTER_ACCOUNT_INDEX=your_account_index
LIGHTER_API_KEY_INDEX=your_api_key_index
```

##### GRVT

```bash
GRVT_TRADING_ACCOUNT_ID=your_trading_account_id
GRVT_PRIVATE_KEY=your_private_key
GRVT_API_KEY=your_api_key
```

##### BingX

```bash
BINGX_API_KEY=your_api_key
BINGX_API_SECRET=your_api_secret
BINGX_ENVIRONMENT=prod  # or testnet
```

##### Extended

```bash
EXTENDED_API_KEY=your_api_key
EXTENDED_STARK_KEY_PUBLIC=your_stark_public_key
EXTENDED_STARK_KEY_PRIVATE=your_stark_private_key
EXTENDED_VAULT=your_vault_id
```

##### Apex

```bash
APEX_API_KEY=your_api_key
APEX_API_KEY_PASSPHRASE=your_passphrase
APEX_API_KEY_SECRET=your_secret
APEX_OMNI_KEY_SEED=your_omni_key_seed
```

### Command-Line Arguments

#### Standard Trading Mode

```bash
python runbot.py [OPTIONS]
```

**Options**:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--exchange` | str | edgex | Exchange to use |
| `--ticker` | str | ETH | Trading pair symbol |
| `--quantity` | Decimal | 0.1 | Order size |
| `--take-profit` | Decimal | 0.02 | Take profit percentage |
| `--direction` | str | buy | Trading direction (buy/sell) |
| `--max-orders` | int | 40 | Maximum concurrent orders |
| `--wait-time` | int | 450 | Wait time between orders (seconds) |
| `--grid-step` | Decimal | -100 | Grid step percentage (-100 = disabled) |
| `--stop-price` | Decimal | -1 | Stop price (-1 = disabled) |
| `--pause-price` | Decimal | -1 | Pause price (-1 = disabled) |
| `--boost` | flag | False | Enable boost mode (Aster/Backpack only) |
| `--env-file` | str | .env | Environment file path |

#### Hedge Mode

```bash
python hedge_mode.py [OPTIONS]
```

**Options**:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--exchange` | str | required | Exchange (backpack/extended/apex/grvt/grvt_bingx/edgex) |
| `--ticker` | str | BTC | Trading pair symbol |
| `--size` | Decimal | required | Order size |
| `--iter` | int | required | Number of iterations |
| `--fill-timeout` | int | 5 | Maker order timeout (seconds) |
| `--sleep` | int | 0 | Sleep time after each trade (seconds) |
| `--tp-roi` | Decimal | None | Take profit ROI percentage |
| `--sl-roi` | Decimal | None | Stop loss ROI percentage |
| `--position-close` | flag | False | Close hedge positions (grvt_bingx only) |
| `--env-file` | str | .env | Environment file path |

---

## Examples

### Example 1: Basic Trading Bot

Simple ETH trading on EdgeX:

```bash
python runbot.py \
  --exchange edgex \
  --ticker ETH \
  --quantity 0.1 \
  --take-profit 0.02 \
  --max-orders 40 \
  --wait-time 450
```

### Example 2: Trading with Grid Step Control

ETH trading with 0.5% grid step to avoid dense orders:

```bash
python runbot.py \
  --exchange edgex \
  --ticker ETH \
  --quantity 0.1 \
  --take-profit 0.02 \
  --max-orders 40 \
  --wait-time 450 \
  --grid-step 0.5
```

### Example 3: Trading with Stop Price

Stop trading when price reaches 5500:

```bash
python runbot.py \
  --exchange edgex \
  --ticker ETH \
  --quantity 0.1 \
  --take-profit 0.02 \
  --max-orders 40 \
  --wait-time 450 \
  --stop-price 5500
```

### Example 4: Boost Mode Trading

High-frequency trading on Backpack:

```bash
python runbot.py \
  --exchange backpack \
  --ticker ETH \
  --direction buy \
  --quantity 0.1 \
  --boost
```

### Example 5: Hedge Mode Trading

Backpack + Lighter hedge strategy:

```bash
python hedge_mode.py \
  --exchange backpack \
  --ticker BTC \
  --size 0.05 \
  --iter 20 \
  --fill-timeout 5
```

### Example 6: Hedge Mode with ROI Targets

Hedge trading with take profit and stop loss:

```bash
python hedge_mode.py \
  --exchange apex \
  --ticker BTC \
  --size 0.05 \
  --iter 20 \
  --tp-roi 0.4 \
  --sl-roi 0.2
```

### Example 7: Using Multiple Accounts

Different accounts with separate `.env` files:

```bash
# Account 1
python runbot.py \
  --exchange edgex \
  --ticker ETH \
  --quantity 0.1 \
  --env-file account1.env

# Account 2
python runbot.py \
  --exchange edgex \
  --ticker BTC \
  --quantity 0.05 \
  --env-file account2.env
```

### Example 8: Programmatic Usage

```python
import asyncio
from decimal import Decimal
from trading_bot import TradingBot, TradingConfig

async def main():
    # Create configuration
    config = TradingConfig(
        ticker="ETH",
        contract_id="",  # Will be resolved by bot
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
        await bot.run()
    except KeyboardInterrupt:
        print("Bot stopped by user")
        await bot.graceful_shutdown("User interruption")
    except Exception as e:
        print(f"Error: {e}")
        await bot.graceful_shutdown(f"Error: {e}")

if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()
    
    asyncio.run(main())
```

### Example 9: Custom Exchange Client

Implement a new exchange:

```python
from exchanges.base import BaseExchangeClient, OrderResult, OrderInfo
from typing import List, Optional, Tuple
from decimal import Decimal

class MyExchangeClient(BaseExchangeClient):
    """Custom exchange implementation."""
    
    def _validate_config(self) -> None:
        """Validate configuration."""
        required = ['api_key', 'api_secret']
        for field in required:
            if not hasattr(self.config, field):
                raise ValueError(f"Missing required config: {field}")
    
    async def connect(self) -> None:
        """Connect to exchange."""
        # Implement WebSocket connection
        pass
    
    async def disconnect(self) -> None:
        """Disconnect from exchange."""
        # Close connections
        pass
    
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
            order_id="123456",
            side=direction,
            size=quantity,
            price=Decimal("2000"),
            status="OPEN"
        )
    
    # Implement other required methods...
    
    def get_exchange_name(self) -> str:
        return "my_exchange"

# Register with factory
from exchanges.factory import ExchangeFactory
ExchangeFactory.register_exchange('my_exchange', MyExchangeClient)
```

### Example 10: Custom Notification Handler

Extend notification system:

```python
import asyncio
from trading_bot import TradingBot, TradingConfig
from helpers import TradingLogger

class CustomBot(TradingBot):
    """Trading bot with custom notifications."""
    
    async def send_notification(self, message: str):
        """Override notification method."""
        # Call parent implementation
        await super().send_notification(message)
        
        # Add custom notification logic
        print(f"[CUSTOM NOTIFICATION] {message}")
        
        # Send to custom webhook
        # await self.send_to_custom_webhook(message)

# Use custom bot
config = TradingConfig(...)
bot = CustomBot(config)
asyncio.run(bot.run())
```

### Example 11: Monitoring and Logging

Monitor bot activity:

```python
import asyncio
from decimal import Decimal
from trading_bot import TradingBot, TradingConfig
from helpers import TradingLogger

async def monitor_bot():
    """Monitor bot with custom logging."""
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
        grid_step=Decimal("-100"),
        stop_price=Decimal("-1"),
        pause_price=Decimal("-1"),
        boost_mode=False
    )
    
    bot = TradingBot(config)
    
    # Create custom logger
    logger = TradingLogger("edgex", "ETH", log_to_console=True)
    
    try:
        logger.log("Starting bot monitoring", "INFO")
        await bot.run()
    except Exception as e:
        logger.log(f"Bot error: {e}", "ERROR")
    finally:
        logger.log("Bot monitoring ended", "INFO")

if __name__ == "__main__":
    asyncio.run(monitor_bot())
```

---

## Advanced Topics

### Error Handling

The bot includes comprehensive error handling:

1. **Retry Logic**: Automatic retry with exponential backoff using `@query_retry` decorator
2. **Graceful Shutdown**: Proper cleanup on errors or interruption
3. **Position Mismatch Detection**: Automatic detection and notification
4. **WebSocket Reconnection**: Automatic reconnection on connection loss
5. **Transaction Logging**: All trades logged to CSV for audit

### Performance Optimization

Tips for optimal performance:

1. **Wait Time**: Adjust based on market volatility and max orders
2. **Grid Step**: Use appropriate grid step to avoid order clustering
3. **Max Orders**: Balance between capital utilization and risk
4. **Fill Timeout**: Adjust based on exchange and market conditions

### Risk Management

Built-in risk management features:

1. **Max Orders Limit**: Prevents excessive position accumulation
2. **Stop Price**: Automatic shutdown at price threshold
3. **Pause Price**: Temporary pause at price threshold
4. **Position Monitoring**: Continuous position/order reconciliation
5. **Grid Step Control**: Prevents dense order placement

### Multi-Account Setup

Running multiple accounts:

1. Create separate `.env` files:
   ```bash
   # account1.env
   ACCOUNT_NAME=account1
   EDGEX_ACCOUNT_ID=...
   
   # account2.env
   ACCOUNT_NAME=account2
   EDGEX_ACCOUNT_ID=...
   ```

2. Run separate instances:
   ```bash
   python runbot.py --env-file account1.env --ticker ETH &
   python runbot.py --env-file account2.env --ticker BTC &
   ```

### Extending the System

Adding new features:

1. **New Exchange**: Implement `BaseExchangeClient` and register with factory
2. **New Strategy**: Extend `TradingBot` class with custom logic
3. **New Notifications**: Implement custom notification handlers
4. **Custom Metrics**: Add logging and monitoring as needed

---

## Support and Contribution

### Getting Help

- Read the main README files: `README.md` and `README_EN.md`
- Check exchange-specific setup guides in `docs/`
- Review example commands and configurations above

### Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request

### License

This project uses a non-commercial license. See `LICENSE` file for details.

### Disclaimer

This software is for educational and research purposes only. Cryptocurrency trading involves significant risk and may result in financial loss. Use at your own risk.

---

## Appendix

### Order Status Values

Common order status values across exchanges:

- `OPEN`: Order placed but not filled
- `FILLED`: Order completely filled
- `PARTIALLY_FILLED`: Order partially filled
- `CANCELED`: Order cancelled
- `REJECTED`: Order rejected by exchange
- `EXPIRED`: Order expired

### Side Values

Order side indicators:

- `buy`: Buy order (long position)
- `sell`: Sell order (short position)

### Direction vs Side

- **Direction**: Bot's trading direction (`buy` = bullish, `sell` = bearish)
- **Side**: Order side (opening or closing position)
- For `buy` direction: open with `buy`, close with `sell`
- For `sell` direction: open with `sell`, close with `buy`

### Logging Files

Generated log files:

- `logs/{exchange}_{ticker}_activity.log`: Detailed activity log
- `logs/{exchange}_{ticker}_orders.csv`: Trade transaction log
- With account: `logs/{exchange}_{ticker}_{account}_*`

### CSV Log Format

Transaction CSV columns:

| Column | Description |
|--------|-------------|
| Timestamp | Transaction timestamp |
| OrderID | Order identifier |
| Side | Buy or sell |
| Quantity | Order size |
| Price | Execution price |
| Status | Order status |

---

**Document Version**: 1.0  
**Last Updated**: 2025-11-23  
**Compatible With**: All current exchange implementations
