# API Documentation - Perp DEX Trading Bot

**Version**: 1.0  
**Last Updated**: 2025-11-23

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Core Architecture](#core-architecture)
4. [Trading Bot API](#trading-bot-api)
5. [Exchange Client API](#exchange-client-api)
6. [Helper Modules](#helper-modules)
7. [Hedge Mode API](#hedge-mode-api)
8. [Configuration Guide](#configuration-guide)
9. [Usage Examples](#usage-examples)
10. [Error Handling](#error-handling)

---

## Overview

The Perp DEX Trading Bot is a modular Python trading system designed to support multiple perpetual futures exchanges. It implements automated market-making strategies with configurable parameters for risk management and position control.

### Key Features

- **Multi-Exchange Support**: EdgeX, Backpack, Paradex, Aster, Lighter, GRVT, Extended, Apex, BingX
- **Flexible Trading Strategies**: Standard market-making and hedge mode
- **Real-time Order Management**: WebSocket-based order updates and monitoring
- **Risk Controls**: Grid step, max orders, stop/pause prices
- **Notification System**: Telegram and Lark bot integration
- **Comprehensive Logging**: CSV trade logs and detailed activity logs

### Architecture

```
perp-dex-tools/
├── runbot.py                 # Main entry point for standard trading
├── hedge_mode.py             # Entry point for hedge mode trading
├── trading_bot.py            # Core TradingBot implementation
├── exchanges/                # Exchange client implementations
│   ├── base.py              # Base exchange interface
│   ├── factory.py           # Exchange factory pattern
│   ├── edgex.py             # EdgeX client
│   ├── backpack.py          # Backpack client
│   ├── paradex.py           # Paradex client
│   └── ...                  # Other exchange clients
├── helpers/                  # Utility modules
│   ├── logger.py            # Trading logger
│   ├── telegram_bot.py      # Telegram notifications
│   └── lark_bot.py          # Lark notifications
└── hedge/                    # Hedge mode implementations
    ├── hedge_mode_bp.py     # Backpack + Lighter hedge
    ├── hedge_mode_ext.py    # Extended + Lighter hedge
    └── ...                  # Other hedge implementations
```

---

## Installation

### Prerequisites

- Python 3.10 - 3.12 (recommended)
- pip package manager
- Virtual environment (recommended)

### Basic Installation

```bash
# Clone the repository
git clone <repository-url>
cd perp-dex-tools

# Create and activate virtual environment
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Exchange-Specific Dependencies

#### GRVT

```bash
pip install grvt-pysdk
```

#### Paradex

```bash
# Create separate virtual environment
python3 -m venv para_env
source para_env/bin/activate
pip install -r para_requirements.txt
```

#### Apex

```bash
pip install -r apex_requirements.txt
```

### Environment Configuration

Create a `.env` file in the project root:

```bash
cp env_example.txt .env
# Edit .env with your API credentials
```

---

## Core Architecture

### Design Patterns

The project uses several design patterns for maintainability and extensibility:

1. **Factory Pattern**: `ExchangeFactory` dynamically creates exchange clients
2. **Abstract Base Class**: `BaseExchangeClient` defines the interface for all exchanges
3. **Data Classes**: `TradingConfig`, `OrderResult`, `OrderInfo` for type-safe data structures
4. **Context Managers**: `TelegramBot` and `LarkBot` for resource management
5. **Dependency Injection**: Configuration passed to bot and exchange clients

### Threading Model

- **Async/Await**: All I/O operations are asynchronous using `asyncio`
- **WebSocket Handlers**: Run in background tasks for real-time updates
- **Thread-Safe Events**: `asyncio.Event` for order state synchronization

---

## Trading Bot API

### TradingBot Class

The main trading bot implementation that orchestrates all trading operations.

#### Class Definition

```python
class TradingBot:
    """Modular Trading Bot - Main trading logic supporting multiple exchanges."""
    
    def __init__(self, config: TradingConfig)
```

#### Constructor Parameters

- **config** (`TradingConfig`): Complete trading configuration object

#### Public Methods

##### `async def run()`

Starts the main trading loop and manages the bot lifecycle.

**Returns**: None (runs until stopped or error occurs)

**Raises**: 
- `ValueError`: Invalid configuration
- `Exception`: Critical trading errors

**Example**:

```python
from trading_bot import TradingBot, TradingConfig
from decimal import Decimal

config = TradingConfig(
    ticker='ETH',
    contract_id='',  # Auto-populated
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

bot = TradingBot(config)
await bot.run()
```

##### `async def graceful_shutdown(reason: str = "Unknown")`

Performs graceful shutdown of the trading bot.

**Parameters**:
- **reason** (`str`, optional): Reason for shutdown. Default: "Unknown"

**Returns**: None

**Example**:

```python
await bot.graceful_shutdown("Stop price reached")
```

##### `async def send_notification(message: str)`

Sends notifications via Telegram and Lark if configured.

**Parameters**:
- **message** (`str`): Notification message to send

**Returns**: None

**Example**:

```python
await bot.send_notification("Position mismatch detected!")
```

#### Private Methods

##### `def _calculate_wait_time() -> Decimal`

Calculates dynamic wait time based on active orders.

**Logic**:
- No wait if orders decreased
- 1 second if max orders reached
- Scales from `wait_time/4` to `2*wait_time` based on order count

##### `async def _place_and_monitor_open_order() -> bool`

Places an open order and monitors its execution.

**Returns**: `True` if successful, `False` otherwise

##### `async def _log_status_periodically()`

Logs position and order status every 60 seconds.

**Returns**: `True` if position mismatch detected, `False` otherwise

##### `async def _meet_grid_step_condition() -> bool`

Checks if new order meets grid step requirements.

**Returns**: `True` if grid step condition met, `False` otherwise

##### `async def _check_price_condition() -> Tuple[bool, bool]`

Checks stop and pause price conditions.

**Returns**: Tuple of `(stop_trading, pause_trading)` booleans

---

### TradingConfig Class

Configuration dataclass for trading parameters.

#### Class Definition

```python
@dataclass
class TradingConfig:
    """Configuration class for trading parameters."""
    ticker: str                    # Trading pair symbol (e.g., 'ETH', 'BTC')
    contract_id: str               # Exchange-specific contract ID
    quantity: Decimal              # Order quantity per trade
    take_profit: Decimal           # Take profit percentage (e.g., 0.02 = 0.02%)
    tick_size: Decimal             # Minimum price increment
    direction: str                 # Trading direction: 'buy' or 'sell'
    max_orders: int                # Maximum concurrent orders
    wait_time: int                 # Base wait time between orders (seconds)
    exchange: str                  # Exchange name (e.g., 'edgex')
    grid_step: Decimal             # Minimum distance to next close order (%)
    stop_price: Decimal            # Price to stop trading (-1 = disabled)
    pause_price: Decimal           # Price to pause trading (-1 = disabled)
    boost_mode: bool               # Enable boost mode (maker open, taker close)
```

#### Properties

##### `close_order_side -> str`

Returns the opposite side for closing positions.

**Returns**: 
- `'buy'` if direction is `'sell'`
- `'sell'` if direction is `'buy'`

**Example**:

```python
config = TradingConfig(direction='buy', ...)
print(config.close_order_side)  # Output: 'sell'
```

---

### OrderMonitor Class

Thread-safe order monitoring state.

#### Class Definition

```python
@dataclass
class OrderMonitor:
    """Thread-safe order monitoring state."""
    order_id: Optional[str] = None
    filled: bool = False
    filled_price: Optional[Decimal] = None
    filled_qty: Decimal = 0.0
```

#### Methods

##### `def reset()`

Resets the monitor state to initial values.

**Example**:

```python
monitor = OrderMonitor()
monitor.order_id = "12345"
monitor.reset()
print(monitor.order_id)  # Output: None
```

---

## Exchange Client API

### BaseExchangeClient (Abstract Base Class)

All exchange implementations inherit from this base class.

#### Class Definition

```python
class BaseExchangeClient(ABC):
    """Base class for all exchange clients."""
    
    def __init__(self, config: Dict[str, Any])
```

#### Abstract Methods

All exchange clients must implement these methods:

##### `async def connect()`

Establishes connection to the exchange (WebSocket, HTTP, etc.).

**Returns**: None

**Raises**: Connection errors specific to the exchange

##### `async def disconnect()`

Closes all connections to the exchange.

**Returns**: None

##### `async def place_open_order(contract_id: str, quantity: Decimal, direction: str) -> OrderResult`

Places an open (entry) order.

**Parameters**:
- **contract_id** (`str`): Exchange-specific contract identifier
- **quantity** (`Decimal`): Order size
- **direction** (`str`): `'buy'` or `'sell'`

**Returns**: `OrderResult` object with order details

**Example**:

```python
result = await exchange_client.place_open_order(
    contract_id="ETH-PERP",
    quantity=Decimal('0.1'),
    direction='buy'
)
if result.success:
    print(f"Order placed: {result.order_id}")
```

##### `async def place_close_order(contract_id: str, quantity: Decimal, price: Decimal, side: str) -> OrderResult`

Places a close (exit) order.

**Parameters**:
- **contract_id** (`str`): Exchange-specific contract identifier
- **quantity** (`Decimal`): Order size
- **price** (`Decimal`): Limit price
- **side** (`str`): `'buy'` or `'sell'`

**Returns**: `OrderResult` object with order details

##### `async def cancel_order(order_id: str) -> OrderResult`

Cancels an active order.

**Parameters**:
- **order_id** (`str`): Order ID to cancel

**Returns**: `OrderResult` with cancellation status

##### `async def get_order_info(order_id: str) -> Optional[OrderInfo]`

Retrieves information about a specific order.

**Parameters**:
- **order_id** (`str`): Order ID to query

**Returns**: `OrderInfo` object or `None` if not found

##### `async def get_active_orders(contract_id: str) -> List[OrderInfo]`

Gets all active orders for a contract.

**Parameters**:
- **contract_id** (`str`): Exchange-specific contract identifier

**Returns**: List of `OrderInfo` objects

##### `async def get_account_positions() -> Decimal`

Gets current net position size.

**Returns**: Net position (positive = long, negative = short)

##### `def setup_order_update_handler(handler: Callable)`

Sets up WebSocket handler for real-time order updates.

**Parameters**:
- **handler** (`Callable`): Callback function for order updates

**Example**:

```python
def my_handler(message):
    print(f"Order update: {message}")

exchange_client.setup_order_update_handler(my_handler)
```

##### `async def get_contract_attributes() -> Tuple[str, Decimal]`

Gets contract ID and tick size for the configured ticker.

**Returns**: Tuple of `(contract_id, tick_size)`

##### `async def fetch_bbo_prices(contract_id: str) -> Tuple[Decimal, Decimal]`

Fetches best bid and offer prices.

**Parameters**:
- **contract_id** (`str`): Exchange-specific contract identifier

**Returns**: Tuple of `(best_bid, best_ask)`

##### `async def get_order_price(direction: str) -> Decimal`

Gets the price for a new order based on direction.

**Parameters**:
- **direction** (`str`): `'buy'` or `'sell'`

**Returns**: Price for the order

##### `async def place_market_order(contract_id: str, quantity: Decimal, side: str) -> OrderResult`

Places a market order (for boost mode).

**Parameters**:
- **contract_id** (`str`): Exchange-specific contract identifier
- **quantity** (`Decimal`): Order size
- **side** (`str`): `'buy'` or `'sell'`

**Returns**: `OrderResult` object

#### Utility Methods

##### `def round_to_tick(price: Decimal) -> Decimal`

Rounds a price to the exchange's tick size.

**Parameters**:
- **price** (`Decimal`): Price to round

**Returns**: Rounded price

**Example**:

```python
rounded = exchange_client.round_to_tick(Decimal('2000.123'))
# If tick_size = 0.01, returns Decimal('2000.12')
```

---

### OrderResult Class

Standardized structure for order operation results.

```python
@dataclass
class OrderResult:
    """Standardized order result structure."""
    success: bool                           # Whether operation succeeded
    order_id: Optional[str] = None          # Exchange order ID
    side: Optional[str] = None              # 'buy' or 'sell'
    size: Optional[Decimal] = None          # Order size
    price: Optional[Decimal] = None         # Order price
    status: Optional[str] = None            # Order status
    error_message: Optional[str] = None     # Error message if failed
    filled_size: Optional[Decimal] = None   # Partially filled amount
```

#### Usage Example

```python
result = await exchange_client.place_open_order(...)
if result.success:
    print(f"Order {result.order_id} placed at {result.price}")
    if result.status == 'FILLED':
        print(f"Immediately filled: {result.filled_size}")
else:
    print(f"Order failed: {result.error_message}")
```

---

### OrderInfo Class

Standardized structure for order information.

```python
@dataclass
class OrderInfo:
    """Standardized order information structure."""
    order_id: str                           # Exchange order ID
    side: str                               # 'buy' or 'sell'
    size: Decimal                           # Total order size
    price: Decimal                          # Order price
    status: str                             # Order status
    filled_size: Decimal = 0.0              # Amount filled
    remaining_size: Decimal = 0.0           # Amount remaining
    cancel_reason: str = ''                 # Reason if canceled
```

---

### ExchangeFactory Class

Factory class for creating exchange clients dynamically.

#### Class Definition

```python
class ExchangeFactory:
    """Factory class for creating exchange clients."""
```

#### Class Methods

##### `classmethod create_exchange(exchange_name: str, config: Dict[str, Any]) -> BaseExchangeClient`

Creates an exchange client instance.

**Parameters**:
- **exchange_name** (`str`): Name of the exchange (e.g., 'edgex', 'backpack')
- **config** (`Dict[str, Any]`): Configuration dictionary

**Returns**: Exchange client instance

**Raises**: 
- `ValueError`: If exchange is not supported
- `ImportError`: If exchange module cannot be imported

**Example**:

```python
from exchanges import ExchangeFactory
from decimal import Decimal

config = {
    'ticker': 'ETH',
    'quantity': Decimal('0.1'),
    'direction': 'buy',
    # ... other config
}

exchange = ExchangeFactory.create_exchange('edgex', config)
await exchange.connect()
```

##### `classmethod get_supported_exchanges() -> List[str]`

Returns list of all supported exchanges.

**Returns**: List of exchange names

**Example**:

```python
exchanges = ExchangeFactory.get_supported_exchanges()
print(exchanges)
# Output: ['edgex', 'backpack', 'paradex', 'aster', 'lighter', 'grvt', 'extended', 'apex', 'bingx']
```

##### `classmethod register_exchange(name: str, exchange_class: Type[BaseExchangeClient])`

Registers a new exchange client (for custom implementations).

**Parameters**:
- **name** (`str`): Exchange name
- **exchange_class** (`Type[BaseExchangeClient]`): Exchange client class

**Raises**: `ValueError` if class doesn't inherit from `BaseExchangeClient`

**Example**:

```python
class MyCustomExchange(BaseExchangeClient):
    # ... implementation ...
    pass

ExchangeFactory.register_exchange('custom', MyCustomExchange)
```

---

### query_retry Decorator

Retry decorator for handling transient failures in API calls.

#### Function Definition

```python
def query_retry(
    default_return: Any = None,
    exception_type: Union[Type[Exception], Tuple[Type[Exception], ...]] = (Exception,),
    max_attempts: int = 5,
    min_wait: float = 1,
    max_wait: float = 10,
    reraise: bool = False
)
```

#### Parameters

- **default_return** (`Any`, optional): Value to return if all retries fail. Default: `None`
- **exception_type** (`Exception` or `Tuple[Exception, ...]`, optional): Exception types to retry. Default: `(Exception,)`
- **max_attempts** (`int`, optional): Maximum retry attempts. Default: `5`
- **min_wait** (`float`, optional): Minimum wait time between retries (seconds). Default: `1`
- **max_wait** (`float`, optional): Maximum wait time between retries (seconds). Default: `10`
- **reraise** (`bool`, optional): Whether to reraise exception after retries. Default: `False`

#### Returns

Decorated function with retry logic

#### Example

```python
from exchanges.base import query_retry

@query_retry(
    default_return={},
    exception_type=(ConnectionError, TimeoutError),
    max_attempts=3,
    min_wait=2,
    max_wait=10
)
async def fetch_market_data():
    # API call that might fail
    return await exchange.get_market_data()

# Usage
data = await fetch_market_data()
```

---

## Helper Modules

### TradingLogger Class

Enhanced logging with structured output for trading operations.

#### Class Definition

```python
class TradingLogger:
    """Enhanced logging with structured output and error handling."""
    
    def __init__(
        self,
        exchange: str,
        ticker: str,
        log_to_console: bool = False
    )
```

#### Constructor Parameters

- **exchange** (`str`): Exchange name
- **ticker** (`str`): Trading pair ticker
- **log_to_console** (`bool`, optional): Whether to log to console. Default: `False`

#### Public Methods

##### `def log(message: str, level: str = "INFO")`

Logs a message with specified level.

**Parameters**:
- **message** (`str`): Message to log
- **level** (`str`, optional): Log level ('DEBUG', 'INFO', 'WARNING', 'ERROR'). Default: 'INFO'

**Example**:

```python
from helpers import TradingLogger

logger = TradingLogger('edgex', 'ETH', log_to_console=True)
logger.log("Starting trading bot", "INFO")
logger.log("Connection error occurred", "ERROR")
```

##### `def log_transaction(order_id: str, side: str, quantity: Decimal, price: Decimal, status: str)`

Logs a transaction to CSV file.

**Parameters**:
- **order_id** (`str`): Order ID
- **side** (`str`): 'buy' or 'sell'
- **quantity** (`Decimal`): Order quantity
- **price** (`Decimal`): Order price
- **status** (`str`): Order status

**CSV Output Format**:

```csv
Timestamp,OrderID,Side,Quantity,Price,Status
2025-11-23 10:30:45,order_123,buy,0.1,2000.50,FILLED
```

**Example**:

```python
logger.log_transaction(
    order_id="order_123",
    side="buy",
    quantity=Decimal('0.1'),
    price=Decimal('2000.50'),
    status="FILLED"
)
```

#### Log Files

- **Activity Log**: `logs/{exchange}_{ticker}_activity.log`
- **Trade Log**: `logs/{exchange}_{ticker}_orders.csv`
- **Multi-Account**: Adds `_{ACCOUNT_NAME}` suffix if `ACCOUNT_NAME` env var is set

---

### TelegramBot Class

Telegram notification bot using context manager pattern.

#### Class Definition

```python
class TelegramBot:
    """Telegram bot for sending notifications."""
    
    def __init__(
        self,
        token: str,
        chat_id: str,
        base_url: Optional[str] = None
    )
```

#### Constructor Parameters

- **token** (`str`): Telegram bot token
- **chat_id** (`str`): Telegram chat ID
- **base_url** (`str`, optional): Custom API base URL. Default: `"https://api.telegram.org/bot"`

#### Public Methods

##### `def send_text(content: str, parse_mode: str = "HTML") -> Dict[str, Any]`

Sends a text message to Telegram.

**Parameters**:
- **content** (`str`): Message content
- **parse_mode** (`str`, optional): Parse mode ('HTML', 'Markdown'). Default: 'HTML'

**Returns**: Dictionary with response data

**Example**:

```python
from helpers.telegram_bot import TelegramBot

token = "your_bot_token"
chat_id = "your_chat_id"

with TelegramBot(token, chat_id) as bot:
    response = bot.send_text("Trading bot started successfully!")
    if response.get("ok"):
        print("Notification sent!")
```

##### `def close()`

Closes the requests session.

#### Context Manager Usage

```python
with TelegramBot(token, chat_id) as bot:
    bot.send_text("<b>Alert:</b> Stop price reached!")
    bot.send_text("Position: 1.5 ETH")
# Session automatically closed
```

---

### LarkBot Class

Lark (Feishu) notification bot using async context manager pattern.

#### Class Definition

```python
class LarkBot:
    """Lark/Feishu bot for sending notifications."""
    
    def __init__(
        self,
        token: str,
        base_url: Optional[str] = None
    )
```

#### Constructor Parameters

- **token** (`str`): Lark webhook token
- **base_url** (`str`, optional): Custom webhook base URL. Default: `"https://www.feishu.cn/flow/api/trigger-webhook/"`

#### Public Methods

##### `async def send_text(content: str) -> Dict[str, Any]`

Sends a text message to Lark.

**Parameters**:
- **content** (`str`): Message content

**Returns**: Dictionary with response data

**Example**:

```python
from helpers.lark_bot import LarkBot
import asyncio

async def send_notification():
    token = "your_lark_token"
    
    async with LarkBot(token) as bot:
        response = await bot.send_text("Trading bot started!")
        if response.get("code") == 0:
            print("Notification sent!")

asyncio.run(send_notification())
```

##### `async def close()`

Closes the aiohttp session.

#### Context Manager Usage

```python
async with LarkBot(token) as bot:
    await bot.send_text("Alert: Position mismatch detected")
    await bot.send_text("Please check your positions manually")
# Session automatically closed
```

---

## Hedge Mode API

### HedgeBot Class (General Interface)

Hedge mode implements a two-leg strategy where positions are simultaneously opened on a primary exchange (with maker orders) and hedged on a secondary exchange (with market orders).

#### Supported Exchange Pairs

- **Backpack + Lighter**: `hedge.hedge_mode_bp.HedgeBot`
- **Extended + Lighter**: `hedge.hedge_mode_ext.HedgeBot`
- **Apex + Lighter**: `hedge.hedge_mode_apex.HedgeBot`
- **GRVT + Lighter**: `hedge.hedge_mode_grvt.HedgeBot`
- **GRVT + BingX**: `hedge.hedge_mode_grvt_bingx.HedgeBot`
- **EdgeX + Lighter**: `hedge.hedge_mode_edgex.HedgeBot`

#### Class Definition (Common Interface)

```python
class HedgeBot:
    """Trading bot that hedges positions between two exchanges."""
    
    def __init__(
        self,
        ticker: str,
        order_quantity: Decimal,
        fill_timeout: int = 5,
        iterations: int = 20,
        sleep_time: int = 0,
        tp_roi: Optional[Decimal] = None,
        sl_roi: Optional[Decimal] = None
    )
```

#### Constructor Parameters

- **ticker** (`str`): Trading pair symbol (e.g., 'BTC', 'ETH')
- **order_quantity** (`Decimal`): Order size per trade
- **fill_timeout** (`int`, optional): Timeout for maker order fills (seconds). Default: `5`
- **iterations** (`int`, optional): Number of trading cycles. Default: `20`
- **sleep_time** (`int`, optional): Sleep time after each step (seconds). Default: `0`
- **tp_roi** (`Decimal`, optional): Take profit ROI percentage. Default: `None`
- **sl_roi** (`Decimal`, optional): Stop loss ROI percentage. Default: `None`

#### Public Methods

##### `async def run()`

Executes the hedge trading strategy for the specified number of iterations.

**Returns**: None

**Raises**: Trading errors

**Example**:

```python
from hedge.hedge_mode_bp import HedgeBot
from decimal import Decimal

bot = HedgeBot(
    ticker='BTC',
    order_quantity=Decimal('0.05'),
    fill_timeout=5,
    iterations=20,
    sleep_time=0,
    tp_roi=Decimal('0.4'),  # 0.4% take profit
    sl_roi=Decimal('0.2')   # 0.2% stop loss
)

await bot.run()
```

##### `async def cleanup()`

Cleans up resources and closes connections.

**Returns**: None

**Example**:

```python
try:
    await bot.run()
finally:
    await bot.cleanup()
```

##### `async def close_positions_with_limit_orders()` (GRVT + BingX only)

Places limit OPEN orders on both exchanges to close existing hedge positions.

**Returns**: None

**Example**:

```python
from hedge.hedge_mode_grvt_bingx import HedgeBot
from decimal import Decimal

bot = HedgeBot(
    ticker='BTC',
    order_quantity=Decimal('0.05'),
    iterations=1
)

try:
    await bot.close_positions_with_limit_orders()
finally:
    await bot.cleanup()
```

#### Hedge Trading Flow

1. **Open Phase**:
   - Place maker order on primary exchange
   - Wait for fill or timeout
   - Immediately hedge with market order on secondary exchange

2. **Hold Phase** (optional with ROI parameters):
   - Monitor unrealized PnL
   - Wait for take profit or stop loss target
   - Or hold for minimum time

3. **Close Phase**:
   - Place maker close order on primary exchange
   - Wait for fill or timeout
   - Close hedge position with market order on secondary exchange

4. **Repeat**: Continue for specified iterations

---

## Configuration Guide

### Environment Variables

#### Required Configuration

Create a `.env` file in the project root:

```bash
# Account identification (optional, for multi-account setups)
ACCOUNT_NAME=main

# Timezone for logging
TIMEZONE=Asia/Shanghai
```

#### Exchange-Specific Configuration

##### EdgeX

```bash
EDGEX_ACCOUNT_ID=your_account_id
EDGEX_STARK_PRIVATE_KEY=your_stark_private_key
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
PARADEX_L1_ADDRESS=your_l1_wallet_address
PARADEX_L2_PRIVATE_KEY=your_l2_private_key
```

##### Aster

```bash
ASTER_API_KEY=your_api_key
ASTER_SECRET_KEY=your_secret_key
```

##### Lighter

```bash
API_KEY_PRIVATE_KEY=your_lighter_private_key
LIGHTER_ACCOUNT_INDEX=your_account_index
LIGHTER_API_KEY_INDEX=your_api_key_index
```

**Finding LIGHTER_ACCOUNT_INDEX**:

1. Visit: `https://mainnet.zklighter.elliot.ai/api/v1/account?by=l1_address&value=YOUR_WALLET_ADDRESS`
2. Search for "account_index" in the response
3. Use the shorter index (main account) or longer index (sub-account)

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
BINGX_ENVIRONMENT=prod  # or 'testnet'
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

#### Notification Configuration (Optional)

##### Telegram

```bash
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

Setup guide: See [docs/telegram-bot-setup.md](docs/telegram-bot-setup.md)

##### Lark (Feishu)

```bash
LARK_TOKEN=your_webhook_token
```

### Command Line Arguments

#### Standard Trading Mode (`runbot.py`)

```bash
python runbot.py [OPTIONS]
```

**Options**:

- `--exchange`: Exchange to use. Choices: `edgex`, `backpack`, `paradex`, `aster`, `lighter`, `grvt`, `extended`, `apex`. Default: `edgex`
- `--ticker`: Trading pair symbol (e.g., `ETH`, `BTC`). Default: `ETH`
- `--quantity`: Order quantity per trade. Default: `0.1`
- `--take-profit`: Take profit percentage (e.g., `0.02` = 0.02%). Default: `0.02`
- `--direction`: Trading direction (`buy` or `sell`). Default: `buy`
- `--max-orders`: Maximum concurrent orders. Default: `40`
- `--wait-time`: Wait time between orders (seconds). Default: `450`
- `--env-file`: Path to `.env` file. Default: `.env`
- `--grid-step`: Minimum distance to next close order (%). Default: `-100` (disabled)
- `--stop-price`: Price to stop trading. Default: `-1` (disabled)
- `--pause-price`: Price to pause trading. Default: `-1` (disabled)
- `--boost`: Enable boost mode (maker open, taker close)

#### Hedge Mode (`hedge_mode.py`)

```bash
python hedge_mode.py [OPTIONS]
```

**Options**:

- `--exchange`: Exchange pair (`backpack`, `extended`, `apex`, `grvt`, `grvt_bingx`, `edgex`). **Required**
- `--ticker`: Trading pair symbol. Default: `BTC`
- `--size`: Order quantity per trade. **Required**
- `--iter`: Number of trading cycles. **Required**
- `--fill-timeout`: Timeout for maker fills (seconds). Default: `5`
- `--sleep`: Sleep time after each step (seconds). Default: `0`
- `--tp-roi`: Take profit ROI percentage. Optional
- `--sl-roi`: Stop loss ROI percentage. Optional
- `--env-file`: Path to `.env` file. Default: `.env`
- `--position-close`: (GRVT+BingX only) Close existing positions with limit orders

---

## Usage Examples

### Example 1: Basic Market Making on EdgeX

```bash
# Long ETH with 0.1 quantity, 0.02% take profit
python runbot.py \
  --exchange edgex \
  --ticker ETH \
  --quantity 0.1 \
  --take-profit 0.02 \
  --direction buy \
  --max-orders 40 \
  --wait-time 450
```

**What it does**:
- Places buy orders slightly above market price
- When filled, immediately places sell order at 0.02% profit
- Maintains up to 40 concurrent close orders
- Waits 450 seconds base time between new orders

### Example 2: Grid Trading with Step Control

```bash
# ETH with 0.5% grid step to prevent close orders bunching
python runbot.py \
  --exchange edgex \
  --ticker ETH \
  --quantity 0.1 \
  --take-profit 0.02 \
  --max-orders 40 \
  --wait-time 450 \
  --grid-step 0.5
```

**Grid step logic**:
- New close order must be 0.5% away from nearest existing close order
- Prevents orders clustering at similar price levels
- Improves fill probability and reduces risk

### Example 3: Stop Price Protection

```bash
# Stop trading if ETH reaches $5500 (prevent opening longs at perceived top)
python runbot.py \
  --exchange edgex \
  --ticker ETH \
  --direction buy \
  --quantity 0.1 \
  --take-profit 0.02 \
  --max-orders 40 \
  --wait-time 450 \
  --stop-price 5500
```

**Stop price behavior**:
- For `buy` direction: Stops if price >= stop_price
- For `sell` direction: Stops if price <= stop_price
- Bot performs graceful shutdown and sends notification

### Example 4: Boost Mode for High Volume

```bash
# Backpack boost mode: maker open, taker close
python runbot.py \
  --exchange backpack \
  --ticker ETH \
  --direction buy \
  --quantity 0.1 \
  --boost
```

**Boost mode**:
- Opens with maker order (lower fees)
- Closes immediately with taker/market order
- Maximizes trading volume at cost of slippage
- Only supported on Backpack and Aster

### Example 5: Multi-Account Setup

**Setup**:
```bash
# Create separate .env files
# account_1.env
ACCOUNT_NAME=MAIN
BACKPACK_PUBLIC_KEY=main_account_key
BACKPACK_SECRET_KEY=main_account_secret

# account_2.env  
ACCOUNT_NAME=SUB
BACKPACK_PUBLIC_KEY=sub_account_key
BACKPACK_SECRET_KEY=sub_account_secret
```

**Usage**:
```bash
# Run main account
python runbot.py --env-file account_1.env --exchange backpack --ticker ETH --quantity 0.1

# Run sub account
python runbot.py --env-file account_2.env --exchange backpack --ticker ETH --quantity 0.05
```

**Log files**:
- Main: `logs/backpack_ETH_MAIN_orders.csv`
- Sub: `logs/backpack_ETH_SUB_orders.csv`

### Example 6: Hedge Mode - Backpack + Lighter

```bash
# Hedge BTC positions between Backpack and Lighter
python hedge_mode.py \
  --exchange backpack \
  --ticker BTC \
  --size 0.05 \
  --iter 20 \
  --fill-timeout 5
```

**Trading flow**:
1. Place maker buy order on Backpack
2. When filled, immediately market sell on Lighter (hedge)
3. Wait briefly
4. Place maker sell order on Backpack (close position)
5. When filled, immediately market buy on Lighter (close hedge)
6. Repeat 20 times

### Example 7: Hedge Mode with ROI Targets

```bash
# Wait for 0.4% profit or 0.2% loss before closing
python hedge_mode.py \
  --exchange apex \
  --ticker BTC \
  --size 0.05 \
  --iter 10 \
  --tp-roi 0.4 \
  --sl-roi 0.2
```

**ROI behavior**:
- After opening hedge position, calculates average entry price
- Monitors unrealized PnL
- Closes position when:
  - PnL reaches +0.4% (take profit), OR
  - PnL reaches -0.2% (stop loss), OR
  - Timeout expires
- Provides protection against adverse price movements

### Example 8: Close Existing Hedge Positions

```bash
# Close existing GRVT + BingX hedge positions
python hedge_mode.py \
  --exchange grvt_bingx \
  --ticker BTC \
  --size 0.05 \
  --iter 1 \
  --position-close
```

**What it does**:
- Reads current positions on both exchanges
- Places limit OPEN orders to close positions:
  - If long on GRVT, sells on GRVT
  - If short on BingX, buys on BingX
- Waits for fills to unwind hedge

### Example 9: Programmatic Usage

```python
import asyncio
from decimal import Decimal
from trading_bot import TradingBot, TradingConfig

async def run_bot():
    # Create configuration
    config = TradingConfig(
        ticker='ETH',
        contract_id='',  # Auto-populated by bot
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
    
    # Create and run bot
    bot = TradingBot(config)
    
    try:
        await bot.run()
    except KeyboardInterrupt:
        print("Bot stopped by user")
        await bot.graceful_shutdown("User interruption")
    except Exception as e:
        print(f"Bot error: {e}")
        await bot.graceful_shutdown(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(run_bot())
```

### Example 10: Custom Exchange Client

```python
from exchanges.base import BaseExchangeClient, OrderResult, OrderInfo
from exchanges import ExchangeFactory
from decimal import Decimal
from typing import List, Optional

class MyCustomExchange(BaseExchangeClient):
    """Custom exchange implementation."""
    
    def _validate_config(self):
        required = ['api_key', 'api_secret']
        for key in required:
            if key not in self.config:
                raise ValueError(f"Missing required config: {key}")
    
    async def connect(self):
        # Implement connection logic
        print("Connecting to custom exchange...")
    
    async def disconnect(self):
        # Implement disconnection logic
        print("Disconnecting from custom exchange...")
    
    async def place_open_order(
        self,
        contract_id: str,
        quantity: Decimal,
        direction: str
    ) -> OrderResult:
        # Implement order placement
        return OrderResult(
            success=True,
            order_id="custom_order_123",
            side=direction,
            size=quantity,
            price=Decimal('2000.00'),
            status='OPEN'
        )
    
    # ... implement other required methods ...
    
    def get_exchange_name(self) -> str:
        return "custom"

# Register and use
ExchangeFactory.register_exchange('custom', MyCustomExchange)

config = {'api_key': 'key', 'api_secret': 'secret', 'ticker': 'ETH'}
exchange = ExchangeFactory.create_exchange('custom', config)
```

---

## Error Handling

### Exception Hierarchy

The bot uses several error handling strategies:

1. **Retry Mechanism**: `query_retry` decorator for transient failures
2. **Graceful Shutdown**: `graceful_shutdown()` for clean termination
3. **Notification System**: Alerts via Telegram/Lark for critical errors
4. **Comprehensive Logging**: All errors logged to activity log

### Common Error Scenarios

#### Connection Errors

```python
try:
    await exchange_client.connect()
except ConnectionError as e:
    logger.log(f"Failed to connect: {e}", "ERROR")
    # Retry logic or notify user
```

#### Order Placement Failures

```python
result = await exchange_client.place_open_order(...)
if not result.success:
    logger.log(f"Order failed: {result.error_message}", "ERROR")
    # Handle failure (skip, retry, or shutdown)
```

#### Position Mismatch Detection

The bot automatically detects position mismatches:

```python
if abs(position_amt - active_close_amount) > (2 * quantity):
    error_message = "Position mismatch detected!"
    logger.log(error_message, "ERROR")
    await bot.send_notification(error_message)
    await bot.graceful_shutdown("Position mismatch")
```

**User action required**: Manually reconcile positions and restart bot

#### WebSocket Disconnections

Exchange clients should handle WebSocket disconnections:

```python
async def handle_websocket_disconnect():
    logger.log("WebSocket disconnected, reconnecting...", "WARNING")
    await asyncio.sleep(5)
    await self.connect()
```

### Best Practices

1. **Always use try/except** around critical operations
2. **Log errors** with full context (order IDs, prices, quantities)
3. **Graceful shutdown** on unrecoverable errors
4. **Notify user** for critical issues (position mismatch, stop price, etc.)
5. **Retry transient failures** with exponential backoff
6. **Clean up resources** in `finally` blocks

### Error Recovery Example

```python
from exchanges.base import query_retry

@query_retry(
    default_return=None,
    exception_type=(ConnectionError, TimeoutError),
    max_attempts=3,
    min_wait=2,
    max_wait=10,
    reraise=False
)
async def place_order_with_retry():
    """Place order with automatic retry on connection errors."""
    result = await exchange_client.place_open_order(
        contract_id=contract_id,
        quantity=quantity,
        direction=direction
    )
    
    if not result.success:
        raise ConnectionError(f"Order failed: {result.error_message}")
    
    return result

# Usage
result = await place_order_with_retry()
if result is None:
    # All retries failed
    logger.log("Failed to place order after retries", "ERROR")
    await bot.send_notification("Critical: Unable to place orders")
```

---

## Advanced Topics

### Custom Trading Strategies

Extend `TradingBot` to implement custom strategies:

```python
from trading_bot import TradingBot, TradingConfig

class CustomStrategyBot(TradingBot):
    """Custom trading bot with modified strategy."""
    
    def __init__(self, config: TradingConfig):
        super().__init__(config)
        self.custom_state = {}
    
    async def _place_and_monitor_open_order(self) -> bool:
        """Override with custom order logic."""
        # Custom pre-order checks
        if not await self._custom_signal_check():
            return False
        
        # Call parent implementation
        return await super()._place_and_monitor_open_order()
    
    async def _custom_signal_check(self) -> bool:
        """Custom signal logic."""
        # Implement your trading signal
        return True
```

### Multi-Pair Trading

Run multiple bots for different pairs:

```python
import asyncio
from trading_bot import TradingBot, TradingConfig
from decimal import Decimal

async def run_multi_pair():
    # ETH bot
    eth_config = TradingConfig(
        ticker='ETH',
        exchange='edgex',
        quantity=Decimal('0.1'),
        # ... other params
    )
    eth_bot = TradingBot(eth_config)
    
    # BTC bot
    btc_config = TradingConfig(
        ticker='BTC',
        exchange='edgex',
        quantity=Decimal('0.05'),
        # ... other params
    )
    btc_bot = TradingBot(btc_config)
    
    # Run concurrently
    await asyncio.gather(
        eth_bot.run(),
        btc_bot.run()
    )

asyncio.run(run_multi_pair())
```

### Performance Optimization

1. **Batch operations** when possible
2. **Use WebSocket** for real-time data instead of polling
3. **Minimize API calls** with local order book management
4. **Connection pooling** for HTTP requests
5. **Async I/O** for all network operations

### Security Best Practices

1. **Never commit `.env` files** to version control
2. **Use API keys with minimal permissions** (only trading, no withdrawals)
3. **Rotate API keys regularly**
4. **Monitor for unusual activity** via logs
5. **Validate all inputs** before API calls
6. **Use IP whitelisting** on exchange API keys when available

---

## Troubleshooting

### Common Issues

#### Issue: Bot not placing orders

**Symptoms**: Bot connects but doesn't place any orders

**Possible causes**:
1. `max_orders` limit reached
2. `grid_step` condition not met
3. `pause_price` triggered
4. Insufficient account balance

**Solution**:
```bash
# Check logs
tail -f logs/edgex_ETH_activity.log

# Check position and orders
# Look for "Current Position" log messages
```

#### Issue: Position mismatch warning

**Symptoms**: Error message about position mismatch

**Cause**: Discrepancy between actual position and close orders

**Solution**:
1. Stop the bot
2. Manually check positions on exchange
3. Manually close or adjust positions
4. Restart bot

#### Issue: Orders not filling

**Symptoms**: Orders placed but never filled

**Possible causes**:
1. Price too far from market
2. Low market liquidity
3. Market moving against orders

**Solution**:
- Adjust order placement logic
- Use boost mode for faster fills
- Reduce `wait_time` for more frequent orders

#### Issue: WebSocket disconnections

**Symptoms**: Frequent reconnection messages

**Possible causes**:
1. Network instability
2. Exchange API issues
3. Rate limiting

**Solution**:
- Check network connection
- Implement exponential backoff for reconnections
- Contact exchange support if persistent

---

## API Version History

### Version 1.0 (Current)
- Initial comprehensive API documentation
- Support for 9 exchanges
- Standard and hedge mode trading
- Notification system (Telegram, Lark)
- Grid step control
- Stop/pause price features
- ROI-based exit strategies
- Boost mode for volume

---

## Contributing

To add a new exchange:

1. Create new file in `exchanges/` (e.g., `exchanges/myexchange.py`)
2. Implement `BaseExchangeClient` interface
3. Add to `ExchangeFactory._registered_exchanges`
4. Add environment variables to `.env` example
5. Update documentation
6. Add tests

Example template:

```python
from exchanges.base import BaseExchangeClient, OrderResult, OrderInfo
from decimal import Decimal
from typing import List, Optional

class MyExchangeClient(BaseExchangeClient):
    def _validate_config(self):
        # Validate required config
        pass
    
    async def connect(self):
        # Establish connections
        pass
    
    async def disconnect(self):
        # Close connections
        pass
    
    # Implement all other required methods...
```

---

## Support & Resources

- **Documentation**: This file and `/docs` folder
- **Issues**: GitHub Issues
- **Twitter**: [@yourQuantGuy](https://x.com/yourQuantGuy)

---

## License

This project is licensed under a non-commercial license. See [LICENSE](LICENSE) file for details.

**Important**: This software is for personal learning and research only. Commercial use is prohibited without explicit permission.

---

## Disclaimer

This software is for educational purposes only. Cryptocurrency trading involves significant risk and can result in substantial financial losses. Use at your own risk and never trade with funds you cannot afford to lose.

The developers and contributors are not responsible for any trading losses or damages resulting from the use of this software.

---

**End of API Documentation**
