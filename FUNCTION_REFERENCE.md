# Function Reference Guide

Quick reference for all public functions, classes, and methods in the Perp DEX Trading Bot.

---

## Table of Contents

- [TradingBot Class](#tradingbot-class)
- [TradingConfig Dataclass](#tradingconfig-dataclass)
- [Exchange Client Methods](#exchange-client-methods)
- [Helper Functions](#helper-functions)
- [Hedge Mode](#hedge-mode)
- [Utility Functions](#utility-functions)

---

## TradingBot Class

### Constructor

```python
TradingBot(config: TradingConfig)
```

Creates a new trading bot instance.

**Parameters:**
- `config`: TradingConfig object with all trading parameters

**Example:**
```python
bot = TradingBot(config)
```

---

### `async run()`

Starts the main trading loop.

**Returns:** None  
**Raises:** Exception on critical errors

**Example:**
```python
await bot.run()
```

---

### `async graceful_shutdown(reason: str = "Unknown")`

Performs graceful shutdown.

**Parameters:**
- `reason` (optional): Shutdown reason

**Example:**
```python
await bot.graceful_shutdown("Stop price reached")
```

---

### `async send_notification(message: str)`

Sends notification via Telegram/Lark.

**Parameters:**
- `message`: Notification text

**Example:**
```python
await bot.send_notification("Position mismatch!")
```

---

## TradingConfig Dataclass

### Constructor

```python
TradingConfig(
    ticker: str,
    contract_id: str,
    quantity: Decimal,
    take_profit: Decimal,
    tick_size: Decimal,
    direction: str,
    max_orders: int,
    wait_time: int,
    exchange: str,
    grid_step: Decimal,
    stop_price: Decimal,
    pause_price: Decimal,
    boost_mode: bool
)
```

**Example:**
```python
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
    stop_price=Decimal('-1'),
    pause_price=Decimal('-1'),
    boost_mode=False
)
```

---

### `close_order_side` Property

Returns opposite side for closing positions.

**Returns:** 'buy' or 'sell'

**Example:**
```python
side = config.close_order_side  # Returns 'sell' if direction is 'buy'
```

---

## Exchange Client Methods

All exchange clients inherit from `BaseExchangeClient` and implement these methods.

---

### `async connect()`

Connects to the exchange.

**Returns:** None

**Example:**
```python
await exchange_client.connect()
```

---

### `async disconnect()`

Disconnects from the exchange.

**Returns:** None

**Example:**
```python
await exchange_client.disconnect()
```

---

### `async place_open_order(contract_id: str, quantity: Decimal, direction: str) -> OrderResult`

Places an open/entry order.

**Parameters:**
- `contract_id`: Contract identifier
- `quantity`: Order size
- `direction`: 'buy' or 'sell'

**Returns:** OrderResult object

**Example:**
```python
result = await exchange_client.place_open_order(
    contract_id="ETH-PERP",
    quantity=Decimal('0.1'),
    direction='buy'
)
```

---

### `async place_close_order(contract_id: str, quantity: Decimal, price: Decimal, side: str) -> OrderResult`

Places a close/exit order.

**Parameters:**
- `contract_id`: Contract identifier
- `quantity`: Order size
- `price`: Limit price
- `side`: 'buy' or 'sell'

**Returns:** OrderResult object

**Example:**
```python
result = await exchange_client.place_close_order(
    contract_id="ETH-PERP",
    quantity=Decimal('0.1'),
    price=Decimal('2050.00'),
    side='sell'
)
```

---

### `async place_market_order(contract_id: str, quantity: Decimal, side: str) -> OrderResult`

Places a market order (for boost mode).

**Parameters:**
- `contract_id`: Contract identifier
- `quantity`: Order size
- `side`: 'buy' or 'sell'

**Returns:** OrderResult object

**Example:**
```python
result = await exchange_client.place_market_order(
    contract_id="ETH-PERP",
    quantity=Decimal('0.1'),
    side='sell'
)
```

---

### `async cancel_order(order_id: str) -> OrderResult`

Cancels an active order.

**Parameters:**
- `order_id`: Order ID to cancel

**Returns:** OrderResult object

**Example:**
```python
result = await exchange_client.cancel_order("order_123")
```

---

### `async get_order_info(order_id: str) -> Optional[OrderInfo]`

Gets order information.

**Parameters:**
- `order_id`: Order ID to query

**Returns:** OrderInfo object or None

**Example:**
```python
info = await exchange_client.get_order_info("order_123")
if info:
    print(f"Status: {info.status}")
```

---

### `async get_active_orders(contract_id: str) -> List[OrderInfo]`

Gets all active orders for a contract.

**Parameters:**
- `contract_id`: Contract identifier

**Returns:** List of OrderInfo objects

**Example:**
```python
orders = await exchange_client.get_active_orders("ETH-PERP")
for order in orders:
    print(f"{order.side} {order.size} @ {order.price}")
```

---

### `async get_account_positions() -> Decimal`

Gets current net position.

**Returns:** Position size (positive = long, negative = short)

**Example:**
```python
position = await exchange_client.get_account_positions()
print(f"Current position: {position}")
```

---

### `async get_contract_attributes() -> Tuple[str, Decimal]`

Gets contract ID and tick size.

**Returns:** Tuple of (contract_id, tick_size)

**Example:**
```python
contract_id, tick_size = await exchange_client.get_contract_attributes()
```

---

### `async fetch_bbo_prices(contract_id: str) -> Tuple[Decimal, Decimal]`

Fetches best bid and offer prices.

**Parameters:**
- `contract_id`: Contract identifier

**Returns:** Tuple of (best_bid, best_ask)

**Example:**
```python
bid, ask = await exchange_client.fetch_bbo_prices("ETH-PERP")
print(f"Bid: {bid}, Ask: {ask}")
```

---

### `async get_order_price(direction: str) -> Decimal`

Gets price for new order based on direction.

**Parameters:**
- `direction`: 'buy' or 'sell'

**Returns:** Price for the order

**Example:**
```python
price = await exchange_client.get_order_price('buy')
```

---

### `setup_order_update_handler(handler: Callable)`

Sets up WebSocket handler for order updates.

**Parameters:**
- `handler`: Callback function

**Example:**
```python
def my_handler(message):
    print(f"Order update: {message}")

exchange_client.setup_order_update_handler(my_handler)
```

---

### `round_to_tick(price: Decimal) -> Decimal`

Rounds price to tick size.

**Parameters:**
- `price`: Price to round

**Returns:** Rounded price

**Example:**
```python
rounded = exchange_client.round_to_tick(Decimal('2000.123'))
```

---

### `get_exchange_name() -> str`

Gets exchange name.

**Returns:** Exchange name string

**Example:**
```python
name = exchange_client.get_exchange_name()  # Returns 'edgex'
```

---

## ExchangeFactory Methods

### `classmethod create_exchange(exchange_name: str, config: Dict[str, Any]) -> BaseExchangeClient`

Creates an exchange client instance.

**Parameters:**
- `exchange_name`: Exchange name
- `config`: Configuration dictionary

**Returns:** Exchange client instance

**Example:**
```python
from exchanges import ExchangeFactory

exchange = ExchangeFactory.create_exchange('edgex', config)
```

---

### `classmethod get_supported_exchanges() -> List[str]`

Gets list of supported exchanges.

**Returns:** List of exchange names

**Example:**
```python
exchanges = ExchangeFactory.get_supported_exchanges()
# ['edgex', 'backpack', 'paradex', ...]
```

---

### `classmethod register_exchange(name: str, exchange_class: Type[BaseExchangeClient])`

Registers a custom exchange client.

**Parameters:**
- `name`: Exchange name
- `exchange_class`: Exchange client class

**Example:**
```python
ExchangeFactory.register_exchange('custom', MyExchangeClass)
```

---

## Helper Functions

### TradingLogger

#### Constructor

```python
TradingLogger(exchange: str, ticker: str, log_to_console: bool = False)
```

**Example:**
```python
logger = TradingLogger('edgex', 'ETH', log_to_console=True)
```

---

#### `log(message: str, level: str = "INFO")`

Logs a message.

**Parameters:**
- `message`: Message text
- `level`: 'DEBUG', 'INFO', 'WARNING', or 'ERROR'

**Example:**
```python
logger.log("Bot started", "INFO")
logger.log("Connection error", "ERROR")
```

---

#### `log_transaction(order_id: str, side: str, quantity: Decimal, price: Decimal, status: str)`

Logs a transaction to CSV.

**Parameters:**
- `order_id`: Order ID
- `side`: 'buy' or 'sell'
- `quantity`: Order quantity
- `price`: Order price
- `status`: Order status

**Example:**
```python
logger.log_transaction(
    order_id="order_123",
    side="buy",
    quantity=Decimal('0.1'),
    price=Decimal('2000.50'),
    status="FILLED"
)
```

---

### TelegramBot

#### Constructor

```python
TelegramBot(token: str, chat_id: str, base_url: Optional[str] = None)
```

**Example:**
```python
bot = TelegramBot("bot_token", "chat_id")
```

---

#### `send_text(content: str, parse_mode: str = "HTML") -> Dict[str, Any]`

Sends text message.

**Parameters:**
- `content`: Message text
- `parse_mode`: 'HTML' or 'Markdown'

**Returns:** Response dictionary

**Example:**
```python
with TelegramBot(token, chat_id) as bot:
    response = bot.send_text("<b>Alert:</b> Stop price reached!")
```

---

#### `close()`

Closes the session.

**Example:**
```python
bot.close()
```

---

### LarkBot

#### Constructor

```python
LarkBot(token: str, base_url: Optional[str] = None)
```

**Example:**
```python
bot = LarkBot("lark_token")
```

---

#### `async send_text(content: str) -> Dict[str, Any]`

Sends text message.

**Parameters:**
- `content`: Message text

**Returns:** Response dictionary

**Example:**
```python
async with LarkBot(token) as bot:
    response = await bot.send_text("Alert: Position mismatch!")
```

---

#### `async close()`

Closes the session.

**Example:**
```python
await bot.close()
```

---

## Hedge Mode

### HedgeBot

#### Constructor

```python
HedgeBot(
    ticker: str,
    order_quantity: Decimal,
    fill_timeout: int = 5,
    iterations: int = 20,
    sleep_time: int = 0,
    tp_roi: Optional[Decimal] = None,
    sl_roi: Optional[Decimal] = None
)
```

**Parameters:**
- `ticker`: Trading pair (e.g., 'BTC')
- `order_quantity`: Order size
- `fill_timeout`: Maker fill timeout (seconds)
- `iterations`: Number of trading cycles
- `sleep_time`: Sleep after each step (seconds)
- `tp_roi`: Take profit ROI %
- `sl_roi`: Stop loss ROI %

**Example:**
```python
from hedge.hedge_mode_bp import HedgeBot

bot = HedgeBot(
    ticker='BTC',
    order_quantity=Decimal('0.05'),
    fill_timeout=5,
    iterations=20,
    tp_roi=Decimal('0.4'),
    sl_roi=Decimal('0.2')
)
```

---

#### `async run()`

Executes hedge trading strategy.

**Returns:** None

**Example:**
```python
await bot.run()
```

---

#### `async cleanup()`

Cleans up resources.

**Returns:** None

**Example:**
```python
try:
    await bot.run()
finally:
    await bot.cleanup()
```

---

#### `async close_positions_with_limit_orders()` (GRVT+BingX only)

Closes existing hedge positions.

**Returns:** None

**Example:**
```python
await bot.close_positions_with_limit_orders()
```

---

## Utility Functions

### query_retry Decorator

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

Retry decorator for API calls.

**Parameters:**
- `default_return`: Return value if all retries fail
- `exception_type`: Exception types to retry
- `max_attempts`: Max retry attempts
- `min_wait`: Min wait between retries (seconds)
- `max_wait`: Max wait between retries (seconds)
- `reraise`: Whether to reraise exception

**Example:**
```python
from exchanges.base import query_retry

@query_retry(
    default_return=None,
    exception_type=(ConnectionError, TimeoutError),
    max_attempts=3
)
async def fetch_data():
    return await api_call()
```

---

## Data Classes

### OrderResult

```python
@dataclass
class OrderResult:
    success: bool
    order_id: Optional[str] = None
    side: Optional[str] = None
    size: Optional[Decimal] = None
    price: Optional[Decimal] = None
    status: Optional[str] = None
    error_message: Optional[str] = None
    filled_size: Optional[Decimal] = None
```

**Example:**
```python
if result.success:
    print(f"Order {result.order_id} @ {result.price}")
else:
    print(f"Error: {result.error_message}")
```

---

### OrderInfo

```python
@dataclass
class OrderInfo:
    order_id: str
    side: str
    size: Decimal
    price: Decimal
    status: str
    filled_size: Decimal = 0.0
    remaining_size: Decimal = 0.0
    cancel_reason: str = ''
```

**Example:**
```python
info = await exchange.get_order_info("order_123")
print(f"Status: {info.status}, Filled: {info.filled_size}")
```

---

### OrderMonitor

```python
@dataclass
class OrderMonitor:
    order_id: Optional[str] = None
    filled: bool = False
    filled_price: Optional[Decimal] = None
    filled_qty: Decimal = 0.0
```

**Methods:**
- `reset()`: Resets all fields to initial values

**Example:**
```python
monitor = OrderMonitor()
monitor.order_id = "order_123"
monitor.reset()  # Clears all fields
```

---

## Command Line Tools

### runbot.py

Main entry point for standard trading.

```bash
python runbot.py [OPTIONS]
```

**Key Options:**
- `--exchange`: Exchange name
- `--ticker`: Trading pair
- `--quantity`: Order size
- `--take-profit`: Take profit %
- `--direction`: 'buy' or 'sell'
- `--max-orders`: Max concurrent orders
- `--wait-time`: Wait between orders (seconds)
- `--grid-step`: Grid step %
- `--stop-price`: Stop price
- `--pause-price`: Pause price
- `--boost`: Enable boost mode

**Example:**
```bash
python runbot.py --exchange edgex --ticker ETH --quantity 0.1 \
  --take-profit 0.02 --max-orders 40 --wait-time 450
```

---

### hedge_mode.py

Entry point for hedge mode trading.

```bash
python hedge_mode.py [OPTIONS]
```

**Key Options:**
- `--exchange`: Exchange pair
- `--ticker`: Trading pair
- `--size`: Order size
- `--iter`: Number of iterations
- `--fill-timeout`: Maker fill timeout
- `--tp-roi`: Take profit ROI %
- `--sl-roi`: Stop loss ROI %
- `--position-close`: Close existing positions (GRVT+BingX)

**Example:**
```bash
python hedge_mode.py --exchange backpack --ticker BTC \
  --size 0.05 --iter 20 --tp-roi 0.4 --sl-roi 0.2
```

---

## Quick Reference by Use Case

### Starting a Basic Bot

```python
from trading_bot import TradingBot, TradingConfig
from decimal import Decimal

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
    grid_step=Decimal('-100'),
    stop_price=Decimal('-1'),
    pause_price=Decimal('-1'),
    boost_mode=False
)

bot = TradingBot(config)
await bot.run()
```

---

### Sending Notifications

```python
# Telegram
from helpers.telegram_bot import TelegramBot

with TelegramBot(token, chat_id) as bot:
    bot.send_text("Alert message")

# Lark
from helpers.lark_bot import LarkBot

async with LarkBot(token) as bot:
    await bot.send_text("Alert message")
```

---

### Logging

```python
from helpers import TradingLogger

logger = TradingLogger('edgex', 'ETH', log_to_console=True)
logger.log("Bot started", "INFO")
logger.log_transaction("order_123", "buy", Decimal('0.1'), 
                       Decimal('2000'), "FILLED")
```

---

### Creating Custom Exchange

```python
from exchanges.base import BaseExchangeClient
from exchanges import ExchangeFactory

class MyExchange(BaseExchangeClient):
    # Implement required methods
    pass

ExchangeFactory.register_exchange('myexchange', MyExchange)
exchange = ExchangeFactory.create_exchange('myexchange', config)
```

---

## Status Codes

### Order Status

- `OPEN`: Order placed, waiting for fill
- `FILLED`: Order completely filled
- `PARTIALLY_FILLED`: Order partially filled
- `CANCELED`: Order canceled
- `REJECTED`: Order rejected by exchange

### Direction/Side

- `buy`: Buy/long direction
- `sell`: Sell/short direction

### Log Levels

- `DEBUG`: Detailed debugging information
- `INFO`: General informational messages
- `WARNING`: Warning messages
- `ERROR`: Error messages

---

**End of Function Reference**
