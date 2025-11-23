# API Reference and Documentation

This document provides a comprehensive reference for the public APIs, functions, components, and usage of the Modular Trading Bot.

## Table of Contents

1. [Overview](#overview)
2. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
   - [Configuration](#configuration)
3. [Command Line Interface (CLI)](#command-line-interface-cli)
   - [Trading Bot (`runbot.py`)](#trading-bot-runbotpy)
   - [Hedge Mode (`hedge_mode.py`)](#hedge-mode-hedge_modepy)
4. [Core Components](#core-components)
   - [TradingBot](#tradingbot)
   - [TradingConfig](#tradingconfig)
5. [Exchange API (Extending the Bot)](#exchange-api-extending-the-bot)
   - [BaseExchangeClient](#baseexchangeclient)
   - [Data Structures](#data-structures)
   - [ExchangeFactory](#exchangefactory)
   - [Adding a New Exchange](#adding-a-new-exchange)
6. [Helper Utilities](#helper-utilities)

---

## Overview

The Modular Trading Bot is a Python-based automated trading system designed to support multiple cryptocurrency exchanges through a unified interface. It features a main trading bot logic, a hedge mode for specific strategies, and a plugin-like architecture for adding new exchanges.

## Getting Started

### Prerequisites

- Python 3.8+
- `pip` package manager

### Installation

1. Clone the repository.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   Note: Specific exchanges may require additional dependencies (e.g., `apex_requirements.txt`, `para_requirements.txt`).

### Configuration

The bot uses environment variables for sensitive configuration (API keys, secrets). Create a `.env` file in the root directory.

**Example `.env`:**

```env
# General
LARK_TOKEN=your_lark_token
TELEGRAM_BOT_TOKEN=your_telegram_token
TELEGRAM_CHAT_ID=your_chat_id

# EdgeX Exchange
EDGEX_ACCOUNT_ID=your_account_id
EDGEX_STARK_PRIVATE_KEY=your_private_key
EDGEX_BASE_URL=https://pro.edgex.exchange
EDGEX_WS_URL=wss://quote.edgex.exchange

# Other exchanges (Apex, Backpack, etc.) follow similar patterns
# check env_example.txt for more details
```

---

## Command Line Interface (CLI)

### Trading Bot (`runbot.py`)

The main entry point for the standard trading bot.

**Usage:**

```bash
python runbot.py [OPTIONS]
```

**Arguments:**

| Argument | Type | Default | Description |
|Data | | | |
| `--exchange` | `str` | `edgex` | Exchange to use. (e.g., `edgex`, `backpack`, `aster`, etc.) |
| `--ticker` | `str` | `ETH` | Trading pair ticker symbol (e.g., `BTC`, `ETH`). |
| `--quantity` | `Decimal` | `0.1` | Order quantity per trade. |
| `--take-profit` | `Decimal` | `0.02` | Take profit percentage (or value depending on logic). |
| `--direction` | `str` | `buy` | Direction of the bot: `buy` (long) or `sell` (short). |
| `--max-orders` | `int` | `40` | Maximum number of active orders allowed. |
| `--wait-time` | `int` | `450` | Cooldown time between orders in seconds. |
| `--grid-step` | `Decimal` | `-100` | Minimum distance (%) to the next close order price. |
| `--stop-price` | `Decimal` | `-1` | Price to stop trading. `-1` to disable. |
| `--pause-price` | `Decimal` | `-1` | Price to pause trading. `-1` to disable. |
| `--boost` | `flag` | `False` | Enable Boost mode (volume boosting). |
| `--env-file` | `str` | `.env` | Path to the `.env` file. |

**Example:**

```bash
python runbot.py --exchange edgex --ticker BTC --quantity 0.001 --direction buy --take-profit 0.5
```

### Hedge Mode (`hedge_mode.py`)

Entry point for the hedge mode trading strategy.

**Usage:**

```bash
python hedge_mode.py --exchange <exchange> --size <size> --iter <iterations> [OPTIONS]
```

**Arguments:**

| Argument | Type | Required | Default | Description |
|Data | | | | |
| `--exchange` | `str` | Yes | - | Exchange to use (`backpack`, `apex`, `grvt`, `edgex`, etc.). |
| `--ticker` | `str` | No | `BTC` | Ticker symbol. |
| `--size` | `str` | Yes | - | Quantity to buy/sell per order. |
| `--iter` | `int` | Yes | - | Number of iterations to run. |
| `--fill-timeout`| `int` | No | `5` | Timeout in seconds for maker order fills. |
| `--sleep` | `int` | No | `0` | Sleep time (seconds) after each step. |
| `--tp-roi` | `Decimal` | No | `None` | Target ROI % for take profit. |
| `--sl-roi` | `Decimal` | No | `None` | Target ROI % for stop loss. |
| `--position-close`| `flag` | No | `False`| (grvt_bingx only) Close existing hedge positions. |

**Example:**

```bash
python hedge_mode.py --exchange edgex --ticker BTC --size 0.001 --iter 10
```

---

## Core Components

### `TradingBot`

`trading_bot.py`

The `TradingBot` class orchestrates the trading logic. It initializes the exchange client, monitors the connection, handles order updates via WebSocket, and executes the trading strategy (open/close orders).

**Key Methods:**

- `run()`: The main async loop. Connects to the exchange and manages the trading cycle.
- `graceful_shutdown(reason)`: Safely disconnects and stops the bot.
- `_place_and_monitor_open_order()`: Places an entry order and waits for fill.
- `_handle_order_result(order_result)`: Logic for when an order is filled (placing TP orders).

### `TradingConfig`

`trading_bot.py`

A dataclass that holds the configuration for the bot instance.

```python
@dataclass
class TradingConfig:
    ticker: str
    contract_id: str
    quantity: Decimal
    take_profit: Decimal
    tick_size: Decimal
    direction: str
    max_orders: int
    wait_time: int
    exchange: str
    grid_step: Decimal
    stop_price: Decimal
    pause_price: Decimal
    boost_mode: bool
```

---

## Exchange API (Extending the Bot)

To add support for a new exchange, you must implement the `BaseExchangeClient` interface.

### `BaseExchangeClient`

`exchanges/base.py`

Abstract base class that defines the methods required for an exchange integration.

**Required Methods to Implement:**

1.  **Configuration & Connection:**
    -   `_validate_config(self)`: Validate env vars or config dict.
    -   `connect(self)`: Establish WebSocket/API connections.
    -   `disconnect(self)`: Cleanup connections.

2.  **Order Management:**
    -   `place_open_order(self, contract_id, quantity, direction) -> OrderResult`
    -   `place_close_order(self, contract_id, quantity, price, side) -> OrderResult`
    -   `cancel_order(self, order_id) -> OrderResult`
    -   `get_order_info(self, order_id) -> Optional[OrderInfo]`
    -   `get_active_orders(self, contract_id) -> List[OrderInfo]`

3.  **Account & Market Data:**
    -   `get_account_positions(self) -> Decimal`
    -   `fetch_bbo_prices(self, contract_id) -> Tuple[Decimal, Decimal]`: Return (best_bid, best_ask).
    -   `get_contract_attributes(self) -> Tuple[str, Decimal]`: Return (contract_id, tick_size).

4.  **Events:**
    -   `setup_order_update_handler(self, handler)`: Register a callback for WebSocket order updates.

### Data Structures

`exchanges/base.py`

#### `OrderResult`

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

#### `OrderInfo`

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

### `ExchangeFactory`

`exchanges/factory.py`

Manages the registration and creation of exchange clients.

-   `create_exchange(exchange_name, config)`: Instantiates the requested exchange client.
-   `register_exchange(name, exchange_class)`: Dynamically register a new exchange.

### Adding a New Exchange

1.  Create a new file in `exchanges/` (e.g., `exchanges/my_exchange.py`).
2.  Create a class inheriting from `BaseExchangeClient`.
3.  Implement all abstract methods.
4.  Register the exchange in `exchanges/factory.py` (add to `_registered_exchanges` dict).

---

## Helper Utilities

### `TradingLogger`

`helpers/logger.py`

Provides standardized logging for the application.

```python
logger = TradingLogger(exchange="edgex", ticker="BTC")
logger.log("Message", level="INFO")
```

### Decorators

#### `@query_retry`

`exchanges/base.py`

A decorator to automatically retry failed API calls.

```python
@query_retry(max_attempts=5, min_wait=1, max_wait=10)
async def my_api_call():
    ...
```

### Notification Bots

Located in `helpers/`.

-   **`LarkBot`**: Sends notifications to Lark/Feishu.
-   **`TelegramBot`**: Sends notifications to Telegram.
