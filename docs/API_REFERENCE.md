# API Reference and Documentation

This document provides a comprehensive reference for the Modular Trading Bot codebase, including public APIs, functions, components, and usage instructions.

## Table of Contents

1. [Overview](#overview)
2. [CLI Usage](#cli-usage)
    - [Trading Bot (`runbot.py`)](#trading-bot-runbotpy)
    - [Hedge Mode (`hedge_mode.py`)](#hedge-mode-hedge_modepy)
3. [Core Components](#core-components)
    - [TradingBot](#tradingbot)
    - [TradingConfig](#tradingconfig)
4. [Exchange Interface](#exchange-interface)
    - [BaseExchangeClient](#baseexchangeclient)
    - [ExchangeFactory](#exchangefactory)
    - [Data Structures](#data-structures)
5. [Hedge Mode Architecture](#hedge-mode-architecture)
6. [Helpers](#helpers)

---

## Overview

The Modular Trading Bot is a Python-based trading system designed to support multiple exchanges through a unified interface. It supports both a standard grid/trading bot mode and a specialized "Hedge Mode" for cross-exchange hedging strategies.

## CLI Usage

### Trading Bot (`runbot.py`)

The main entry point for running the standard trading bot.

**Usage:**
```bash
python runbot.py [options]
```

**Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--exchange` | str | `edgex` | Exchange to use (e.g., `edgex`, `backpack`, `paradex`, `aster`, `lighter`, `grvt`, `extended`, `apex`, `bingx`). |
| `--ticker` | str | `ETH` | Ticker symbol to trade. |
| `--quantity` | Decimal | `0.1` | Order quantity per trade. |
| `--take-profit` | Decimal | `0.02` | Take profit percentage. |
| `--direction` | str | `buy` | Direction of the bot (`buy` or `sell`). |
| `--max-orders` | int | `40` | Maximum number of active orders. |
| `--wait-time` | int | `450` | Wait time between orders in seconds. |
| `--env-file` | str | `.env` | Path to the `.env` file containing credentials. |
| `--grid-step` | str | `-100` | Minimum distance in percentage to the next close order price. |
| `--stop-price` | Decimal | `-1` | Price to stop trading and exit (safety mechanism). |
| `--pause-price` | Decimal | `-1` | Price to pause trading. |
| `--boost` | flag | `False` | Enable Boost mode for volume boosting (only `aster` and `backpack`). |

**Example:**
```bash
python runbot.py --exchange backpack --ticker SOL --quantity 1 --direction buy --take-profit 0.5
```

### Hedge Mode (`hedge_mode.py`)

Entry point for the hedge mode strategies.

**Usage:**
```bash
python hedge_mode.py --exchange <exchange> --size <size> --iter <iterations> [options]
```

**Arguments:**

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--exchange` | str | Yes | Exchange to use (`backpack`, `extended`, `apex`, `grvt`, `grvt_bingx`, `edgex`). |
| `--size` | str | Yes | Number of tokens to buy/sell per order. |
| `--iter` | int | Yes | Number of iterations to run. |
| `--ticker` | str | No | Ticker symbol (default: `BTC`). |
| `--fill-timeout` | int | No | Timeout in seconds for maker order fills (default: `5`). |
| `--sleep` | int | No | Sleep time in seconds after each step (default: `0`). |
| `--tp-roi` | Decimal | No | Target ROI % for take profit based on avg entry price. |
| `--sl-roi` | Decimal | No | Target ROI % for stop loss based on avg entry price. |
| `--env-file` | str | No | Path to `.env` file (default: `.env`). |
| `--position-close` | flag | No | (grvt_bingx only) Close existing hedge positions with limit orders. |

**Example:**
```bash
python hedge_mode.py --exchange backpack --ticker SOL --size 1 --iter 100
```

---

## Core Components

### TradingBot

`trading_bot.py`

The main class orchestrating the trading logic.

#### `class TradingBot(config: TradingConfig)`

**Methods:**

-   `async run()`: Starts the main trading loop.
-   `async graceful_shutdown(reason: str)`: Stops the bot and closes connections.
-   `async send_notification(message: str)`: Sends alerts via Lark or Telegram.

**Internal Logic:**
-   Connects to the exchange.
-   Monitors active orders and positions.
-   Places open orders based on logic and configuration.
-   Places close orders (take-profit) when open orders are filled.
-   Handles WebSocket updates for real-time order tracking.

### TradingConfig

`trading_bot.py`

Data class holding the configuration for `TradingBot`.

**Properties:**
-   `ticker`: Trading pair symbol.
-   `contract_id`: Exchange-specific contract identifier.
-   `quantity`: Order size.
-   `take_profit`: Take profit percentage.
-   `direction`: 'buy' or 'sell'.
-   `max_orders`: Max concurrent orders.
-   `exchange`: Exchange name.
-   `stop_price` / `pause_price`: Risk management triggers.

---

## Exchange Interface

The system uses a factory pattern to support multiple exchanges. All exchange clients inherit from `BaseExchangeClient`.

### BaseExchangeClient

`exchanges/base.py`

Abstract base class defining the standard interface for all exchanges.

#### `async connect()`
Establishes WebSocket or API connections.

#### `async disconnect()`
Closes all connections.

#### `async place_open_order(contract_id, quantity, direction) -> OrderResult`
Places an opening order (e.g., entering a position).

#### `async place_close_order(contract_id, quantity, price, side) -> OrderResult`
Places a closing order (e.g., take profit).

#### `async cancel_order(order_id) -> OrderResult`
Cancels a specific order.

#### `async get_order_info(order_id) -> OrderInfo`
Retrieves details about a specific order.

#### `async get_active_orders(contract_id) -> List[OrderInfo]`
Returns a list of currently active (open) orders.

#### `async get_account_positions() -> Decimal`
Returns the current net position for the account.

#### `setup_order_update_handler(handler)`
Registers a callback function to receive real-time order updates via WebSocket.

### ExchangeFactory

`exchanges/factory.py`

#### `create_exchange(exchange_name, config) -> BaseExchangeClient`
Factory method to instantiate the correct exchange client based on the name string.

**Supported Exchanges:**
-   `edgex`
-   `backpack`
-   `paradex`
-   `aster`
-   `lighter`
-   `grvt`
-   `extended`
-   `apex`
-   `bingx`

### Data Structures

`exchanges/base.py`

#### `OrderResult`
-   `success`: bool
-   `order_id`: str (optional)
-   `error_message`: str (optional)
-   `price`: Decimal (optional)
-   `status`: str (optional)

#### `OrderInfo`
-   `order_id`: str
-   `side`: str
-   `size`: Decimal
-   `price`: Decimal
-   `status`: str
-   `filled_size`: Decimal

---

## Hedge Mode Architecture

Hedge mode is designed to arbitrage or hedge between a "Maker" exchange (usually the one passed in `--exchange`) and a "Taker" exchange (often Lighter or BingX, hardcoded in the specific hedge implementation).

**Key Class: `HedgeBot`**
(Implementations found in `hedge/hedge_mode_*.py`)

**Typical Flow:**
1.  **Step 1:** Place a Post-Only order on the Maker exchange.
2.  **Wait:** Wait for the Maker order to fill.
3.  **Hedge:** Immediately place a Market order on the Taker exchange to hedge the position.
4.  **ROI Check:** Optionally wait for a specific Return on Investment (ROI) threshold.
5.  **Step 2:** Place a closing Post-Only order on the Maker exchange.
6.  **Hedge Close:** Fill the closing order on the Taker exchange.

**Common Methods in HedgeBot:**
-   `trading_loop()`: Main logic cycle.
-   `place_backpack_post_only_order()`: Strategy for placing maker orders.
-   `place_lighter_market_order()`: Strategy for placing taker orders.
-   `handle_backpack_order_update()`: WebSocket callback.

---

## Helpers

### TradingLogger

`helpers/logger.py`

Provides structured logging to both console and files (`logs/`).

**Usage:**
```python
logger = TradingLogger(exchange="backpack", ticker="SOL", log_to_console=True)
logger.log("Message", "INFO")
logger.log_transaction(order_id, side, quantity, price, status)
```

**Features:**
-   Timezone support.
-   Separate CSV file for transaction history.
-   Separate log file for debug logs.

### Notification Bots

`helpers/lark_bot.py` & `helpers/telegram_bot.py`

Used by `TradingBot` to send alerts (e.g., stop price triggered, position mismatch).

**Environment Variables Required:**
-   `LARK_TOKEN`
-   `TELEGRAM_BOT_TOKEN`
-   `TELEGRAM_CHAT_ID`
