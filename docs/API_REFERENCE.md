# API Reference

This document explains how to work with the public entry points, configuration objects, helpers, exchange clients, and hedge-mode components that make up the modular perp-dex trading bot. It complements the quick-start material in `README*.md` by diving into the Python APIs and showing how components fit together.

---

## Command-Line Entry Points

### `runbot.py`

`runbot.py` is the maker/taker volume bot for a single exchange. The script parses all trading parameters, validates boost-mode usage, loads `.env` files, and launches `TradingBot`.

```17:53:runbot.py
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Modular Trading Bot - Supports multiple exchanges')

    # Exchange selection
    parser.add_argument('--exchange', type=str, default='edgex',
                        choices=ExchangeFactory.get_supported_exchanges(),
                        help='Exchange to use (default: edgex). '
                             f'Available: {", ".join(ExchangeFactory.get_supported_exchanges())}')

    # Trading parameters
    parser.add_argument('--ticker', type=str, default='ETH',
                        help='Ticker (default: ETH)')
    parser.add_argument('--quantity', type=Decimal, default=Decimal(0.1),
                        help='Order quantity (default: 0.1)')
    parser.add_argument('--take-profit', type=Decimal, default=Decimal(0.02),
                        help='Take profit in USDT (default: 0.02)')
    parser.add_argument('--direction', type=str, default='buy', choices=['buy', 'sell'],
                        help='Direction of the bot (default: buy)')
    parser.add_argument('--max-orders', type=int, default=40,
                        help='Maximum number of active orders (default: 40)')
    parser.add_argument('--wait-time', type=int, default=450,
                        help='Wait time between orders in seconds (default: 450)')
    parser.add_argument('--env-file', type=str, default=".env",
                        help=".env file path (default: .env)")
    parser.add_argument('--grid-step', type=str, default='-100',
                        help='The minimum distance in percentage to the next close order price (default: -100)')
    parser.add_argument('--stop-price', type=Decimal, default=-1,
                        help='Price to stop trading and exit. Buy: exits if price >= stop-price.'
                        'Sell: exits if price <= stop-price. (default: -1, no stop)')
    parser.add_argument('--pause-price', type=Decimal, default=-1,
                        help='Pause trading and wait. Buy: pause if price >= pause-price.'
                        'Sell: pause if price <= pause-price. (default: -1, no pause)')
    parser.add_argument('--boost', action='store_true',
                        help='Use the Boost mode for volume boosting')
```

- **Boost mode** only works for `aster` and `backpack`. The script fails fast for other exchanges to avoid unwanted taker fees.
- `--env-file` allows you to keep multiple credential sets side-by-side; the script verifies the file exists before starting.
- Logging defaults to `WARNING` but can be changed inside `setup_logging`.

Common usage patterns:

```bash
# EdgeX ETH with grid spacing and stop/pause controls
python runbot.py --exchange edgex --ticker ETH --quantity 0.1 \
  --take-profit 0.02 --max-orders 40 --wait-time 450 \
  --grid-step 0.5 --stop-price 5500 --pause-price 5200

# Backpack boost mode (maker open, taker close)
python runbot.py --exchange backpack --ticker ETH --direction buy \
  --quantity 0.1 --boost
```

### `hedge_mode.py`

`hedge_mode.py` is the entry point for cross-exchange hedge loops (maker on one venue, taker on another). It loads `.env`, validates the requested exchange (`backpack`, `extended`, `apex`, `grvt`, `grvt_bingx`, or `edgex`), imports the corresponding `HedgeBot`, and launches it asynchronously.

- Shared CLI flags:
  - `--ticker`, `--size`, `--iter`, `--fill-timeout`, `--sleep`
  - Optional ROI-based waits: `--tp-roi`, `--sl-roi`
  - `--position-close` (grvt_bingx only) submits limit OPEN orders on both venues to flatten latent positions
- Example:

```bash
python hedge_mode.py --exchange grvt_bingx --ticker BTC --size 0.05 \
  --iter 20 --tp-roi 0.4 --sl-roi 0.2 --sleep 5
```

---

## Core Trading Loop (`trading_bot.py`)

### `TradingConfig` and `OrderMonitor`

`TradingConfig` gathers all runtime parameters. The dataclass exposes a helper property `close_order_side` so strategies do not have to recompute the opposite side.

```19:55:trading_bot.py
@dataclass
class TradingConfig:
    """Configuration class for trading parameters."""
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

    @property
    def close_order_side(self) -> str:
        """Get the close order side based on bot direction."""
        return 'buy' if self.direction == "sell" else 'sell'
```

`OrderMonitor` tracks fills/cancellations across threads so the async trading loop knows when to place the next close order. Use it if you build custom strategies on top of the trading bot.

### `TradingBot`

Key public methods:

- `__init__(config: TradingConfig)` wires up logging, instantiates an exchange via `ExchangeFactory`, and installs websocket handlers.
- `run()` is the async trading loop: connects to the exchange, refreshes contract metadata, enforces grid spacing, handles pause/stop levels, and manages open/close orders (or boost-mode taker closes).
- `graceful_shutdown(reason: str)` disconnects websocket/REST clients and prevents new orders.
- `send_notification(message: str)` autodetects `LARK_TOKEN` and/or `TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID` so telemetry can be fanned out to chat apps.

Basic programmatic usage:

```python
from decimal import Decimal
import asyncio
from trading_bot import TradingConfig, TradingBot
from exchanges import ExchangeFactory

config = TradingConfig(
    ticker="ETH",
    contract_id="",
    quantity=Decimal("0.2"),
    take_profit=Decimal("0.02"),
    tick_size=Decimal("0"),
    direction="buy",
    max_orders=20,
    wait_time=300,
    exchange="extended",
    grid_step=Decimal("0.3"),
    stop_price=Decimal("2800"),
    pause_price=Decimal("-1"),
    boost_mode=False,
)

async def main():
    bot = TradingBot(config)
    await bot.run()

asyncio.run(main())
```

Implementation highlights:

- `_setup_websocket_handlers` calls `exchange_client.setup_order_update_handler` and pushes every fill/cancel into `TradingLogger` CSVs.
- `_place_and_monitor_open_order` handles retries, cancels, and partial fills. In boost-mode, close orders are immediate takers via `place_market_order`.
- `_meet_grid_step_condition` protects profitability by spacing close orders relative to best bid/ask.
- `_check_price_condition` enforces `--stop-price` (shutdown) and `--pause-price` (temporary delay).

---

## Logging and Notification Helpers (`helpers/`)

- `helpers/logger.py::TradingLogger` creates `logs/{exchange}_{ticker}*_orders.csv` and `*_activity.log`, supports timezone overrides via `TIMEZONE`, and offers `log()` plus `log_transaction()` for CSV rows.
- `helpers/telegram_bot.py::TelegramBot` is a context manager around the Telegram Bot API. Use `send_text(content, parse_mode="HTML")` to push alerts. Provide `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`.
- `helpers/lark_bot.py::LarkBot` is an async context manager for Feishu/Lark robot webhooks. Provide `LARK_TOKEN` (and optionally `base_url`) and call `await send_text(message)` to notify.

Example notification pattern inside custom code:

```python
from helpers.telegram_bot import TelegramBot

with TelegramBot(token=os.environ["TELEGRAM_BOT_TOKEN"],
                 chat_id=os.environ["TELEGRAM_CHAT_ID"]) as bot:
    bot.send_text("Trading loop restarted ✅")
```

---

## Exchange Abstraction Layer (`exchanges/`)

### Base Structures and Retry Decorator

`BaseExchangeClient` defines the contract every exchange client must satisfy. The shared dataclasses and retry helper live in `exchanges/base.py`.

```13:125:exchanges/base.py
def query_retry(
    default_return: Any = None,
    exception_type: Union[Type[Exception], Tuple[Type[Exception], ...]] = (Exception,),
    max_attempts: int = 5,
    min_wait: float = 1,
    max_wait: float = 10,
    reraise: bool = False
):
    def retry_error_callback(retry_state: RetryCallState):
        print(f"Operation: [{retry_state.fn.__name__}] failed after {retry_state.attempt_number} retries, "
              f"exception: {str(retry_state.outcome.exception())}")
        return default_return

    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception_type(exception_type),
        retry_error_callback=retry_error_callback,
        reraise=reraise
    )

@dataclass
class OrderResult:
    """Standardized order result structure."""
    success: bool
    order_id: Optional[str] = None
    side: Optional[str] = None
    size: Optional[Decimal] = None
    price: Optional[Decimal] = None
    status: Optional[str] = None
    error_message: Optional[str] = None
    filled_size: Optional[Decimal] = None

@dataclass
class OrderInfo:
    """Standardized order information structure."""
    order_id: str
    side: str
    size: Decimal
    price: Decimal
    status: str
    filled_size: Decimal = 0.0
    remaining_size: Decimal = 0.0
    cancel_reason: str = ''

class BaseExchangeClient(ABC):
    ...
```

- Decorate network-bound methods with `@query_retry` to automatically back off using Tenacity.
- `OrderResult` travels through the trading bot whenever orders are placed, canceled, or filled.
- `OrderInfo` powers periodic status logs and grid-spacing checks.

Every concrete client implements:

1. Connection lifecycle: `_validate_config`, `connect`, `disconnect`, `setup_order_update_handler`, `get_exchange_name`
2. Order workflow: `place_open_order`, `place_close_order`, `cancel_order`, `place_market_order` (when supported), `get_order_info`, `get_active_orders`
3. Market/account data: `fetch_bbo_prices`, `get_account_positions`, `get_contract_attributes`, helper utilities such as `round_to_tick`

### Exchange Factory

`ExchangeFactory` lazily imports clients and exposes `get_supported_exchanges()` for CLI validation.

```10:82:exchanges/factory.py
class ExchangeFactory:
    """Factory class for creating exchange clients."""

    _registered_exchanges = {
        'edgex': 'exchanges.edgex.EdgeXClient',
        'backpack': 'exchanges.backpack.BackpackClient',
        'paradex': 'exchanges.paradex.ParadexClient',
        'aster': 'exchanges.aster.AsterClient',
        'lighter': 'exchanges.lighter.LighterClient',
        'grvt': 'exchanges.grvt.GrvtClient',
        'extended': 'exchanges.extended.ExtendedClient',
        'apex': 'exchanges.apex.ApexClient',
        'bingx': 'exchanges.bingx.BingxClient',
    }

    @classmethod
    def create_exchange(cls, exchange_name: str, config: Dict[str, Any]) -> BaseExchangeClient:
        ...

    @classmethod
    def get_supported_exchanges(cls) -> list:
        """Get list of supported exchanges."""
        return list(cls._registered_exchanges.keys())

    @classmethod
    def register_exchange(cls, name: str, exchange_class: type) -> None:
        ...
```

To add a new venue, implement `BaseExchangeClient`, update the registry, and follow `docs/ADDING_EXCHANGES.md`.

### Exchange Implementations

Each client lives in `exchanges/{name}.py` and follows the same flow: validate env vars, instantiate SDKs, wire websocket updates, implement order placement with maker-friendly pricing, and convert SDK payloads into `OrderInfo/OrderResult`.

| Client | File | SDK / Dependencies | Required Environment | Highlights & Usage |
| --- | --- | --- | --- | --- |
| `EdgeXClient` | `exchanges/edgex.py` | `edgex_sdk` | `EDGEX_ACCOUNT_ID`, `EDGEX_STARK_PRIVATE_KEY`, optional `EDGEX_BASE_URL`, `EDGEX_WS_URL` | Auto-reconnecting private WebSocket with `trade-event` handler, POST_ONLY limit orders, `python runbot.py --exchange edgex ...` |
| `BackpackClient` | `exchanges/backpack.py` | `bpx` SDK + custom `BackpackWebSocketManager` | `BACKPACK_PUBLIC_KEY`, `BACKPACK_SECRET_KEY` (base64) | Handles ED25519 auth for websocket, includes `place_market_order` for boost-mode close legs, boost mode is supported |
| `AsterClient` | `exchanges/aster.py` | REST (`aiohttp`) + websockets | `ASTER_API_KEY`, `ASTER_SECRET_KEY` | Binance-style listen keys with keepalive, ping health checks, maker/taker price adjustments |
| `ApexClient` | `exchanges/apex.py` | `apexomni` REST/WS | `APEX_API_KEY`, `APEX_API_KEY_PASSPHRASE`, `APEX_API_KEY_SECRET`, `APEX_OMNI_KEY_SEED`, optional `APEX_ENVIRONMENT` | HTTP and WS auto-reconnect, uses `_ApexWebSocketManager`, maker-only defaults |
| `LighterClient` | `exchanges/lighter.py` + `lighter_custom_websocket.py` | Official `lighter` SDK + custom stream | `API_KEY_PRIVATE_KEY`, `LIGHTER_ACCOUNT_INDEX`, `LIGHTER_API_KEY_INDEX` | Fully manages order book locally, supports `place_limit_order`, uses custom websocket to monitor open/close states (used both by `runbot` and hedge bots) |
| `ExtendedClient` | `exchanges/extended.py` | `x10.perpetual` Starknet SDK | `EXTENDED_API_KEY`, `EXTENDED_STARK_KEY_PUBLIC`, `EXTENDED_STARK_KEY_PRIVATE`, `EXTENDED_VAULT` | Maintains two websocket tasks (account + orderbook), tracks open orders locally due to REST lag, cancels buy orders on disconnect to avoid stale exposure |
| `ParadexClient` | `exchanges/paradex.py` | `paradex_py` | `PARADEX_L1_ADDRESS`, `PARADEX_L2_PRIVATE_KEY`, optional `PARADEX_L2_ADDRESS`, `PARADEX_ENVIRONMENT` | Uses L2 credentials only, patches HTTP client to silence noisy logs, streaming order updates via `ParadexWebsocketChannel.ORDERS` |
| `GrvtClient` | `exchanges/grvt.py` | `grvt-pysdk` (`GrvtCcxt`, `GrvtCcxtWS`) | `GRVT_TRADING_ACCOUNT_ID`, `GRVT_PRIVATE_KEY`, `GRVT_API_KEY`, optional `GRVT_ENVIRONMENT` | Handles custom websocket events to map statuses, ensures maker pricing via tick checks |
| `BingxClient` | `exchanges/bingx.py` | `ccxt.async_support` | `BINGX_API_KEY`, `BINGX_API_SECRET`, optional `BINGX_ENVIRONMENT`, tuning vars for polling | Polls REST instead of websockets, exposes helpers for TP/SL payloads, configurable limit/taker hedging via env such as `BINGX_ORDER_POLL_INTERVAL` |

**Usage tip:** as long as `config.exchange` matches one of the above names, `TradingBot` automatically instantiates the correct client. You rarely need to import clients directly unless you’re writing new hedge-mode logic.

---

## Hedge Mode Bots (`hedge/`)

Each hedge-mode module defines a `Config` shim (to satisfy exchange constructors) plus a `HedgeBot` class customized for one maker venue and one hedge venue. All bots share:

- Structured logging into `logs/{primary}_{ticker}_hedge_mode*.{txt,csv}`
- ROI-based waits (`tp_roi`, `sl_roi`)
- Graceful shutdown (`signal` handlers)
- Order book listeners for both venues to keep hedged exposure synchronized

Variants:

- `hedge_mode_bp.py`: Backpack maker + Lighter taker. Includes websocket depth streaming for Backpack so orders can be priced off live BBO.
- `hedge_mode_ext.py`: Extended + Lighter.
- `hedge_mode_apex.py`: Apex + Lighter (similar to Backpack but using `ApexClient`).
- `hedge_mode_grvt.py`: GRVT + Lighter, uses GRVT-specific websocket feed parsing.
- `hedge_mode_edgex.py`: EdgeX + Lighter.
- `hedge_mode_grvt_bingx.py`: GRVT maker + BingX hedge via `BingxClient`.

`hedge_mode_grvt_bingx.py` exposes an additional management routine that can be triggered from the CLI (`--position-close`). It flattens residual hedged positions by placing limit **open** orders on both exchanges until both accounts are neutral.

```1005:1045:hedge/hedge_mode_grvt_bingx.py
    async def close_positions_with_limit_orders(self) -> None:
        """
        Place limit OPEN orders on both GRVT and BingX to flatten existing positions.
        """
        self.logger.info(
            "🔚 Initiating GRVT+BingX limit-open position close routine (strict=%s).",
            "ON" if self.strict_mode else "OFF"
        )

        if self.grvt_client is None or self.bingx_client is None:
            self.initialize_clients()

        if self.grvt_contract_id is None or self.bingx_contract_id is None:
            try:
                await self.load_contract_metadata()
            except Exception as exc:
                self.logger.error(f"Unable to load contract metadata for position close: {exc}")
                return

        tolerance = self.position_tolerance
        pending_close = False
        previous_positions: Optional[Tuple[Decimal, Decimal]] = None
        last_submission_time: Optional[float] = None
        start_time = time.time()

        while not self.stop_flag:
            grvt_position, bingx_position = await self._fetch_signed_positions()
            ...
```

Each bot’s constructor accepts:

- `ticker`, `order_quantity`, `fill_timeout`, `iterations`, `sleep_time`
- Optional ROI controls (`tp_roi`, `sl_roi`)
- Venue-specific knobs (e.g., BingX hedge order type, time-in-force, attach TP/SL flags, retry delays)

Run any variant via `python hedge_mode.py --exchange <variant> ...`. Use `--position-close` for `grvt_bingx` when you want to exit both legs immediately without running a full hedging loop.

---

## Testing Utilities

`tests/test_query_retry.py` demonstrates how to validate retry behavior using async mocks and Tenacity’s instrumentation.

```18:71:tests/test_query_retry.py
@query_retry(default_return='failed')
async def success_function():
    return "success"

@query_retry(default_return="default", max_attempts=3)
async def network_error_function():
    # raise NetworkError("模拟网络错误")
    raise asyncio.TimeoutError()

@query_retry(default_return=0, exception_type=(NetworkError,))
async def business_error_function():
    raise BusinessError("业务错误")

@query_retry(default_return=None, min_wait=1, max_wait=5, exception_type=(NetworkError,))
async def timing_function():
    raise NetworkError()
```

Use `python -m pytest tests/test_query_retry.py` (or run the module directly) to ensure any new `query_retry` usage behaves as expected.

---

## Related Documentation and Files

- `docs/ADDING_EXCHANGES.md` contains a step-by-step guide for onboarding new venues, including environment variables and best practices.
- `docs/telegram-bot-setup*.md` outlines how to provision Telegram credentials referenced by `helpers/telegram_bot.py`.
- `env_example.txt` lists every environment variable the project expects; copy it to `.env` (or the file specified by `--env-file`) before running bots.

With the above references, you can compose new strategies, integrate additional exchanges, or customize the hedge bots while reusing the existing abstractions for logging, notifications, and order management.
