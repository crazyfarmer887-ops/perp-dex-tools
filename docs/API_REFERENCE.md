# API Reference

This reference explains how the composite trading system fits together, what each public API exposes, and how to run or extend the tooling. Every section links back to the authoritative implementation so you can verify behaviours directly in the source tree.

## Architecture Overview

- `trading_bot.TradingBot` owns the main asynchronous loop: it bootstraps an exchange client via `ExchangeFactory`, wires websocket callbacks, enforces grid/ROI guardrails, and dispatches notifications before shutting down gracefully.
- All exchange adapters inherit `BaseExchangeClient`, which standardises crates such as `OrderResult`, `OrderInfo`, retry semantics, and the coroutine surface the bot relies on.
- Helper utilities (`helpers.logger.TradingLogger`, `helpers.lark_bot.LarkBot`, `helpers.telegram_bot.TelegramBot`) encapsulate logging and alert fan‑out so strategies can remain exchange-agnostic.
- CLI entry points (`runbot.py`, `hedge_mode.py`) turn the library surface into runnable programs: the first for single-exchange grid trading, the second for hedging two venues.
- Optional hedge bots under `hedge/` share a `HedgeBot` façade that pairs a maker venue with a hedge venue and adds ROI-based exit controls.

## Trading Bot Core

### TradingConfig and OrderMonitor

`TradingConfig` captures all tunable parameters, while `OrderMonitor` supplies a re-usable state container for tracking fills:

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


@dataclass
class OrderMonitor:
    """Thread-safe order monitoring state."""
    order_id: Optional[str] = None
    filled: bool = False
    filled_price: Optional[Decimal] = None
    filled_qty: Decimal = 0.0

    def reset(self):
        """Reset the monitor state."""
        self.order_id = None
        self.filled = False
        self.filled_price = None
        self.filled_qty = 0.0
```

### TradingBot lifecycle

- **Construction** wires logging, instantiates the selected exchange through `ExchangeFactory`, and prepares the asynchronous state machine (order/fill events, loop handle, shutdown flag).
- **Websocket handlers** (`_setup_websocket_handlers`) fan out exchange-specific messages into a normalised dict so downstream logic can react without bespoke parsing.
- **Order placement** (`_place_and_monitor_open_order` + `_handle_order_result`) enforces maker-only entry logic, retries cancellations, and spawns take-profit or boost-mode exits.
- **Risk/health checks** (`_log_status_periodically`, `_meet_grid_step_condition`, `_check_price_condition`) block new orders until exposure, price bands, and grid spacing are acceptable.
- **Notifications** (`send_notification`) forwards anomalies to Lark and/or Telegram on demand.
- **Main loop** (`run`) sequences the above building blocks until a stop signal or critical error occurs.

```58:162:trading_bot.py
class TradingBot:
    """Modular Trading Bot - Main trading logic supporting multiple exchanges."""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = TradingLogger(config.exchange, config.ticker, log_to_console=True)

        # Create exchange client
        try:
            self.exchange_client = ExchangeFactory.create_exchange(
                config.exchange,
                config
            )
        except ValueError as e:
            raise ValueError(f"Failed to create exchange client: {e}")

        # Trading state
        self.active_close_orders = []
        self.last_close_orders = 0
        self.last_open_order_time = 0
        self.last_log_time = 0
        self.current_order_status = None
        self.order_filled_event = asyncio.Event()
        self.order_canceled_event = asyncio.Event()
        self.shutdown_requested = False
        self.loop = None

        # Register order callback
        self._setup_websocket_handlers()
```

```101:162:trading_bot.py
    def _setup_websocket_handlers(self):
        """Setup WebSocket handlers for order updates."""
        def order_update_handler(message):
            """Handle order updates from WebSocket."""
            try:
                # Check if this is for our contract
                if message.get('contract_id') != self.config.contract_id:
                    return
                ...
                if status in ['OPEN', 'PARTIALLY_FILLED', 'FILLED', 'CANCELED']:
                    self.logger.log(...)
                    self.logger.log_transaction(...)
            except Exception as e:
                self.logger.log(f"Error handling order update: {e}", "ERROR")
                self.logger.log(f"Traceback: {traceback.format_exc()}", "ERROR")

        # Setup order update handler
        self.exchange_client.setup_order_update_handler(order_update_handler)
```

```193:359:trading_bot.py
    async def _place_and_monitor_open_order(self) -> bool:
        """Place an order and monitor its execution."""
        try:
            self.order_filled_event.clear()
            self.current_order_status = 'OPEN'
            self.order_filled_amount = 0.0
            order_result = await self.exchange_client.place_open_order(
                self.config.contract_id,
                self.config.quantity,
                self.config.direction
            )
            if not order_result.success:
                return False
            if order_result.status == 'FILLED':
                return await self._handle_order_result(order_result)
            elif not self.order_filled_event.is_set():
                try:
                    await asyncio.wait_for(self.order_filled_event.wait(), timeout=10)
                except asyncio.TimeoutError:
                    pass
            return await self._handle_order_result(order_result)
        except Exception as e:
            self.logger.log(f"Error placing order: {e}", "ERROR")
            self.logger.log(f"Traceback: {traceback.format_exc()}", "ERROR")
            return False

    async def _handle_order_result(self, order_result) -> bool:
        """Handle the result of an order placement."""
        order_id = order_result.order_id
        filled_price = order_result.price
        ...
        if self.order_filled_event.is_set() or order_result.status == 'FILLED':
            if self.config.boost_mode:
                close_order_result = await self.exchange_client.place_market_order(...)
            else:
                ...
                close_order_result = await self.exchange_client.place_close_order(...)
        else:
            new_order_price = await self.exchange_client.get_order_price(self.config.direction)
            ...
            self.logger.log(f"[OPEN] [{order_id}] Cancelling order and placing a new order", "INFO")
            ...
            if self.order_filled_amount > 0:
                close_side = self.config.close_order_side
                ...
                close_order_result = await self.exchange_client.place_close_order(...)
        return False
```

```478:579:trading_bot.py
    async def send_notification(self, message: str):
        lark_token = os.getenv("LARK_TOKEN")
        if lark_token:
            async with LarkBot(lark_token) as lark_bot:
                await lark_bot.send_text(message)

        telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if telegram_token and telegram_chat_id:
            with TelegramBot(telegram_token, telegram_chat_id) as tg_bot:
                tg_bot.send_text(message)

    async def run(self):
        """Main trading loop."""
        try:
            self.config.contract_id, self.config.tick_size = await self.exchange_client.get_contract_attributes()
            ...
            while not self.shutdown_requested:
                active_orders = await self.exchange_client.get_active_orders(self.config.contract_id)
                ...
                stop_trading, pause_trading = await self._check_price_condition()
                if stop_trading:
                    ...
                    await self.graceful_shutdown(msg)
                    continue
                ...
                if not mismatch_detected:
                    wait_time = self._calculate_wait_time()
                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        meet_grid_step_condition = await self._meet_grid_step_condition()
                        if not meet_grid_step_condition:
                            await asyncio.sleep(1)
                            continue

                        await self._place_and_monitor_open_order()
                        self.last_close_orders += 1
        except KeyboardInterrupt:
            ...
        finally:
            try:
                await self.exchange_client.disconnect()
            except Exception as e:
                self.logger.log(f"Error disconnecting from exchange: {e}", "ERROR")
```

### Programmatic usage example

```python
import asyncio
from decimal import Decimal
from trading_bot import TradingBot, TradingConfig

async def main():
    config = TradingConfig(
        ticker="ETH",
        contract_id="",          # auto-populated in TradingBot.run()
        quantity=Decimal("0.1"),
        take_profit=Decimal("0.5"),
        tick_size=Decimal("0"),  # filled later
        direction="buy",
        max_orders=40,
        wait_time=300,
        exchange="edgex",
        grid_step=Decimal("0.2"),
        stop_price=Decimal("-1"),
        pause_price=Decimal("-1"),
        boost_mode=False,
    )
    bot = TradingBot(config)
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())
```

Provision the environment variables required by the selected exchange client (see the Exchange section) plus optional alert tokens (`LARK_TOKEN`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`) before running.

## CLI Entrypoints

### `runbot.py`

`runbot.py` parses CLI flags, loads `.env`, initialises logging, and hands control to `TradingBot.run()`:

```17:134:runbot.py
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Modular Trading Bot - Supports multiple exchanges')
    parser.add_argument('--exchange', type=str, default='edgex',
                        choices=ExchangeFactory.get_supported_exchanges(),
                        help='Exchange to use (default: edgex). '
                             f'Available: {", ".join(ExchangeFactory.get_supported_exchanges())}')
    ...

async def main():
    """Main entry point."""
    args = parse_arguments()
    setup_logging("WARNING")
    ...
    bot = TradingBot(config)
    try:
        await bot.run()
    except Exception as e:
        print(f"Bot execution failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
```

Example:

```bash
python runbot.py \
  --exchange edgex \
  --ticker ETH \
  --quantity 0.2 \
  --take-profit 0.4 \
  --direction buy \
  --max-orders 20 \
  --wait-time 120 \
  --grid-step 0.1 \
  --env-file .env
```

### `hedge_mode.py`

`hedge_mode.py` is the umbrella CLI for the specialised `hedge.HedgeBot` implementations. It validates the requested pair, loads environment variables, imports the matching module (Backpack+Lighter, Extended+Lighter, Apex+Lighter, GRVT+Lighter, edgeX+Lighter, or GRVT+BingX), and awaits either `run()` or `close_positions_with_limit_orders()` when supported:

```31:173:hedge_mode.py
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Hedge Mode Trading Bot Entry Point',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python hedge_mode.py --exchange backpack --ticker BTC --size 0.002 --iter 10
    ...
""")
    parser.add_argument('--exchange', type=str, required=True,
                        help='Exchange to use (backpack, extended, apex, grvt, grvt_bingx, or edgex)')
    ...

async def main():
    """Main entry point that creates and runs the appropriate hedge bot."""
    args = parse_arguments()
    ...
    HedgeBotClass = get_hedge_bot_class(args.exchange)
    ...
    bot = HedgeBotClass(...)
    if args.position_close:
        if args.exchange.lower() != 'grvt_bingx':
            ...
        await bot.close_positions_with_limit_orders()
    else:
        await bot.run()
```

Example:

```bash
python hedge_mode.py --exchange grvt_bingx --ticker BTC --size 0.05 --iter 5 --tp-roi 0.4 --sl-roi 0.2
```

## Helper Utilities

### TradingLogger

`TradingLogger` centralises CSV trade logs and rotating activity logs, automatically deriving filenames per exchange/ticker (and optionally `ACCOUNT_NAME`) and respecting `TIMEZONE` for timestamps:

```13:113:helpers/logger.py
class TradingLogger:
    """Enhanced logging with structured output and error handling."""

    def __init__(self, exchange: str, ticker: str, log_to_console: bool = False):
        self.exchange = exchange
        self.ticker = ticker
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        logs_dir = os.path.join(project_root, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        ...
        self.timezone = pytz.timezone(os.getenv('TIMEZONE', 'Asia/Shanghai'))
        self.logger = self._setup_logger(log_to_console)
    ...
    def log(self, message: str, level: str = "INFO"):
        ...

    def log_transaction(self, order_id: str, side: str, quantity: Decimal, price: Decimal, status: str):
        """Log a transaction to CSV file."""
        ...
```

### LarkBot

`helpers.lark_bot.LarkBot` is an async context manager that posts messages to Feishu/Lark webhooks using a hardened SSL context (`certifi`) and short HTTP timeouts:

```10:66:helpers/lark_bot.py
BASE_URL = "https://www.feishu.cn/flow/api/trigger-webhook/"

class LarkBot:
    def __init__(self, token: str, base_url: Optional[str]=None):
        self.token = token
        ...
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            timeout=aiohttp.ClientTimeout(total=5),
            trust_env=True
        )

    async def send_text(self, content: str) -> Dict[str, Any]:
        payload = {
            "msg_type": "text",
            "content": {"text": content}
        }
        return await self._send_message(payload)
```

### TelegramBot

`helpers.telegram_bot.TelegramBot` performs synchronous Telegram calls using `requests.Session` with `certifi` verification—ideal for notification fallbacks or manual alerts:

```10:55:helpers/telegram_bot.py
BASE_URL = "https://api.telegram.org/bot"

class TelegramBot:
    def __init__(self, token: str, chat_id: str, base_url: Optional[str] = None):
        self.token = token
        self.chat_id = chat_id
        ...
        self.session = requests.Session()
        self.session.verify = certifi.where()
        self.session.timeout = 10

    def send_text(self, content: str, parse_mode: str = "HTML") -> Dict[str, Any]:
        """Send a text message to Telegram"""
        payload = {"chat_id": self.chat_id, "text": content, "parse_mode": parse_mode}
        return self._send_message("sendMessage", payload)
```

## Exchange Abstractions

### Retry helper, result dataclasses, and base client

```13:58:exchanges/base.py
def query_retry(
    default_return: Any = None,
    exception_type: Union[Type[Exception], Tuple[Type[Exception], ...]] = (Exception,),
    max_attempts: int = 5,
    min_wait: float = 1,
    max_wait: float = 10,
    reraise: bool = False
):
    ...
    return retry(...)

@dataclass
class OrderResult:
    """Standardized order result structure."""
    success: bool
    order_id: Optional[str] = None
    ...

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
```

```61:129:exchanges/base.py
class BaseExchangeClient(ABC):
    """Base class for all exchange clients."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the exchange client with configuration."""
        self.config = config
        self._validate_config()

    def round_to_tick(self, price) -> Decimal:
        ...

    @abstractmethod
    async def connect(self) -> None:
        ...
    @abstractmethod
    async def disconnect(self) -> None:
        ...
    @abstractmethod
    async def place_open_order(...):
        ...
    @abstractmethod
    async def place_close_order(...):
        ...
    @abstractmethod
    async def cancel_order(...):
        ...
    @abstractmethod
    async def get_order_info(...):
        ...
    @abstractmethod
    async def get_active_orders(...):
        ...
    @abstractmethod
    async def get_account_positions(...):
        ...
    @abstractmethod
    def setup_order_update_handler(...):
        ...
    @abstractmethod
    def get_exchange_name(self) -> str:
        ...
```

### ExchangeFactory

```9:101:exchanges/factory.py
class ExchangeFactory:
    """Factory class for creating exchange clients."""

    _registered_exchanges = {
        'edgex': 'exchanges.edgex.EdgeXClient',
        'backpack': 'exchanges.backpack.BackpackClient',
        ...
    }

    @classmethod
    def create_exchange(cls, exchange_name: str, config: Dict[str, Any]) -> BaseExchangeClient:
        ...
        exchange_class_path = cls._registered_exchanges[exchange_name]
        exchange_class = cls._import_exchange_class(exchange_class_path)
        return exchange_class(config)

    @classmethod
    def get_supported_exchanges(cls) -> list:
        ...

    @classmethod
    def register_exchange(cls, name: str, exchange_class: type) -> None:
        ...
```

### Built-in exchange clients

Each adapter constrains the same coroutine surface but injects venue-specific authentication, websocket handling, and order semantics.

#### EdgeXClient

```17:64:exchanges/edgex.py
class EdgeXClient(BaseExchangeClient):
    """EdgeX exchange client implementation."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.account_id = os.getenv('EDGEX_ACCOUNT_ID')
        self.stark_private_key = os.getenv('EDGEX_STARK_PRIVATE_KEY')
        self.base_url = os.getenv('EDGEX_BASE_URL', 'https://pro.edgex.exchange')
        self.ws_url = os.getenv('EDGEX_WS_URL', 'wss://quote.edgex.exchange')
        if not self.account_id or not self.stark_private_key:
            raise ValueError(...)
        self.client = Client(...)
        self.ws_manager = WebSocketManager(...)
```

#### BackpackClient

```152:181:exchanges/backpack.py
class BackpackClient(BaseExchangeClient):
    """Backpack exchange client implementation."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.public_key = os.getenv('BACKPACK_PUBLIC_KEY')
        self.secret_key = os.getenv('BACKPACK_SECRET_KEY')
        if not self.public_key or not self.secret_key:
            raise ValueError(...)
        self.public_client = Public()
        self.account_client = Account(public_key=self.public_key, secret_key=self.secret_key)
```

#### ApexClient

```23:95:exchanges/apex.py
class ApexClient(BaseExchangeClient):
    """Apex exchange client implementation"""

    def __init__(self, config: Dict[str, any]):
        super().__init__(config)
        self.api_key = os.getenv('APEX_API_KEY')
        self.api_key_passphrase = os.getenv('APEX_API_KEY_PASSPHRASE')
        self.api_key_secret = os.getenv('APEX_API_KEY_SECRET')
        self.omni_key_seed = os.getenv('APEX_OMNI_KEY_SEED')
        self.environment = os.getenv('APEX_ENVIRONMENT', 'prod')
        ...
```

#### AsterClient

```323:350:exchanges/aster.py
class AsterClient(BaseExchangeClient):
    """Aster exchange client implementation."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = os.getenv('ASTER_API_KEY')
        self.secret_key = os.getenv('ASTER_SECRET_KEY')
        self.base_url = 'https://fapi.asterdex.com'
        if not self.api_key or not self.secret_key:
            raise ValueError("ASTER_API_KEY and ASTER_SECRET_KEY must be set...")
        self.logger = TradingLogger(...)
```

#### ExtendedClient

```72:113:exchanges/extended.py
class ExtendedClient(BaseExchangeClient):
    """Extended exchange client implementation."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        vault = os.getenv('EXTENDED_VAULT')
        private_key = os.getenv('EXTENDED_STARK_KEY_PRIVATE')
        public_key = os.getenv('EXTENDED_STARK_KEY_PUBLIC')
        api_key = os.getenv('EXTENDED_API_KEY')
        self.stark_account = StarkPerpetualAccount(...)
        self.perpetual_trading_client = PerpetualTradingClient(...)
        self.logger = TradingLogger(...)
```

#### ParadexClient

```49:139:exchanges/paradex.py
class ParadexClient(BaseExchangeClient):
    """Simplified Paradex exchange client - L2 credentials only."""

    def __init__(self, config: Dict[str, Any]):
        from paradex_py import Paradex
        ...
        self.l1_address = os.getenv('PARADEX_L1_ADDRESS')
        self.l2_private_key_hex = os.getenv('PARADEX_L2_PRIVATE_KEY')
        self.l2_address = os.getenv('PARADEX_L2_ADDRESS')
        self.environment = os.getenv('PARADEX_ENVIRONMENT', 'prod')
        ...
```

#### GrvtClient

```18:79:exchanges/grvt.py
class GrvtClient(BaseExchangeClient):
    """GRVT exchange client implementation."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.trading_account_id = os.getenv('GRVT_TRADING_ACCOUNT_ID')
        self.private_key = os.getenv('GRVT_PRIVATE_KEY')
        self.api_key = os.getenv('GRVT_API_KEY')
        self.environment = os.getenv('GRVT_ENVIRONMENT', 'prod')
        if not self.trading_account_id or not self.private_key or not self.api_key:
            raise ValueError(...)
        self.rest_client = GrvtCcxt(...)
```

#### LighterClient

```30:69:exchanges/lighter.py
class LighterClient(BaseExchangeClient):
    """Lighter exchange client implementation."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key_private_key = os.getenv('API_KEY_PRIVATE_KEY')
        self.account_index = int(os.getenv('LIGHTER_ACCOUNT_INDEX', '0'))
        self.api_key_index = int(os.getenv('LIGHTER_API_KEY_INDEX', '0'))
        self.base_url = "https://mainnet.zklighter.elliot.ai"
        if not self.api_key_private_key:
            raise ValueError("API_KEY_PRIVATE_KEY must be set ...")
        self.logger = TradingLogger(...)
```

#### BingxClient

```23:57:exchanges/bingx.py
class BingxClient(BaseExchangeClient):
    """BingX exchange client implementation based on ccxt async support."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = os.getenv('BINGX_API_KEY')
        self.api_secret = os.getenv('BINGX_API_SECRET')
        self.environment = os.getenv('BINGX_ENVIRONMENT', 'prod').lower()
        self.logger = TradingLogger(exchange="bingx", ticker=self.config.ticker, log_to_console=False)
        self.exchange = ccxt.bingx({...})
```

#### LighterCustomWebSocketManager

Advanced consumers can manage raw book/account streams outside the SDK using the custom websocket manager:

```13:174:exchanges/lighter_custom_websocket.py
class LighterCustomWebSocketManager:
    """Custom WebSocket manager for Lighter order updates and order book without SDK."""

    def __init__(self, config: Dict[str, Any], order_update_callback: Optional[Callable] = None):
        self.config = config
        ...
    def update_order_book(self, side: str, updates: List[Dict[str, Any]]):
        ...
    async def connect(self):
        """Connect to Lighter WebSocket using custom implementation."""
        ...
```

## Hedge Mode Bots

Each module in `hedge/` exports a `HedgeBot` tailored to a venue pair. All follow the same lifecycle: set up makers, subscribe to websockets, hedge fills, enforce ROI or timeout guards, and optionally dump CSV logs for both venues.

### Backpack + Lighter

```25:174:hedge/hedge_mode_bp.py
class HedgeBot:
    """Trading bot that places post-only orders on Backpack and hedges with market orders on Lighter."""

    def __init__(..., iterations: int = 20, ...):
        self.backpack_client = None
        self.lighter_client = None
        self.backpack_order_book = {'bids': {}, 'asks': {}}
        self.lighter_order_book = {"bids": {}, "asks": {}}
        ...
```

This implementation streams both venues (including Backpack depth) and alternates between opening and closing legs in three steps (`place_backpack_post_only_order`, `place_lighter_market_order`, ROI wait, final flatten).

### Extended + Lighter

```26:185:hedge/hedge_mode_ext.py
class HedgeBot:
    """Trading bot that places post-only orders on Extended and hedges with market orders on Lighter."""

    def __init__(...):
        self.extended_client = None
        self.extended_order_book = {'bids': {}, 'asks': {}}
        self.lighter_client = None
        ...
```

Adds Extended’s websocket snapshot reconciliation (including offset validation) before mirroring fills on Lighter.

### Apex + Lighter

```24:189:hedge/hedge_mode_apex.py
class HedgeBot:
    """Trading bot that places post-only orders on Apex and hedges with market orders on Lighter."""

    def __init__(...):
        self.apex_client = None
        self.apex_position = Decimal('0')
        self.lighter_client = None
        ...
```

### GRVT + Lighter

```26:200:hedge/hedge_mode_grvt.py
class HedgeBot:
    """Trading bot that places post-only orders on GRVT and hedges with market orders on Lighter."""

    def __init__(...):
        self.grvt_client = None
        self.grvt_position = Decimal('0')
        self.lighter_client = None
        ...
```

### edgeX + Lighter

```25:190:hedge/hedge_mode_edgex.py
class HedgeBot:
    """Trading bot that places post-only orders on edgeX and hedges with market orders on Lighter."""

    def __init__(...):
        self.edgex_client = None
        self.edgex_order_book = {"bids": {}, "asks": {}}
        self.lighter_client = None
        ...
```

### GRVT + BingX (dual venues)

```24:279:hedge/hedge_mode_grvt_bingx.py
class HedgeBot:
    """
    Hedge bot that runs maker orders on GRVT and hedges fills on BingX.
    """

    def __init__(..., bingx_order_type: Optional[str] = None, ...):
        self.grvt_client: Optional[GrvtClient] = None
        self.bingx_client: Optional[BingxClient] = None
        self.bingx_hedge_order_type = ...
        self.position_tolerance = ...
        ...
```

It can also place limit orders simultaneously on both venues to flatten leftover positions, which is triggered through `--position-close` on the CLI:

```1005:1089:hedge/hedge_mode_grvt_bingx.py
    async def close_positions_with_limit_orders(self) -> None:
        """
        Place limit OPEN orders on both GRVT and BingX to flatten existing positions.
        """
        ...
        while not self.stop_flag:
            grvt_position, bingx_position = await self._fetch_signed_positions()
            ...
            tasks = []
            if abs(grvt_position) > tolerance:
                tasks.append(self._place_grvt_limit_close(grvt_position))
            ...
            results = await asyncio.gather(*tasks, return_exceptions=True)
            ...
```

## Testing & Quality

`tests/test_query_retry.py` demonstrates how to wrap async functions with the `query_retry` decorator, mock sleep for deterministic waits, and assert retry/backoff behaviour. Use it as a how-to for adopting the decorator in new code.

```12:71:tests/test_query_retry.py
class NetworkError(Exception):
    pass

class BusinessError(Exception):
    pass

@query_retry(default_return='failed')
async def success_function():
    return "success"

@query_retry(default_return="default", max_attempts=3)
async def network_error_function():
    raise asyncio.TimeoutError()

@query_retry(default_return=0, exception_type=(NetworkError,))
async def business_error_function():
    raise BusinessError("业务错误")

@query_retry(default_return=None, min_wait=1, max_wait=5, exception_type=(NetworkError,))
async def timing_function():
    raise NetworkError()
```

Run the suite via `pytest tests/test_query_retry.py` to validate retry behaviour after modifying exchange clients.

## Example Workflows

- **Register a custom exchange** by implementing `BaseExchangeClient` and calling `ExchangeFactory.register_exchange("mydex", MyDexClient)`. `TradingBot` (and the CLI) can then target `--exchange mydex`.
- **Send notifications** whenever you hit a manual kill-switch by calling `await bot.send_notification("message")`—the helper will fan out to any configured tokens.
- **Hedge instantly**: `python hedge_mode.py --exchange backpack --ticker BTC --size 0.01 --iter 3 --tp-roi 0.2` quickly tests a handful of maker/hedge cycles with ROI guarding.
- **Automate cleanup** on GRVT+BingX accounts by invoking `python hedge_mode.py --exchange grvt_bingx --ticker BTC --size 0.05 --iter 1 --position-close`, which launches `close_positions_with_limit_orders` instead of the normal loop.

With these APIs and examples you can script new strategies, plug in exchanges, and extend the hedging toolkit without reverse-engineering the codebase.
