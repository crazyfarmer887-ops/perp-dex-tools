#!/usr/bin/env python3
"""
Modular Trading Bot - Supports multiple exchanges
"""

import argparse
import asyncio
import logging
from pathlib import Path
import sys
import dotenv
from decimal import Decimal
from trading_bot import TradingBot, TradingConfig
from exchanges import ExchangeFactory


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
    parser.add_argument('--dual-sided', action='store_true',
                        help='Enable simultaneous buy & sell limit grids (BingX and GRVT only)')
    parser.add_argument('--tp-roi', type=Decimal, default=None,
                        help='ROI percentage relative to average fill required before placing take-profit orders')
    parser.add_argument('--sl-roi', type=Decimal, default=None,
                        help='ROI percentage relative to average fill that triggers stop-loss handling')
    parser.add_argument('--roi-poll-interval', type=float, default=2.0,
                        help='Seconds between ROI checks when ROI targets are enabled (default: 2.0)')
    parser.add_argument('--roi-max-wait', type=int, default=0,
                        help='Maximum seconds to wait for ROI targets (0 disables the timeout)')

    return parser.parse_args()


def setup_logging(log_level: str):
    """Setup global logging configuration."""
    # Convert string level to logging constant
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Clear any existing handlers to prevent duplicates
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Configure root logger WITHOUT adding a console handler
    # This prevents duplicate logs when TradingLogger adds its own console handler
    root_logger.setLevel(level)

    # Suppress websockets debug logs unless DEBUG level is explicitly requested
    if log_level.upper() != 'DEBUG':
        logging.getLogger('websockets').setLevel(logging.WARNING)

    # Suppress other noisy loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)

    # Suppress Lighter SDK debug logs
    logging.getLogger('lighter').setLevel(logging.WARNING)
    # Also suppress any root logger DEBUG messages that might be coming from Lighter
    if log_level.upper() != 'DEBUG':
        # Set root logger to WARNING to suppress DEBUG messages from Lighter SDK
        root_logger.setLevel(logging.WARNING)


async def main():
    """Main entry point."""
    args = parse_arguments()

    # Setup logging first
    setup_logging("WARNING")

    exchange_name = args.exchange.lower()

    # Validate boost-mode can only be used with aster and backpack exchange
    if args.boost and exchange_name not in {'aster', 'backpack'}:
        print(f"Error: --boost can only be used when --exchange is 'aster' or 'backpack'. "
              f"Current exchange: {args.exchange}")
        sys.exit(1)

    if args.dual_sided and exchange_name not in {'bingx', 'grvt'}:
        print("Error: --dual-sided is currently supported only for BingX or GRVT exchanges.")
        sys.exit(1)

    if args.dual_sided and args.boost:
        print("Error: --dual-sided cannot be combined with --boost mode.")
        sys.exit(1)

    if args.tp_roi is not None and args.tp_roi <= 0:
        print("Error: --tp-roi must be a positive decimal percentage.")
        sys.exit(1)

    if args.sl_roi is not None and args.sl_roi <= 0:
        print("Error: --sl-roi must be a positive decimal percentage.")
        sys.exit(1)

    if args.roi_poll_interval <= 0:
        print("Error: --roi-poll-interval must be positive.")
        sys.exit(1)

    env_path = Path(args.env_file)
    if not env_path.exists():
        print(f"Env file not find: {env_path.resolve()}")
        sys.exit(1)
    dotenv.load_dotenv(args.env_file)

    roi_max_wait = args.roi_max_wait if args.roi_max_wait > 0 else None
    directions = ['buy', 'sell'] if args.dual_sided else [args.direction.lower()]
    bots = []

    for direction in directions:
        config = TradingConfig(
            ticker=args.ticker.upper(),
            contract_id='',  # will be set in the bot's run method
            tick_size=Decimal(0),
            quantity=args.quantity,
            take_profit=args.take_profit,
            direction=direction,
            max_orders=args.max_orders,
            wait_time=args.wait_time,
            exchange=exchange_name,
            grid_step=Decimal(args.grid_step),
            stop_price=Decimal(args.stop_price),
            pause_price=Decimal(args.pause_price),
            boost_mode=args.boost,
            dual_sided=args.dual_sided,
            tp_roi=args.tp_roi,
            sl_roi=args.sl_roi,
            roi_poll_interval=args.roi_poll_interval,
            roi_max_wait=roi_max_wait
        )
        bots.append(TradingBot(config))

    # Create and run the bot(s)
    try:
        await asyncio.gather(*(bot.run() for bot in bots))
    except Exception as e:
        print(f"Bot execution failed: {e}")
        # The bot's run method already handles graceful shutdown
        return


if __name__ == "__main__":
    asyncio.run(main())
