from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Awaitable, Callable, Optional, Tuple


FetchBBOCallable = Callable[[], Awaitable[Tuple[Decimal, Decimal]]]
StopConditionCallable = Callable[[], bool]


async def wait_for_roi_threshold(
    *,
    tp_roi: Optional[Decimal],
    sl_roi: Optional[Decimal],
    entry_side: str,
    avg_entry_price: Decimal,
    fetch_bbo_prices: FetchBBOCallable,
    logger,
    check_interval: float = 1.0,
    timeout: Optional[float] = None,
    stop_condition: Optional[StopConditionCallable] = None,
) -> Optional[str]:
    """
    Wait until either the take-profit or stop-loss ROI threshold is reached.

    Args:
        tp_roi: Take-profit ROI percentage (positive). If None, TP is ignored.
        sl_roi: Stop-loss ROI percentage (positive). If None, SL is ignored.
        entry_side: 'buy' for long entries, 'sell' for short entries.
        avg_entry_price: Weighted average entry price for the position.
        fetch_bbo_prices: Coroutine returning (best_bid, best_ask) as Decimals.
        logger: Logger with .info/.warning methods.
        check_interval: Seconds between ROI checks.
        timeout: Optional timeout in seconds. None waits indefinitely.
        stop_condition: Optional callable returning True to abort early.

    Returns:
        'tp' if take-profit threshold hit, 'sl' if stop-loss hit,
        'timeout' if timeout reached, None if aborted by stop_condition.
    """

    if tp_roi is None and sl_roi is None:
        return None

    tp_roi = tp_roi if tp_roi is not None else None
    sl_roi = sl_roi if sl_roi is not None else None
    entry_side = entry_side.lower()

    if entry_side not in ('buy', 'sell'):
        raise ValueError(f"Unsupported entry side for ROI tracking: {entry_side}")

    start_time = asyncio.get_event_loop().time()
    check_interval = max(check_interval, 0.1)
    timeout = timeout if timeout is not None and timeout > 0 else None

    while True:
        if stop_condition and stop_condition():
            logger.info("ROI wait aborted by stop condition")
            return None

        try:
            best_bid, best_ask = await fetch_bbo_prices()
        except Exception as exc:
            logger.warning(f"Failed to fetch BBO prices during ROI wait: {exc}")
            await asyncio.sleep(check_interval)
            continue

        if entry_side == 'buy':
            current_price = best_bid
            roi = ((current_price - avg_entry_price) / avg_entry_price) * Decimal('100')
        else:
            current_price = best_ask
            roi = ((avg_entry_price - current_price) / avg_entry_price) * Decimal('100')

        if tp_roi is not None and roi >= tp_roi:
            logger.info(f"Take-profit ROI reached: current {roi:.4f}% >= target {tp_roi}%")
            return 'tp'

        if sl_roi is not None and roi <= -sl_roi:
            logger.info(f"Stop-loss ROI reached: current {roi:.4f}% <= -{sl_roi}%")
            return 'sl'

        if timeout is not None and (asyncio.get_event_loop().time() - start_time) >= timeout:
            logger.warning("ROI wait timed out")
            return 'timeout'

        await asyncio.sleep(check_interval)
