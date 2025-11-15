"""
Base exchange client interface.
All exchange implementations should inherit from this class.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Type, Union
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from tenacity import RetryCallState, retry, retry_if_exception_type, stop_after_attempt, wait_exponential


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
    """Base class for all exchange clients."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the exchange client with configuration."""
        self.config = config
        self._validate_config()

    def round_to_tick(self, price) -> Decimal:
        try:
            price = Decimal(str(price))
        except (ValueError, TypeError, InvalidOperation):
            raise ValueError(f"Invalid price value for rounding: {price}")

        tick = getattr(self.config, 'tick_size', None)
        if tick is None:
            return price
        
        try:
            tick = Decimal(str(tick))
        except (ValueError, TypeError, InvalidOperation):
            # Fallback to default tick size if config tick_size is invalid
            tick = Decimal('0.01')
        
        if tick <= 0:
            # Invalid tick size, return price as-is
            return price
        
        try:
            # quantize forces price to be a multiple of tick
            # Normalize tick to ensure it has proper precision for quantize
            return price.quantize(tick, rounding=ROUND_HALF_UP)
        except InvalidOperation:
            # If quantize fails, try to normalize the tick size
            # Use Decimal's as_tuple() to get the exponent (precision)
            try:
                tick_tuple = tick.as_tuple()
                if tick_tuple.exponent is not None:
                    # Calculate decimal places from exponent
                    # Exponent is negative for decimal places (e.g., -2 means 2 decimal places)
                    decimal_places = abs(tick_tuple.exponent)
                    # Limit to reasonable precision (max 8 decimal places)
                    decimal_places = min(decimal_places, 8)
                else:
                    decimal_places = 2  # Default fallback
            except (AttributeError, TypeError):
                # Fallback: try to extract from string representation
                tick_str = str(tick).rstrip('0').rstrip('.')
                if '.' in tick_str:
                    decimal_places = min(len(tick_str.split('.')[1]), 8)
                else:
                    decimal_places = 2
            
            # Use a normalized tick based on decimal places
            if decimal_places > 0:
                normalized_tick = Decimal('1') / (Decimal('10') ** Decimal(str(decimal_places)))
            else:
                normalized_tick = Decimal('1')
            
            try:
                return price.quantize(normalized_tick, rounding=ROUND_HALF_UP)
            except InvalidOperation:
                # Last resort: round to reasonable precision (2 decimal places)
                return price.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

    @abstractmethod
    def _validate_config(self) -> None:
        """Validate the exchange-specific configuration."""
        pass

    @abstractmethod
    async def connect(self) -> None:
        """Connect to the exchange (WebSocket, etc.)."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the exchange."""
        pass

    @abstractmethod
    async def place_open_order(self, contract_id: str, quantity: Decimal, direction: str) -> OrderResult:
        """Place an open order."""
        pass

    @abstractmethod
    async def place_close_order(self, contract_id: str, quantity: Decimal, price: Decimal, side: str) -> OrderResult:
        """Place a close order."""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> OrderResult:
        """Cancel an order."""
        pass

    @abstractmethod
    async def get_order_info(self, order_id: str) -> Optional[OrderInfo]:
        """Get order information."""
        pass

    @abstractmethod
    async def get_active_orders(self, contract_id: str) -> List[OrderInfo]:
        """Get active orders for a contract."""
        pass

    @abstractmethod
    async def get_account_positions(self) -> Decimal:
        """Get account positions."""
        pass

    @abstractmethod
    def setup_order_update_handler(self, handler) -> None:
        """Setup order update handler for WebSocket."""
        pass

    @abstractmethod
    def get_exchange_name(self) -> str:
        """Get the exchange name."""
        pass
