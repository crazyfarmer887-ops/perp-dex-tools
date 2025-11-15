from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP, localcontext
from types import SimpleNamespace
from typing import List, Optional
import unittest
import importlib.util

BASE_MODULE_PATH = Path(__file__).parent.parent / "exchanges" / "base.py"
spec = importlib.util.spec_from_file_location("exchanges.base", BASE_MODULE_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Unable to load BaseExchangeClient from {BASE_MODULE_PATH}")
base_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base_module)

BaseExchangeClient = base_module.BaseExchangeClient
OrderInfo = base_module.OrderInfo
OrderResult = base_module.OrderResult


class DummyClient(BaseExchangeClient):
    """Concrete stub to exercise BaseExchangeClient helpers."""

    def __init__(self, tick_size):
        super().__init__(SimpleNamespace(tick_size=Decimal(str(tick_size))))

    def _validate_config(self) -> None:
        pass

    async def connect(self) -> None:  # pragma: no cover
        raise NotImplementedError

    async def disconnect(self) -> None:  # pragma: no cover
        raise NotImplementedError

    async def place_open_order(self, contract_id: str, quantity: Decimal, direction: str) -> OrderResult:  # pragma: no cover
        raise NotImplementedError

    async def place_close_order(self, contract_id: str, quantity: Decimal, price: Decimal, side: str) -> OrderResult:  # pragma: no cover
        raise NotImplementedError

    async def cancel_order(self, order_id: str) -> OrderResult:  # pragma: no cover
        raise NotImplementedError

    async def get_order_info(self, order_id: str) -> Optional[OrderInfo]:  # pragma: no cover
        raise NotImplementedError

    async def get_active_orders(self, contract_id: str) -> List[OrderInfo]:  # pragma: no cover
        raise NotImplementedError

    async def get_account_positions(self) -> Decimal:  # pragma: no cover
        raise NotImplementedError

    def setup_order_update_handler(self, handler) -> None:  # pragma: no cover
        raise NotImplementedError

    def get_exchange_name(self) -> str:  # pragma: no cover
        return "dummy"


class RoundToTickTests(unittest.TestCase):

    def test_round_to_tick_power_of_ten(self):
        client = DummyClient('0.01')
        result = client.round_to_tick(Decimal('123.456'))
        self.assertEqual(result, Decimal('123.46'))

    def test_round_to_tick_non_power_of_ten(self):
        tick = Decimal('0.7943282347242815020659182831')
        client = DummyClient(tick)
        price = Decimal('96070.6')

        result = client.round_to_tick(price)

        with localcontext() as ctx:
            ctx.prec = 60
            steps = (price / tick).quantize(Decimal('1'), rounding=ROUND_HALF_UP)
            expected = steps * tick

        self.assertEqual(result, expected)

    def test_round_to_tick_invalid_tick_returns_input(self):
        client = DummyClient('0.01')
        client.config.tick_size = Decimal('0')
        price = Decimal('42.4242')

        self.assertEqual(client.round_to_tick(price), price)


if __name__ == "__main__":
    unittest.main()
