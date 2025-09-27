import unittest
import os
from unittest.mock import patch, AsyncMock
import asyncio
from decimal import Decimal

from exchanges.grvt import GRVTClient
from exchanges.base import OrderInfo, OrderResult

class TestGRVTClient(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.config = {
            'contract_id': 'BTC-PERP',
            'tick_size': Decimal('0.1'),
        }
        # Set dummy environment variables
        os.environ["GRVT_API_KEY"] = "test_api_key"
        os.environ["GRVT_PRIVATE_KEY"] = "test_private_key"
        os.environ["GRVT_TRADING_ACCOUNT_ID"] = "test_account_id"

    def tearDown(self):
        # Unset dummy environment variables safely
        os.environ.pop("GRVT_API_KEY", None)
        os.environ.pop("GRVT_PRIVATE_KEY", None)
        os.environ.pop("GRVT_TRADING_ACCOUNT_ID", None)

    def test_validate_config_missing_key(self):
        """
        Test that _validate_config raises ValueError if a required key is missing.
        """
        os.environ.pop("GRVT_API_KEY", None)
        with self.assertRaisesRegex(ValueError, "Missing required environment variable: GRVT_API_KEY"):
            # Instantiation calls _validate_config
            GRVTClient(self.config)

    @patch('exchanges.grvt.GrvtCcxtPro')
    async def test_connect(self, MockGrvtCcxtPro):
        """
        Test the connect method to ensure it initializes and connects the client.
        """
        # Arrange
        mock_instance = MockGrvtCcxtPro.return_value
        mock_instance.load_markets = AsyncMock()
        client = GRVTClient(self.config)

        # Act
        await client.connect()

        # Assert
        MockGrvtCcxtPro.assert_called_once()
        mock_instance.load_markets.assert_awaited_once()
        self.assertIsNotNone(client.grvt_client)

    @patch('exchanges.grvt.GrvtCcxtPro')
    async def test_place_open_order(self, MockGrvtCcxtPro):
        """
        Test placing an open limit order.
        """
        # Arrange
        mock_grvt_client = MockGrvtCcxtPro.return_value
        mock_grvt_client.fetch_ticker = AsyncMock(return_value={'last': '50000'})
        mock_grvt_client.create_order = AsyncMock(return_value={
            'id': '12345', 'side': 'buy', 'amount': 0.1, 'price': 49999.9, 'status': 'open',
            'filled': 0.0, 'remaining': 0.1
        })
        mock_grvt_client.markets = {'BTC-PERP': {'precision': {'price': '0.1'}}}

        client = GRVTClient(self.config)
        client.grvt_client = mock_grvt_client

        # Act
        result = await client.place_open_order(
            contract_id='BTC-PERP', quantity=Decimal('0.1'), direction='buy'
        )

        # Assert
        self.assertTrue(result.success)
        self.assertEqual(result.order_id, '12345')
        mock_grvt_client.create_order.assert_awaited_once()

    @patch('exchanges.grvt.GrvtCcxtPro')
    async def test_get_account_positions(self, MockGrvtCcxtPro):
        """
        Test fetching account positions.
        """
        # Arrange
        mock_grvt_client = MockGrvtCcxtPro.return_value
        mock_grvt_client.fetch_positions = AsyncMock(return_value=[
            {'symbol': 'BTC-PERP', 'contracts': '1.5', 'side': 'long'}
        ])
        client = GRVTClient(self.config)
        client.grvt_client = mock_grvt_client

        # Act
        position = await client.get_account_positions()

        # Assert
        self.assertEqual(position, Decimal('1.5'))
        mock_grvt_client.fetch_positions.assert_awaited_once_with(symbols=['BTC-PERP'])


if __name__ == '__main__':
    unittest.main()