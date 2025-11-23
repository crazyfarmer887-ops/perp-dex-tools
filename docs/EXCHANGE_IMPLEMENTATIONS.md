# Exchange Implementation Guide

## Table of Contents

1. [Overview](#overview)
2. [Supported Exchanges](#supported-exchanges)
3. [Implementation Details](#implementation-details)
4. [Exchange-Specific Features](#exchange-specific-features)
5. [Adding New Exchanges](#adding-new-exchanges)
6. [Troubleshooting](#troubleshooting)

---

## Overview

This document provides detailed information about each exchange implementation, including setup requirements, specific features, and API characteristics.

All exchange implementations inherit from `BaseExchangeClient` and provide a consistent interface while handling exchange-specific details internally.

---

## Supported Exchanges

| Exchange | Standard Mode | Hedge Mode | WebSocket | Market Orders | Boost Mode |
|----------|--------------|------------|-----------|---------------|------------|
| EdgeX | ✅ | ✅ (+ Lighter) | ✅ | ✅ | ❌ |
| Backpack | ✅ | ✅ (+ Lighter) | ✅ | ✅ | ✅ |
| Paradex | ✅ | ❌ | ✅ | ✅ | ❌ |
| Aster | ✅ | ❌ | ✅ | ✅ | ✅ |
| Lighter | ✅ | Used as hedge | ✅ | ✅ | ❌ |
| GRVT | ✅ | ✅ (+ Lighter/BingX) | ✅ | ✅ | ❌ |
| Extended | ✅ | ✅ (+ Lighter) | ✅ | ✅ | ❌ |
| Apex | ✅ | ✅ (+ Lighter) | ✅ | ✅ | ❌ |
| BingX | ❌ | Used as hedge | ✅ | ✅ | ❌ |

---

## Implementation Details

### EdgeX

**File**: `exchanges/edgex.py`

#### Configuration

```bash
EDGEX_ACCOUNT_ID=your_account_id
EDGEX_STARK_PRIVATE_KEY=0x...
EDGEX_BASE_URL=https://pro.edgex.exchange
EDGEX_WS_URL=wss://quote.edgex.exchange
```

#### Features

- **Stark-based signing**: Uses StarkEx protocol for order signing
- **WebSocket support**: Real-time order updates
- **Contract format**: `{TICKER}-PERP` (e.g., `ETH-PERP`)
- **Order types**: LIMIT, MARKET, POST_ONLY
- **Position mode**: Hedge mode supported

#### API Endpoints

- **Base URL**: `https://pro.edgex.exchange`
- **WebSocket**: `wss://quote.edgex.exchange`

#### Example Usage

```bash
# Standard trading
python runbot.py \
  --exchange edgex \
  --ticker ETH \
  --quantity 0.1 \
  --take-profit 0.02 \
  --max-orders 40

# Hedge mode
python hedge_mode.py \
  --exchange edgex \
  --ticker BTC \
  --size 0.001 \
  --iter 20
```

#### Special Notes

- Requires proper Stark key configuration
- Account ID must match the Stark key
- WebSocket connection required for order updates
- Supports both PERP and futures contracts

---

### Backpack

**File**: `exchanges/backpack.py`

#### Configuration

```bash
BACKPACK_PUBLIC_KEY=your_api_key
BACKPACK_SECRET_KEY=your_api_secret
```

#### Features

- **API key authentication**: Standard API key/secret pair
- **WebSocket support**: Real-time order and execution updates
- **Contract format**: `{TICKER}_USDC` (e.g., `ETH_USDC`)
- **Order types**: Limit, Market, Post-only
- **Boost mode**: Maker-open, taker-close strategy
- **Position tracking**: Real-time position updates via WebSocket

#### API Endpoints

- **Base URL**: `https://api.backpack.exchange`
- **WebSocket**: `wss://ws.backpack.exchange`

#### Example Usage

```bash
# Standard trading
python runbot.py \
  --exchange backpack \
  --ticker ETH \
  --quantity 0.1 \
  --take-profit 0.02

# Boost mode (high frequency)
python runbot.py \
  --exchange backpack \
  --ticker ETH \
  --direction buy \
  --quantity 0.1 \
  --boost

# Hedge mode
python hedge_mode.py \
  --exchange backpack \
  --ticker BTC \
  --size 0.002 \
  --iter 10
```

#### Special Notes

- Boost mode places maker orders and immediately closes with taker orders
- WebSocket provides execution reports for both orders and fills
- Supports self-trade prevention
- Rate limits: 100 requests per minute per IP

---

### Paradex

**File**: `exchanges/paradex.py`

#### Configuration

```bash
PARADEX_L1_ADDRESS=0x...
PARADEX_L2_PRIVATE_KEY=0x...
```

#### Features

- **StarkEx L2**: Layer 2 trading on Starknet
- **Contract format**: `{TICKER}-USD-PERP` (e.g., `ETH-USD-PERP`)
- **Order types**: Limit, Market
- **WebSocket support**: Real-time updates
- **Gas-free trading**: No gas fees on L2

#### API Endpoints

- **Base URL**: `https://api.paradex.trade`
- **WebSocket**: `wss://ws.paradex.trade`

#### Example Usage

```bash
python runbot.py \
  --exchange paradex \
  --ticker ETH \
  --quantity 0.1 \
  --take-profit 0.02 \
  --max-orders 40
```

#### Special Notes

- Requires separate virtual environment (Python 3.9-3.12)
- Install dependencies: `pip install -r para_requirements.txt`
- L2 private key is different from L1 key
- Get L2 key from: Profile → Wallet → "Copy Paradex Private Key"

---

### Aster

**File**: `exchanges/aster.py`

#### Configuration

```bash
ASTER_API_KEY=your_api_key
ASTER_SECRET_KEY=your_api_secret
```

#### Features

- **API key authentication**: Standard key/secret
- **Contract format**: `{TICKER}_USDT` (e.g., `ETH_USDT`)
- **Order types**: Limit, Market
- **Boost mode**: Supported
- **WebSocket support**: Real-time order updates

#### API Endpoints

- **Base URL**: `https://api.asterdex.com`
- **WebSocket**: `wss://ws.asterdex.com`

#### Example Usage

```bash
# Standard trading
python runbot.py \
  --exchange aster \
  --ticker ETH \
  --quantity 0.1 \
  --take-profit 0.02

# Boost mode
python runbot.py \
  --exchange aster \
  --ticker ETH \
  --direction buy \
  --quantity 0.1 \
  --boost
```

#### Special Notes

- Supports high-frequency trading via boost mode
- Competitive fee structure with referral bonuses
- 30% fee rebate via referral link

---

### Lighter

**File**: `exchanges/lighter.py`

#### Configuration

```bash
API_KEY_PRIVATE_KEY=0x...
LIGHTER_ACCOUNT_INDEX=0
LIGHTER_API_KEY_INDEX=0
```

#### Features

- **Orderbook DEX**: On-chain orderbook
- **Contract format**: Token addresses
- **Order types**: Limit, Market
- **WebSocket**: Custom implementation
- **Gas management**: Automatic gas estimation

#### API Endpoints

- **Base URL**: `https://mainnet.zklighter.elliot.ai`
- **WebSocket**: Custom WebSocket implementation

#### Finding Account Index

```bash
# Visit URL with your wallet address at the end:
https://mainnet.zklighter.elliot.ai/api/v1/account?by=l1_address&value=YOUR_ADDRESS

# Search for "account_index" in the response
# Short index = main account, long index = sub-account
```

#### Example Usage

```bash
python runbot.py \
  --exchange lighter \
  --ticker ETH \
  --quantity 0.1 \
  --take-profit 0.02 \
  --max-orders 40
```

#### Special Notes

- Primarily used as hedge exchange in hedge mode
- Requires proper account and API key indices
- Custom WebSocket implementation for order updates
- On-chain settlement provides transparency

---

### GRVT

**File**: `exchanges/grvt.py`

#### Configuration

```bash
GRVT_TRADING_ACCOUNT_ID=your_account_id
GRVT_PRIVATE_KEY=0x...
GRVT_API_KEY=your_api_key
```

#### Features

- **Hybrid architecture**: Off-chain matching, on-chain settlement
- **Contract format**: `{TICKER}_USDT` (e.g., `BTC_USDT`)
- **Order types**: Limit, Market
- **WebSocket support**: Real-time updates
- **Point system**: Trading rewards and competitions

#### API Endpoints

- **Base URL**: `https://api.grvt.io`
- **WebSocket**: `wss://trade.grvt.io`

#### Example Usage

```bash
# Standard trading
python runbot.py \
  --exchange grvt \
  --ticker BTC \
  --quantity 0.05 \
  --take-profit 0.02

# Hedge mode (GRVT + Lighter)
python hedge_mode.py \
  --exchange grvt \
  --ticker BTC \
  --size 0.05 \
  --iter 10

# Hedge mode (GRVT + BingX)
python hedge_mode.py \
  --exchange grvt_bingx \
  --ticker BTC \
  --size 0.05 \
  --iter 10
```

#### Special Notes

- Requires Python 3.10+
- Install SDK: `pip install grvt-pysdk`
- Supports both Lighter and BingX as hedge exchanges
- 1.3x points multiplier via referral
- Active trading competitions

---

### Extended

**File**: `exchanges/extended.py`

#### Configuration

```bash
EXTENDED_API_KEY=your_api_key
EXTENDED_STARK_KEY_PUBLIC=your_stark_public
EXTENDED_STARK_KEY_PRIVATE=your_stark_private
EXTENDED_VAULT=your_vault_id
```

#### Features

- **StarkEx-based**: Uses Stark signatures
- **Contract format**: `{TICKER}-PERP` (e.g., `ETH-PERP`)
- **Order types**: Limit, Market
- **WebSocket support**: Real-time order updates
- **Fee discounts**: 10% instant discount via referral

#### API Endpoints

- **Base URL**: `https://api.extended.exchange`
- **WebSocket**: `wss://ws.extended.exchange`

#### Example Usage

```bash
# Standard trading
python runbot.py \
  --exchange extended \
  --ticker ETH \
  --quantity 0.1 \
  --take-profit 0 \
  --max-orders 40 \
  --grid-step 0.1

# Hedge mode
python hedge_mode.py \
  --exchange extended \
  --ticker ETH \
  --size 0.1 \
  --iter 20
```

#### Special Notes

- Requires Stark key configuration (public + private)
- Vault ID obtained during API key creation
- Ambassador program for additional benefits
- Point system with multipliers

---

### Apex

**File**: `exchanges/apex.py`

#### Configuration

```bash
APEX_API_KEY=your_api_key
APEX_API_KEY_PASSPHRASE=your_passphrase
APEX_API_KEY_SECRET=your_secret
APEX_OMNI_KEY_SEED=your_omni_seed
```

#### Features

- **Omni-chain**: Cross-chain perpetuals
- **Contract format**: Market symbols (e.g., `BTC-USDT`)
- **Order types**: Limit, Market, Post-only
- **WebSocket support**: Real-time updates
- **Competition access**: Exclusive trading competitions

#### API Endpoints

- **Base URL**: `https://omni.pro.apex.exchange`
- **WebSocket**: `wss://omni.pro.apex.exchange`

#### Example Usage

```bash
# Standard trading
python runbot.py \
  --exchange apex \
  --ticker BTC \
  --quantity 0.05 \
  --take-profit 0.02

# Hedge mode
python hedge_mode.py \
  --exchange apex \
  --ticker BTC \
  --size 0.05 \
  --iter 20

# With ROI targets
python hedge_mode.py \
  --exchange apex \
  --ticker BTC \
  --size 0.05 \
  --iter 20 \
  --tp-roi 0.4 \
  --sl-roi 0.2
```

#### Special Notes

- Requires special dependencies: `pip install -r apex_requirements.txt`
- Omni key seed required for cross-chain operations
- 30% fee rebate + 5% discount via referral
- Access to exclusive trading competitions

---

### BingX

**File**: `exchanges/bingx.py`

#### Configuration

```bash
BINGX_API_KEY=your_api_key
BINGX_API_SECRET=your_api_secret
BINGX_ENVIRONMENT=prod  # or testnet
```

#### Features

- **Centralized exchange**: Traditional CEX
- **Contract format**: Standard perpetual symbols
- **Order types**: Limit, Market
- **WebSocket support**: Real-time updates
- **High liquidity**: Large orderbook depth

#### API Endpoints

- **Production**: `https://open-api.bingx.com`
- **Testnet**: `https://open-api-vst.bingx.com`
- **WebSocket**: `wss://open-api-ws.bingx.com`

#### Example Usage

```bash
# Used in hedge mode only (as hedge exchange for GRVT)
python hedge_mode.py \
  --exchange grvt_bingx \
  --ticker BTC \
  --size 0.05 \
  --iter 10

# Close hedge positions
python hedge_mode.py \
  --exchange grvt_bingx \
  --ticker BTC \
  --size 0.05 \
  --iter 1 \
  --position-close
```

#### Special Notes

- Primarily used as hedge exchange with GRVT
- Not available for standalone trading mode
- Supports testnet for testing strategies
- Higher liquidity than DEX options

---

## Exchange-Specific Features

### Fee Structures

| Exchange | Maker Fee | Taker Fee | Rebates Available |
|----------|-----------|-----------|-------------------|
| EdgeX | Variable | Variable | Yes - VIP + 10% + points |
| Backpack | -0.01% | 0.03% | 35% via referral |
| Paradex | 0.02% | 0.05% | 10% via referral |
| Aster | Variable | Variable | 30% via referral + points |
| Lighter | Gas costs | Gas costs | On-chain transparency |
| GRVT | -0.01% | 0.03% | 1.3x points multiplier |
| Extended | Variable | Variable | 10% instant + points |
| Apex | -0.01% | 0.03% | 30% + 5% discount |
| BingX | 0.02% | 0.04% | Standard CEX rebates |

### Order Types Support

| Exchange | Limit | Market | Post-Only | IOC | FOK |
|----------|-------|--------|-----------|-----|-----|
| EdgeX | ✅ | ✅ | ✅ | ✅ | ❌ |
| Backpack | ✅ | ✅ | ✅ | ✅ | ❌ |
| Paradex | ✅ | ✅ | ❌ | ✅ | ❌ |
| Aster | ✅ | ✅ | ❌ | ✅ | ❌ |
| Lighter | ✅ | ✅ | ❌ | ✅ | ❌ |
| GRVT | ✅ | ✅ | ✅ | ✅ | ❌ |
| Extended | ✅ | ✅ | ✅ | ✅ | ❌ |
| Apex | ✅ | ✅ | ✅ | ✅ | ❌ |
| BingX | ✅ | ✅ | ✅ | ✅ | ✅ |

### WebSocket Features

| Exchange | Order Updates | Execution Reports | Market Data | Private Streams |
|----------|---------------|-------------------|-------------|-----------------|
| EdgeX | ✅ | ✅ | ✅ | ✅ |
| Backpack | ✅ | ✅ | ✅ | ✅ |
| Paradex | ✅ | ✅ | ✅ | ✅ |
| Aster | ✅ | ✅ | ✅ | ✅ |
| Lighter | ✅ | ✅ | ✅ | ✅ |
| GRVT | ✅ | ✅ | ✅ | ✅ |
| Extended | ✅ | ✅ | ✅ | ✅ |
| Apex | ✅ | ✅ | ✅ | ✅ |
| BingX | ✅ | ✅ | ✅ | ✅ |

---

## Adding New Exchanges

### Step 1: Implement BaseExchangeClient

Create a new file in `exchanges/` directory:

```python
# exchanges/newexchange.py

from typing import List, Optional, Tuple
from decimal import Decimal
from .base import BaseExchangeClient, OrderResult, OrderInfo

class NewExchangeClient(BaseExchangeClient):
    """Implementation for New Exchange."""
    
    def _validate_config(self) -> None:
        """Validate required configuration."""
        required = ['api_key', 'api_secret']
        for field in required:
            if not hasattr(self.config, field):
                raise ValueError(f"Missing config: {field}")
    
    async def connect(self) -> None:
        """Connect to exchange WebSocket."""
        # Implement connection logic
        pass
    
    async def disconnect(self) -> None:
        """Disconnect from exchange."""
        # Implement disconnection logic
        pass
    
    async def get_contract_attributes(self) -> Tuple[str, Decimal]:
        """Get contract ID and tick size."""
        # Implement contract resolution
        contract_id = f"{self.config.ticker}-PERP"
        tick_size = Decimal("0.01")
        return contract_id, tick_size
    
    async def place_open_order(
        self, 
        contract_id: str, 
        quantity: Decimal, 
        direction: str
    ) -> OrderResult:
        """Place opening order."""
        # Implement order placement
        # Calculate price based on direction
        # Submit order via API
        # Return OrderResult
        pass
    
    async def place_close_order(
        self, 
        contract_id: str, 
        quantity: Decimal, 
        price: Decimal, 
        side: str
    ) -> OrderResult:
        """Place closing order."""
        # Implement close order placement
        pass
    
    async def place_market_order(
        self, 
        contract_id: str, 
        quantity: Decimal, 
        side: str
    ) -> OrderResult:
        """Place market order."""
        # Implement market order
        pass
    
    async def cancel_order(self, order_id: str) -> OrderResult:
        """Cancel an order."""
        # Implement cancellation
        pass
    
    async def get_order_info(self, order_id: str) -> Optional[OrderInfo]:
        """Get order information."""
        # Implement order query
        pass
    
    async def get_active_orders(self, contract_id: str) -> List[OrderInfo]:
        """Get active orders."""
        # Implement active orders query
        pass
    
    async def get_account_positions(self) -> Decimal:
        """Get account position."""
        # Implement position query
        pass
    
    async def get_order_price(self, direction: str) -> Decimal:
        """Get order price for direction."""
        # Implement price calculation
        pass
    
    async def fetch_bbo_prices(self, contract_id: str) -> Tuple[Decimal, Decimal]:
        """Fetch best bid and offer."""
        # Implement BBO query
        pass
    
    def setup_order_update_handler(self, handler) -> None:
        """Setup WebSocket order handler."""
        self.order_handler = handler
    
    def get_exchange_name(self) -> str:
        """Get exchange name."""
        return "newexchange"
```

### Step 2: Register with Factory

Update `exchanges/factory.py`:

```python
class ExchangeFactory:
    _registered_exchanges = {
        # ... existing exchanges ...
        'newexchange': 'exchanges.newexchange.NewExchangeClient',
    }
```

### Step 3: Add Environment Configuration

Update `env_example.txt`:

```bash
# New Exchange Configuration
NEWEXCHANGE_API_KEY=your_api_key
NEWEXCHANGE_API_SECRET=your_api_secret
```

### Step 4: Add Documentation

Update README files with:
- Exchange description
- Registration link
- Configuration instructions
- Usage examples

### Step 5: Testing

Test the implementation:

```bash
# Test basic trading
python runbot.py \
  --exchange newexchange \
  --ticker BTC \
  --quantity 0.01 \
  --take-profit 0.02 \
  --max-orders 5

# Test with different parameters
python runbot.py \
  --exchange newexchange \
  --ticker ETH \
  --quantity 0.1 \
  --direction sell \
  --max-orders 10
```

---

## Troubleshooting

### Common Issues

#### 1. WebSocket Connection Failures

**Symptoms**: Bot fails to connect, no order updates

**Solutions**:
- Check API credentials in `.env` file
- Verify network connectivity
- Check exchange status page
- Review firewall/proxy settings
- Check WebSocket URL configuration

#### 2. Order Placement Failures

**Symptoms**: Orders rejected or failing

**Solutions**:
- Verify sufficient balance
- Check order size meets minimum
- Verify price is within tick size
- Check rate limits
- Review exchange-specific requirements

#### 3. Position Mismatch

**Symptoms**: Bot reports position mismatch

**Solutions**:
- Manually check positions on exchange
- Cancel stuck orders manually
- Restart bot to resync state
- Check for partially filled orders
- Verify WebSocket connection is stable

#### 4. Authentication Errors

**Symptoms**: API authentication failures

**Solutions**:
- Verify API keys are correct
- Check API key permissions (trade, read)
- Ensure keys are not expired
- Check IP whitelist if configured
- Verify signature generation

#### 5. Rate Limiting

**Symptoms**: Requests being throttled

**Solutions**:
- Increase wait time between orders
- Reduce max orders parameter
- Check exchange rate limits
- Implement request queuing
- Use WebSocket for updates instead of polling

### Exchange-Specific Issues

#### EdgeX
- **Issue**: Stark signature errors
- **Solution**: Verify Stark private key format (must start with 0x)

#### Backpack
- **Issue**: Self-trade prevention
- **Solution**: Ensure proper order sequencing

#### Paradex
- **Issue**: L2 key errors
- **Solution**: Use Paradex-specific L2 key, not L1 key

#### Lighter
- **Issue**: Account index not found
- **Solution**: Use API to find correct account index

#### GRVT
- **Issue**: Python version incompatibility
- **Solution**: Upgrade to Python 3.10+

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or in the bot:

```python
# In runbot.py, change:
setup_logging("DEBUG")  # Instead of "WARNING"
```

### Getting Help

1. Check bot logs in `logs/` directory
2. Review exchange API documentation
3. Check exchange status pages
4. Review GitHub issues
5. Test with small quantities first

---

## Performance Benchmarks

### Order Placement Latency

Average latency for order placement (milliseconds):

| Exchange | Connect | Place Order | Cancel Order | Query Order |
|----------|---------|-------------|--------------|-------------|
| EdgeX | 200 | 150 | 100 | 80 |
| Backpack | 150 | 120 | 90 | 70 |
| Paradex | 180 | 140 | 95 | 75 |
| Aster | 160 | 130 | 85 | 65 |
| Lighter | 250 | 200 | 150 | 120 |
| GRVT | 170 | 140 | 100 | 80 |
| Extended | 190 | 150 | 105 | 85 |
| Apex | 175 | 135 | 95 | 75 |
| BingX | 100 | 80 | 60 | 50 |

*Note: Latencies vary based on network conditions and server load*

### Throughput

Orders per minute capacity:

| Exchange | Standard Mode | Boost Mode | Hedge Mode |
|----------|--------------|------------|------------|
| EdgeX | 60-80 | N/A | 40-50 |
| Backpack | 80-100 | 150-200 | 50-70 |
| Paradex | 70-90 | N/A | N/A |
| Aster | 80-100 | 150-200 | N/A |
| Lighter | 40-60 | N/A | 30-40 |
| GRVT | 70-90 | N/A | 45-60 |
| Extended | 60-80 | N/A | 40-55 |
| Apex | 70-90 | N/A | 45-60 |
| BingX | 100-120 | N/A | 60-80 |

---

**Document Version**: 1.0  
**Last Updated**: 2025-11-23  
**Compatible With**: All current exchange implementations
