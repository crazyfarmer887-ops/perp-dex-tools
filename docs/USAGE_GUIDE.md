# Comprehensive Usage Guide

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Usage](#basic-usage)
3. [Advanced Strategies](#advanced-strategies)
4. [Configuration Guide](#configuration-guide)
5. [Parameter Tuning](#parameter-tuning)
6. [Best Practices](#best-practices)
7. [Common Scenarios](#common-scenarios)
8. [Monitoring and Maintenance](#monitoring-and-maintenance)
9. [FAQ](#faq)

---

## Getting Started

### Prerequisites

- **Python Version**: 3.8+ (3.10-3.12 recommended)
  - GRVT requires Python 3.10+
  - Paradex requires Python 3.9-3.12
- **Operating System**: Linux, macOS, or Windows
- **Network**: Stable internet connection
- **Capital**: Sufficient balance on target exchange

### Installation

#### 1. Clone Repository

```bash
git clone <repository-url>
cd perp-dex-tools
```

#### 2. Create Virtual Environment

```bash
# Deactivate any existing virtual environment
deactivate

# Create new virtual environment
python3 -m venv env

# Activate virtual environment
# On Linux/Mac:
source env/bin/activate

# On Windows:
env\Scripts\activate
```

#### 3. Install Dependencies

**For most exchanges:**
```bash
pip install -r requirements.txt
```

**For GRVT (additional):**
```bash
pip install grvt-pysdk
```

**For Paradex (separate environment):**
```bash
# Create Paradex-specific environment
python3 -m venv para_env
source para_env/bin/activate  # Windows: para_env\Scripts\activate
pip install -r para_requirements.txt
```

**For Apex (additional):**
```bash
pip install -r apex_requirements.txt
```

#### 4. Configure Environment

```bash
# Copy example configuration
cp env_example.txt .env

# Edit .env with your credentials
nano .env  # or use your preferred editor
```

### Quick Start

Test the bot with minimal settings:

```bash
# Standard trading (EdgeX, ETH)
python runbot.py \
  --exchange edgex \
  --ticker ETH \
  --quantity 0.01 \
  --take-profit 0.02 \
  --max-orders 5 \
  --wait-time 300

# Hedge mode (Backpack + Lighter)
python hedge_mode.py \
  --exchange backpack \
  --ticker BTC \
  --size 0.001 \
  --iter 5
```

---

## Basic Usage

### Standard Trading Mode

#### Command Structure

```bash
python runbot.py \
  --exchange <EXCHANGE> \
  --ticker <SYMBOL> \
  --quantity <SIZE> \
  --take-profit <PERCENT> \
  --direction <buy|sell> \
  --max-orders <NUMBER> \
  --wait-time <SECONDS> \
  [--grid-step <PERCENT>] \
  [--stop-price <PRICE>] \
  [--pause-price <PRICE>] \
  [--boost] \
  [--env-file <FILE>]
```

#### Required Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--exchange` | Exchange name | `edgex`, `backpack`, `grvt` |
| `--ticker` | Trading pair symbol | `ETH`, `BTC`, `SOL` |
| `--quantity` | Order size | `0.1` |
| `--take-profit` | Take profit % | `0.02` (means 0.02%) |

#### Optional Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--direction` | Trading direction | `buy` |
| `--max-orders` | Max concurrent orders | `40` |
| `--wait-time` | Wait between orders (s) | `450` |
| `--grid-step` | Grid spacing % | `-100` (disabled) |
| `--stop-price` | Stop trading price | `-1` (disabled) |
| `--pause-price` | Pause trading price | `-1` (disabled) |
| `--boost` | Enable boost mode | `False` |
| `--env-file` | Config file path | `.env` |

#### Basic Examples

**Example 1: Buy ETH with defaults**
```bash
python runbot.py \
  --exchange edgex \
  --ticker ETH \
  --quantity 0.1 \
  --take-profit 0.02
```

**Example 2: Sell BTC with custom settings**
```bash
python runbot.py \
  --exchange backpack \
  --ticker BTC \
  --direction sell \
  --quantity 0.05 \
  --take-profit 0.03 \
  --max-orders 30 \
  --wait-time 600
```

**Example 3: Conservative ETH trading**
```bash
python runbot.py \
  --exchange grvt \
  --ticker ETH \
  --quantity 0.05 \
  --take-profit 0.05 \
  --max-orders 20 \
  --wait-time 900
```

### Hedge Mode

#### Command Structure

```bash
python hedge_mode.py \
  --exchange <EXCHANGE> \
  --ticker <SYMBOL> \
  --size <SIZE> \
  --iter <NUMBER> \
  [--fill-timeout <SECONDS>] \
  [--sleep <SECONDS>] \
  [--tp-roi <PERCENT>] \
  [--sl-roi <PERCENT>] \
  [--env-file <FILE>]
```

#### Required Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--exchange` | Primary exchange | `backpack`, `grvt`, `apex` |
| `--ticker` | Trading pair symbol | `BTC`, `ETH` |
| `--size` | Order size | `0.05` |
| `--iter` | Number of cycles | `20` |

#### Optional Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--fill-timeout` | Order fill timeout (s) | `5` |
| `--sleep` | Sleep after each trade (s) | `0` |
| `--tp-roi` | Take profit ROI % | `None` |
| `--sl-roi` | Stop loss ROI % | `None` |
| `--position-close` | Close positions (grvt_bingx) | `False` |
| `--env-file` | Config file path | `.env` |

#### Hedge Examples

**Example 1: Basic hedge (Backpack + Lighter)**
```bash
python hedge_mode.py \
  --exchange backpack \
  --ticker BTC \
  --size 0.002 \
  --iter 10
```

**Example 2: Hedge with ROI targets**
```bash
python hedge_mode.py \
  --exchange apex \
  --ticker BTC \
  --size 0.05 \
  --iter 20 \
  --tp-roi 0.4 \
  --sl-roi 0.2
```

**Example 3: GRVT + BingX hedge**
```bash
python hedge_mode.py \
  --exchange grvt_bingx \
  --ticker BTC \
  --size 0.05 \
  --iter 15 \
  --fill-timeout 10
```

---

## Advanced Strategies

### Grid Step Strategy

Grid step controls the minimum distance between close orders, preventing order clustering.

#### Configuration

```bash
--grid-step 0.5  # 0.5% minimum spacing
```

#### How It Works

For **buy** direction (long):
- New close order price must be at least 0.5% **below** nearest existing close order
- Example: If nearest close order is at $2000, new order must be ≤ $1990

For **sell** direction (short):
- New close order price must be at least 0.5% **above** nearest existing close order
- Example: If nearest close order is at $2000, new order must be ≥ $2010

#### Use Cases

**Tight Grid (0.1-0.3%):**
- High volatility markets
- Small take profit targets
- Frequent order fills

```bash
python runbot.py \
  --exchange extended \
  --ticker ETH \
  --quantity 0.1 \
  --take-profit 0.02 \
  --grid-step 0.1
```

**Medium Grid (0.5-1.0%):**
- Moderate volatility
- Standard take profit
- Balanced approach

```bash
python runbot.py \
  --exchange edgex \
  --ticker BTC \
  --quantity 0.05 \
  --take-profit 0.05 \
  --grid-step 0.5
```

**Wide Grid (1.0-2.0%):**
- Low volatility markets
- Large take profit targets
- Conservative positioning

```bash
python runbot.py \
  --exchange backpack \
  --ticker ETH \
  --quantity 0.1 \
  --take-profit 0.1 \
  --grid-step 1.5
```

### Stop and Pause Prices

Control trading based on price levels to manage risk.

#### Stop Price

Exits the bot when price reaches threshold.

**Buy Direction:**
- Stops when `price >= stop_price`
- Use case: Exit if price goes too high (avoid entering at top)

**Sell Direction:**
- Stops when `price <= stop_price`
- Use case: Exit if price goes too low (avoid entering at bottom)

**Example:**
```bash
# Stop buying ETH if price reaches 3500
python runbot.py \
  --exchange edgex \
  --ticker ETH \
  --direction buy \
  --quantity 0.1 \
  --stop-price 3500
```

#### Pause Price

Pauses trading temporarily when price reaches threshold, resumes when price moves back.

**Buy Direction:**
- Pauses when `price >= pause_price`
- Resumes when `price < pause_price`

**Sell Direction:**
- Pauses when `price <= pause_price`
- Resumes when `price > pause_price`

**Example:**
```bash
# Pause buying ETH if price reaches 3200, resume below
python runbot.py \
  --exchange edgex \
  --ticker ETH \
  --direction buy \
  --quantity 0.1 \
  --pause-price 3200
```

#### Combined Strategy

```bash
# Pause at 3200, stop at 3500
python runbot.py \
  --exchange edgex \
  --ticker ETH \
  --direction buy \
  --quantity 0.1 \
  --pause-price 3200 \
  --stop-price 3500
```

### Boost Mode

Boost mode maximizes trading volume by using maker-taker pairs.

#### How It Works

1. Place **maker** order (gets rebate)
2. Immediately close with **taker** order (pays fee)
3. Repeat rapidly

#### Cost

- One maker fee (often negative = rebate)
- One taker fee
- Slippage

#### When to Use

- Volume requirements or competitions
- Maker rebates available
- High liquidity markets
- Short-term campaigns

#### Supported Exchanges

- Backpack
- Aster

#### Example

```bash
python runbot.py \
  --exchange backpack \
  --ticker ETH \
  --direction buy \
  --quantity 0.1 \
  --boost
```

### Dynamic Wait Time Strategy

The bot automatically adjusts wait time based on active orders.

#### Logic

| Active Orders | Wait Time Multiplier |
|---------------|---------------------|
| ≥ max_orders | 1 second (nearly stop) |
| ≥ 2/3 max | 2.0x |
| ≥ 1/3 max | 1.0x |
| ≥ 1/6 max | 0.5x |
| < 1/6 max | 0.25x |

#### Example Calculation

With `--max-orders 40` and `--wait-time 450`:

- 40+ orders: Wait 1 second
- 27-39 orders: Wait 900 seconds
- 14-26 orders: Wait 450 seconds
- 7-13 orders: Wait 225 seconds
- 0-6 orders: Wait 112.5 seconds

#### Tuning

**Aggressive (fast accumulation):**
```bash
--max-orders 60 --wait-time 300
```

**Moderate (balanced):**
```bash
--max-orders 40 --wait-time 450
```

**Conservative (slow, controlled):**
```bash
--max-orders 20 --wait-time 900
```

---

## Configuration Guide

### Single Exchange, Single Account

Simplest setup for one exchange and one account.

**1. Create `.env` file:**
```bash
# Exchange credentials
EDGEX_ACCOUNT_ID=your_account_id
EDGEX_STARK_PRIVATE_KEY=your_private_key

# Optional notifications
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id
```

**2. Run bot:**
```bash
python runbot.py --exchange edgex --ticker ETH --quantity 0.1
```

### Multiple Exchanges, Single Account Each

Different exchanges with one account per exchange.

**1. Create `.env` file:**
```bash
# EdgeX
EDGEX_ACCOUNT_ID=your_edgex_account
EDGEX_STARK_PRIVATE_KEY=your_edgex_key

# Backpack
BACKPACK_PUBLIC_KEY=your_backpack_key
BACKPACK_SECRET_KEY=your_backpack_secret

# GRVT
GRVT_TRADING_ACCOUNT_ID=your_grvt_account
GRVT_PRIVATE_KEY=your_grvt_key
GRVT_API_KEY=your_grvt_api_key
```

**2. Run different exchanges:**
```bash
# Terminal 1: EdgeX
python runbot.py --exchange edgex --ticker ETH --quantity 0.1

# Terminal 2: Backpack
python runbot.py --exchange backpack --ticker BTC --quantity 0.05

# Terminal 3: GRVT
python runbot.py --exchange grvt --ticker ETH --quantity 0.1
```

### Single Exchange, Multiple Accounts

Multiple accounts on the same exchange.

**1. Create separate env files:**

`account1.env`:
```bash
ACCOUNT_NAME=main_account
EDGEX_ACCOUNT_ID=account1_id
EDGEX_STARK_PRIVATE_KEY=account1_key
```

`account2.env`:
```bash
ACCOUNT_NAME=secondary_account
EDGEX_ACCOUNT_ID=account2_id
EDGEX_STARK_PRIVATE_KEY=account2_key
```

**2. Run with different env files:**
```bash
# Terminal 1: Account 1, ETH
python runbot.py \
  --env-file account1.env \
  --exchange edgex \
  --ticker ETH \
  --quantity 0.1

# Terminal 2: Account 2, BTC
python runbot.py \
  --env-file account2.env \
  --exchange edgex \
  --ticker BTC \
  --quantity 0.05
```

### Multiple Contracts, Same Account

Trade multiple pairs with one account.

**1. Create `.env` file:**
```bash
EDGEX_ACCOUNT_ID=your_account_id
EDGEX_STARK_PRIVATE_KEY=your_private_key
```

**2. Run multiple instances:**
```bash
# Terminal 1: ETH
python runbot.py \
  --exchange edgex \
  --ticker ETH \
  --quantity 0.1 \
  --max-orders 30

# Terminal 2: BTC
python runbot.py \
  --exchange edgex \
  --ticker BTC \
  --quantity 0.05 \
  --max-orders 20

# Terminal 3: SOL
python runbot.py \
  --exchange edgex \
  --ticker SOL \
  --quantity 1.0 \
  --max-orders 25
```

---

## Parameter Tuning

### Quantity (`--quantity`)

Order size for each trade.

#### Considerations

- Exchange minimum order size
- Account balance
- Risk tolerance
- Gas costs (for DEXes)

#### Guidelines

**Small (0.001-0.01):**
- Testing strategies
- Low balance
- High volatility
- DEX with high gas

**Medium (0.01-0.1):**
- Standard trading
- Moderate balance
- Normal volatility

**Large (0.1+):**
- High balance
- Low volatility
- Volume requirements

### Take Profit (`--take-profit`)

Profit target as percentage.

#### Common Values

| Value | Use Case |
|-------|----------|
| 0.01-0.02% | High frequency, tight spread |
| 0.02-0.05% | Standard trading |
| 0.05-0.1% | Conservative, wider spread |
| 0.1%+ | Very conservative |

#### Factors to Consider

- Exchange fees
- Market volatility
- Spread width
- Volume vs profit balance

**Example Calculation:**
```
Entry: $2000
Take profit: 0.02%
Target: $2000 × (1 + 0.02/100) = $2000.40
Profit: $0.40 per unit
```

### Max Orders (`--max-orders`)

Maximum concurrent closing orders.

#### Impact

**Low (5-15):**
- Lower capital requirements
- Less exposure
- Faster order cycling
- Higher wait times
- More conservative

**Medium (20-40):**
- Balanced approach
- Moderate capital
- Standard exposure
- Moderate wait times

**High (40-100):**
- High capital requirements
- Maximum exposure
- Lower wait times
- More aggressive

#### Tuning Tips

1. Start low (10-20)
2. Monitor position accumulation
3. Increase gradually
4. Consider available capital
5. Match with wait time

### Wait Time (`--wait-time`)

Seconds between orders (base wait time).

#### Common Values

| Seconds | Frequency | Use Case |
|---------|-----------|----------|
| 60-180 | High | Volume campaigns |
| 300-600 | Medium | Standard trading |
| 600-1200 | Low | Conservative |

#### Factors

- Market volatility
- Max orders setting
- Risk tolerance
- Volume requirements

#### Relationship with Max Orders

```bash
# Aggressive: Quick fill, many orders
--max-orders 60 --wait-time 300

# Moderate: Balanced
--max-orders 40 --wait-time 450

# Conservative: Slow fill, few orders
--max-orders 20 --wait-time 900
```

### Grid Step (`--grid-step`)

Minimum distance between close orders (%).

#### Values

| Value | Spacing | Use Case |
|-------|---------|----------|
| -100 | Disabled | No restriction |
| 0.1-0.3 | Tight | High volatility |
| 0.5-1.0 | Medium | Standard |
| 1.0-2.0 | Wide | Low volatility |

#### Selection Guide

1. **High volatility** → Tight grid (0.1-0.3%)
2. **Medium volatility** → Medium grid (0.5-1.0%)
3. **Low volatility** → Wide grid (1.0-2.0%)
4. **No preference** → Disabled (-100)

---

## Best Practices

### 1. Start Small

Always begin with minimal parameters:

```bash
python runbot.py \
  --exchange edgex \
  --ticker ETH \
  --quantity 0.01 \
  --max-orders 5 \
  --wait-time 600
```

### 2. Monitor Regularly

Check logs and positions:

```bash
# View real-time logs
tail -f logs/edgex_ETH_activity.log

# Check transaction history
cat logs/edgex_ETH_orders.csv
```

### 3. Use Notifications

Enable Telegram or Lark:

```bash
# In .env
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id
```

### 4. Manage Risk

- Set appropriate `--max-orders`
- Use `--stop-price` for risk limits
- Monitor position size
- Keep sufficient balance
- Don't overtrade

### 5. Optimize Parameters

- Test with different `--wait-time`
- Adjust `--grid-step` based on volatility
- Balance `--take-profit` with fees
- Scale `--quantity` appropriately

### 6. Maintain Balance

- Keep enough balance for max orders
- Consider exchange fees
- Account for gas costs (DEXes)
- Reserve buffer for market moves

### 7. Handle Errors Gracefully

- Check logs immediately
- Don't panic on position mismatch
- Manually verify on exchange
- Restart bot if needed
- Contact support if necessary

### 8. Regular Maintenance

- Update dependencies
- Review trading performance
- Adjust parameters based on results
- Clean up old logs
- Backup configurations

---

## Common Scenarios

### Scenario 1: Volume Campaign

**Goal**: Maximize trading volume quickly

**Configuration:**
```bash
python runbot.py \
  --exchange backpack \
  --ticker ETH \
  --quantity 0.1 \
  --take-profit 0.01 \
  --max-orders 80 \
  --wait-time 180 \
  --grid-step 0.2 \
  --boost
```

**Strategy:**
- High max orders
- Short wait time
- Tight take profit
- Boost mode enabled
- Tight grid step

### Scenario 2: Conservative Long-term

**Goal**: Steady, low-risk trading

**Configuration:**
```bash
python runbot.py \
  --exchange grvt \
  --ticker BTC \
  --quantity 0.02 \
  --take-profit 0.1 \
  --max-orders 15 \
  --wait-time 1200 \
  --grid-step 1.5
```

**Strategy:**
- Low max orders
- Long wait time
- Wide take profit
- Wide grid step
- Conservative quantity

### Scenario 3: Range-bound Market

**Goal**: Trade within specific range

**Configuration:**
```bash
python runbot.py \
  --exchange edgex \
  --ticker ETH \
  --direction buy \
  --quantity 0.1 \
  --take-profit 0.05 \
  --max-orders 30 \
  --wait-time 450 \
  --pause-price 3200 \
  --grid-step 0.5
```

**Strategy:**
- Pause at range top
- Medium parameters
- Balanced grid step

### Scenario 4: Trending Market

**Goal**: Follow trend with stop

**Configuration:**
```bash
python runbot.py \
  --exchange backpack \
  --ticker BTC \
  --direction buy \
  --quantity 0.05 \
  --take-profit 0.03 \
  --max-orders 35 \
  --wait-time 400 \
  --stop-price 65000
```

**Strategy:**
- Stop at potential reversal
- Medium aggressiveness
- Standard parameters

### Scenario 5: Multi-account Arbitrage

**Goal**: Trade on multiple accounts

**Setup:**
```bash
# Terminal 1: Main account
python runbot.py \
  --env-file main.env \
  --exchange edgex \
  --ticker ETH \
  --quantity 0.15 \
  --max-orders 40

# Terminal 2: Sub account
python runbot.py \
  --env-file sub.env \
  --exchange edgex \
  --ticker ETH \
  --quantity 0.1 \
  --max-orders 30
```

---

## Monitoring and Maintenance

### Log Files

#### Activity Log

Location: `logs/{exchange}_{ticker}_activity.log`

**What it contains:**
- Order placements
- Order fills
- Cancellations
- Position updates
- Errors and warnings

**How to monitor:**
```bash
# Real-time monitoring
tail -f logs/edgex_ETH_activity.log

# Search for errors
grep ERROR logs/edgex_ETH_activity.log

# View recent activity
tail -n 100 logs/edgex_ETH_activity.log
```

#### Transaction Log

Location: `logs/{exchange}_{ticker}_orders.csv`

**What it contains:**
- Timestamp
- Order ID
- Side (buy/sell)
- Quantity
- Price
- Status

**How to analyze:**
```bash
# View recent trades
tail -n 20 logs/edgex_ETH_orders.csv

# Count filled orders
grep FILLED logs/edgex_ETH_orders.csv | wc -l

# Calculate total volume
# Use spreadsheet or custom script
```

### Performance Metrics

#### Key Metrics to Track

1. **Fill Rate**: Filled orders / Total orders
2. **Average Fill Time**: Time to fill orders
3. **Total Volume**: Cumulative trading volume
4. **Profit/Loss**: Net P&L from trading
5. **Fee Costs**: Total fees paid
6. **Position Size**: Current open position

#### Calculating Metrics

**From CSV:**
```python
import pandas as pd

# Load data
df = pd.read_csv('logs/edgex_ETH_orders.csv')

# Fill rate
fill_rate = len(df[df['Status'] == 'FILLED']) / len(df)

# Total volume
total_volume = df[df['Status'] == 'FILLED']['Quantity'].sum()

# Average price
avg_price = df[df['Status'] == 'FILLED']['Price'].mean()

print(f"Fill Rate: {fill_rate:.2%}")
print(f"Total Volume: {total_volume}")
print(f"Average Price: ${avg_price:.2f}")
```

### Health Checks

#### Daily Checks

1. **Position Status**
   - Check position on exchange
   - Verify matches bot expectations
   - Review pending orders

2. **Log Review**
   - Check for errors
   - Review fill rates
   - Monitor wait times

3. **Balance Check**
   - Verify sufficient balance
   - Check fee reserves
   - Monitor margin usage

#### Weekly Checks

1. **Performance Review**
   - Calculate P&L
   - Review trading volume
   - Analyze fill rates
   - Assess parameter effectiveness

2. **Parameter Optimization**
   - Adjust based on results
   - Test new configurations
   - Fine-tune settings

3. **System Maintenance**
   - Clean old logs
   - Update dependencies
   - Backup configurations

### Troubleshooting

#### Position Mismatch

**Symptoms:**
- Bot reports position ≠ active close orders
- Alert notification sent

**Steps:**
1. Check exchange position
2. Check active orders
3. Cancel problematic orders manually
4. Restart bot
5. Monitor for recurrence

#### WebSocket Disconnection

**Symptoms:**
- No order updates
- Bot continues placing orders
- Missing fill notifications

**Steps:**
1. Check internet connection
2. Verify exchange status
3. Restart bot
4. Check firewall/proxy
5. Review API credentials

#### Rate Limiting

**Symptoms:**
- Requests failing
- Slow order placement
- API errors

**Steps:**
1. Increase `--wait-time`
2. Reduce `--max-orders`
3. Check exchange rate limits
4. Implement backoff strategy
5. Contact exchange support

---

## FAQ

### General Questions

**Q: Can I run multiple bots simultaneously?**  
A: Yes, you can run multiple instances with different tickers, exchanges, or accounts.

**Q: What happens if my internet disconnects?**  
A: The bot will attempt to reconnect WebSocket. Existing orders remain on exchange.

**Q: Can I stop the bot anytime?**  
A: Yes, use Ctrl+C for graceful shutdown. Existing orders remain on exchange.

**Q: Do I need to manually close positions?**  
A: No, the bot automatically places close orders for filled open orders.

### Trading Strategy

**Q: What's the best wait time?**  
A: Depends on strategy. Start with 450s, adjust based on performance. Use 300-600s for standard trading.

**Q: Should I use grid step?**  
A: Yes, recommended for organized order placement. Use 0.5-1.0% for most cases.

**Q: When should I use boost mode?**  
A: For volume campaigns on Backpack/Aster. Costs more in fees but generates high volume.

**Q: Can the bot make consistent profits?**  
A: The bot is designed for volume generation, not guaranteed profits. Market risk applies.

### Technical Questions

**Q: Which Python version should I use?**  
A: Python 3.10-3.12 is recommended. GRVT requires 3.10+, Paradex needs 3.9-3.12.

**Q: Can I run on Windows?**  
A: Yes, the bot supports Windows, macOS, and Linux.

**Q: Do I need a VPS?**  
A: Not required, but recommended for 24/7 operation and stable connection.

**Q: How much RAM does it need?**  
A: Typically 100-200MB per bot instance. Most systems can run 5-10 bots simultaneously.

### Troubleshooting

**Q: Bot says "Position mismatch", what do I do?**  
A: Check your position on the exchange manually. Cancel any stuck orders. Restart bot to resync.

**Q: Orders not filling, why?**  
A: Check order prices are competitive. Review tick size. Verify sufficient balance.

**Q: WebSocket keeps disconnecting?**  
A: Check internet stability. Verify exchange status. Review firewall settings. Try different network.

**Q: Bot crashes on start?**  
A: Verify API credentials in `.env`. Check Python version. Install all dependencies. Review error logs.

### Exchange-Specific

**Q: EdgeX says "Invalid Stark key"?**  
A: Ensure key starts with `0x`. Verify key matches account ID. Check key format.

**Q: Paradex won't connect?**  
A: Use Paradex L2 private key, not L1. Ensure Python 3.9-3.12. Use separate virtual environment.

**Q: Lighter account index not found?**  
A: Use API to find correct index. Check wallet address. Verify account exists.

**Q: GRVT requires Python 3.10?**  
A: Yes, upgrade Python to 3.10+. Create new virtual environment with correct version.

---

## Conclusion

This guide covers comprehensive usage of the trading bot. Remember:

1. **Start small** and scale gradually
2. **Monitor regularly** for optimal performance
3. **Adjust parameters** based on results
4. **Manage risk** appropriately
5. **Use notifications** for alerts
6. **Maintain systems** regularly
7. **Learn continuously** from trading data

For additional help:
- Review API documentation
- Check exchange-specific guides
- Join community discussions
- Contact support when needed

Happy trading! 🚀

---

**Document Version**: 1.0  
**Last Updated**: 2025-11-23  
**Compatibility**: All current exchange implementations
