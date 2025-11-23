# Quick Start Guide

Get started with the Perp DEX Trading Bot in 5 minutes.

---

## Prerequisites

- Python 3.10 - 3.12
- API credentials from your chosen exchange
- Basic understanding of perpetual futures trading

---

## Installation Steps

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd perp-dex-tools

# Create virtual environment
python3 -m venv env
source env/bin/activate  # Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example environment file
cp env_example.txt .env

# Edit .env with your API credentials
nano .env  # or use your preferred editor
```

**Example `.env` configuration for EdgeX:**

```bash
# EdgeX Configuration
EDGEX_ACCOUNT_ID=your_account_id_here
EDGEX_STARK_PRIVATE_KEY=your_private_key_here
EDGEX_BASE_URL=https://pro.edgex.exchange
EDGEX_WS_URL=wss://quote.edgex.exchange

# Optional: Telegram Notifications
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

### 3. Run Your First Bot

```bash
# Start trading ETH with default parameters
python runbot.py --exchange edgex --ticker ETH --quantity 0.1 \
  --take-profit 0.02 --max-orders 40 --wait-time 450
```

**What this does:**
- Trades ETH perpetual on EdgeX
- Places 0.1 ETH per order
- Takes profit at 0.02% gain
- Maintains up to 40 concurrent orders
- Waits 450 seconds between new orders

---

## Common Use Cases

### Use Case 1: Conservative Market Making

**Goal:** Slow and steady trading with low risk

```bash
python runbot.py \
  --exchange edgex \
  --ticker ETH \
  --quantity 0.05 \
  --take-profit 0.02 \
  --max-orders 20 \
  --wait-time 600 \
  --grid-step 0.5
```

**Parameters explained:**
- Small quantity (0.05 ETH) = lower risk
- Low max orders (20) = limited exposure
- Long wait time (600s) = patient execution
- Grid step (0.5%) = prevents order clustering

### Use Case 2: Aggressive Volume Generation

**Goal:** Maximum trading volume, higher risk

```bash
python runbot.py \
  --exchange backpack \
  --ticker ETH \
  --quantity 0.2 \
  --take-profit 0.01 \
  --max-orders 60 \
  --wait-time 300 \
  --boost
```

**Parameters explained:**
- Larger quantity (0.2 ETH) = more volume
- Higher max orders (60) = more concurrent trades
- Shorter wait time (300s) = faster execution
- Boost mode = immediate taker closes

### Use Case 3: Range-Bound Trading

**Goal:** Trade only within price range

```bash
python runbot.py \
  --exchange edgex \
  --ticker BTC \
  --direction buy \
  --quantity 0.05 \
  --take-profit 0.02 \
  --max-orders 40 \
  --wait-time 450 \
  --stop-price 105000 \
  --pause-price 100000
```

**Parameters explained:**
- Direction buy = only long positions
- Stop price (105000) = exit if BTC reaches $105k
- Pause price (100000) = pause if BTC reaches $100k

### Use Case 4: Hedge Mode Trading

**Goal:** Risk-free volume with hedged positions

```bash
python hedge_mode.py \
  --exchange backpack \
  --ticker BTC \
  --size 0.05 \
  --iter 20 \
  --fill-timeout 5 \
  --tp-roi 0.3 \
  --sl-roi 0.2
```

**Parameters explained:**
- Backpack for maker orders
- Lighter for hedge (automatic)
- 20 iterations = 20 complete cycles
- Take profit at 0.3% ROI
- Stop loss at 0.2% loss

---

## Understanding Key Parameters

### --quantity / --size

Order size per trade.

**Examples:**
- ETH: `0.1` (0.1 ETH per order)
- BTC: `0.05` (0.05 BTC per order)
- SOL: `5` (5 SOL per order)

**Tip:** Start small, increase gradually

### --take-profit

Profit target as a percentage.

**Examples:**
- `0.01` = 0.01% profit target
- `0.02` = 0.02% profit target
- `0.05` = 0.05% profit target

**Tip:** Smaller = faster fills, larger = more profit per trade

### --max-orders

Maximum concurrent close orders.

**Examples:**
- `20` = Conservative, low risk
- `40` = Moderate, balanced
- `60` = Aggressive, high volume

**Tip:** More orders = more capital tied up

### --wait-time

Base wait time between orders (seconds).

**Examples:**
- `300` = 5 minutes (fast)
- `450` = 7.5 minutes (moderate)
- `600` = 10 minutes (slow)

**Note:** Actual wait time scales with order count

### --grid-step

Minimum distance between close orders (%).

**Examples:**
- `-100` = No restriction (default)
- `0.3` = 0.3% minimum distance
- `0.5` = 0.5% minimum distance

**Tip:** Use to prevent order clustering

### --direction

Trading direction.

**Options:**
- `buy` = Open long positions
- `sell` = Open short positions

**Tip:** Match your market outlook

### --stop-price

Exit bot if price reaches this level.

**Examples:**
- `-1` = Disabled (default)
- `5500` = Exit if price reaches $5500

**Logic:**
- Buy direction: Exit if price >= stop_price
- Sell direction: Exit if price <= stop_price

### --pause-price

Pause bot if price reaches this level.

**Examples:**
- `-1` = Disabled (default)
- `5000` = Pause if price reaches $5000

**Difference from stop-price:** Resumes when price moves away

---

## Monitoring Your Bot

### Log Files

Located in `logs/` directory:

```
logs/
├── edgex_ETH_activity.log       # Detailed activity log
└── edgex_ETH_orders.csv         # Trade history
```

### View Live Logs

```bash
# Follow activity log
tail -f logs/edgex_ETH_activity.log

# View recent trades
tail -20 logs/edgex_ETH_orders.csv
```

### Console Output

The bot prints status every 60 seconds:

```
--------------------------------
[EDGEX_ETH] Current Position: 1.5 | Active closing amount: 1.5 | Order quantity: 15
--------------------------------
```

**What to monitor:**
- Position should match active closing amount
- If mismatch > 2x quantity, bot will alert and stop

---

## Stopping the Bot

### Graceful Stop

Press `Ctrl+C` to gracefully stop the bot.

**What happens:**
1. Bot stops placing new orders
2. Existing orders remain active
3. WebSocket connections closed
4. Final status logged

### Emergency Stop

If bot is unresponsive:

```bash
# Find process ID
ps aux | grep runbot.py

# Kill process
kill <PID>
```

**Note:** Active orders remain on exchange

---

## Telegram Notifications (Optional)

### Setup Steps

1. **Create Telegram Bot:**
   - Message [@BotFather](https://t.me/BotFather)
   - Send `/newbot` and follow instructions
   - Save the bot token

2. **Get Chat ID:**
   - Message your bot
   - Visit: `https://api.telegram.org/bot<TOKEN>/getUpdates`
   - Find `"chat":{"id":123456789}`

3. **Add to `.env`:**
   ```bash
   TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
   TELEGRAM_CHAT_ID=123456789
   ```

4. **Test:**
   ```python
   from helpers.telegram_bot import TelegramBot
   
   with TelegramBot(token, chat_id) as bot:
       bot.send_text("Test message!")
   ```

**See:** [docs/telegram-bot-setup.md](docs/telegram-bot-setup.md) for detailed guide

---

## Multi-Account Setup

### Scenario: Multiple Accounts on Same Exchange

**Setup:**

```bash
# Create account-specific .env files
# account_main.env
ACCOUNT_NAME=MAIN
BACKPACK_PUBLIC_KEY=main_key
BACKPACK_SECRET_KEY=main_secret

# account_sub.env
ACCOUNT_NAME=SUB
BACKPACK_PUBLIC_KEY=sub_key
BACKPACK_SECRET_KEY=sub_secret
```

**Usage:**

```bash
# Terminal 1: Main account
python runbot.py --env-file account_main.env --exchange backpack --ticker ETH --quantity 0.1

# Terminal 2: Sub account
python runbot.py --env-file account_sub.env --exchange backpack --ticker ETH --quantity 0.05
```

**Logs:**
- Main: `logs/backpack_ETH_MAIN_*`
- Sub: `logs/backpack_ETH_SUB_*`

---

## Troubleshooting

### Issue: "Env file not found"

**Solution:**
```bash
# Check file exists
ls -la .env

# Verify path
python runbot.py --env-file .env  # Explicit path
```

### Issue: "Failed to create exchange client"

**Causes:**
1. Missing API credentials in `.env`
2. Invalid credentials
3. Network connectivity issue

**Solution:**
```bash
# Verify .env file
cat .env

# Test credentials manually on exchange
```

### Issue: Bot connects but doesn't trade

**Possible causes:**
1. Insufficient balance
2. Max orders reached
3. Grid step blocking orders
4. Pause price triggered

**Solution:**
```bash
# Check logs
tail -f logs/exchange_ticker_activity.log

# Look for messages about:
# - "Current Position"
# - "Order quantity"
# - Grid step conditions
```

### Issue: Orders not filling

**Causes:**
1. Price too far from market
2. Low liquidity
3. Post-only order rejection

**Solution:**
- Reduce wait time for more frequent orders
- Use boost mode for immediate fills
- Check exchange order book depth

### Issue: Position mismatch warning

**Message:**
```
ERROR: Position mismatch detected
current position: 2.5 | active closing amount: 1.0
```

**Causes:**
1. Manual trades outside bot
2. Partial fills not tracked
3. WebSocket disconnection

**Solution:**
1. Stop the bot
2. Manually reconcile positions on exchange
3. Close or adjust orders to match
4. Restart bot

---

## Safety Tips

### Start Small

- Begin with minimum quantities
- Test with small max-orders (10-20)
- Use longer wait times (600+)

### Monitor Regularly

- Check logs every few hours
- Verify position matches close orders
- Watch for error messages

### Set Limits

- Use `--stop-price` to cap exposure
- Set reasonable `--max-orders`
- Don't exceed your risk tolerance

### Understand Risks

- **No stop loss:** Bot doesn't automatically cut losses
- **Directional risk:** Wrong market direction = losses
- **Funding rates:** Long-term positions incur funding
- **Slippage:** Market orders in boost mode have slippage

---

## Next Steps

### Once Comfortable

1. **Experiment with parameters:**
   - Try different take-profit percentages
   - Adjust wait times and max orders
   - Test grid step values

2. **Try multiple pairs:**
   - Run ETH and BTC simultaneously
   - Use different exchanges
   - Compare performance

3. **Explore hedge mode:**
   - Learn two-exchange hedging
   - Test with small sizes first
   - Understand ROI targets

4. **Customize the bot:**
   - Read the API documentation
   - Modify trading logic
   - Add custom strategies

### Resources

- **Full API Docs:** [API_DOCUMENTATION.md](API_DOCUMENTATION.md)
- **Function Reference:** [FUNCTION_REFERENCE.md](FUNCTION_REFERENCE.md)
- **Chinese README:** [README.md](README.md)
- **English README:** [README_EN.md](README_EN.md)
- **Twitter:** [@yourQuantGuy](https://x.com/yourQuantGuy)

---

## Example: Complete First-Time Workflow

### Step-by-Step

```bash
# 1. Setup
git clone <repo-url>
cd perp-dex-tools
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt

# 2. Configure
cp env_example.txt .env
nano .env  # Add your EdgeX credentials

# 3. Start conservatively
python runbot.py \
  --exchange edgex \
  --ticker ETH \
  --quantity 0.05 \
  --take-profit 0.02 \
  --max-orders 20 \
  --wait-time 600 \
  --grid-step 0.5

# 4. Monitor in another terminal
tail -f logs/edgex_ETH_activity.log

# 5. Stop gracefully when done
# Press Ctrl+C in bot terminal
```

### What to Expect

**First 5 minutes:**
- Bot connects to exchange
- Logs initial configuration
- Checks existing positions
- Places first order

**After 1 hour:**
- Several orders opened and closed
- Position builds up gradually
- Close orders accumulate
- Periodic status updates every 60s

**After several hours:**
- Steady trading rhythm established
- Close orders being filled
- Volume accumulating
- Check for any warnings in logs

---

## Quick Command Reference

### Standard Trading

```bash
# Basic
python runbot.py --exchange EXCHANGE --ticker SYMBOL --quantity SIZE

# Full parameters
python runbot.py \
  --exchange edgex \
  --ticker ETH \
  --quantity 0.1 \
  --take-profit 0.02 \
  --direction buy \
  --max-orders 40 \
  --wait-time 450 \
  --grid-step 0.5 \
  --stop-price 5500 \
  --pause-price 5000
```

### Hedge Mode

```bash
# Basic
python hedge_mode.py --exchange EXCHANGE --ticker SYMBOL --size SIZE --iter COUNT

# With ROI targets
python hedge_mode.py \
  --exchange backpack \
  --ticker BTC \
  --size 0.05 \
  --iter 20 \
  --tp-roi 0.4 \
  --sl-roi 0.2
```

### Monitoring

```bash
# Watch logs
tail -f logs/exchange_ticker_activity.log

# View trades
cat logs/exchange_ticker_orders.csv

# Check bot process
ps aux | grep runbot.py
```

---

## Getting Help

### Check Logs First

Logs contain detailed information about:
- Connection status
- Order placements and fills
- Position updates
- Error messages

### Common Error Messages

**"LIGHTER_ACCOUNT_INDEX not found"**
- See Configuration Guide to find your account index

**"API rate limit exceeded"**
- Increase wait time
- Reduce order frequency

**"Insufficient balance"**
- Deposit more funds
- Reduce order quantity

### Community Support

- Open GitHub issue for bugs
- Follow [@yourQuantGuy](https://x.com/yourQuantGuy) for updates
- Read full documentation for advanced topics

---

## Summary Checklist

- [ ] Python 3.10+ installed
- [ ] Repository cloned
- [ ] Virtual environment created and activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] `.env` file created with API credentials
- [ ] Tested with small quantity first
- [ ] Monitoring logs in separate terminal
- [ ] Understanding of key parameters
- [ ] Know how to stop bot gracefully
- [ ] Aware of risks and limitations

**Ready to trade!** 🚀

---

**Note:** Always trade responsibly. Start small, monitor closely, and never risk more than you can afford to lose.

