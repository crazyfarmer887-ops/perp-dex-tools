#!/bin/bash

# BingX Hedge Mode Bot Runner Script

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== BingX Hedge Position Mode Bot ===${NC}"
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${RED}Error: .env file not found!${NC}"
    echo "Please copy env_example.txt to .env and configure your BingX API credentials."
    exit 1
fi

# Check if BingX credentials are set
if ! grep -q "BINGX_API_KEY=" .env || ! grep -q "BINGX_API_SECRET=" .env; then
    echo -e "${YELLOW}Warning: BingX API credentials may not be configured in .env${NC}"
    echo "Make sure BINGX_API_KEY and BINGX_API_SECRET are set."
    echo ""
fi

# Default values
TICKER="BTC"
QUANTITY="0.001"
TP_ROI="10"
SL_ROI="10"
ITERATIONS="1"
SLEEP="0"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --ticker)
            TICKER="$2"
            shift 2
            ;;
        --quantity)
            QUANTITY="$2"
            shift 2
            ;;
        --tp-roi)
            TP_ROI="$2"
            shift 2
            ;;
        --sl-roi)
            SL_ROI="$2"
            shift 2
            ;;
        --iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        --sleep)
            SLEEP="$2"
            shift 2
            ;;
        --testnet)
            export BINGX_ENVIRONMENT="testnet"
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --ticker SYMBOL      Trading pair (default: BTC)"
            echo "  --quantity AMOUNT    Position size for each side (default: 0.001)"
            echo "  --tp-roi PERCENT     Take profit ROI % (default: 10)"
            echo "  --sl-roi PERCENT     Stop loss ROI % (default: 10)"
            echo "  --iterations N       Number of cycles (default: 1)"
            echo "  --sleep SECONDS      Sleep between iterations (default: 0)"
            echo "  --testnet           Use testnet environment"
            echo "  --help              Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --ticker BTC --quantity 0.001 --tp-roi 5 --sl-roi 5"
            echo "  $0 --ticker ETH --quantity 0.1 --tp-roi 10 --sl-roi 10 --iterations 3"
            echo "  $0 --testnet --ticker BTC --quantity 0.01"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Display configuration
echo -e "${YELLOW}Configuration:${NC}"
echo "  Ticker: $TICKER"
echo "  Quantity: $QUANTITY"
echo "  TP ROI: $TP_ROI%"
echo "  SL ROI: $SL_ROI%"
echo "  Iterations: $ITERATIONS"
echo "  Sleep: $SLEEP seconds"
echo "  Environment: ${BINGX_ENVIRONMENT:-prod}"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the bot
echo -e "${GREEN}Starting BingX Hedge Bot...${NC}"
echo ""

python3 hedge/hedge_mode_bingx.py "$TICKER" "$QUANTITY" \
    --tp-roi "$TP_ROI" \
    --sl-roi "$SL_ROI" \
    --iterations "$ITERATIONS" \
    --sleep "$SLEEP"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}Bot finished successfully${NC}"
else
    echo ""
    echo -e "${RED}Bot exited with error${NC}"
    exit 1
fi