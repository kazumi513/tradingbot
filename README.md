# Enhanced Trading Bot with LSTM and Sentiment Analysis

## Overview
This script implements an automated trading bot that combines:
- **LSTM-based trend prediction** for technical analysis
- **Sentiment analysis** of news headlines using BERT
- **Traditional technical indicators** (RSI, SMA)
- **Risk management** features including sector diversification and trade limits

## Features

### Technical Analysis
- **LSTM Model**: Predicts price trends using a 60-day window with 2 LSTM layers (128 neurons each)
- **Technical Indicators**:
  - Relative Strength Index (RSI)
  - Simple Moving Average (SMA)

### Sentiment Analysis
- Uses BERT-based sentiment analysis pipeline
- Aggregates sentiment scores from recent news headlines
- Only considers signals when sufficient news data is available

### Risk Management
- Daily trade limits (5 trades/day max)
- Sector diversification (2 trades/sector max)
- Automatic position monitoring with:
  - Stop-loss (-3%)
  - Take-profit (+5%)

### Supported Stocks
The bot trades a diversified portfolio of 7 stocks across sectors:
- Tech: AAPL, MSFT
- Healthcare: JNJ
- Finance: JPM
- Energy: XOM
- Retail: HD
- Consumer: PG

## Requirements
- Python 3.7+
- Required packages:
  ```
  numpy pandas requests python-dotenv nltk alpaca-trade-api ta tensorflow transformers
  ```
- API keys needed:
  - Alpaca Market Data
  - NewsAPI

## Setup
1. Create a `.env` file with your API keys:
   ```
   ALPCA_API_KEY=your_key
   ALPCA_SECRET_KEYS=your_secret
   ALPCA_API_BASE_URL=https://paper-api.alpaca.markets
   NEWS_API_KEY=your_news_api_key
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the script:
```bash
python tradingscript.py
```

The bot will:
1. Monitor existing positions for stop-loss/take-profit conditions
2. Analyze each stock in the universe
3. Execute trades when strong signals are detected
4. Log all actions to `enhanced_trading_bot.log`

## Decision Logic
Trades are executed when all conditions are met:

**Buy Signal**:
- Positive sentiment (> 0.3)
- LSTM predicts upward trend (> 0.01 delta)
- Price above SMA
- RSI < 70 (not overbought)

**Sell Signal**:
- Negative sentiment (< -0.3)
- LSTM predicts downward trend (< -0.01 delta)
- Price below SMA
- RSI > 30 (not oversold)

## Notes
- The bot uses Alpaca's paper trading environment by default
- The LSTM model is saved/loaded from `enhanced_lstm_model.h5`
- All trading activity is logged with timestamps for audit purposes
- Risk parameters can be adjusted in the script (STOP_LOSS_THRESHOLD, TAKE_PROFIT_THRESHOLD, etc.)
