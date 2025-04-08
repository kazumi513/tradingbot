# S&P 500 Trading Bot with Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Alpaca](https://img.shields.io/badge/Alpaca-Trading%20API-green)

An automated trading bot that analyzes market sentiment and executes trades on S&P 500 stocks using real-time news analysis and technical indicators.

## Features

- **Real-time Sentiment Analysis**: Uses VADER sentiment analysis on financial news headlines
- **Multi-Stock Support**: Tracks and trades 50+ S&P 500 stocks including major tech companies
- **Alpaca Integration**: Executes paper trades through Alpaca's API
- **Yahoo Finance Data**: Pulls real-time stock price data for analysis
- **Intelligent Decision Making**: Combines market-wide and stock-specific sentiment

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sp500-trading-bot.git
   cd sp500-trading-bot
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your API keys:
   - Create a `.env` file in the project directory
   - Add your Alpaca and NewsAPI credentials:
     ```
     ALPACA_API_KEY=your_api_key_here
     ALPACA_SECRET_KEY=your_secret_key_here
     NEWS_API_KEY=your_newsapi_key_here
     ```

## Usage

Run the trading bot:
```bash
python tradingscript.py
```

The bot will:
1. Fetch overall market sentiment from financial news
2. Analyze sentiment for each individual stock
3. Make trading decisions based on combined sentiment scores
4. Execute paper trades through Alpaca

## Configuration

Modify the following variables in `desi.py` to customize behavior:

```python
# Trading parameters
SENTIMENT_BUY_THRESHOLD = 0.1    # Buy when sentiment > 0.1
SENTIMENT_SELL_THRESHOLD = -0.1  # Sell when sentiment < -0.1
TRADE_QUANTITY = 1               # Number of shares to trade

# Stock selection
SP500_TICKERS = [
    'META', 'NVDA', 'AMD', ...  # Modify this list as needed
]
```

## Included Stocks

The bot currently tracks these major S&P 500 components:

- **Tech**: META, NVDA, AMD, MSFT, GOOGL, AMZN
- **Semiconductors**: AVGO, TSM, ARM
- **Financials**: JPM, BAC, COIN
- **Retail**: WMT, COST, ULTA
- **Transportation**: UBER, LYFT, DAL
- **Energy**: CEG
- **Healthcare**: PFE, HIMS

(Full list in the script)

## Requirements

- Python 3.8+
- Alpaca Trading Account
- NewsAPI Key
- Libraries listed in `requirements.txt`

## Disclaimer

⚠️ **This is for educational purposes only** ⚠️

- This is a paper trading script using mock orders
- Past performance is not indicative of future results
- Always test strategies thoroughly before using real money
- The author is not responsible for any trading losses

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
