# Stock Trading Bot with LSTM and Sentiment Analysis

A Python trading bot that combines **LSTM neural networks** for stock price prediction with **real-time sentiment analysis** from financial news. Uses **Alpaca API** for paper trading and **Yahoo Finance** for market data. Features:  

📈 **LSTM Model** – Optimized with Optuna for price forecasting  
📰 **News Sentiment Analysis** – VADER-powered market mood detection  
⚖️ **Risk Assessment** – Volatility-based risk evaluation  
💹 **Automated Trading** – Executes buy/sell orders via Alpaca  

Ideal for algorithmic trading experiments, AI finance projects, or learning ML in trading. **Paper trading only** – use at your own risk with real funds.  

*Key tools: TensorFlow, Alpaca API, NewsAPI, yfinance, Optuna*  

# Netflix Clone

<video width="100%" controls>
  <source src="https://files.catbox.moe/k6uuo6.webm" type="video/webm">
  Your browser does not support the video tag.
</video>

---  
**🚀 Try it out:** Clone, add API keys to `.env`, and run `python desi.py` (default: analyzes TSLA). Contributions welcome!

## Overview

This Python script implements a stock trading bot that combines LSTM neural networks for price prediction with sentiment analysis from financial news to make trading decisions. The bot uses Alpaca API for executing trades and Yahoo Finance for market data.

## Features

- **Real-time Stock Data Fetching**: Retrieves live stock data using Yahoo Finance API
- **Sentiment Analysis**: Analyzes market sentiment from financial news using VADER
- **LSTM Model**: Predicts stock prices using an optimized LSTM neural network
- **Risk Assessment**: Evaluates stock volatility for risk management
- **Portfolio Analysis**: Monitors portfolio health and risk levels
- **Automated Trading**: Executes trades through Alpaca's paper trading API

```markdown
## 🛠 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/stock-trading-bot.git
   cd stock-trading-bot
   ```

2. **Set up a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install numpy pandas yfinance tensorflow scikit-learn vaderSentiment requests matplotlib seaborn optuna alpaca-trade-api python-dotenv
   ```

4. **Set up environment variables**:
   - Create a `.env` file in the project root
   - Add your API keys:
     ```
     NEWS_API_KEY=your_newsapi_key_here
     ALPACA_API_KEY=your_alpaca_key_here
     ALPACA_SECRET_KEY=your_alpaca_secret_here
     ```

5. **Run the application**:
   ```bash
   python desi.py
   ```
```

## Prerequisites

Before running the script, ensure you have:

1. Python 3.8 or higher
2. The following API keys in your `.env` file:
   - NewsAPI key
   - Alpaca API key and secret key

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/stock-trading-bot.git
   cd stock-trading-bot
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root with your API keys:
   ```
   NEWS_API_KEY=your_news_api_key
   ALPACA_API_KEY=your_alpaca_key
   ALPACA_SECRET_KEY=your_alpaca_secret
   ```

## Usage

1. Run the script:
   ```bash
   python desi.py
   ```

2. By default, the script will:
   - Fetch market sentiment from financial news
   - Train an LSTM model on Tesla (TSLA) stock data
   - Execute a buy/sell order based on the sentiment analysis

3. To customize:
   - Change the `ticker` variable at the bottom of the script to analyze different stocks
   - Modify the trading logic in the `execute_trade` function

## Code Structure

The main components of the script are:

1. **API Configuration**:
   - Sets up Alpaca and NewsAPI connections
   - Loads environment variables

2. **Data Fetching**:
   - `fetch_real_time_stock_data()`: Gets stock data from Yahoo Finance
   - `get_market_sentiment()`: Retrieves and analyzes financial news sentiment

3. **Machine Learning**:
   - `optimize_lstm()`: Uses Optuna for hyperparameter optimization
   - `train_model()`: Trains the LSTM model on stock data

4. **Trading Functions**:
   - `assess_risk()`: Evaluates stock volatility
   - `portfolio_health()`: Analyzes portfolio performance
   - `execute_trade()`: Places orders through Alpaca

## Configuration Options

You can modify these parameters in the script:

- `ticker`: Stock symbol to analyze (default: "TSLA")
- `interval`: Data frequency (default: "1m" for 1-minute intervals)
- `period`: Historical data period (default: "7d" for 7 days)
- LSTM hyperparameters in the `optimize_lstm()` function

## Limitations

1. This is a demonstration script using paper trading - use at your own risk with real money
2. The LSTM model is trained on limited data (7 days by default)
3. News sentiment analysis is based on a simple keyword search ("stock market")

## Future Enhancements

1. Add more sophisticated technical indicators
2. Implement portfolio rebalancing logic
3. Add backtesting functionality
4. Support for multiple stocks and asset classes
5. Improved error handling and logging

```markdown
## 🚨 Troubleshooting

**CUDA errors**: If you encounter GPU-related errors:
```bash
export CUDA_VISIBLE_DEVICES="-1"  # Force CPU usage
```

**Missing dependencies**: Ensure pip is up to date:
```bash
pip install --upgrade pip
```

**Alpaca API errors**: Verify your:
- Paper trading account is activated
- API keys are correctly set in `.env`
```

```markdown
---
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

For questions or issues, please open an issue on the GitHub repository.
