import os
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import alpaca_trade_api as tradeapi
from tqdm import tqdm
import time

# 🔹 Force TensorFlow to use CPU (Fixes CUDA Errors)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN optimizations

# 🔹 Hardcoded Dummy API Keys (Replace with real ones in production)
NEWS_API_KEY = "b50fe45c07784ffa937faf5fe426b62c"
ALPACA_API_KEY = "PKR447NDRZNIBNP8553T"
ALPACA_SECRET_KEY = "YKSqGYAIj12Omnqe5ZRAsrP7R9oDBH10v6jpxvMU"

# 🔹 Alpaca API Setup
alpaca_api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url="https://paper-api.alpaca.markets/")

# 🔹 Load Sentiment Analysis Tool
sentiment_analyzer = SentimentIntensityAnalyzer()

# 🔹 List of S&P 500 stocks with focus on requested companies
SP500_TICKERS = [
    # Technology
    'META', 'NVDA', 'AMD', 'CRM', 'GOOGL', 'MSFT', 'ARM', 'AVGO', 'MSTR', 
    'APP', 'AMZN', 'SHOP', 'COST', 'PYPL', 'COIN', 'HOOD', 'TSM', 'ASTL',
    
    # Financial Services
    'JPM', 'BAC', 'NU', 
    
    # Consumer Discretionary
    'WMT', 'ULTA', 'CVNA', 'LYFT', 'UBER', 'TSLA', 'DAL', 
    
    # Healthcare
    'PFE', 'RPRX', 'WBA', 'HIMS', 
    
    # Energy
    'CEG', 'SSL', 
    
    # Entertainment
    'SPOT', 'RDDT', 'DUOL', 
    
    # Industrials
    'STRL', 'G'
]

# 🔹 Fetch Market Sentiment from NewsAPI for specific stocks
def get_market_sentiment(ticker=None):
    """Fetch real-time financial news and analyze sentiment."""
    query = "stock market" if ticker is None else f"{ticker} stock"
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
    try:
        response = requests.get(url)
        news_data = response.json()
        if "articles" in news_data:
            headlines = [article["title"] for article in news_data["articles"][:10]]
            sentiment_scores = [sentiment_analyzer.polarity_scores(headline)["compound"] for headline in headlines]
            return np.mean(sentiment_scores)
        else:
            return 0  # Neutral sentiment if API fails
    except Exception as e:
        print(f"⚠️ Error fetching news for {ticker if ticker else 'market'}: {e}")
        return 0  # Neutral sentiment if there's an error

# 🔹 Fetch Real-Time Stock Data
def fetch_real_time_stock_data(ticker, interval="1m", period="7d"):
    """Fetches live stock data for high-frequency trading."""
    try:
        df = yf.download(ticker, interval=interval, period=period, auto_adjust=True)
        if df.empty:
            print(f"⚠️ No stock data available for {ticker}")
            return None
        df["Return"] = df["Close"].pct_change()
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"⚠️ Error fetching data for {ticker}: {e}")
        return None

# 🔹 Execute Trade Using Alpaca API
def execute_trade(ticker, action):
    """Places a buy or sell order on Alpaca."""
    try:
        alpaca_api.submit_order(
            symbol=ticker, 
            qty=1, 
            side=action.lower(), 
            type="market", 
            time_in_force="gtc"
        )
        print(f"✅ Order placed: {action} 1 share of {ticker}")
    except Exception as e:
        print(f"⚠️ Failed to execute {action} order for {ticker}: {e}")

# 🔹 Main Trading Function
def run_trading_bot():
    """Run the trading bot for all S&P 500 stocks."""
    print("📈 Starting S&P 500 Trading Bot")
    
    # Get general market sentiment
    market_sentiment = get_market_sentiment()
    print(f"📊 Overall Market Sentiment Score: {market_sentiment:.2f}")
    
    for ticker in tqdm(SP500_TICKERS, desc="Processing Stocks"):
        try:
            # Get stock-specific sentiment
            stock_sentiment = get_market_sentiment(ticker)
            combined_sentiment = (market_sentiment + stock_sentiment) / 2
            
            # Fetch stock data
            df = fetch_real_time_stock_data(ticker)
            if df is None:
                continue
                
            # Simple trading decision based on sentiment
            if combined_sentiment > 0.1:  # Positive sentiment threshold
                execute_trade(ticker, "BUY")
            elif combined_sentiment < -0.1:  # Negative sentiment threshold
                execute_trade(ticker, "SELL")
            
            # Add delay to avoid rate limiting
            time.sleep(1)
            
        except Exception as e:
            print(f"⚠️ Error processing {ticker}: {e}")
            continue

# 🔹 Run the trading bot
if __name__ == "__main__":
    run_trading_bot()
