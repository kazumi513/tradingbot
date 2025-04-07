# Enhanced LSTM Model and Sentiment Analysis Script

import os
import time
import requests
import numpy as np
import pandas as pd
import logging
from dotenv import load_dotenv
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline  # For sentiment analysis with BERT
from alpaca_trade_api.rest import REST, TimeFrame
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# Load environment variables
load_dotenv()
ALPACA_API_KEY = os.getenv("ALPCA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPCA_SECRET_KEYS")
ALPACA_BASE_URL = os.getenv("ALPCA_API_BASE_URL")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=ALPACA_BASE_URL)

# Sentiment Analysis with BERT
sentiment_pipeline = pipeline("sentiment-analysis")

# Set up logging
logging.basicConfig(filename='enhanced_trading_bot.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Stock universe (diversified)
ticker_to_name = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "JNJ": "Johnson & Johnson",
    "JPM": "JPMorgan",
    "XOM": "ExxonMobil",
    "HD": "Home Depot",
    "PG": "Procter & Gamble"
}

# Risk limits
MAX_TRADES_PER_DAY = 5
MAX_TRADES_PER_SECTOR = 2
sectors = {
    "AAPL": "Tech",
    "MSFT": "Tech",
    "JNJ": "Healthcare",
    "JPM": "Finance",
    "XOM": "Energy",
    "HD": "Retail",
    "PG": "Consumer"
}
sector_trade_count = {}
trades_executed = 0
open_positions = {}
STOP_LOSS_THRESHOLD = -0.03  # -3%
TAKE_PROFIT_THRESHOLD = 0.05  # +5%

# --- Optimized Sentiment Analysis --- #
def get_stock_sentiment(stock_name):
    url = f"https://newsapi.org/v2/everything?q={stock_name}&language=en&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
    try:
        response = requests.get(url)
        news_data = response.json()
        headlines = [article["title"] for article in news_data.get("articles", [])]
        if len(headlines) > 5:  # Ensure enough data is fetched
            sentiment_scores = [sentiment_pipeline(h)[0]["score"] * (1 if sentiment_pipeline(h)[0]["label"] == "POSITIVE" else -1) for h in headlines]
            return np.mean(sentiment_scores) if sentiment_scores else 0
        else:
            logging.warning(f"Not enough headlines for {stock_name} sentiment.")
            return 0
    except Exception as e:
        logging.error(f"Error fetching sentiment for {stock_name}: {e}")
        return 0

# --- Technical Indicators --- #
def apply_indicators(df):
    df["RSI"] = RSIIndicator(close=df["close"], window=14).rsi()
    df["SMA"] = SMAIndicator(close=df["close"], window=20).sma_indicator()
    return df

# --- Enhanced LSTM Model --- #
def predict_trend(df):
    data = df["close"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    sequence_length = 60  # Window size
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i - sequence_length:i])
        y.append(scaled_data[i])
    if not X:
        return 0

    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Load or improve model
    model_path = "enhanced_lstm_model.h5"
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(X.shape[1], 1)),  # Increased neurons
            Dropout(0.3),  # Higher dropout for regularization
            LSTM(128),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=20, batch_size=64, verbose=0)  # Optimized training parameters
        model.save(model_path)

    last_60 = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)
    predicted = model.predict(last_60)
    return predicted[0][0] - scaled_data[-1][0]  # Return delta

# --- Data Fetching --- #
def fetch_stock_data(ticker):
    try:
        bars = api.get_bars(ticker, TimeFrame.Day, limit=100).df
        return bars.reset_index()[["timestamp", "open", "high", "low", "close", "volume"]]
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

# --- Trade Execution --- #
def execute_trade(ticker, action):
    global trades_executed, sector_trade_count, open_positions

    sector = sectors.get(ticker, "Other")
    if trades_executed >= MAX_TRADES_PER_DAY:
        logging.warning(f"Trade limit reached. Skipping {ticker}.")
        return
    if sector_trade_count.get(sector, 0) >= MAX_TRADES_PER_SECTOR:
        logging.warning(f"Sector trade limit reached for {sector}. Skipping {ticker}.")
        return

    qty = 1
    try:
        api.submit_order(
            symbol=ticker,
            qty=qty,
            side=action.lower(),
            type="market",
            time_in_force="gtc"
        )
        trades_executed += 1
        sector_trade_count[sector] = sector_trade_count.get(sector, 0) + 1
        open_positions[ticker] = {
            "side": action.lower(),
            "entry_price": api.get_latest_trade(ticker).price
        }
        logging.info(f"Executed {action} on {ticker} in {sector} sector.")
    except Exception as e:
        logging.error(f"Failed to execute trade for {ticker}: {e}")

# --- Auto-Close Logic --- #
def monitor_positions():
    for ticker, pos in open_positions.copy().items():
        try:
            current_price = api.get_latest_trade(ticker).price
            entry_price = pos["entry_price"]
            side = pos["side"]
            change = (current_price - entry_price) / entry_price

            if side == "buy" and (change >= TAKE_PROFIT_THRESHOLD or change <= STOP_LOSS_THRESHOLD):
                api.submit_order(symbol=ticker, qty=1, side="sell", type="market", time_in_force="gtc")
                logging.info(f"Closed BUY {ticker} @ {current_price:.2f} | P/L: {change:.2%}")
                open_positions.pop(ticker)
            elif side == "sell" and (-change >= TAKE_PROFIT_THRESHOLD or -change <= STOP_LOSS_THRESHOLD):
                api.submit_order(symbol=ticker, qty=1, side="buy", type="market", time_in_force="gtc")
                logging.info(f"Closed SELL {ticker} @ {current_price:.2f} | P/L: {-change:.2%}")
                open_positions.pop(ticker)
        except Exception as e:
            logging.error(f"Error monitoring position for {ticker}: {e}")
            continue

# --- Main Logic --- #
def main():
    monitor_positions()
    for ticker, name in ticker_to_name.items():
        logging.info(f"\nProcessing {name} ({ticker})")

        sentiment = get_stock_sentiment(name)
        logging.info(f"Sentiment Score: {sentiment:.3f}")

        df = fetch_stock_data(ticker)
        if df.empty or len(df) < 65:
            logging.warning(f"Insufficient data for {ticker}. Skipping.")
            continue

        df = apply_indicators(df)
        rsi = df.iloc[-1]["RSI"]
        sma = df.iloc[-1]["SMA"]
        current_price = df.iloc[-1]["close"]

        lstm_trend = predict_trend(df)
        logging.info(f"LSTM Trend Delta: {lstm_trend:.4f}")
        logging.info(f"RSI: {rsi:.2f}, SMA: {sma:.2f}, Price: {current_price:.2f}")

        # Decision rules
        if sentiment > 0.3 and lstm_trend > 0.01 and current_price > sma and rsi < 70:
            execute_trade(ticker, "Buy")
        elif sentiment < -0.3 and lstm_trend < -0.01 and current_price < sma and rsi > 30:
            execute_trade(ticker, "Sell")
        else:
            logging.info("No strong signal. Skipping.")

if __name__ == "__main__":
    main()
