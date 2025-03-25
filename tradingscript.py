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
from dotenv import load_dotenv  # Load .env variables

# 🔹 Force TensorFlow to use CPU (Fixes CUDA Errors)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN optimizations

# 🔹 Load environment variables from .env file
load_dotenv()

# 🔹 Fetch API Keys from Environment Variables
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# 🔹 Check if API keys are loaded correctly
if not NEWS_API_KEY or not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    raise ValueError("🚨 Missing API keys! Ensure they are in the .env file.")

# 🔹 Alpaca API Setup
alpaca_api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url="https://paper-api.alpaca.markets/v2")

# 🔹 Load Sentiment Analysis Tool
sentiment_analyzer = SentimentIntensityAnalyzer()

# 🔹 Fetch Market Sentiment from NewsAPI
def get_market_sentiment():
    """Fetch real-time financial news and analyze sentiment."""
    url = f"https://newsapi.org/v2/everything?q=stock market&language=en&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
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
        print(f"⚠️ Error fetching news: {e}")
        return 0  # Neutral sentiment if there's an error

# 🔹 Fetch Real-Time Stock Data
def fetch_real_time_stock_data(ticker, interval="1m", period="7d"):
    """Fetches live stock data for high-frequency trading."""
    df = yf.download(ticker, interval=interval, period=period, auto_adjust=True)
    if df.empty:
        print(f"⚠️ No stock data available for {ticker}")
    df["Return"] = df["Close"].pct_change()
    df.dropna(inplace=True)
    return df

# 🔹 Risk Assessment Function
def assess_risk(df):
    """Calculates stock volatility for risk assessment."""
    if df.empty or "Return" not in df.columns:
        return "Unknown Risk"
    volatility = df["Return"].std()
    return "High Risk" if volatility > 0.02 else "Low Risk"

# 🔹 Portfolio Health Function
def portfolio_health(portfolio):
    """Analyzes a portfolio’s performance."""
    total_value = 0
    risk_levels = []
    for stock in portfolio:
        df = fetch_real_time_stock_data(stock["ticker"])
        if not df.empty and not df["Close"].dropna().empty:
            latest_price = df["Close"].dropna().iloc[-1].item()
            total_value += stock["shares"] * latest_price
            risk_levels.append(assess_risk(df))
        else:
            print(f"⚠️ No data for {stock['ticker']}!")
            risk_levels.append("Unknown Risk")
    return {"Total Portfolio Value": total_value, "Risk Levels": risk_levels}

# 🔹 Optimize LSTM Hyperparameters with Optuna
def optimize_lstm(trial):
    """Optimize LSTM hyperparameters using Optuna."""
    units = trial.suggest_int("units", 10, 100)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    epochs = trial.suggest_int("epochs", 5, 20)

    model = Sequential([
        Input(shape=(60, 1)),  # ✅ Use explicit Input layer
        LSTM(units, return_sequences=True),
        Dropout(dropout),
        LSTM(units, return_sequences=False),
        Dropout(dropout),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")

    X_dummy = np.random.rand(100, 60, 1)
    y_dummy = np.random.rand(100, 1)

    history = model.fit(X_dummy, y_dummy, batch_size=32, epochs=epochs, verbose=0)
    return history.history["loss"][-1]

# 🔹 Train LSTM Model
def train_model(ticker):
    """Trains LSTM model on real stock price data."""
    df = fetch_real_time_stock_data(ticker)
    if df.empty:
        return None, None, df

    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df[["Close"]])

    X, y = [], []
    lookback = 60
    for i in range(lookback, len(df_scaled)):
        X.append(df_scaled[i - lookback:i, 0])
        y.append(df_scaled[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    study = optuna.create_study(direction="minimize")
    study.optimize(optimize_lstm, n_trials=10)

    best_params = study.best_params if len(study.trials) > 0 else {"units": 50, "dropout": 0.2, "epochs": 10}

    model = Sequential([
        Input(shape=(60, 1)),
        LSTM(best_params["units"], return_sequences=True),
        Dropout(best_params["dropout"]),
        LSTM(best_params["units"], return_sequences=False),
        Dropout(best_params["dropout"]),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X, y, batch_size=32, epochs=best_params["epochs"], verbose=1)

    return model, scaler, df

# 🔹 Execute Trade Using Alpaca API
def execute_trade(ticker, action):
    """Places a buy or sell order on Alpaca."""
    alpaca_api.submit_order(symbol=ticker, qty=1, side=action.lower(), type="market", time_in_force="gtc")
    print(f"✅ Order placed: {action} 1 share of {ticker}")

# 🔹 Run Model on a Stock
ticker = "TSLA"
market_sentiment = get_market_sentiment()
print(f"📊 Market Sentiment Score: {market_sentiment}")

model, scaler, df = train_model(ticker)
if df.empty:
    print(f"⚠️ No stock data available for {ticker}")
else:
    execute_trade(ticker, "Buy" if market_sentiment > 0 else "Sell")