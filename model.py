import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib

# Load trained LSTM model and scaler
MODEL_PATH = "financial_risk_model.h5"
SCALER_PATH = "scaler.pkl"

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Configuration
STOCK_SYMBOL = "AAPL"  # Change this for different stocks
SEQ_LENGTH = 60  # Same as in train.py


# Fetch latest stock data
def fetch_latest_stock_data(stock_symbol, period="90d"):
    stock = yf.Ticker(stock_symbol)
    df = stock.history(period=period)
    return df[['Close']]  # Use only closing prices


# Prepare data for prediction
def prepare_data_for_prediction(df):
    scaled_data = scaler.transform(df)  # Normalize data

    X_input = []
    X_input.append(scaled_data[-SEQ_LENGTH:])  # Use last 60 days
    X_input = np.array(X_input)
    X_input = np.reshape(X_input, (X_input.shape[0], X_input.shape[1], 1))

    return X_input


# Predict future stock price
def predict_stock_price(stock_symbol):
    print(f"Fetching latest stock data for {stock_symbol}...")
    df = fetch_latest_stock_data(stock_symbol)

    print("Preparing data for prediction...")
    X_input = prepare_data_for_prediction(df)

    print("Predicting stock price...")
    predicted_price = model.predict(X_input)

    # Inverse transform to get actual price
    predicted_price_actual = scaler.inverse_transform(predicted_price.reshape(-1, 1))

    return predicted_price_actual[0][0]


# Risk Assessment Logic
def assess_risk(current_price, predicted_price):
    change = ((predicted_price - current_price) / current_price) * 100

    if change > 5:
        return "Low Risk (Expected Growth)"
    elif -5 <= change <= 5:
        return "Medium Risk (Stable)"
    else:
        return "High Risk (Possible Decline)"


# Run prediction and risk assessment
if __name__ == "__main__":
    latest_price = fetch_latest_stock_data(STOCK_SYMBOL).iloc[-1, 0]
    predicted_price = predict_stock_price(STOCK_SYMBOL)

    print(f"📈 Latest Price: ${latest_price:.2f}")
    print(f"🔮 Predicted Price: ${predicted_price:.2f}")

    risk_level = assess_risk(latest_price, predicted_price)
    print(f"⚠️ Risk Assessment: {risk_level}")
