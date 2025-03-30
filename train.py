import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import joblib

# Configuration
STOCK_SYMBOL = "AAPL"  # Change to any stock ticker
SEQ_LENGTH = 60  # Number of days to look back for predictions
EPOCHS = 20
BATCH_SIZE = 32


# Fetch stock data
def fetch_stock_data(stock_symbol, period="5y"):
    stock = yf.Ticker(stock_symbol)
    df = stock.history(period=period)
    return df[['Close']]  # Use only closing price for prediction


# Preprocess Data
def preprocess_data(data):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Creating sequences for LSTM
    X, y = [], []
    for i in range(len(data_scaled) - SEQ_LENGTH):
        X.append(data_scaled[i:i + SEQ_LENGTH])
        y.append(data_scaled[i + SEQ_LENGTH])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape for LSTM

    return X, y, scaler


# Define LSTM Model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# Main Training Process
def train_model():
    print("Fetching stock data...")
    df = fetch_stock_data(STOCK_SYMBOL)

    print("Preprocessing data...")
    X, y, scaler = preprocess_data(df)

    print("Building LSTM model...")
    model = build_lstm_model((X.shape[1], 1))

    print("Training model...")
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # Save Model & Scaler
    model.save("financial_risk_model.keras")
    joblib.dump(scaler, "scaler.pkl")

    print("✅ Model and scaler saved successfully!")


if __name__ == "__main__":
    train_model()
