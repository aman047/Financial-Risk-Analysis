import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from data_fetch import fetch_stock_data
from preprocess import prepare_data_for_testing

# Constants
MODEL_PATH = "financial_risk_model.keras"  # Updated to .keras format
SCALER_PATH = "scaler.pkl"
STOCK_SYMBOL = "AAPL"  # Example stock symbol
TEST_DAYS = "90d"  # Fetch past 90 days of data
SEQ_LENGTH = 60  # Must match training sequence length

# Load trained model and scaler
try:
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Model and scaler loaded successfully!")
except Exception as e:
    print(f"🚨 Error loading model or scaler: {e}")
    exit()

# Fetch stock data
df = fetch_stock_data(STOCK_SYMBOL, TEST_DAYS)
print(f"📊 Data for {STOCK_SYMBOL} loaded. Shape: {df.shape}")

# Validate DataFrame
if df.shape[0] < SEQ_LENGTH:
    print(f"🚨 Not enough data for testing! Required: {SEQ_LENGTH}, Found: {df.shape[0]}")
    exit()

# Preprocess data
X_test, y_test = prepare_data_for_testing(df, scaler)

# Validate X_test shape
if X_test.shape[0] == 0:
    print("🚨 Error: X_test is empty after preprocessing! Check data and sequence length.")
    exit()

# Make predictions
y_pred = model.predict(X_test)

# Convert back to actual stock prices
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_actual = scaler.inverse_transform(y_pred.reshape(-1, 1))

# Evaluate performance
mae = mean_absolute_error(y_test_actual, y_pred_actual)
mse = mean_squared_error(y_test_actual, y_pred_actual)
rmse = np.sqrt(mse)

print(f"\n📊 Model Evaluation for {STOCK_SYMBOL}")
print(f"✅ Mean Absolute Error (MAE): {mae:.4f}")
print(f"✅ Mean Squared Error (MSE): {mse:.4f}")
print(f"✅ Root Mean Squared Error (RMSE): {rmse:.4f}")
