import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import matplotlib.pyplot as plt
import datetime

# ------------------------ 🛠️ Load Model & Scaler ------------------------
MODEL_PATH = "financial_risk_model.keras"
SCALER_PATH = "scaler.pkl"

try:
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    model_loaded = True
except Exception as e:
    st.error(f"🚨 Error loading model or scaler: {e}")
    model_loaded = False

# ------------------------ 🎨 Streamlit Page Config ------------------------
st.set_page_config(page_title="Financial Risk AI", layout="wide")
st.title("📊 Financial Risk Assessment & AI-Driven Decision Making")

# Sidebar Inputs
st.sidebar.header("📈 Stock Prediction Settings")
stock_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL, TSLA)", "AAPL").upper()
prediction_days = st.sidebar.slider("🔮 Days to Predict Ahead", 1, 30, 7)


# ------------------------ 📊 Function: Fetch Stock Data ------------------------
def fetch_stock_data(stock_symbol, period="5y"):
    try:
        stock = yf.Ticker(stock_symbol)
        df = stock.history(period=period)
        return df[['Close']]
    except Exception as e:
        st.error(f"⚠️ Error fetching stock data: {e}")
        return None


# ------------------------ 🔢 Function: Prepare Data for LSTM ------------------------
def prepare_data_for_prediction(df, seq_length=60):
    if df is None or len(df) < seq_length:
        st.warning("⚠️ Not enough data to make predictions.")
        return None
    scaled_data = scaler.transform(df)
    X_input = np.array([scaled_data[-seq_length:]])
    X_input = np.reshape(X_input, (X_input.shape[0], X_input.shape[1], 1))
    return X_input


# ------------------------ 🤖 Function: Predict Stock Price ------------------------
def predict_stock_price(stock_symbol, days=prediction_days):
    df = fetch_stock_data(stock_symbol, "90d")
    X_input = prepare_data_for_prediction(df)
    if X_input is None:
        return None

    predictions = []
    for _ in range(days):
        predicted_price = model.predict(X_input)
        predicted_price_actual = scaler.inverse_transform(predicted_price.reshape(-1, 1))
        predictions.append(predicted_price_actual[0][0])
        X_input = np.roll(X_input, shift=-1, axis=1)  # Shift window
        X_input[0, -1, 0] = predicted_price[0, 0]  # Append new prediction

    return predictions


# ------------------------ ⚠️ Function: Assess Financial Risk ------------------------
def assess_risk(current_price, predicted_price):
    change = ((predicted_price - current_price) / current_price) * 100
    if change > 5:
        return "🟢 Low Risk (Expected Growth)"
    elif -5 <= change <= 5:
        return "🟡 Medium Risk (Stable)"
    else:
        return "🔴 High Risk (Possible Decline)"


# ------------------------ 🚀 Run Predictions & Display Results ------------------------
if model_loaded:
    df = fetch_stock_data(stock_symbol)

    if df is not None:
        latest_price = df.iloc[-1, 0]
        predicted_prices = predict_stock_price(stock_symbol)

        if predicted_prices:
            predicted_price = predicted_prices[-1]  # Last day’s predicted price
            risk_level = assess_risk(latest_price, predicted_price)

            # Display Metrics
            st.subheader(f"📌 {stock_symbol} Stock Analysis")
            st.metric("📈 Latest Stock Price", f"${latest_price:.2f}")
            st.metric("🔮 Predicted Price (in 7 Days)", f"${predicted_price:.2f}")
            st.metric("⚠️ Risk Level", risk_level)

            # Display Future Predictions
            future_dates = [df.index[-1] + datetime.timedelta(days=i) for i in range(1, prediction_days + 1)]
            pred_df = pd.DataFrame({"Date": future_dates, "Predicted Price": predicted_prices})
            pred_df.set_index("Date", inplace=True)

            st.subheader(f"📉 Predicted Stock Prices for {stock_symbol} (Next {prediction_days} Days)")
            st.line_chart(pred_df)

            # Stock Price Trend Chart
            st.subheader("📊 Stock Price Trend (Last 5 Years)")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df.index, df["Close"], label="Stock Price", color="blue")
            ax.axhline(y=latest_price, color="red", linestyle="--", label="Current Price")
            ax.legend()
            st.pyplot(fig)

            # Investment Advice
            st.subheader("📌 Investment Decision")
            if risk_level == "🟢 Low Risk (Expected Growth)":
                st.success("✅ This stock shows **growth potential**. Consider **investing.**")
            elif risk_level == "🟡 Medium Risk (Stable)":
                st.warning("⚠️ The stock is relatively stable. Consider holding or short-term trading.")
            else:
                st.error("❌ High risk detected! Consider **avoiding or short-selling.**")

st.sidebar.write("Developed by **Aman Jain** 🚀")
