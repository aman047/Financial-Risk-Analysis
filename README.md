# 📊 Financial Risk Assessment & AI Decision-Making

## 🚀 Overview
This project uses **deep learning** to predict stock prices and assess financial risk using **LSTM neural networks**. The model is trained on historical stock data fetched from **Yahoo Finance** and deployed in an interactive **Streamlit web app**.

## 🏗️ Project Structure
```
Financial_risk_ai_project/
│── main.py           # Streamlit web app for risk assessment & visualization
│── train.py          # Train LSTM model and save it as .keras file
│── test.py           # Evaluate trained model on test data
│── model.py          # Define and build LSTM model
│── data_fetch.py     # Fetch stock data from Yahoo Finance
│── preprocess.py     # Preprocess data (scaling, splitting, reshaping)
│── utils.py          # Helper functions for logging, metrics, etc.
│── requirements.txt  # Python dependencies
│── scaler.pkl        # Pre-trained MinMaxScaler for input normalization
│── financial_risk_model.keras  # Saved trained model
```

## ⚡ Features
✅ **Stock Price Prediction**: Predict future prices for up to 30 days ahead.
✅ **Financial Risk Assessment**: Classifies risk levels as **low, medium, or high**.
✅ **LSTM-Based AI Model**: Uses deep learning (LSTM) for better time-series forecasting.
✅ **Interactive Web Dashboard**: Built with **Streamlit** for real-time user interaction.
✅ **Historical Data Visualization**: Fetches and visualizes stock trends from Yahoo Finance.
✅ **Investment Insights**: Provides recommendations on buying, holding, or selling stocks.

## 📦 Installation
Clone this repository and install dependencies:
```sh
git clone https://github.com/your-repo/financial-risk-ai.git
cd financial-risk-ai
pip install -r requirements.txt
```

## 🚀 Usage
1️⃣ **Train the Model:**
```sh
python train.py
```
2️⃣ **Evaluate Model:**
```sh
python test.py
```
3️⃣ **Run Web App:**
```sh
streamlit run main.py
```

## 📊 How It Works
1. Fetches **historical stock data** using Yahoo Finance.
2. Preprocesses data (**scaling, reshaping**) for LSTM model.
3. Trains **LSTM model** on stock price time-series data.
4. Uses trained model to **predict future stock prices**.
5. Assesses risk based on **predicted vs. current price changes**.
6. Displays insights using **interactive graphs & risk levels** in Streamlit.

## 🔮 Future Improvements
🚀 Add **real-time stock news sentiment analysis**.
📈 Enhance **model accuracy** with hybrid deep learning techniques.
📊 Implement **multi-stock portfolio risk assessment**.

## 🤖 Technologies Used
- **Python** (Data Processing & AI Model)
- **TensorFlow/Keras** (LSTM Model)
- **Yahoo Finance API** (Stock Data)
- **Streamlit** (Web App)
- **Matplotlib & Seaborn** (Data Visualization)

## 👨‍💻 Developed By
**Aman Jain** 🚀

