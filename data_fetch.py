import yfinance as yf
import pandas as pd

# Function to fetch stock data
def fetch_stock_data(stock_symbol, period="5y"):
    stock = yf.Ticker(stock_symbol)
    df = stock.history(period=period)
    return df[['Close']]

# Test function
if __name__ == "__main__":
    stock_symbol = "AAPL"
    df = fetch_stock_data(stock_symbol, "90d")
    print(df.tail())
