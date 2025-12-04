import yfinance as yf
from datetime import datetime
import pandas as pd
import numpy as np

def fetch_process_stock_data(ticker):
    """
    Fetches and process stock data into returns
    Returns: Dictionary with all necessary data
    :param ticker:
    :return:
    """

    #Set up stock
    stock = yf.Ticker(ticker)

    #Set date range to 5 years before current trading day
    endDate = pd.to_datetime('today')
    startDate = endDate - pd.DateOffset(years=5)
    historical_data = stock.history(start=startDate, end=endDate)

    # Checks if data is empty
    if historical_data.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    #Returns and prices
    close_dict = {}
    close_dict[ticker] = historical_data['Close']
    close_prices = pd.DataFrame(close_dict)

    #Calculate Returns
    returns = close_prices.pct_change().dropna()

    #Declare prices variable that matches length to returns
    prices = close_prices.iloc[1:]

    train_size = int(0.75 * len(returns))

    # For returns data split
    train_returns = returns[:train_size]
    test_returns = returns[train_size:]

    # For prices data split
    train_prices = prices[:train_size]
    test_prices = prices[train_size:]

    return {
        'ticker': ticker,
        'raw_data': historical_data,
        'prices': prices,
        'returns': returns,
        'train_prices': train_prices,
        'test_prices': test_prices,
        'train_returns': train_returns,
        'test_returns': test_returns,
        'split_index': train_size
    }
