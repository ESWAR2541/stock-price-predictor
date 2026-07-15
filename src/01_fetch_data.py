"""
STEP 1: FETCH LIVE STOCK DATA
================================
Uses yfinance - a free Python library that pulls real, current data
directly from Yahoo Finance. Every time you run this, it fetches the
LATEST available data, not a stale snapshot - this is what makes the
project "real-time" rather than a one-time static dataset.

NOTE: This script needs internet access to run - it will only work on
your own computer, not in a sandboxed/offline environment.
"""

import yfinance as yf
import pandas as pd

TICKER = "AAPL"
PERIOD = "5y"  # 5 years of daily history - enough data to train on,
                # while still reflecting relatively recent market behavior

print(f"Fetching live data for {TICKER}...")
data = yf.download(TICKER, period=PERIOD, interval="1d", auto_adjust=True)

# yfinance returns a MultiIndex column structure sometimes - flatten it
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

data = data.reset_index()

print(f"\nFetched {len(data)} trading days")
print(f"Date range: {data['Date'].min().date()} to {data['Date'].max().date()}")
print(f"\nMost recent close price: ${data['Close'].iloc[-1]:.2f}")
print(data.tail())

data.to_csv("data/stock_data.csv", index=False)
print("\nSaved -> data/stock_data.csv")
print("\nRun this script again anytime to refresh with the latest trading data.")
