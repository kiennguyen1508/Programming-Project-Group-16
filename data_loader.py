"""
Shared data for our group dashboard.

Run once to download: python data_loader.py
Then in any panel: from data_loader import get_prices, get_returns, SPLIT
"""

from pathlib import Path

import numpy as np
import pandas as pd

TICKERS = {"Gold": "GC=F", "Silver": "SI=F", "Oil": "BZ=F", "S&P 500": "^GSPC"}
START = "2015-01-01"
END = "2026-11-01" # Since yfinance is not always up to date, we set the end date to a future date to ensure we get the latest data.
SPLIT = "2025-01-01" # "before" vs "since 2025"

DATA_DIR = Path(__file__).parent / "data"
CSV = DATA_DIR / "prices.csv"

def download():
    """
    Download daily closing prices from Yahoo Finance and save them to data/prices.csv.
    """
    import yfinance as yf

    raw = yf.download(list(TICKERS.values()), start = START , end = END, 
                      auto_adjust = True , progress = False ) ["Close"]

    prices = raw.rename(columns = {v: k for k , v in TICKERS.items()})
    prices = prices[list(TICKERS)] # fixed column order
    prices = prices.dropna() # keep only days when all four traded
    if prices.empty:
        raise ValueError("No data download. Check the tickers and check your internet connect")
    prices.index.name = "Date"

    DATA_DIR.mkdir(exist_ok = True)
    prices.round(4).to_csv(CSV)
    print(f"Saved {len(prices)} rows to {CSV}")
    print(f"From {prices.index[0].date()} to {prices.index[-1].date()}")
    print(prices.tail())

if __name__ == "__main__":
    download()